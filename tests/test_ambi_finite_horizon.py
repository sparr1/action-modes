from copy import deepcopy

import pytest
import torch

from RL.tdmpc2_core.common.latent_buffer import LatentReplayBuffer
from tests.test_ambi_root_local_sac import _tiny_model


def _prepared(**kwargs):
    model = _tiny_model(inner_finite_horizon=True, **kwargs)
    engine = model.agent.inner_engine
    with engine.rng.fork('initialization'):
        engine._prepare_workspace(t0=True)
    return model, engine


@pytest.mark.parametrize('target_mode,interior', [('reward_only', 5.0), ('entropy_augmented', 5.5)])
@pytest.mark.parametrize('representation', ['scalar', 'distributional'])
def test_horizon_hands_off_to_outer_priors_and_true_terminal_masks_both(
    monkeypatch, target_mode, interior, representation,
):
    model, engine = _prepared(inner_sac_critic_target=target_mode, q_representation=representation)
    batch = {
        'z': torch.zeros(4, model.cfg.latent_dim),
        'action': torch.zeros(4, model.cfg.action_dim),
        'reward': torch.full((4, 1), 3.0),
        'next_z': torch.ones(4, model.cfg.latent_dim),
        'terminated': torch.tensor([[0.], [0.], [1.], [1.]]),
        'horizon_end': torch.tensor([[0.], [1.], [0.], [1.]]),
    }
    calls = []
    def policy(z, **kwargs):
        calls.append(kwargs)
        outer = kwargs.get('policy') is None
        return torch.full((len(z), 1), 0.75 if outer else -0.5), {'log_prob': torch.full((len(z), 1), -2.)}
    def outer_q(z, action, **kwargs):
        assert kwargs == {'reduction': model.cfg.mppi_terminal_q_reduction}
        torch.testing.assert_close(action, torch.full_like(action, 0.75))
        return torch.full((len(z), 1), 11.)
    monkeypatch.setattr(engine.model, 'pi', policy)
    monkeypatch.setattr(engine.model, 'Q', outer_q)
    monkeypatch.setattr(engine, '_bootstrap_q', lambda z, action: torch.full((len(z), 1), 5.))
    captured = []
    original = engine.model.critic_loss
    def loss(predictions, target):
        captured.append(target.clone())
        return original(predictions, target)
    monkeypatch.setattr(engine.model, 'critic_loss', loss)
    engine._sac_critic_step(batch, torch.tensor(0.25))
    expected = 3. + model.agent.discount * torch.tensor([[interior], [11.], [0.], [0.]])
    torch.testing.assert_close(captured[0], expected)
    assert calls[0]['policy'] is engine.state.actor
    assert set(calls[1]) == {'noise'}  # Outer log-std semantics, no inner overrides.
    assert all(p.grad is None for p in engine.model._Qs.parameters())


@pytest.mark.parametrize('horizon', [1, 3])
@pytest.mark.parametrize('mode', ['none', 'frozen_random', 'shared_mixture', 'separate_critics', 'adaptive_param_noise'])
def test_all_collectors_label_only_last_depth(monkeypatch, horizon, mode):
    extra = {'inner_param_noise_actor_count': 1} if mode == 'adaptive_param_noise' else {}
    model, engine = _prepared(
        inner_rollout_horizon=horizon, inner_explorer_mode=mode,
        inner_rollouts_per_round=2, inner_replay_capacity=32, **extra,
    )
    root = torch.zeros(1, model.cfg.latent_dim)
    result = engine._collect_round(root)
    replay = engine.state.replay
    assert result['transition_count'] == 2 * horizon
    assert replay.horizon_end[:replay.size].sum() == 2
    assert not replay.terminated[:replay.size].any()
    # Collection order may be population-major; each population ends at H.
    if mode == 'none':
        expected = torch.zeros(horizon, 2, 1)
        expected[-1] = 1
        torch.testing.assert_close(replay.horizon_end[:replay.size], expected.flatten(0, 1))


def test_early_model_terminal_is_not_a_horizon_cutoff(monkeypatch):
    model, engine = _prepared(episodic=True, inner_rollout_horizon=3, inner_replay_capacity=32)
    monkeypatch.setattr(engine.model, 'termination', lambda z: z.new_ones(len(z), 1))
    result = engine._collect_round(torch.zeros(1, model.cfg.latent_dim))
    replay = engine.state.replay
    assert result['transition_count'] == 2
    assert replay.terminated[:replay.size].all()
    assert not replay.horizon_end[:replay.size].any()


@pytest.mark.parametrize('source', [False, True])
def test_horizon_replay_wrap_and_exact_restore(source):
    replay = LatentReplayBuffer(4, 2, 1, 'cpu', store_horizon=True, store_source=source)
    z = torch.arange(12.).reshape(6, 2)
    flags = torch.tensor([[0.], [0.], [1.], [0.], [0.], [1.]])
    replay.add_batch(z, z[:, :1], z[:, :1], z + 1, torch.zeros(6, 1),
                     horizon_end=flags, **({'source': 1} if source else {}))
    restored = LatentReplayBuffer(4, 2, 1, 'cpu', store_horizon=True, store_source=source)
    state = replay.training_state_dict()
    restored.load_training_state_dict(state)
    for key in ('z', 'horizon_end', 'terminated'):
        torch.testing.assert_close(getattr(replay, key), getattr(restored, key))
    batch = restored.sample(4, indices=torch.arange(4))
    torch.testing.assert_close(batch['horizon_end'], flags[batch['sample_ids']])
    corrupt = deepcopy(state)
    corrupt['state']['horizon_end'][0] = 2.
    with pytest.raises(ValueError, match='horizon_end'):
        restored.load_training_state_dict(corrupt)
    torch.testing.assert_close(replay.horizon_end, restored.horizon_end)


@pytest.mark.parametrize('mode', ['none', 'shared_mixture', 'separate_critics'])
def test_finite_horizon_act_updates_without_mutating_outer_priors(mode):
    model = _tiny_model(inner_finite_horizon=True, inner_explorer_mode=mode,
                        inner_rounds=1, inner_updates_per_round=1)
    before = {k: v.clone() for k, v in model.agent.model.state_dict().items()}
    model.agent.act(torch.zeros(3))
    assert model.agent.last_inner_metrics['inner_critic_optimizer_steps'] == 1
    for key, value in model.agent.model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


@pytest.mark.parametrize('mode', ['none', 'shared_mixture', 'separate_critics'])
@pytest.mark.parametrize('representation', ['scalar', 'distributional'])
def test_all_targets_mask_nonfinite_unused_continuations(monkeypatch, mode, representation):
    model, engine = _prepared(
        inner_explorer_mode=mode, inner_mixture_target_estimator='weighted',
        inner_sac_critic_target='entropy_augmented', q_representation=representation,
    )
    try:
        batch = {
            'z': torch.zeros(4, model.cfg.latent_dim),
            'action': torch.zeros(4, model.cfg.action_dim),
            'reward': torch.full((4, 1), 3.),
            'next_z': torch.arange(4.).unsqueeze(1).expand(-1, model.cfg.latent_dim),
            'terminated': torch.tensor([[0.], [0.], [1.], [1.]]),
            'horizon_end': torch.tensor([[0.], [1.], [0.], [1.]]),
        }
        def policy(z, **kwargs):
            action = z.new_zeros(len(z), model.cfg.action_dim)
            return action, {'log_prob': z.new_zeros(len(z), 1), 'pre_tanh_action': action}
        inner = torch.tensor([[5.], [float('nan')], [float('inf')], [float('nan')]])
        prior = torch.tensor([[float('nan')], [11.], [float('nan')], [float('inf')]])
        monkeypatch.setattr(engine.model, 'pi', policy)
        monkeypatch.setattr(engine.model, 'mixture_log_prob', lambda *args: args[0].new_zeros(len(args[0]), 1))
        monkeypatch.setattr(engine, '_bootstrap_q', lambda z, *args, **kwargs: inner[z[:, 0].long()])
        monkeypatch.setattr(engine, '_q_with', lambda z, *args, **kwargs: inner[z[:, 0].long()])
        monkeypatch.setattr(engine.model, 'Q', lambda z, *args, **kwargs: prior[z[:, 0].long()])
        captured = []
        original = engine.model.critic_loss
        def loss(predictions, target):
            captured.append(target.clone())
            return original(predictions, target)
        monkeypatch.setattr(engine.model, 'critic_loss', loss)
        if mode == 'none':
            engine._sac_critic_step(batch, torch.tensor(.25))
        elif mode == 'shared_mixture':
            engine._shared_mixture_critic_step(batch, torch.tensor(.25))
        else:
            engine._separate_critics_step(batch, update_primary=True, update_explorer=True)
        assert len(captured) == (2 if mode == 'separate_critics' else 1)
        expected = 3. + model.agent.discount * torch.tensor([[5.], [11.], [0.], [0.]])
        for target in captured:
            torch.testing.assert_close(target, expected)
        assert all(p.grad is None for p in engine.model.parameters())
    finally:
        model.env.close()
