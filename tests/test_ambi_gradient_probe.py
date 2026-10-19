"""Directional interventions must distinguish a misleading critic from dynamics."""
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import probe_ambi_gradients as cli
from evaluate_ambi_checkpoint import _outer_state_digest
from tests.test_ambi_real_calibration import AnalyticEnv
from tests.test_ambi_root_local_sac import _tiny_model
from utils.ambi_gradient_probe import (
    PriorRewardObjective, frozen_parameters, gradient_directions, probe_root, summarize_roots, summarize_calibration,
)
from utils.ambi_real_calibration import capture_simulator_snapshot


class QuadraticEnv(AnalyticEnv):
    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)
        return obs, -float((action[0] - .5) ** 2), terminated, truncated, info


class QuadraticObjective:
    discount = .5

    @staticmethod
    def tensor(value):
        return torch.as_tensor(value, dtype=torch.float64)

    @staticmethod
    def actor(observations, noise):
        return np.tanh(.1 * noise)

    @staticmethod
    def real_q(observations, actions):
        return (actions[:, 0] - .5) ** 2

    def scores(self, observation, action, noise, horizon):
        # Deliberately reversed critic slope; exact immediate reward gradient.
        critic = ((action - .5) ** 2).sum()
        reward = -critic.expand(noise.shape[1])
        for step in range(1, horizon):
            reward = reward - self.discount ** step * (torch.tanh(.1 * noise[step, :, 0]) - .5) ** 2
        bootstrap = self.discount ** horizon * (torch.tanh(.1 * noise[horizon, :, 0]) - .5) ** 2
        return dict(critic=critic, reward=reward, bootstrap=bootstrap, total=reward + bootstrap)


def test_paired_real_interventions_detect_wrong_critic_and_keep_tanh_bounds():
    envs = [QuadraticEnv(), QuadraticEnv()]
    envs[0].reset(seed=101)
    snapshot = capture_simulator_snapshot(envs[0])
    noise = np.random.default_rng(7).normal(size=(9, 2, 1))
    result = probe_root(QuadraticObjective(), envs, snapshot, [0.], np.array([.1]),
        horizon=3, gradient_noise=np.ones((4, 7, 1)), evaluation_noise=noise,
        epsilons=[.03, .1], random_directions=1, direction_seed=13, original_remaining_steps=3)
    assert result['gradients']['cosine_action']['model_h1'] == pytest.approx(-1.)
    assert result['work']['simulator_decisions'] == 17 * 2 * 9
    contrasts = {(r['direction'], r['epsilon']): r for r in result['contrasts']}
    assert contrasts['critic', .1]['real_mc_return_gain'] < 0
    assert contrasts['critic', .1]['critic_gain'] > 0
    for name in ('model_h1', 'model_h3'):
        assert contrasts[name, .1]['real_mc_return_gain'] > 0
        assert contrasts[name, .1]['real_mc_return_gain'] == pytest.approx(contrasts[name, .1]['model_return_gain'])
    baseline = [row for row in result['rows'] if row['direction'] == 'baseline']
    for row in result['rows']:
        assert np.max(np.abs(row['action'])) <= 1
        if row['direction'] == 'model_h3' and row['sign'] == 1:
            other = baseline[row['rollout_index']]
            # Same continuation noise at every intervention, including the Q action.
            assert row['endpoint_action'] == other['endpoint_action']
            assert row['real_mc_return'] - other['real_mc_return'] == pytest.approx(
                -(row['action'][0] - .5) ** 2 + (other['action'][0] - .5) ** 2)


def test_unexpected_truncation_cannot_produce_complete_probe():
    env = QuadraticEnv(truncated_at=1)
    env.reset()
    with pytest.raises(ValueError, match='truncation'):
        probe_root(QuadraticObjective(), [env], capture_simulator_snapshot(env), [0.], np.array([.1]),
            horizon=1, gradient_noise=np.zeros((2, 1, 1)), evaluation_noise=np.zeros((3, 1, 1)),
            epsilons=[.1], random_directions=0, direction_seed=1, original_remaining_steps=3)


@pytest.mark.parametrize('representation', ['scalar', 'distributional'])
def test_actual_auxiliary_gradients_match_finite_differences_without_parameter_or_rng_mutation(representation):
    model = _tiny_model(aux_return_mode='sac', q_representation=representation,
                        num_q=2 if representation == 'scalar' else 3)
    try:
        with torch.no_grad():
            for p in model.agent.model._aux_return_Qs.parameters():
                p.normal_(0, .15)
            for p in model.agent.model._reward.parameters():
                p.normal_(0, .15)
        objective = PriorRewardObjective(model, reduction='mean_pair', seed=77)
        assert objective.source == 'aux_return'
        assert objective.critic is model.agent.model._aux_return_Qs
        modes = [m.training for m in model.agent.model.modules()]
        flags = [p.requires_grad for p in model.agent.model.parameters()]
        digest, rng = _outer_state_digest(model), torch.random.get_rng_state().clone()
        noise = objective.tensor(np.random.default_rng(2).normal(size=(4, 8, 1)))
        obs, action = np.array([.2, -.1, .7]), objective.tensor([.15]).requires_grad_(True)
        with frozen_parameters(model.agent.model):
            for horizon in (1, 3):
                scores = objective.scores(obs, action, noise, horizon)
                derivative = torch.autograd.grad(scores['total'].mean(), action)[0]
                epsilon = .003
                finite = (objective.scores(obs, action.detach() + epsilon, noise, horizon)['total'].mean()
                          - objective.scores(obs, action.detach() - epsilon, noise, horizon)['total'].mean()) / (2 * epsilon)
                assert float(derivative) == pytest.approx(float(finite), abs=3e-4, rel=.03)
            gradient_directions(objective, obs, np.array([.15]), noise, 3, random_directions=1, seed=99)
        assert _outer_state_digest(model) == digest
        assert torch.equal(torch.random.get_rng_state(), rng)
        assert [m.training for m in model.agent.model.modules()] == modes
        assert [p.requires_grad for p in model.agent.model.parameters()] == flags
        assert all(p.grad is None for p in model.agent.model.parameters())
    finally:
        model.env.close()


def test_frozen_context_restores_mixed_modes_and_flags_after_error():
    module = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Dropout(.5))
    module[0].eval()
    module[0].weight.requires_grad_(False)
    with pytest.raises(RuntimeError):
        with frozen_parameters(module):
            assert not any(m.training for m in module.modules())
            assert not any(p.requires_grad for p in module.parameters())
            raise RuntimeError('test')
    assert module.training and not module[0].training and module[1].training
    assert not module[0].weight.requires_grad and module[0].bias.requires_grad


def test_reward_semantics_fail_closed_for_soft_q():
    model = _tiny_model(outer_critic_target='entropy_augmented')
    try:
        with pytest.raises(ValueError, match='soft Q'):
            PriorRewardObjective(model, reduction='mean_all', seed=1)
    finally:
        model.env.close()


def test_aggregation_does_not_treat_roots_as_independent_episodes():
    def row(episode, gain):
        return dict(episode_id=episode, contrasts=[dict(direction='critic', epsilon=.1, real_mc_return_gain=gain)])
    summary = summarize_roots([row('a', 1), row('a', 3), row('b', -4)])[0]
    assert summary['means']['real_mc_return_gain'] == -1
    assert summary['roots'] == 3 and summary['source_episodes'] == 2
    assert summary['source_episodes_with_positive_mc_gain'] == 1
    assert summary['mc_gain_se_across_source_episodes'] == pytest.approx(3)
    assert summarize_roots([row('a', 1)])[0]['mc_gain_se_across_source_episodes'] is None


def test_calibration_aggregation_retains_missing_zero_gradient_cosines():
    def row(episode, bias, cosine):
        return dict(episode_id=episode, calibration={'critic_bias': bias},
                    gradients={'cosine_action': {'model_h1': cosine}, 'cosine_pretanh': {'model_h1': None}})
    result = summarize_calibration([row('a', 1, .5), row('a', 3, None), row('b', -4, -.5)])
    assert result['critic_bias']['mean'] == -1
    assert result['critic_bias']['source_episodes'] == 2
    assert result['cosine_action_model_h1']['mean'] == 0
    assert result['cosine_action_model_h1']['roots'] == 2
    assert 'cosine_pretanh_model_h1' not in result


def _args(tmp_path, checkpoint, *extra):
    return cli.build_parser().parse_args(['--checkpoint', str(checkpoint), '--output', str(tmp_path / 'bundle'), *extra])


def test_options_reject_ambiguous_bank_coverage_and_invalid_steps(tmp_path):
    for extra, pattern in [(['--epsilons', 'nan'], 'epsilons'), (['--rollouts', '0'], 'rollouts'),
                           (['--root-bank', 'x', '--seeds', '101'], 'coverage')]:
        with pytest.raises(ValueError, match=pattern):
            cli._validate_options(_args(tmp_path, 'a', *extra))


def test_plan_only_uses_saved_auxiliary_semantics_without_loading_checkpoint(tmp_path, monkeypatch):
    metadata = dict(schema_version=1, trial_run_params=dict(alg='AMBITDMPC2/AMBITDMPC2', env='DMControl-v0',
        alg_params=dict(aux_return_mode='sac', outer_critic_target='entropy_augmented')),
        experiment_params=dict(env_params=dict(task='humanoid-walk', obs='state')),
        checkpoint=dict(kind='periodic', step=400000, episode=1, best_score=None, best_window=1))
    checkpoint = tmp_path / 'missing-weights.pt'
    (tmp_path / 'missing-weights.pt.metadata.json').write_text(json.dumps(metadata))
    monkeypatch.setattr(cli, '_make_env', lambda *a: pytest.fail('plan-only must not construct an environment'))
    plan = cli.run_probe(_args(tmp_path, checkpoint, '--plan-only'))
    assert plan['branch_simulator_decisions_upper_bound'] == 4 * 17 * 4 * 1003
    assert not (tmp_path / 'bundle').exists()


@pytest.mark.skipif(os.environ.get('AMBI_RUN_REAL_DMCONTROL_TESTS') != '1', reason='opt-in real DMControl runtime')
def test_real_humanoid_bundle_and_cross_checkpoint_bank(tmp_path):
    import gymnasium as gym
    import domains  # noqa: F401
    from RL.AMBITDMPC2 import AMBITDMPC2
    from tests.test_ambi_root_local_sac import _tiny_params

    env = gym.make('DMControl-v0', task='humanoid-walk', obs='state')
    params = _tiny_params(aux_return_mode='sac', outer_critic_target='entropy_augmented')
    config = dict(alg='AMBITDMPC2/AMBITDMPC2', env='DMControl-v0', seed=13, device='cpu', total_steps=10, alg_params=params)
    model = AMBITDMPC2('AMBITDMPC2', env, params, config, {})
    checkpoint = tmp_path / 'tiny.pt'
    try:
        # Avoid vacuous zero-gradient integration tests from zero-initialized Q/reward heads.
        with torch.no_grad():
            for p in model.agent.model._aux_return_Qs.parameters():
                p.normal_(0, .1)
            for p in model.agent.model._reward.parameters():
                p.normal_(0, .1)
        model.agent.save(str(checkpoint))
        second = tmp_path / 'second.pt'
        with torch.no_grad():
            next(model.agent.model._pi.parameters()).add_(.05)
        model.agent.save(str(second))
    finally:
        env.close()
    sidecar = dict(schema_version=1, trial_run_params=config,
        experiment_params={'env_params': {'task': 'humanoid-walk', 'obs': 'state'}},
        checkpoint=dict(kind='periodic', step=10, episode=2, best_score=None, best_window=1))
    for path in (checkpoint, second):
        path.with_name(path.name + '.metadata.json').write_text(json.dumps(sidecar))
    extra = ['--horizon', '3', '--tail-steps', '4', '--rollouts', '2', '--gradient-rollouts', '3',
             '--epsilons', '.1', '--random-directions', '0']
    args = _args(tmp_path, checkpoint, '--seeds', '101', '--decisions', '0', '2', *extra)
    first = cli.run_probe(args)
    assert first['complete'] and first['outer_state_unchanged']
    assert first['branch_simulator_decisions'] == 2 * 7 * 2 * 7
    reuse = _args(tmp_path / 'reuse', second, '--root-bank', str(args.output / 'root-bank.json'), *extra)
    other = cli.run_probe(reuse)
    assert other['root_bank_id'] == first['root_bank_id']
    assert other['checkpoint']['sha256'] != first['checkpoint']['sha256']
    assert other['root_bank_checkpoint_sha256'] == first['checkpoint']['sha256']
    assert other['timing']['root_collection_simulator_decisions'] == 0
    for i in range(2):
        a = json.loads((args.output / 'roots' / f'{i:04d}.json').read_text())
        b = json.loads((reuse.output / 'roots' / f'{i:04d}.json').read_text())
        assert a['rows'][0]['action'] == b['rows'][0]['action']
        assert a['snapshot_sha256'] == b['snapshot_sha256']
        assert a['noise_seeds'] == b['noise_seeds']
        assert a['critic_source'] == b['critic_source'] == 'aux_return'
    with pytest.raises(FileExistsError):
        cli.run_probe(args)
    raw = json.loads((args.output / 'root-bank.json').read_text())
    raw['roots'][0]['anchor_pretanh_action'][0] += 1
    bad = tmp_path / 'corrupted.json'
    bad.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='corrupted'):
        cli.load_shared_bank(bad, cli._resolved(SimpleNamespace(trial_run_params=config,
            experiment_params=sidecar['experiment_params'])))
    incompatible = cli._resolved(SimpleNamespace(trial_run_params=config,
        experiment_params=sidecar['experiment_params']))
    incompatible['environment']['params']['action_repeat'] = 3
    with pytest.raises(ValueError, match='incompatible environment'):
        cli.load_shared_bank(args.output / 'root-bank.json', incompatible)
