"""Opt-in measurements must not perturb SAC, its sample identities, or frozen state."""

from copy import deepcopy

import pytest
import torch

from RL.tdmpc2_core.horizon_diagnostics import horizon_sums
from RL.tdmpc2_core.inner_trace import InnerActionTrace, metric_catalog
from tests.test_ambi_latency_contract import _pool_snapshot, _clone_tree, _assert_tree_equal
from tests.test_ambi_root_local_sac import _tiny_model
from tests.test_ambi_horizon_config import _build_cfg, ALGORITHM
from utils.eval_series_data import planner_identity
from utils.resume_identity import scientific_trial_parameters


def snapshot(agent):
    result = _pool_snapshot(agent)
    pool = agent.inner_engine._action_pool
    result.update(log_alpha=_clone_tree(pool.log_alpha),
                  temperature_optim=_clone_tree(pool.temperature_optim.state_dict()),
                  outer=_clone_tree(agent.model.state_dict()),
                  rng=_clone_tree(agent.inner_engine.rng.training_state_dict()))
    return result


def test_hand_reductions_and_absent_horizons():
    h = torch.tensor([[3], [1], [3]])
    values = torch.tensor([[[2.], [6.], [8.]], [[4.], [2.], [10.]]], requires_grad=True)
    target = torch.tensor([[1.], [5.], [12.]])
    metrics = horizon_sums('critic', h, 3, predicted_q=values.mean(0), target_q=target,
                           td_error_abs=(values-target).abs().mean(0))
    expected = {1: (1, 2, 4, 5), 2: (0, 0, 0, 0), 3: (2, 5, 12, 13)}
    for step, row in expected.items():
        for suffix, value in zip(('sample_count', 'td_error_abs_sum', 'predicted_q_sum', 'target_q_sum'), row):
            actual = metrics[f'critic_horizon_{step}_{suffix}']
            assert actual.item() == value and not actual.requires_grad
    actor = horizon_sums('actor', h, 3, entropy=torch.tensor([[2.], [-1.], [3.]]),
                         objective_q=torch.tensor([[4.], [1.], [6.]]))
    assert actor['actor_horizon_3_entropy_sum'] == 5
    assert actor['actor_horizon_3_objective_q_sum'] == 10
    assert actor['actor_horizon_2_sample_count'] == 0
    catalog = metric_catalog(metrics | actor)
    assert catalog['critic_horizon_3_target_q_sum']['preferred_axis'] == 'critic_updates'
    assert catalog['actor_horizon_3_entropy_sum']['preferred_axis'] == 'actor_updates'
    assert all(catalog[k]['sampling_phase'] == 'pre_update_minibatch' for k in metrics | actor)


@pytest.mark.parametrize('value', [None, 0, 1, 'true'])
def test_flag_is_strict_boolean(value):
    with pytest.raises((TypeError, ValueError), match='inner_horizon_diagnostics'):
        _build_cfg(inner_horizon_diagnostics=value)


def test_diagnostic_flag_is_operational_only():
    configs = [vars(_build_cfg(inner_finite_horizon=True, inner_horizon_diagnostics=flag))
               for flag in (False, True)]
    assert planner_identity(configs[0], {}, ALGORITHM, "tanh_mean") == planner_identity(configs[1], {}, ALGORITHM, "tanh_mean")
    assert scientific_trial_parameters(dict(alg=ALGORITHM, alg_params=configs[0])) == scientific_trial_parameters(dict(alg=ALGORITHM, alg_params=configs[1]))
    assert _build_cfg().inner_horizon_diagnostics is False


@pytest.mark.parametrize('conditioning', ['none', 'one_hot'])
@pytest.mark.parametrize('horizon', [1, 3])
@pytest.mark.parametrize('timing', ['round', 'step'])
def test_enabled_disabled_exact_learning_rng_and_frozen_state(conditioning, horizon, timing, monkeypatch):
    options = dict(inner_horizon_conditioning=conditioning, inner_finite_horizon=True,
                   inner_rollout_horizon=horizon, train_unroll_horizon=3,
                   inner_rounds=2, inner_updates_per_round=2, inner_replay_capacity=24,
                   dropout=.2, num_q=3, q_representation='distributional', aux_return_mode='sac', inner_terminal_entropy='outer',
                   inner_update_timing=timing)
    if timing == 'step':
        options.update(inner_steps_per_update=2, inner_updates_per_round=None)
    ordinary = _tiny_model(**options)
    observed = _tiny_model(**options, inner_horizon_diagnostics=True)
    sampled = [[], []]
    for i, model in enumerate((ordinary, observed)):
        engine = model.agent.inner_engine
        original = engine._sample_batch
        def record(indices=None, original=original, log=sampled[i]):
            log.append(indices.clone())
            return original(indices)
        monkeypatch.setattr(engine, "_sample_batch", record)
    before = deepcopy(observed.agent.checkpoint_state())
    global_rng = torch.random.get_rng_state().clone()
    try:
        for decision in range(2):
            traces = [InnerActionTrace(), InnerActionTrace()]
            actions = [model.agent.act(torch.full((3,), decision*.1), t0=decision == 0,
                                       collect_diagnostics=False, trace=trace)
                       for model, trace in zip((ordinary, observed), traces)]
            torch.testing.assert_close(*actions, rtol=0, atol=0)
            a, b = snapshot(ordinary.agent), snapshot(observed.agent)
            if conditioning == 'none':
                assert 'remaining_horizon' not in a['replay']
                b['replay'].pop('remaining_horizon')
                b['replay'].pop('remaining_horizon_max')
            _assert_tree_equal(a, b)
            for component in ('actor', 'critic'):
                pa = getattr(ordinary.agent.inner_engine._action_pool, component)
                pb = getattr(observed.agent.inner_engine._action_pool, component)
                for x, y in zip(pa.parameters(), pb.parameters()):
                    if x.grad is not None:
                        torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
            for x, y in zip(traces[0].events, traces[1].events):
                assert x['phase'] == y['phase']
                extra = {k:v for k,v in y['metrics'].items() if '_horizon_' in k}
                common = {k:v for k,v in y['metrics'].items() if k not in extra}
                # Timing fields are observational; all learning/probe values are exact.
                assert {k:v for k,v in x['metrics'].items() if 'seconds' not in k} == {k:v for k,v in common.items() if 'seconds' not in k}
                for component in ('critic', 'actor'):
                    if y.get('updated_'+component):
                        assert sum(extra[f'{component}_horizon_{h}_sample_count'] for h in range(1,horizon+1)) == observed.cfg.inner_batch_size
            replay = observed.agent.inner_engine._action_pool.replay
            assert replay.size == 2*2*horizon
            assert set(replay.remaining_horizon[:replay.size].flatten().tolist()) == set(range(1,horizon+1))
        _assert_tree_equal(sampled[0], sampled[1])
        _assert_tree_equal(observed.agent.checkpoint_state(), before)
        torch.testing.assert_close(global_rng, torch.random.get_rng_state(), rtol=0, atol=0)
    finally:
        ordinary.env.close(); observed.env.close()


@pytest.mark.parametrize('scaled_entropy', [False, True])
@pytest.mark.parametrize('scale_mode', ['none', 'per_action', 'per_update'])
def test_actor_payload_uses_existing_objective_and_exact_entropy(scaled_entropy, scale_mode):
    model = _tiny_model(inner_finite_horizon=True, inner_horizon_diagnostics=True,
                        inner_actor_entropy_mode='tdmpc2_scaled' if scaled_entropy else 'squashed',
                        ent_coef=.2, inner_temperature_mode='inherit_outer',
                        sac_actor_loss_scale_mode='none' if scale_mode == 'none' else 'tdmpc2_percentile_range',
                        inner_actor_loss_scale_update='per_update' if scale_mode == 'per_update' else 'per_action')
    engine = model.agent.inner_engine
    try:
        with engine.rng.fork('initialization'): engine._prepare_workspace(t0=True)
        z = torch.zeros(4, model.cfg.latent_dim)
        scale = None if scale_mode == 'none' else torch.tensor([2.])
        outputs = engine._sac_actor_kernel(z, torch.tensor(.2), torch.zeros(4,model.cfg.action_dim), None, True, q_scale=scale)
        objective_q = outputs[-2] if scale_mode == 'per_update' else outputs[-1]
        entropy = outputs[8] if scaled_entropy else outputs[1]
        torch.testing.assert_close(outputs[2], (-objective_q - .2*entropy).mean())
        assert not objective_q.requires_grad
        expected = 8 + 2 + int(scaled_entropy) + 1 + int(scale_mode == 'per_update')
        assert len(outputs) == expected
        # Exercise eager parsing with simultaneous scale, entropy and sample payloads.
        metrics = engine._sac_policy_step({'z':z, 'remaining_horizon':torch.ones(4,1)},
            update_actor=True, update_temperature=False, alpha=torch.tensor(.2), actor_loss_scale=scale)
        assert metrics['actor_horizon_1_sample_count'] == 4
    finally:
        model.env.close()


def test_auxiliary_conditioned_checkpoint_and_resume_preflight():
    options = dict(aux_return_mode='sac', inner_finite_horizon=True, inner_terminal_entropy='outer',
                   inner_horizon_conditioning='one_hot', inner_horizon_diagnostics=True)
    source, restored = _tiny_model(**options), _tiny_model(**options)
    try:
        before = deepcopy(source.agent.checkpoint_state())
        restored.agent.load(before)
        source.agent.act(torch.zeros(3), collect_diagnostics=False)
        _assert_tree_equal(before, source.agent.checkpoint_state())
        source.agent.prepare_training_resume_boundary()
        state = deepcopy(source.agent.training_state_dict())
        assert state['inner']['version'] == 7 and 'control_sources' in state['inner']
        restored.agent.load_training_state_dict(state)
        bad = deepcopy(state); bad['inner']['control_sources']['inner_critic_source'] = 'aux_return'
        target_before = deepcopy(restored.agent.training_state_dict())
        with pytest.raises(ValueError): restored.agent.load_training_state_dict(bad)
        _assert_tree_equal(target_before, restored.agent.training_state_dict())
        source.agent.reset(); restored.agent.reset()
        torch.testing.assert_close(source.agent.act(torch.zeros(3)), restored.agent.act(torch.zeros(3)), rtol=0, atol=0)
    finally:
        source.env.close(); restored.env.close()


@pytest.mark.parametrize("conditioning", ["none", "one_hot"])
@pytest.mark.parametrize("source", ["inner_actor_source", "inner_critic_source", "inner_horizon_actor_source", "inner_horizon_critic_source"])
def test_auxiliary_backbone_support_is_soft_sources_only(conditioning, source):
    params = dict(aux_return_mode="sac", inner_finite_horizon=True,
                  inner_horizon_conditioning=conditioning, inner_horizon_diagnostics=True)
    assert _build_cfg(**params).aux_return_mode == "sac"
    params[source] = "return_actor" if "actor" in source else "aux_return"
    with pytest.raises(ValueError): _build_cfg(**params)
