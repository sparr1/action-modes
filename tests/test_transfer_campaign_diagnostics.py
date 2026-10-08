"""Sampled campaign metrics must be usable without changing the controller."""
from copy import deepcopy
import math

import numpy as np
import pytest
import torch

from tests.test_ambi_inner_decoupling import _assert_tree_equal, _clone_tree
from tests.test_ambi_root_local_sac import _model_from_params
from tests.test_aux_critic_transfer import critic_params
from utils.transfer_campaign import evaluate_episode
from utils.transfer_campaign_diagnostics import (
    CampaignDiagnostics, diagnostic_settings, relative_feature_metrics,
    verify_episode_diagnostics, verify_observational_isolation, OBSERVATIONAL_TIMING_METRICS,
)


SETTINGS = dict(decisions=[0, 1, 25], stationary_decisions=[25], mc_rollouts=2,
    action_count=4, state_count=8, fit_steps=2)


def test_feature_metrics_report_sample_ceiling_and_are_scale_relative():
    features = torch.tensor([[1., 0., .001], [2., 0., .002], [-1., 0., -.001], [-2., 0., -.002]])
    metrics = relative_feature_metrics(features)
    scaled = relative_feature_metrics(features * 1000.)
    assert metrics['rank_ceiling'] == 3
    assert metrics['effective_rank'] == pytest.approx(1.)
    assert metrics['effective_rank_fraction'] == pytest.approx(1 / 3)
    assert metrics['relative_low_activity_fraction'] == pytest.approx(2 / 3)
    for key in metrics:
        assert scaled[key] == pytest.approx(metrics[key], abs=1e-8)


@pytest.mark.parametrize('component', ['actor', 'critic', 'joint'])
@pytest.mark.parametrize('metric_policy', ['legacy', 'all_scalars'])
def test_sampled_diagnostics_preserve_bernoulli_trajectory_learner_and_rng(component, metric_policy):
    wrapped = _model_from_params(critic_params('return', inner_critic_scope='action',
        inner_rounds=1, inner_rollout_horizon=2))
    arm = {}
    if component in ('actor', 'joint'):
        arm['actor_bernoulli_p'] = .5
    if component in ('critic', 'joint'):
        arm['critic_bernoulli_p'] = .5
    try:
        backbone = _clone_tree(wrapped.agent.model.state_dict())
        records = []
        outputs = []
        rngs = []
        final_states = []
        for enabled in (False, True):
            rows = []
            diagnostic = CampaignDiagnostics(SETTINGS, episode_seed=101, controller_seed=55, smoke=True) if enabled else None
            before = torch.get_rng_state().clone()
            result = evaluate_episode(wrapped, wrapped.env, arm, episode_seed=101,
                controller_seed=55, max_steps=3, on_step=rows.append, smoke=True, diagnostics=diagnostic,
                metric_policy=metric_policy)
            torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
            outputs.append(result)
            records.append(rows)
            rngs.append(deepcopy(wrapped.agent.inner_engine.rng.training_state_dict()))
            final_states.append(wrapped.agent.inner_engine.export_diagnostic_state())
        for left, right in zip(*records):
            np.testing.assert_array_equal(left['action'], right['action'])
            assert left['reward'] == right['reward']
            if metric_policy == 'legacy':
                assert left['metrics'] == right['metrics']
            else:
                assert 'inner_action_seconds' in left['metrics'] and 'inner_action_seconds' in right['metrics']
        verify_observational_isolation(records[0], records[1],
            dict(learner=final_states[0], rng=rngs[0]), dict(learner=final_states[1], rng=rngs[1]))
        _assert_tree_equal(rngs[0], rngs[1])
        _assert_tree_equal(final_states[0], final_states[1])
        _assert_tree_equal(backbone, wrapped.agent.model.state_dict())
        episode = outputs[1]
        verify_episode_diagnostics(episode, SETTINGS, smoke=True)
        assert episode['diagnostics']['samples'] == 2
        assert episode['diagnostics']['completed_stationary_decisions'] == [1]
        assert episode['diagnostic_seconds'] > 0
        assert records[1][2]['diagnostic_seconds'] == 0
        for index, row in enumerate(records[1][:2]):
            diagnostic = row['diagnostics']
            assert set(diagnostic['stages']) == {'prior', 'initial', 'final'} | ({'donor'} if index else set())
            assert diagnostic['reference']['state_count'] == 8
            for stage in diagnostic['stages'].values():
                assert stage['features']['actor']['rank_ceiling'] == 7
                assert math.isfinite(stage['fixed_target']['rmse'])
                assert math.isfinite(stage['action_gradient']['paired_return_gain'])
            assert len(diagnostic['target_sha256']) == 64
        bad = deepcopy(episode)
        bad['diagnostics']['summary'].pop('initial_critic_rmse')
        with pytest.raises(RuntimeError, match='summary'):
            verify_episode_diagnostics(bad, SETTINGS, smoke=True)
        bad = deepcopy(episode)
        bad['diagnostics']['completed_stationary_decisions'] = []
        with pytest.raises(RuntimeError, match='stationary'):
            verify_episode_diagnostics(bad, SETTINGS, smoke=True)
    finally:
        wrapped.close()


@pytest.mark.parametrize('settings', [dict(enabled=False), dict(state_count=7),
    dict(stationary_decisions=[2], decisions=[0,1]), dict(mc_rollouts=1), dict(invented=True)])
def test_invalid_diagnostic_configuration_fails_closed(settings):
    with pytest.raises(ValueError):
        diagnostic_settings(settings)


def test_timing_exceptions_are_exact_engine_timer_keys():
    import ast
    from pathlib import Path
    engine = ast.parse((Path(__file__).resolve().parents[1] / "RL/tdmpc2_core/inner_improvement.py").read_text())
    stops = [node for node in ast.walk(engine) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "_timer_stop"]
    assert all(isinstance(node.args[0], ast.Constant) for node in stops)
    assert {node.args[0].value for node in stops} == OBSERVATIONAL_TIMING_METRICS


def _isolation_fixture():
    rows = [dict(decision=0, action=[.2], reward=1., terminated=False, truncated=False,
                 metrics={"inner_actor_loss": 2., "inner_actor_optimizer_steps": 4.,
                          **{key: .1 for key in OBSERVATIONAL_TIMING_METRICS}})]
    state = dict(learner={"actor": torch.tensor([1.])}, rng={"training": torch.tensor([2])})
    return rows, state


def test_timing_differences_are_ignored_without_removing_saved_metrics():
    before, state = _isolation_fixture()
    after = deepcopy(before)
    after[0]["metrics"].update({key: 9. for key in OBSERVATIONAL_TIMING_METRICS})
    saved_before, saved_after = deepcopy(before), deepcopy(after)
    verify_observational_isolation(before, after, state, deepcopy(state))
    assert before == saved_before and after == saved_after


@pytest.mark.parametrize("change,match", [
    (lambda rows, state: rows[0]["metrics"].update(inner_actor_loss=3.), "metrics"),
    (lambda rows, state: rows[0]["metrics"].update(inner_actor_optimizer_steps=5.), "metrics"),
    (lambda rows, state: rows[0]["metrics"].update(inner_unknown_seconds=3.), "metrics"),
    (lambda rows, state: rows[0].update(action=[.3]), "action"),
    (lambda rows, state: rows[0].update(reward=2.), "reward"),
    (lambda rows, state: state["learner"].update(actor=torch.tensor([3.])), "learner_state.learner"),
    (lambda rows, state: state["rng"].update(training=torch.tensor([3])), "learner_state.rng"),
])
def test_isolation_still_rejects_scientific_metric_action_learner_and_rng_changes(change, match):
    before, state = _isolation_fixture()
    after, after_state = deepcopy(before), deepcopy(state)
    change(after, after_state)
    with pytest.raises(RuntimeError, match=match):
        verify_observational_isolation(before, after, state, after_state)
