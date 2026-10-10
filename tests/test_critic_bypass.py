"""Bug-sensitive selector tests, including a real fresh SAC solve on tiny nets."""
from copy import deepcopy
from itertools import combinations
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from evaluate_ambi_checkpoint import _outer_state_digest
from tests.test_ambi_root_local_sac import _model_from_params, _tiny_params
from tests.test_matched_action_audit import AnalyticReference, Critic
from utils.critic_bypass import (
    ARMS, SOLVE_CONTRACT, complete_replay_actions, learned_q_scores,
    model_return_samples, run_bypass_decision, stable_argmax,
    validate_bypass_controller,
)
from utils.matched_action_audit import score_action_bank
from utils.transfer_diagnostics import Reference, solve_fork


def test_fast_h1_score_matches_independent_audit_and_preserves_rng_modes():
    reference = AnalyticReference(noise_scale=.2)
    root, actions = torch.tensor([[0.]]), torch.tensor([[0.], [-.5], [.5]])
    before = torch.random.get_rng_state().clone()
    model_state = deepcopy(reference.model.state_dict())
    expected = score_action_bank(reference, [0.], actions=actions,
        labels=["zero", "negative", "positive"], critics={"q": Critic()},
        seed=19, mc_rollouts=32, horizons=(1,), max_expanded_batch=32)
    actual = model_return_samples(reference, root, actions, seed=19,
                                  mc_rollouts=32, max_expanded_batch=32)
    target = torch.tensor([row["model"]["h1"]["draws"] for row in expected["actions"]]).T
    torch.testing.assert_close(actual, target, atol=0., rtol=0.)
    large = model_return_samples(reference, root, actions, seed=19, mc_rollouts=32)
    assert torch.equal(actual, large)
    assert not torch.equal(actual, model_return_samples(reference, root, actions, seed=20))
    assert torch.equal(torch.random.get_rng_state(), before)
    assert reference.model.training and reference.engine._horizon_actor.training
    assert all(torch.equal(value, model_state[key]) for key, value in reference.model.state_dict().items())


def test_termination_masks_bootstrap_and_transition_is_not_mc_repeated():
    reference = AnalyticReference(noise_scale=.2)
    calls = []
    def transition(z, actions):
        calls.append(len(actions))
        return actions, z, torch.ones_like(actions, dtype=torch.bool)
    reference.transition = transition
    reference.tail = lambda z, noise: torch.full((len(z), 1), float("nan"))
    actions = torch.tensor([[-.25], [.1], [.3]])
    values = model_return_samples(reference, torch.zeros(1, 1), actions, seed=5)
    assert calls == [3]
    assert torch.equal(values, actions.T.expand(32, -1))


def test_q_selector_integrates_mean_pair_not_min_and_disables_dropout():
    class MultiHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = nn.Dropout(.9)
            self.seen_modes = []
        def _forward_eager(self, z, actions):
            self.seen_modes.append((self.training, self.dropout.training))
            signs = actions.new_tensor([1., 2., 3., 4., -4.])[:, None, None]
            return self.dropout(signs * actions)
    reference, critic = AnalyticReference(), MultiHead()
    actions, root = torch.tensor([[-1.], [1.], [.5]]), torch.zeros(1, 1)
    before = torch.random.get_rng_state().clone()
    scores = learned_q_scores(reference, root, actions, critic)
    heads = actions.new_tensor([1., 2., 3., 4., -4.])[:, None] * actions.T
    pair_mean = torch.stack([heads[list(pair)].mean(0) for pair in combinations(range(5), 2)]).mean(0)
    torch.testing.assert_close(scores, pair_mean)
    assert stable_argmax(scores) == 1
    assert stable_argmax(heads.min(0).values) != 1
    assert critic.seen_modes == [(False, False)]
    assert critic.training and critic.dropout.training
    assert torch.equal(torch.random.get_rng_state(), before)


def test_exact_ties_prefer_actor_then_prior_and_nan_fails():
    assert stable_argmax(torch.tensor([2., 2., 2.])) == 0
    assert stable_argmax(torch.tensor([1., 2., 2.])) == 1
    with pytest.raises(ValueError, match="finite"):
        stable_argmax(torch.tensor([float("nan"), 1.]))


def replay_fixture():
    root = torch.tensor([[.2, .5]])
    replay = SimpleNamespace(size=3, next_sample_id=3, horizon_end=torch.ones(3, 1),
        z=root.expand(3, -1).clone(), action=torch.tensor([[.1], [.3], [.7]]))
    reference = SimpleNamespace(cfg=SimpleNamespace(action_dim=1), engine=SimpleNamespace(
        state=SimpleNamespace(replay=None), _action_pool=SimpleNamespace(replay=replay)))
    return reference, root, replay


def test_complete_replay_bank_is_copied_and_matches_current_root():
    reference, root, replay = replay_fixture()
    actions = complete_replay_actions(reference, root, expected=3)
    actions.zero_()
    assert replay.action[1] == .3
    with pytest.raises(RuntimeError, match="current root"):
        complete_replay_actions(reference, root + .1, expected=3)


@pytest.mark.parametrize("field,value,match", [
    ("size", 2, "incomplete"), ("next_sample_id", 4, "overflowed"),
    ("horizon_end", torch.zeros(3, 1), "non-boundary"),
    ("z", torch.tensor([[.2, .5], [.3, .5], [.2, .5]]), "current root"),
    ("action", torch.tensor([[2.], [.1], [.2]]), "invalid normalized"),
])
def test_incomplete_or_nonroot_replay_rejected(field, value, match):
    reference, root, replay = replay_fixture()
    setattr(replay, field, value)
    with pytest.raises(RuntimeError, match=match):
        complete_replay_actions(reference, root, expected=3)


@pytest.fixture
def controller():
    options = _tiny_params(**SOLVE_CONTRACT, aux_return_mode="sac",
        inner_replay_capacity=768, inner_actor_entropy_mode="squashed",
        inner_finite_horizon=True, inner_component_update_order="critic_first",
        inner_log_std_mapping="direct_clamp", log_std_mapping="direct_clamp",
        inner_critic_dropout_enabled=True, dropout=.1)
    options.pop("inner_updates_per_round")
    wrapped = _model_from_params(options)
    # Auxiliary Q's default zero output makes all terminal draws identical.
    # Give the frozen fixture a nonconstant action-dependent value function.
    generator = torch.Generator().manual_seed(44)
    with torch.no_grad():
        for parameter in wrapped.agent.inner_engine._horizon_critic.parameters():
            parameter.add_(.02 * torch.randn(parameter.shape, generator=generator))
    try:
        yield wrapped
    finally:
        wrapped.close()


def call(wrapped, *, arm="actor_mean", shadow=False, **kwargs):
    options = dict(arm=arm, shadow=shadow, solve_seed=999, selection_seed=1234,
                   validation_seed=4567, mc_rollouts=32)
    options.update(kwargs)
    return run_bypass_decision(wrapped, np.array([1., .2, -.1], dtype=np.float32), **options)


def test_real_solve_matches_audit_fork_and_keeps_outer_state(controller):
    reference = Reference(controller)
    rng = reference.engine._new_rng(999).training_state_dict()
    before = _outer_state_digest(controller)
    torch_before = torch.random.get_rng_state().clone()
    np_before, random_before = deepcopy(np.random.get_state()), random.getstate()
    expected_action, _, final = solve_fork(controller, [1., .2, -.1], rng, capture_rounds=())
    expected_replay = reference.engine._action_pool.replay.action[:768].detach().clone()
    result = call(controller)
    np.testing.assert_array_equal(result["action"], expected_action)
    assert torch.equal(reference.engine._action_pool.replay.action[:768], expected_replay)
    for key, value in final.state_dict("critic").items():
        assert torch.equal(value, reference.engine._action_pool.critic.state_dict()[key])
    assert result["work"] == dict(actor_updates=24, critic_updates=96, model_steps=768)
    assert result["replay_count"] == 768 and result["bank_count"] == 770
    assert result["diagnostics"] is None
    assert result["selected_index"] == 0
    assert _outer_state_digest(controller) == before
    assert torch.equal(torch.random.get_rng_state(), torch_before)
    assert np.array_equal(np_before[1], np.random.get_state()[1])
    assert np_before[2:] == np.random.get_state()[2:]
    assert random_before == random.getstate()


def test_shadow_and_other_arm_scoring_do_not_change_solve_or_execution(controller):
    before = _outer_state_digest(controller)
    torch_before = torch.random.get_rng_state().clone()
    plain = call(controller)
    private_before = deepcopy(controller.agent.inner_engine.rng.training_state_dict())
    results = {arm: call(controller, arm=arm, shadow=True) for arm in ARMS}
    np.testing.assert_array_equal(plain["action"], results["actor_mean"]["action"])
    hashes = {result["diagnostics"]["bank_sha256"] for result in results.values()}
    assert len(hashes) == 1
    final_rng = controller.agent.inner_engine.rng.training_state_dict()
    for kind in ("streams", "phase_streams"):
        assert all(torch.equal(value, final_rng[kind][key])
                   for key, value in private_before[kind].items())
    for arm, result in results.items():
        diag = result["diagnostics"]
        own = diag["choices"][arm]
        np.testing.assert_array_equal(result["normalized_action"], own["normalized_action"])
        assert result["selected_index"] == own["index"]
        assert len(diag["selection_model_means"]) == len(diag["selection_q_scores"]) == 770
        assert diag["selection_seed"] != diag["validation_seed"]
        assert all(len(choice["validation_model"]["draws"]) == 32 for choice in diag["choices"].values())
        own_mean = own["selection_model"]["mean"]
        if arm == "model_score":
            assert own_mean >= max(diag["selection_model_means"]) - 1e-6
        if arm == "learned_q":
            assert own["learned_q"] == max(diag["selection_q_scores"])
        assert result["timing"]["control_seconds"] == pytest.approx(sum(
            result["timing"][key] for key in ("solve_seconds", "bank_seconds", "selection_seconds")))
        assert result["timing"]["diagnostics_seconds"] > 0
        json.dumps(diag, allow_nan=False)
    assert _outer_state_digest(controller) == before
    assert torch.equal(torch.random.get_rng_state(), torch_before)
    changed = call(controller, arm="model_score", shadow=True, validation_seed=4568)
    np.testing.assert_array_equal(changed["action"], results["model_score"]["action"])
    assert changed["diagnostics"]["choices"]["actor_mean"]["validation_model"]["draws"] != (
        results["model_score"]["diagnostics"]["choices"]["actor_mean"]["validation_model"]["draws"])


def test_config_and_streams_fail_closed_before_solve(controller):
    validate_bypass_controller(controller)
    with pytest.raises(ValueError, match="independent"):
        call(controller, validation_seed=1234)
    controller.cfg.inner_rounds = 4
    with pytest.raises(ValueError, match="fixed fresh H1/J6"):
        call(controller)
