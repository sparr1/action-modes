"""Analytical paired-action checks independent of checkpoint and W&B state."""
from copy import deepcopy
import json
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from tests.test_transfer_diagnostic_real import Accumulator
from utils.matched_action_audit import mppi_candidate, score_action_bank, score_real_candidates
from utils.transfer_diagnostic_real import capture_simulator_snapshot
from utils.transfer_diagnostics import Reference


class Critic(nn.Module):
    def __init__(self, slope=1., bias=.5):
        super().__init__()
        self.register_buffer("slope", torch.tensor(slope))
        self.register_buffer("bias", torch.tensor(bias))

    def _forward_eager(self, z, actions):
        value = self.slope * actions + self.bias
        return torch.stack((value - .25, value + .25))


class Policy(nn.Module):
    def __init__(self, noise_scale):
        super().__init__()
        self.noise_scale = noise_scale


class AnalyticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_backend = SimpleNamespace(pair_size=2)
        self.seen_encode_training = []
        self.seen_policy_options = []

    def encode(self, observations):
        self.seen_encode_training.append(self.training)
        return observations

    def pi(self, z, *, policy, noise, **kwargs):
        self.seen_policy_options.append(kwargs)
        action = .5 + policy.noise_scale * noise.tanh()
        return action, {"log_prob": z.new_zeros((len(z), 1))}

    def pi_action(self, z, *, policy, generator, **kwargs):
        noise = torch.randn((len(z), 1), generator=generator, device=z.device)
        return self.pi(z, policy=policy, noise=noise)[0]

    def Q(self, z, actions, *, qs, reduction, generator=None):
        heads = qs(z, actions)
        return heads if reduction == "all" else heads.mean(0)


class AnalyticReference(Reference):
    """Reward=a, state irrelevant, gamma=.5, fixed prior mean=.5.

    Its exact return critic is Q(a)=a+.5; frozen model equals real dynamics
    for all reward/value calculations even when state transition is noisy.
    """
    def __init__(self, noise_scale=0.):
        self.cfg = SimpleNamespace(action_dim=1, inner_sac_critic_target="reward_only",
            inner_terminal_entropy="none", inner_horizon_critic_source="aux_return",
            inner_q_actor_reduction="mean_pair", mppi_terminal_q_reduction="mean_pair",
            episodic=False, inner_termination_threshold=.5)
        self.device, self.discount, self.rollouts = torch.device("cpu"), .5, 8
        self.bounds = {}
        self.model = AnalyticModel()
        actor, critic = Policy(noise_scale), Critic()
        self.engine = SimpleNamespace(_actor_base=actor, _horizon_actor=actor,
            _horizon_critic=critic, _actor_options={}, _horizon_actor_options={})
        self.wrapped = SimpleNamespace(_unscale_action=lambda a: a.copy(),
                                       _scale_action=lambda a: a.copy())
        self.batch_sizes = []

    def transition(self, z, actions):
        self.batch_sizes.append(len(z))
        return actions, z, torch.zeros_like(actions, dtype=torch.bool)


def bank(reference, **kwargs):
    defaults = dict(actions=[[0.], [-.5], [.5]], labels=["prior", "negative", "positive"],
        critics={"correct": Critic(), "reversed": Critic(-1., .5)}, seed=17,
        mc_rollouts=8, horizons=(1, 3), max_expanded_batch=16)
    defaults.update(kwargs)
    return score_action_bank(reference, [0.], **defaults)


def test_analytic_critic_rank_regret_and_root_only_scoring():
    reference = AnalyticReference()
    result = bank(reference)
    correct = next(row for row in result["critics"] if row["name"] == "correct" and row["horizon"] == 1)
    reversed_q = next(row for row in result["critics"] if row["name"] == "reversed" and row["horizon"] == 1)
    assert correct["spearman"] == pytest.approx(1.)
    assert correct["rmse"] == pytest.approx(0.)
    assert correct["top_action_label"] == "positive"
    assert correct["top_action_regret"] == 0.
    assert reversed_q["spearman"] == pytest.approx(-1.)
    assert reversed_q["top_action_label"] == "negative"
    assert reversed_q["top_action_regret"] == pytest.approx(1.)
    assert reversed_q["selected_action_gain"] == pytest.approx(-.5)
    for row in result["actions"]:
        for h in (1, 3):
            assert row["model"][f"h{h}"]["mean"] == pytest.approx(row["action"][0] + .5)
        assert row["q_head_std"]["correct"] == pytest.approx(.25)
    assert max(reference.batch_sizes) <= 16
    assert not any(reference.model.seen_encode_training)
    json.dumps(result, allow_nan=False)


def test_common_noise_is_chunk_invariant_and_paired_before_reduction():
    reference = AnalyticReference(noise_scale=.2)
    state, model_state = torch.random.get_rng_state().clone(), deepcopy(reference.model.state_dict())
    result = bank(reference, max_expanded_batch=8)
    large = bank(reference, max_expanded_batch=512)
    assert result["actions"] == large["actions"]
    assert result["critics"] == large["critics"]
    positive = result["actions"][2]
    assert positive["model"]["h1"]["se"] > 0.
    assert positive["model"]["h3"]["se"] > 0.
    for h in (1, 3):
        assert positive["model"][f"h{h}"]["gain_vs_baseline_mean"] == pytest.approx(.5)
        assert positive["model"][f"h{h}"]["gain_vs_baseline_se"] < 1e-7
    assert reference.model.training
    assert reference.engine._actor_base.training
    assert torch.equal(torch.random.get_rng_state(), state)
    assert all(torch.equal(value, model_state[key]) for key, value in reference.model.state_dict().items())
    other = bank(reference, seed=18)
    assert other["actions"][0]["model"]["h1"]["draws"] != result["actions"][0]["model"]["h1"]["draws"]


def test_mppi_candidate_supports_read_only_persistent_mean_and_seeded_restart():
    reference = AnalyticReference(noise_scale=.2)
    options = dict(iterations=2, num_samples=8, num_elites=2, num_pi_trajs=1,
        temperature=.5, min_std=.05, max_std=1.)
    rng_before = torch.random.get_rng_state().clone()
    first = mppi_candidate(reference, [0.], horizon=3, seed=44, planner_options=options)
    repeat = mppi_candidate(reference, [0.], horizon=3, seed=44, planner_options=options)
    assert first == repeat
    assert first["next_mean"] == first["proposal_mean"]
    assert first["normalized_action"] == first["next_mean"][0]
    previous = torch.as_tensor(first["next_mean"])
    before = previous.clone()
    warm = mppi_candidate(reference, [0.], horizon=3, seed=45, planner_options=options,
                          previous_mean=previous)
    assert warm["warm_start"] and not first["warm_start"]
    assert torch.equal(previous, before)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert reference.model.training
    assert not any(reference.model.seen_encode_training)
    json.dumps(warm, allow_nan=False)


def real(reference, env, snapshot, **kwargs):
    values = dict(actions=[[0.], [.5]], labels=["prior", "positive"], seed=23,
        real_rollouts=4, tail_steps=4, horizons=(1, 3), max_expanded_batch=4)
    values.update(kwargs)
    return score_real_candidates(reference, env, snapshot, **values)


def test_real_pair_decomposition_finite_cutoff_and_simulator_restoration():
    env = gym.wrappers.TimeLimit(Accumulator(), max_episode_steps=2)
    env.reset(seed=9)
    root = capture_simulator_snapshot(env)
    env.step(np.array([.3]))  # Caller state intentionally differs from audited root.
    before = capture_simulator_snapshot(env).sha256
    reference = AnalyticReference()
    torch_before = torch.random.get_rng_state().clone()
    result = real(reference, env, root)
    assert result["simulator_unchanged"] and result["continuing"]
    assert capture_simulator_snapshot(env).sha256 == before
    assert env._max_episode_steps == 2
    assert torch.equal(torch.random.get_rng_state(), torch_before)
    for row in result["actions"]:
        assert row["complete"]
        assert row["model_prefix_bias_mean"] == pytest.approx(0.)
        assert row["terminal_bias_mean"] == pytest.approx(.5 ** (row["horizon"] + 4))
        assert row["total_bias_mean"] == pytest.approx(row["terminal_bias_mean"])
        assert row["real_decisions_requested"] == row["horizon"] + 4
        assert row["truncated_replicates"] == 0
        assert row["model_samples"] == row["real_tail_samples"] == 4
        if row["label"] == "positive":
            assert row["real_tail_gain_vs_baseline_mean"] == pytest.approx(.5)
            assert row["real_tail_gain_vs_baseline_se"] == 0.
    for row in result["replicates"]:
        assert row["prefix_decisions"] == row["horizon"]
        assert row["tail_decisions"] == 4
        assert row["mc_tail_has_bootstrap"] is False
    assert "finite cutoff" in result["semantics"]["cutoff"]
    json.dumps(result, allow_nan=False)


def test_real_and_model_share_noise_with_paired_gain_uncertainty():
    env, reference = Accumulator(), AnalyticReference(noise_scale=.2)
    result = real(reference, env, capture_simulator_snapshot(env), tail_steps=2)
    assert any(row["model_se"] > 0 for row in result["actions"])
    for row in result["actions"]:
        assert abs(row["model_prefix_bias_mean"]) < 1e-7
        assert row["model_prefix_bias_se"] < 1e-7
        if row["label"] == "positive":
            for metric in ("model", "real_prefix_value", "real_tail"):
                assert row[f"{metric}_gain_vs_baseline_mean"] == pytest.approx(.5, abs=1e-7)
                assert row[f"{metric}_gain_vs_baseline_se"] < 1e-7
    # The first prior action after a forced action has the same draw whether it
    # is H1's first tail action or H3's first continuation action.
    rows = {(row["label"], row["horizon"], row["replicate"]): row for row in result["replicates"]}
    for replicate in range(4):
        short, long = rows["prior", 1, replicate], rows["prior", 3, replicate]
        assert short["real_bootstrapped_return"] == pytest.approx(short["model_return"], abs=1e-7)
        assert long["real_bootstrapped_return"] == pytest.approx(long["model_return"], abs=1e-7)
        # H1 + two tail actions and H3's prefix are exactly the same three
        # real decisions, including the continuation noise at the boundary.
        assert short["real_mc_return"] == long["real_prefix_return"]


def test_real_error_restores_simulator_and_model_modes():
    class FailingAccumulator(Accumulator):
        def step(self, action):
            super().step(action)
            raise RuntimeError("deliberate simulator failure")
    env, reference = FailingAccumulator(), AnalyticReference()
    before = capture_simulator_snapshot(env)
    with pytest.raises(RuntimeError, match="deliberate simulator failure"):
        real(reference, env, before)
    assert capture_simulator_snapshot(env).sha256 == before.sha256
    assert reference.model.training
    assert reference.engine._actor_base.training


@pytest.mark.parametrize("changes,match", [
    ({"actions": [[2.], [0.], [.5]]}, "finite normalized"),
    ({"labels": ["same", "same", "positive"]}, "unique nonempty"),
    ({"max_expanded_batch": 4}, "max_expanded_batch"),
    ({"baseline_index": 3}, "valid baseline"),
])
def test_invalid_banks_fail_before_scoring(changes, match):
    with pytest.raises(ValueError, match=match):
        bank(AnalyticReference(), **changes)


def test_non_return_semantics_fail_closed():
    reference = AnalyticReference()
    reference.cfg.inner_horizon_critic_source = "outer"
    with pytest.raises(ValueError, match="auxiliary return"):
        bank(reference)


@pytest.mark.parametrize("outer_options", [{}, {"log_std_min": -3.}])
def test_frozen_prior_continuation_keeps_outer_policy_bounds(outer_options):
    reference = AnalyticReference(noise_scale=.2)
    reference.bounds = {"log_std_min": -.1}
    reference.engine._actor_options = deepcopy(outer_options)
    reference.engine._horizon_actor_options = deepcopy(outer_options)
    bank(reference)
    env = Accumulator()
    real(reference, env, capture_simulator_snapshot(env))
    assert reference.model.seen_policy_options
    assert all(value == outer_options for value in reference.model.seen_policy_options)
    assert reference.bounds == {"log_std_min": -.1}


def test_actual_distributional_model_scores_aux_return_without_updates():
    from tests.test_aux_critic_transfer import critic_params
    from tests.test_ambi_root_local_sac import _model_from_params
    wrapped = _model_from_params(critic_params(inner_critic_scope="action"))
    try:
        reference = Reference(wrapped, rollouts=3)
        model = wrapped.agent.model
        before = {key: value.clone() for key, value in model.state_dict().items()}
        rng = torch.random.get_rng_state().clone()
        observation = np.array([1., .2, -.1], dtype=np.float32)
        candidate = mppi_candidate(reference, observation, horizon=3, seed=29,
            planner_options=dict(iterations=1, num_samples=6, num_elites=2,
                num_pi_trajs=1, temperature=.5, min_std=.05, max_std=1.))
        result = score_action_bank(reference, observation,
            actions=[[0.], candidate["normalized_action"]], labels=["zero", "mppi_h3"],
            critics={"aux_return": reference.engine._horizon_critic}, seed=31,
            mc_rollouts=3, max_expanded_batch=3)
        assert len(result["actions"]) == len(result["critics"]) == 2
        json.dumps(result, allow_nan=False)
        assert torch.equal(torch.random.get_rng_state(), rng)
        assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
    finally:
        wrapped.close()
