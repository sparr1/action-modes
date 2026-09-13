"""Scientific invariants for the isolated fixed-critic entropy experiment."""
from contextlib import contextmanager
import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from evaluate_ambi_checkpoint import _outer_state_digest
from RL.tdmpc2_core.common import math as td_math
from RL.tdmpc2_core.inner_improvement import InnerImprovementEngine
from tests.test_ambi_latency_contract import _assert_tree_equal
from tests.test_ambi_root_local_sac import _tiny_component_model
from utils.ambi_entropy_probe import _digest, _entropy, entropy_arms, run_root_entropy_probe


@contextmanager
def tiny(**overrides):
    params = dict(inner_rounds=1, inner_rollouts_per_round=8, inner_rollout_horizon=1,
                  inner_replay_capacity=8, inner_finite_horizon=True,
                  inner_critic_updates_per_round=2, inner_actor_updates_per_round=16,
                  inner_temperature_mode="fixed", inner_temperature=0,
                  inner_actor_initialization="prior", inner_critic_initialization="prior",
                  outer_actor_entropy_mode="tdmpc2_scaled", ent_coef=.003,
                  num_q=3, q_representation="distributional", dropout=.2,
                  inner_critic_dropout_enabled=True,
                  inner_q_actor_reduction="min_pair", mppi_terminal_q_reduction="mean_pair")
    params.update(overrides)
    model = _tiny_component_model(**params)
    try:
        yield model
    finally:
        model.env.close()


def solve(model, arms=None, **overrides):
    args = dict(arms=entropy_arms(model, include_gaussian=True)[0] if arms is None else arms,
                fit_seed=11, actor_seed=22, probe_seed=33,
                snapshot_updates=(0, 1, 4, 16), probe_rollouts=8)
    args.update(overrides)
    return run_root_entropy_probe(model, [.1, .2, .3], **args)


def identities(result):
    return {(r["arm"], r["actor_updates"]): r["actor_state_sha256"]
            for r in result["snapshots"]}


def test_resolver_uses_actual_saved_alpha_and_joint_gaussian_dimension_matching():
    model = SimpleNamespace(cfg=SimpleNamespace(outer_actor_entropy_mode="tdmpc2_scaled", action_dim=21),
                            agent=SimpleNamespace(alpha=torch.tensor(.0001)))
    arms, metadata = entropy_arms(model, include_gaussian=True)
    assert arms[1]["mode"] == "tdmpc2_scaled"
    assert arms[2]["alpha"] == pytest.approx(.0021)
    assert arms[3]["alpha"] == arms[2]["alpha"]
    assert metadata["aliases"] == {}
    model.cfg.outer_actor_entropy_mode = "squashed"
    arms, metadata = entropy_arms(model)
    assert arms[2]["alpha"] == arms[1]["alpha"]
    assert arms[2]["alias_of"] == "prior_recipe"
    assert metadata["aliases"] == {"squashed_matched": "prior_recipe"}


def test_analytic_saturated_mean_has_true_entropy_pressure_but_native_cancels():
    mean = torch.tensor([[20., -20.]], requires_grad=True)
    log_std = torch.zeros_like(mean, requires_grad=True)
    noise = torch.tensor([[.25, -.25]])
    u = mean + noise * log_std.exp()
    gaussian = td_math.gaussian_logprob(noise, log_std)
    _, action, logprob = td_math.squash(mean, u, gaussian)
    info = dict(log_std=log_std, entropy=-logprob,
                scaled_entropy=td_math.tdmpc2_scaled_entropy(gaussian, action))
    assert torch.equal(action.abs(), torch.ones_like(action))
    native_mean, native_std = torch.autograd.grad(_entropy(info, noise, "tdmpc2_scaled").sum(),
                                                 (mean, log_std), retain_graph=True)
    true_mean, = torch.autograd.grad(_entropy(info, noise, "squashed").sum(), (mean,))
    torch.testing.assert_close(native_mean, torch.zeros_like(mean), atol=1e-7, rtol=0)
    torch.testing.assert_close(native_std, torch.full_like(mean, 2.), atol=1e-6, rtol=0)
    torch.testing.assert_close(true_mean, torch.tensor([[-2., 2.]]), atol=1e-6, rtol=0)


def test_collect_fit_once_actual_initialization_immutable_snapshots_and_all_update_points(monkeypatch):
    collected, fits, snapshots = [], [], []
    original_collect = InnerImprovementEngine._collect_round
    original_fit = InnerImprovementEngine._run_component_update_counts
    def collect(self, root):
        collected.append(root.clone())
        return original_collect(self, root)
    def fit(self, **kwargs):
        fits.append(kwargs)
        return original_fit(self, **kwargs)
    monkeypatch.setattr(InnerImprovementEngine, "_collect_round", collect)
    monkeypatch.setattr(InnerImprovementEngine, "_run_component_update_counts", fit)
    with tiny() as model:
        outer_hash = _digest(model.agent.model._pi.state_dict())
        critic_ids, critic_hashes = [], []
        def callback(metadata, snapshot, critic, root):
            snapshots.append((metadata, snapshot))
            critic_ids.append(id(critic))
            critic_hashes.append(_digest(critic.state_dict()))
            assert not any(p.requires_grad for p in critic.parameters())
            assert root.shape == (1, model.cfg.latent_dim)
        result = solve(model, on_snapshot=callback)
        assert len(collected) == len(fits) == 1
        assert fits[0]["critic_count"] == 2 and fits[0]["actor_count"] == 0
        assert result["collection_transitions"] == result["replay_size"] == 8
        assert result["critic_updates"] == 2 and result["actor_optimizer_updates"] == 64
        assert len(set(critic_ids)) == len(set(critic_hashes)) == 1
        for arm in ("off", "prior_recipe", "squashed_matched", "gaussian_control"):
            rows = [r for r in result["snapshots"] if r["arm"] == arm]
            assert [r["actor_updates"] for r in rows] == [0, 1, 4, 16]
            assert rows[0]["actor_state_sha256"] == outer_hash
            assert rows[0]["metrics"]["parameter_displacement_l2"] == 0
        saved = snapshots[0][1]
        policy = saved.make_policy()
        with torch.no_grad():
            next(policy.parameters()).add_(100)
        assert _digest(saved.make_policy().state_dict()) == outer_hash
        assert result["timing"]["callback_seconds"] > 0
        assert result["optimization_seconds"] < result["total_seconds"]


def test_arm_order_probe_frequency_and_callback_rng_do_not_change_actor_updates():
    with tiny() as model:
        engine = InnerImprovementEngine(model.agent)
        arms = entropy_arms(model, include_gaussian=True)[0]
        first = solve(model, arms, fit_engine=engine)
        def noisy_callback(metadata, snapshot, critic, root):
            random.random()
            np.random.normal(size=8)
            torch.randn(8)
            # Deliberately change borrowed modes; scope must restore them.
            critic.eval()
            model.agent.model.train()
        second = solve(model, list(reversed(arms)), fit_engine=engine,
                       snapshot_updates=(0, 16), probe_rollouts=3,
                       on_snapshot=noisy_callback)
        assert second["critic_state_sha256"] == first["critic_state_sha256"]
        assert second["replay_sha256"] == first["replay_sha256"]
        for key, value in identities(second).items():
            assert identities(first)[key] == value
        assert engine.state.actor is engine.state.critic is engine.state.replay is None
        assert engine.rng._action_fork_depth == 0


@pytest.mark.parametrize("fail", [False, True])
def test_global_rng_modes_outer_optimizers_and_live_engine_survive_success_and_error(fail):
    with tiny() as model:
        world = model.agent.model
        world.train()
        world._pi.eval()
        modes = [m.training for m in world.modules()]
        torch_rng = torch.random.get_rng_state().clone()
        py_rng, np_rng = random.getstate(), copy.deepcopy(np.random.get_state())
        before = _outer_state_digest(model)
        live = model.agent.inner_engine
        live_state = live.state
        live_rng = live.rng.training_state_dict()
        dedicated = InnerImprovementEngine(model.agent)
        def callback(*args):
            random.random(); np.random.normal(); torch.randn(2)
            world.eval()
            if fail:
                raise RuntimeError("injected callback failure")
        if fail:
            with pytest.raises(RuntimeError, match="injected callback failure"):
                solve(model, snapshot_updates=(0, 1), on_snapshot=callback, fit_engine=dedicated)
        else:
            solve(model, snapshot_updates=(0, 1), on_snapshot=callback, fit_engine=dedicated)
        assert [m.training for m in world.modules()] == modes
        assert _outer_state_digest(model) == before
        assert live.state is live_state
        _assert_tree_equal(live.rng.training_state_dict(), live_rng)
        assert torch.equal(torch.random.get_rng_state(), torch_rng)
        assert random.getstate() == py_rng
        current_np = np.random.get_state()
        assert current_np[0] == np_rng[0] and np.array_equal(current_np[1], np_rng[1])
        assert current_np[2:] == np_rng[2:]
        assert dedicated.rng._action_fork_depth == 0
        assert dedicated.state.actor is dedicated.state.critic is None


def test_old_squashed_alias_reuses_identical_actor_snapshots_and_only_two_optimizers():
    with tiny(outer_actor_entropy_mode="squashed") as model:
        result = solve(model, entropy_arms(model)[0], snapshot_updates=(0, 1))
        assert result["optimizer_arms"] == 2
        assert len(result["snapshots"]) == 6
        for update in (0, 1):
            prior = next(r for r in result["snapshots"] if r["arm"] == "prior_recipe" and r["actor_updates"] == update)
            alias = next(r for r in result["snapshots"] if r["arm"] == "squashed_matched" and r["actor_updates"] == update)
            assert prior["actor_sha256"] == alias["actor_sha256"]
            assert prior["metrics"] == alias["metrics"]
        assert result["preflight"]["relevant_to_saved_recipe"] is False
        assert result["preflight"]["needs_control"] is False


def test_fixed_q_scale_and_handoff_reductions_preserved():
    with tiny(sac_actor_loss_scale_mode="tdmpc2_percentile_range", inner_q_actor_reduction="mean_pair",
              outer_q_actor_reduction="mean_pair", outer_critic_target="reward_only",
              mppi_terminal_q_reduction="min_pair") as model:
        model.agent.actor_loss_scale.fill_(7.)
        result = solve(model, snapshot_updates=(0, 1))
        assert result["q_scale"] == 7.
        assert result["actor_q_reduction"] == "mean_pair"
        assert result["terminal_q_reduction"] == "min_pair"
        for row in result["snapshots"]:
            assert row["metrics"]["fitted_q_scaled_mean"] == pytest.approx(row["metrics"]["fitted_q_mean"] / 7., abs=1e-6)
        assert float(model.agent.actor_loss_scale) == 7.


def test_refuses_live_engine_policy_bound_override_and_non_h1_protocol():
    with tiny() as model:
        with pytest.raises(ValueError, match="live inner engine"):
            solve(model, fit_engine=model.agent.inner_engine)
        model.cfg.inner_log_std_min -= 1
        with pytest.raises(ValueError, match="mapping and bounds"):
            solve(model)
        model.cfg.inner_log_std_min += 1
        model.cfg.inner_rollout_horizon = 2
        with pytest.raises(ValueError, match="inner_rollout_horizon"):
            solve(model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable; CPU checks do not establish GPU readiness")
def test_cuda_frozen_entropy_probe_preserves_all_cuda_rng_streams():
    with tiny(device="cuda") as model:
        before = torch.cuda.get_rng_state_all()
        result = solve(model, snapshot_updates=(0, 1))
        assert len(result["snapshots"]) == 8
        for actual, expected in zip(torch.cuda.get_rng_state_all(), before):
            assert torch.equal(actual, expected)
