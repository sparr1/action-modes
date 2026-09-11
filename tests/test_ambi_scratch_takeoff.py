"""H1 scratch pilot: exact exploration initialization and controlled update order."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from evaluate_ambi_checkpoint import _validate_checkpoint_contract
from RL.tdmpc2_core.inner_trace import InnerActionTrace
from tests.test_ambi_inner_initialization import _prepare
from tests.test_ambi_latency_contract import _assert_tree_equal
from tests.test_ambi_root_local_sac import _build_cfg, _tiny_model, _tiny_component_model
from utils.ambi_research import load_preset_matrix, resolve_preset
from utils.eval_series_data import planner_identity
from utils.resume_identity import scientific_trial_parameters


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs/research/ambi_scratch_takeoff_h1.json"


def _source_context():
    source = json.loads((ROOT / "configs/dmcontrol/algs/td_ambi_prior_reward_qscale.json").read_text())
    return SimpleNamespace(source="source.json", trial_run_params=source,
                           experiment_params={"env_params": {"task": "humanoid-walk", "obs": "state"}})


def test_takeoff_matrix_pins_backbone_and_independent_component_schedule():
    matrix = load_preset_matrix(MATRIX)
    resolved = resolve_preset(MATRIX, "initialization/scratch", matrix,
                              checkpoint_context=_source_context())
    cfg = _build_cfg(**resolved["algorithm_config"]["alg_params"])
    assert (cfg.inner_rounds, cfg.inner_rollouts_per_round,
            cfg.inner_rollout_horizon, cfg.inner_batch_size) == (4, 128, 1, 256)
    assert cfg.inner_replay_capacity == 2048 and cfg.inner_model_step_budget == 512
    assert cfg.inner_update_timing == "round" and cfg.inner_steps_per_update is None
    assert cfg.inner_component_update_schedule
    assert cfg.inner_critic_updates_per_action == 128 and cfg.inner_actor_updates_per_action == 16
    assert cfg.inner_actor_initialization == cfg.inner_critic_initialization == "random"
    assert cfg.inner_actor_initial_std == 0.3 and cfg.inner_critic_target_initialization == "online"
    assert cfg.inner_finite_horizon and cfg.inner_sac_critic_target == "reward_only"
    assert cfg.inner_actor_loss_scale_update == "per_action"
    assert cfg.inner_temperature_mode == "fixed" and cfg.inner_temperature == 1e-4
    assert cfg.inner_temperature_updates_per_action == 0
    assert cfg.inner_actor_lr == cfg.inner_critic_lr == 3e-4
    assert cfg.inner_q_actor_reduction == cfg.mppi_terminal_q_reduction == "mean_pair"
    assert cfg.inner_q_target_reduction == "min_pair" and cfg.inner_critic_dropout_enabled
    assert (cfg.num_q, cfg.q_num_bins, cfg.dropout) == (5, 101, .01)
    assert [c["step"] for c in matrix["checkpoint_contract"]["checkpoints"]] == [
        100000, 125000, 150000, 200000, 300000, 500000, 2000000]
    assert all(len(c["sha256"]) == 64 for c in matrix["checkpoint_contract"]["checkpoints"])
    assert "saved_q_scale" not in matrix["checkpoint_contract"]
    assert matrix["real_calibration"]["rounds"] == [0, 1, 2, 3, 4]
    prior = resolve_preset(MATRIX, "controller/prior", matrix,
                           checkpoint_context=_source_context())
    prior_cfg = _build_cfg(**prior["algorithm_config"]["alg_params"])
    assert prior_cfg.inner_operator == "none" and prior_cfg.inner_model_step_budget == 0
    assert prior_cfg.inner_actor_initial_std is None


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf"), 10, 1e-10])
def test_initial_std_rejects_invalid_or_unrepresentable_values(value):
    with pytest.raises(ValueError, match="inner_actor_initial_std"):
        _build_cfg(inner_actor_initialization="random", inner_actor_initial_std=value)


def test_initial_std_requires_random_actor_and_preserves_omitted_identities():
    with pytest.raises(ValueError, match="requires inner_actor_initialization='random'"):
        _build_cfg(inner_actor_initial_std=.3)
    for identity in (
        lambda p: scientific_trial_parameters({"alg": "AMBITDMPC2/AMBITDMPC2", "alg_params": p}),
        lambda p: planner_identity(p, {}, "AMBITDMPC2/AMBITDMPC2", "tanh_mean"),
    ):
        base = {"inner_operator": "sac", "inner_actor_initialization": "random"}
        assert identity(base) == identity({**base, "inner_actor_initial_std": None})
        assert identity(base) != identity({**base, "inner_actor_initial_std": .3})


@pytest.mark.parametrize("mapping", ["direct_clamp", "tdmpc2_tanh"])
def test_initial_std_exact_without_changing_mean_critic_rng_or_outer(mapping):
    params = dict(inner_actor_initialization="random", inner_critic_initialization="random",
                  inner_log_std_mapping=mapping, inner_log_std_min=-10, inner_log_std_max=2)
    default = _tiny_model(**params)
    explicit = _tiny_model(**params, inner_actor_initial_std=.3)
    try:
        before = deepcopy(explicit.agent.model.state_dict())
        modes = [m.training for m in explicit.agent.model.modules()]
        rng = torch.random.get_rng_state().clone()
        left, right = _prepare(default.agent.inner_engine), _prepare(explicit.agent.inner_engine)
        _assert_tree_equal(left.critic.state_dict(), right.critic.state_dict())
        _assert_tree_equal(default.agent.inner_engine.rng.training_state_dict(),
                           explicit.agent.inner_engine.rng.training_state_dict())
        z = torch.ones(17, explicit.cfg.latent_dim)
        left_mu, _ = default.agent.model._policy_parameters(z, policy=left.actor)
        right_mu, log_std = explicit.agent.model._policy_parameters(
            z, policy=right.actor, log_std_mapping=mapping, log_std_min=-10, log_std_max=2)
        torch.testing.assert_close(left_mu, right_mu, rtol=0, atol=0)
        torch.testing.assert_close(log_std.exp(), torch.full_like(log_std, .3))
        _assert_tree_equal(explicit.agent.model.state_dict(), before)
        assert [m.training for m in explicit.agent.model.modules()] == modes
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
        # Reusing the allocation resets its trained scale without another RNG draw.
        explicit.agent.inner_engine._clear_expired(t0=False, include_action=True)
        with torch.no_grad():
            explicit.agent.inner_engine._action_pool.actor[-1].bias.fill_(1)
        restored = _prepare(explicit.agent.inner_engine, t0=False)
        _, restored_std = explicit.agent.model._policy_parameters(
            z, policy=restored.actor, log_std_mapping=mapping, log_std_min=-10, log_std_max=2)
        torch.testing.assert_close(restored_std.exp(), torch.full_like(restored_std, .3))
    finally:
        default.env.close()
        explicit.env.close()


@pytest.mark.parametrize("representation", ["scalar", "distributional"])
def test_h1_takeoff_cadence_snapshot_and_probe_noninterference(representation):
    matrix = load_preset_matrix(MATRIX)
    params = {k: v for k, v in matrix["shared_alg_params"].items() if v is not None}
    params.update(sac_actor_loss_scale_mode="tdmpc2_percentile_range",
                  ent_coef=1e-4, outer_critic_target="reward_only",
                  q_representation=representation, num_q=2 if representation == "scalar" else 5)
    plain, traced = _tiny_component_model(**params), _tiny_component_model(**params)
    trace = InnerActionTrace(probes=True, probe_mode="outer_tail", probe_rollouts=32,
                             probe_horizon=1, capture_actors=True)
    try:
        expected = plain.agent.act(torch.zeros(3), t0=True, collect_diagnostics=False)
        actual = traced.agent.act(torch.zeros(3), t0=True, collect_diagnostics=False, trace=trace)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        for name in ("actor", "critic", "critic_target", "actor_optim", "critic_optim", "replay"):
            _assert_tree_equal(getattr(plain.agent.inner_engine._action_pool, name).state_dict(),
                               getattr(traced.agent.inner_engine._action_pool, name).state_dict())
        _assert_tree_equal(plain.agent.model.state_dict(), traced.agent.model.state_dict())
        _assert_tree_equal(plain.agent.inner_engine.rng.training_state_dict(),
                           traced.agent.inner_engine.rng.training_state_dict())
        probes = [e for e in trace.events if e["phase"] == "probe"]
        assert [e["actor_updates"] for e in probes] == [0, 4, 8, 12, 16]
        assert [e["critic_updates"] for e in probes] == [0, 32, 64, 96, 128]
        assert sum(e["metrics"]["probe_model_steps"] for e in probes) == 192
        phases = [(e["updated_critic"], e["updated_actor"])
                  for e in trace.events if e["phase"] == "update"]
        assert phases == ([(True, False)] * 32 + [(False, True)] * 4) * 4
        init = trace.actor_snapshots[0]
        _, log_std = traced.agent.model._policy_parameters(
            torch.zeros(5, traced.cfg.latent_dim), policy=init.make_policy(), **init.policy_bounds)
        torch.testing.assert_close(log_std.exp(), torch.full_like(log_std, .3))
        assert [s.actor_updates for s in trace.actor_snapshots] == [0, 4, 8, 12, 16]
    finally:
        plain.env.close()
        traced.env.close()


@pytest.mark.parametrize("direct", [False, True])
def test_initial_std_checkpoint_protocol_rejects_mismatch_without_mutation(direct):
    plain = _tiny_model(inner_actor_initialization="random")
    explicit = _tiny_model(inner_actor_initialization="random", inner_actor_initial_std=.3)
    try:
        source, target = plain.agent, explicit.agent
        if direct:
            source, target = source.inner_engine, target.inner_engine
        before = deepcopy(target.training_state_dict())
        with pytest.raises(ValueError):
            target.load_training_state_dict(source.training_state_dict())
        _assert_tree_equal(target.training_state_dict(), before)
        target.load_training_state_dict(before)
        _assert_tree_equal(target.training_state_dict(), before)
    finally:
        plain.env.close()
        explicit.env.close()


def test_panel_contract_validates_step_and_hash_without_weakening_single(monkeypatch):
    matrix = load_preset_matrix(MATRIX)
    pinned = matrix["checkpoint_contract"]["checkpoints"][0]
    context = SimpleNamespace(metadata={"checkpoint": {"step": pinned["step"]}})
    resolved = [{"algorithm_config": {"alg_params": matrix["checkpoint_contract"]["alg_params"]}}]
    monkeypatch.setattr("evaluate_ambi_checkpoint._file_sha256", lambda _: pinned["sha256"])
    _validate_checkpoint_contract(matrix, "unused", context, resolved)
    context.metadata["checkpoint"]["step"] = 25000
    with pytest.raises(ValueError, match="panel"):
        _validate_checkpoint_contract(matrix, "unused", context, resolved)
    context.metadata["checkpoint"]["step"] = pinned["step"]
    monkeypatch.setattr("evaluate_ambi_checkpoint._file_sha256", lambda _: "bad")
    with pytest.raises(ValueError, match="SHA256"):
        _validate_checkpoint_contract(matrix, "unused", context, resolved)
    single = {"checkpoint_contract": pinned}
    with pytest.raises(ValueError, match="SHA256"):
        _validate_checkpoint_contract(single, "unused", context, [])
