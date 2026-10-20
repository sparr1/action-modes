"""Joint entropy/KL observation must preserve every ordinary solve and its RNG."""
from copy import copy, deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import evaluate_ambi_entropy_kl as cli
from tests.test_ambi_entropy_probe_cuda import _assert_strict_regions_ran
from tests.test_ambi_entropy_probe_invariants import _fixture, _snapshot, _assert_unchanged
from tests.test_ambi_prior_kl_cuda import _PlainWithSnapshots
from tests.test_ambi_prior_kl_probe import _saved
from utils import ambi_entropy_kl as probe
from utils.ambi_entropy_probe import _hash, _preserve_environment


def _assert_joint_parity(wrapper, monkeypatch):
    observation = np.asarray([.2, -.1, .7], dtype=np.float32)
    device = wrapper.agent.device
    wrapper.agent.act(torch.as_tensor(observation, device=device), t0=True, eval_mode=True)
    params = list(wrapper.agent.model.parameters())
    params[0].requires_grad_(False)
    params[1].grad = torch.full_like(params[1], .125)
    before = _snapshot(wrapper)
    config_before = deepcopy(vars(wrapper.cfg))
    cuda_before = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    observed = []
    original = probe._EntropyKLProbeEngine

    class Captured(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            observed.append(self)

    monkeypatch.setattr(probe, "_EntropyKLProbeEngine", Captured)
    result = probe.probe_entropy_kl_root(wrapper, observation, seed=23, anchor_samples=8)
    _assert_unchanged(wrapper, before)
    assert vars(wrapper.cfg) == config_before
    assert result["paired_first_phase"] and result["paired_private_rng_draw_counts"]
    assert not result["hard_kl_budget_enforced"]
    assert not result["counterfactuals_are_closed_loop_evaluations"]
    assert list(result["full_solves"]) == ["beta_0x", "beta_1x", "beta_3x"]
    json.dumps(result, allow_nan=False)
    for observer, multiplier in zip(observed, probe.DEFAULT_BETA_MULTIPLIERS):
        arm = result["full_solves"][probe.beta_multiplier_key(multiplier)]
        beta = multiplier * result["initial_alpha"]
        assert arm["config_overrides"] == {"inner_outer_policy_kl_coef": beta}
        assert arm["beta"] == beta and arm["entropy_enabled"]
        assert arm["inner_temperature_lr"] == .0003
        assert arm["actor_steps"] == arm["temperature_steps"] == arm["critic_steps"] == 4
        assert arm["actor_updates"][0]["temperature_before"]["adam_step"] == 0
        assert arm["actor_updates"][0]["alpha_before"] == result["initial_alpha"]
        for anchor in arm["anchors_initial"].values():
            assert anchor["kl_to_prior"] == pytest.approx(0., abs=1e-6)
        for index, row in enumerate(arm["actor_updates"], 1):
            assert row["beta"] == beta  # Never recompute beta from adapted alpha.
            assert row["temperature_after"]["adam_step"] == index
            assert row["temperature_entropy_residual"] == pytest.approx(
                row["metrics"]["actor_entropy"] - result["target_entropy"])
            expected = (-row["metrics"]["actor_q_mean"]
                        - row["alpha_before"] * row["metrics"]["actor_entropy"]
                        + beta * row["kl_before"])
            assert row["metrics"]["actor_loss"] == pytest.approx(expected, abs=2e-6)
            p = row["gradient_projections"]
            assert p["regularized_objective_first_order_change"] == pytest.approx(
                p["return_q_first_order_change"] + row["alpha_before"] * p["entropy_first_order_change"]
                - beta * p["kl_first_order_change"], abs=2e-7)
            for anchor in row["anchors_after"].values():
                assert anchor["entropy_mc"] == pytest.approx(
                    anchor["entropy_gaussian_sampled"] + anchor["entropy_tanh_correction_mc"], abs=1e-6)
                assert anchor["kl_to_prior"] >= -1e-6
        proxy = copy(wrapper.agent)
        proxy.cfg = deepcopy(wrapper.agent.cfg)
        proxy.cfg.inner_outer_policy_kl_coef = beta
        with _preserve_environment(wrapper.agent.model, device), torch.enable_grad():
            with torch.no_grad():
                root = wrapper.agent.model.encode(torch.as_tensor(observation, device=device)[None]).detach()
            plain = _PlainWithSnapshots(proxy)
            plain.rng = plain._new_rng(23)
            action, metrics, _ = plain.act(root, t0=True, eval_mode=True, collect_diagnostics=False)
            np.testing.assert_array_equal(arm["action"], action.detach().cpu().numpy().squeeze())
            assert arm["first_replay_sha256"] == _hash(plain.initial_replay)
            assert arm["initial_actor_sha256"] == _hash(plain.initial_actor_state)
            assert arm["first_critic_sha256"] == _hash(plain.first_critic_state)
            for component in ("actor", "critic"):
                assert arm[f"final_{component}_sha256"] == _hash(getattr(plain._action_pool, component).state_dict())
            assert arm["final_private_rng_sha256"] == _hash(plain.rng.training_state_dict())
            for metric in ("inner_alpha_final", "inner_actor_entropy", "inner_temperature_loss"):
                assert arm["metrics"][metric] == float(metrics[metric])
            if device.type == "cuda":
                _assert_strict_regions_ran(observer)
                _assert_strict_regions_ran(plain)
        _assert_unchanged(wrapper, before)
        if cuda_before is not None:
            assert torch.equal(torch.cuda.get_rng_state(device), cuda_before)


@pytest.mark.parametrize("representation", ["scalar", "distributional"])
def test_joint_probe_matches_all_plain_engines_and_preserves_state(representation, monkeypatch):
    wrapper = _fixture(representation, ent_coef="auto_0.0033654973", inner_temperature_lr=.0003)
    try:
        _assert_joint_parity(wrapper, monkeypatch)
    finally:
        wrapper.env.close()


@pytest.mark.skipif(os.environ.get("AMBI_RUN_ENTROPY_KL_CUDA_TESTS") != "1" or not torch.cuda.is_available(),
                    reason="Requires AMBI_RUN_ENTROPY_KL_CUDA_TESTS=1 on an allocated CUDA GPU.")
def test_joint_probe_strict_cuda_parity(monkeypatch):
    wrapper = _fixture("distributional", device="cuda", compile=True, compile_strict=True,
                       ent_coef="auto_0.0033654973", inner_temperature_lr=.0003)
    try:
        _assert_joint_parity(wrapper, monkeypatch)
    finally:
        wrapper.env.close()


def test_failed_joint_probe_restores_outer_state_and_rng(monkeypatch):
    wrapper = _fixture(inner_temperature_lr=.0003)
    try:
        before = _snapshot(wrapper)
        original = wrapper.agent.model.pi
        calls = 0

        def interrupted(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("injected joint entropy-KL failure")
            return original(*args, **kwargs)

        monkeypatch.setattr(wrapper.agent.model, "pi", interrupted)
        with pytest.raises(RuntimeError, match="injected joint entropy-KL failure"):
            probe.probe_entropy_kl_root(wrapper, [.2, -.1, .7], seed=23, anchor_samples=8)
        _assert_unchanged(wrapper, before)
    finally:
        wrapper.env.close()


@pytest.mark.parametrize("values", [[], [-1], [float("inf")], [float("nan")], [1, 1]])
def test_invalid_beta_multipliers_rejected(values):
    with pytest.raises(ValueError, match="Beta multipliers"):
        probe.validate_beta_multipliers(values)


def test_cli_plan_preserves_original_saved_bank_and_does_not_load_model(tmp_path, monkeypatch):
    path, profile, resolved, bank, _ = _saved(tmp_path)
    profile.update(beta_multipliers=[0., 1., 3.], actor_steps=32, anchor_samples=128)
    selected = {**profile["checkpoints"]["test"], "step": 100000, "source_run": "original/run"}
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile))
    args = SimpleNamespace(profile=profile_path, root_bank=path, checkpoint_id="test",
                           environment_seed=101, plan_only=True)
    monkeypatch.setattr(cli, "_prepare", lambda args: (profile, selected, tmp_path / "checkpoint", None, resolved, [0]))

    def forbidden(*args, **kwargs):
        raise AssertionError("Plan-only must not load a model or simulator")

    monkeypatch.setattr(cli, "_make_env", forbidden)
    monkeypatch.setattr(cli, "_initialize_frozen_model", forbidden)
    plan = cli.run(args)
    assert plan["root_bank_id"] == bank["id"]
    assert plan["original_source_git_commit"] == "original"
    assert plan["source_controller_solves"] == plan["source_simulator_decisions"] == 0
    assert plan["full_solves_per_root"] == 3
    assert plan["beta_multipliers"] == [0., 1., 3.]
    assert plan["inner_temperature_lr"] == .0003
    assert plan["changed_parameter"] == "inner_outer_policy_kl_coef"
    assert not plan["hard_kl_budget_enforced"]


def test_profile_changes_only_description_and_beta_grid_from_pinned_original():
    root = Path(__file__).resolve().parents[1]
    current = json.loads((root / "configs/research/ambi_entropy_kl_probe.json").read_text())
    original = json.loads((root / "configs/research/ambi_prior_kl_probe.json").read_text())
    assert current.pop("beta_multipliers") == [0., 1., 3.]
    current.pop("description")
    original.pop("description")
    assert current == original
