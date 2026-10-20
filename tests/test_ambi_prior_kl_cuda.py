"""Same-process parity gate for the prior-KL observer and ordinary solvers.

The CUDA case requires AMBI_RUN_PRIOR_KL_CUDA_TESTS=1 on an allocated GPU.
Both sides use strict compilation, the same device, seed, and loaded weights.
Exact equality here is an observer-fidelity check, not a claim that historical
hashes are portable across GPUs, processes, compiler versions, or fusion choices.
The CPU case checks this test harness locally; it does not replace the CUDA gate.
"""

from copy import copy, deepcopy
import os

import numpy as np
import pytest
import torch

from RL.tdmpc2_core.inner_improvement import InnerImprovementEngine
from tests.test_ambi_entropy_probe_cuda import _assert_strict_regions_ran
from tests.test_ambi_entropy_probe_invariants import _fixture, _snapshot, _assert_unchanged
from utils import ambi_prior_kl as probe
from utils.ambi_entropy_probe import _hash, _preserve_environment


class _PlainWithSnapshots(InnerImprovementEngine):
    """Only copy initial tensors; no extra policy/Q forward or differentiation."""

    def _run_component_update_counts(self, **kwargs):
        if not hasattr(self, "initial_replay"):
            self.initial_replay = self.state.replay.z[:self.state.replay.size].detach().clone()
            self.initial_actor_state = deepcopy(self.state.actor.state_dict())
        return super()._run_component_update_counts(**kwargs)

    def _sac_policy_step(self, batch, **kwargs):
        if not hasattr(self, "first_critic_state"):
            assert self.state.actor_steps == 0
            assert self.state.critic_steps == self.cfg.inner_critic_updates_per_round
            self.first_critic_state = deepcopy(self.state.critic.state_dict())
        return super()._sac_policy_step(batch, **kwargs)


def _assert_state_exact(actual, expected, *, label):
    assert actual.keys() == expected.keys(), label
    largest_error = 0.0
    for key in actual:
        left, right = actual[key], expected[key]
        if torch.is_tensor(left):
            if left.numel() and left.is_floating_point():
                largest_error = max(largest_error, float((left - right).abs().max()))
            torch.testing.assert_close(left, right, rtol=0, atol=0,
                                       msg=lambda message: f"{label}/{key}: {message}")
        else:
            assert left == right, f"{label}/{key}"
    return largest_error


def _check_same_process_parity(wrapper, monkeypatch, record_property):
    observation = np.asarray([0.2, -0.1, 0.7], dtype=np.float32)
    seed = 197
    before = _snapshot(wrapper)
    device = wrapper.agent.device
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    observed = []
    original_observer = probe._PriorKLEngine

    class CapturedObserver(original_observer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            observed.append(self)

    monkeypatch.setattr(probe, "_PriorKLEngine", CapturedObserver)
    result = probe.probe_prior_kl_root(wrapper, observation, seed=seed,
                                      actor_steps=3, anchor_samples=8)
    assert len(observed) == 2
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)
    _assert_unchanged(wrapper, before)

    initial_alpha = float(wrapper.agent.alpha.detach())
    methods = (("baseline_entropy", 0.0, True), ("prior_kl", initial_alpha, False))
    for observed_engine, (name, beta, entropy_enabled) in zip(observed, methods):
        proxy = copy(wrapper.agent)
        proxy.cfg = deepcopy(wrapper.agent.cfg)
        proxy.cfg.inner_outer_policy_kl_coef = beta
        proxy.cfg.inner_entropy_enabled = entropy_enabled
        if not entropy_enabled:
            proxy.cfg.inner_temperature_mode = "inherit_outer"
        with _preserve_environment(wrapper.agent.model, device), torch.enable_grad():
            with torch.no_grad():
                root = wrapper.agent.model.encode(torch.as_tensor(observation, device=device)[None]).detach()
            plain = _PlainWithSnapshots(proxy)
            plain.rng = plain._new_rng(seed)
            action, metrics, _ = plain.act(root, t0=True, eval_mode=True, collect_diagnostics=False)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                _assert_strict_regions_ran(observed_engine)
                _assert_strict_regions_ran(plain)

            measured = result["full_solves"][name]
            assert measured["initial_actor_sha256"] == _hash(plain.initial_actor_state)
            assert measured["first_replay_sha256"] == _hash(plain.initial_replay)
            assert measured["first_critic_sha256"] == _hash(plain.first_critic_state)
            assert measured["final_private_rng_sha256"] == _hash(plain.rng.training_state_dict())
            np.testing.assert_array_equal(np.asarray(measured["action"]),
                                          action.detach().cpu().numpy().squeeze())
            for component in ("actor", "critic"):
                actual = getattr(observed_engine._action_pool, component).state_dict()
                expected = getattr(plain._action_pool, component).state_dict()
                error = _assert_state_exact(actual, expected, label=f"{name}/{component}")
                record_property(f"{name}_{component}_max_absolute_difference", error)
                assert measured[f"final_{component}_sha256"] == _hash(expected)
            keys = ["inner_actor_entropy", "inner_actor_loss", "inner_alpha_final"]
            keys.append("inner_temperature_loss" if entropy_enabled else "inner_outer_policy_kl")
            for key in keys:
                assert measured["metrics"][key] == float(metrics[key]), f"{name}/{key}"
            assert measured["temperature_steps"] == plain.state.temperature_steps == (4 if entropy_enabled else 0)
            assert measured["actor_steps"] == plain.state.actor_steps == 4
            assert measured["temperature_allocated"] == (plain._action_pool.log_alpha is not None)
            if not entropy_enabled:
                assert plain._action_pool.temperature_optim is None
                assert all(row["alpha_before"] == row["alpha_after"] == 0.0
                           for row in measured["actor_updates"])
            for row in measured["actor_updates"]:
                for key in ("return_q_first_order_change", "entropy_first_order_change",
                            "kl_first_order_change", "regularized_objective_first_order_change"):
                    assert np.isfinite(row["gradient_projections"][key]), f"{name}/{key}"
        _assert_unchanged(wrapper, before)
        if device.type == "cuda":
            assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)
    assert result["paired_first_phase"]


def test_prior_kl_parity_gate_harness_matches_eager_cpu_engines(monkeypatch, record_property):
    wrapper = _fixture("distributional", ent_coef="auto_0.0033654973")
    try:
        _check_same_process_parity(wrapper, monkeypatch, record_property)
    finally:
        wrapper.env.close()


@pytest.mark.skipif(
    os.environ.get("AMBI_RUN_PRIOR_KL_CUDA_TESTS") != "1" or not torch.cuda.is_available(),
    reason="Requires AMBI_RUN_PRIOR_KL_CUDA_TESTS=1 and an allocated CUDA GPU.",
)
def test_strict_cuda_prior_kl_observer_matches_both_plain_engines(monkeypatch, record_property):
    wrapper = _fixture("distributional", device="cuda", compile=True, compile_strict=True,
                       ent_coef="auto_0.0033654973")
    try:
        assert wrapper.agent.device.type == "cuda"
        _check_same_process_parity(wrapper, monkeypatch, record_property)
    finally:
        wrapper.env.close()
