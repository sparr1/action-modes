"""Opt-in strict CUDA compilation gate for entropy-observer fidelity.

Run on an allocated GPU with AMBI_RUN_ENTROPY_PROBE_CUDA_TESTS=1. This compares
two compiled engines; eager/compiled RNG equivalence is not the contract.
"""
import os

import numpy as np
import pytest
import torch

from RL.tdmpc2_core.inner_improvement import InnerImprovementEngine
from tests.test_ambi_entropy_probe_invariants import _fixture
from utils import ambi_entropy_probe as probe


pytestmark = pytest.mark.skipif(
    os.environ.get("AMBI_RUN_ENTROPY_PROBE_CUDA_TESTS") != "1"
    or not torch.cuda.is_available(),
    reason="Requires opt-in AMBI_RUN_ENTROPY_PROBE_CUDA_TESTS=1 and allocated CUDA GPU.",
)


def _assert_strict_regions_ran(engine):
    # Ordinary finite-horizon SAC inlines its prior bootstrap inside the critic
    # region. The separate prior_value region belongs to explorer paths.
    for name in ("rollout", "critic", "actor"):
        region = engine._compile_regions[name]
        assert region.enabled, f"{name} was not compiled"
        assert region.strict, f"{name} allowed an eager fallback"
        assert not region.failed, f"{name} fell back to eager"
        assert region._compiled is not None, f"{name} never executed its compiled callable"


def test_strict_cuda_compiled_observer_matches_plain_engine_with_dropout(monkeypatch):
    wrapper = _fixture("distributional", device="cuda", compile=True, compile_strict=True)
    observation = np.asarray([0.2, -0.1, 0.7], dtype=np.float32)
    observed_engines = []
    base_class = probe._EntropyProbeEngine

    class CapturedProbeEngine(base_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            observed_engines.append(self)

    monkeypatch.setattr(probe, "_EntropyProbeEngine", CapturedProbeEngine)
    try:
        device = wrapper.agent.device
        assert device.type == "cuda"
        cuda_rng = torch.cuda.get_rng_state(device).clone()
        # Its actor observer performs two autograd.grad calls on the same
        # compiled forward before the original loss.backward. Unsupported AOT
        # multiple-backward behavior must fail here; no eager retry is allowed.
        result = probe.probe_entropy_root(
            wrapper, observation, seed=197, actor_steps=3, anchor_samples=8,
            exact_engine=True,
        )
        torch.cuda.synchronize(device)
        assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)
        assert len(observed_engines) == 1
        _assert_strict_regions_ran(observed_engines[0])
        with probe._preserve_environment(wrapper.agent.model, device):
            with torch.no_grad():
                root = wrapper.agent.model.encode(
                    torch.as_tensor(observation, device=device)[None]
                ).detach()
            plain = InnerImprovementEngine(wrapper.agent)
            plain.rng = plain._new_rng(197)
            action, metrics, _ = plain.act(
                root, t0=True, eval_mode=True, collect_diagnostics=False,
            )
            torch.cuda.synchronize(device)
            _assert_strict_regions_ran(plain)
            measured = result["exact_engine"]
            np.testing.assert_array_equal(
                np.asarray(measured["action"]), action.detach().cpu().numpy().squeeze(),
            )
            assert measured["final_actor_sha256"] == probe._hash(plain._action_pool.actor.state_dict())
            assert measured["final_critic_sha256"] == probe._hash(plain._action_pool.critic.state_dict())
            assert measured["final_private_rng_sha256"] == probe._hash(plain.rng.training_state_dict())
            for key in ("inner_actor_entropy", "inner_actor_loss", "inner_alpha_final",
                        "inner_temperature_loss"):
                assert measured["metrics"][key] == float(metrics[key])
        assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)
        updates = result["exact_engine"]["actor_updates"]
        assert len(updates) == 4
        assert all(row["actual_update_critic_training"] for row in updates)
        assert all(row["gradient_mode"] == "actual_actor_training_forward_same_noise_dropout_and_head_pair"
                   for row in updates)
        for row in updates:
            projections = row["gradient_projections"]
            assert np.isfinite(projections["return_q_first_order_change"])
            assert np.isfinite(projections["entropy_first_order_change"])
    finally:
        wrapper.env.close()
