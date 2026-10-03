"""Local checks for the CUDA gate's distinction between parity and lifecycle."""

from types import SimpleNamespace

import pytest
import torch

from tests.test_ambi_critic_transfer_cuda_gate import (
    _assert_close_modules,
    _dropout_rng_probe,
    _global_rng,
    _run_fixture_gate,
)
from tests.test_ambi_inner_decoupling import _assert_tree_equal


@pytest.fixture
def eager_compile(monkeypatch):
    original = torch.compile
    monkeypatch.setattr(torch, "compile", lambda function, **kwargs:
                        original(function, backend="eager", **kwargs))


def test_dropout_probe_observes_backend_masks_without_requiring_a_difference(eager_compile):
    before = _global_rng(torch.device("cpu"))
    result = _dropout_rng_probe("cpu")
    assert result["compiled_same_seed_replay_exact"] and result["global_rng_preserved"]
    # The eager compiler backend deliberately has the same RNG implementation.
    assert result["same_seed_eager_compiled_masks_equal"]
    assert result["same_seed_eager_compiled_post_rng_equal"]
    assert result["differing_mask_elements"] == 0
    _assert_tree_equal(_global_rng(torch.device("cpu")), before)


def test_cross_backend_dropout_parity_is_rejected_before_model_allocation():
    with pytest.raises(ValueError, match="Cross-backend numerical parity"):
        _run_fixture_gate("return", 3, device="cpu", inner_critic_dropout_enabled=True)


@pytest.mark.parametrize("critic", ["soft", "return"])
@pytest.mark.parametrize("stochastic", [False, True])
def test_split_gate_preserves_original_tolerances_and_exact_stochastic_replication(
    eager_compile, critic, stochastic,
):
    report = _run_fixture_gate(
        critic, 3, device="cpu", reference_compile=stochastic,
        inner_critic_dropout_enabled=stochastic,
    )
    assert report["passed"] and report["compile_strict"] and not report["compile_fallback"]
    assert report["comparison"] == (
        "compiled_stochastic_reproducibility" if stochastic else "eager_compiled_deterministic_parity")
    assert report["inner_critic_dropout_enabled"] == stochastic
    assert report["module_dropout_probability"] == .01
    assert report["comparison_atol"] == (0 if stochastic else 2e-5)
    assert report["comparison_rtol"] == (0 if stochastic else 2e-4)
    assert report["paired_action_max_absolute_difference"] == 0
    assert report["private_rng_states_equal"]
    assert report["solves_per_controller"] == 6
    assert report["checkpoint_load_followup_solves_per_controller"] == 1
    assert set(report["paired_module_maximum_drift"]) == {"actor", "critic", "critic_target"}
    assert all(value["max_absolute_difference"] == 0
               for value in report["paired_module_maximum_drift"].values())


def test_parameter_parity_failures_identify_context_component_and_name():
    def engine():
        state = SimpleNamespace(**{name: torch.nn.Linear(2, 2) for name in (
            "actor", "critic", "critic_target")})
        return SimpleNamespace(state=state, _action_pool=SimpleNamespace())

    left, right = engine(), engine()
    for name in ("actor", "critic", "critic_target"):
        getattr(right.state, name).load_state_dict(getattr(left.state, name).state_dict())
    with torch.no_grad():
        right.state.critic.weight.add_(.01)
    with pytest.raises(AssertionError, match=r"seed=101, decision=0: critic\.weight"):
        _assert_close_modules(left, right, context="seed=101, decision=0")
