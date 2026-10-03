"""Opt-in GPU gate for fresh critic heads; run on an allocated CUDA device."""

import json
import math
import os

import pytest
import torch

from tests import test_ambi_critic_transfer_cuda_gate as gate
from tests.test_aux_critic_transfer import assert_optimizer_reset, component_ids
from tests.test_ambi_inner_decoupling import _assert_tree_equal


def _instrument_hidden_boundaries(model):
    engine = model.agent.inner_engine
    prepare = engine._prepare_workspace
    observed = {"previous": None, "solves": 0, "allocations": []}

    def checked_prepare(*, t0):
        prepare(t0=t0)
        state = engine.state
        if t0:
            observed["solves"] = 0
        expected = engine._critic_base.state_dict() if t0 else observed["previous"]
        assert expected is not None
        head_prefixes = tuple(f"modules_list.{index}.{len(member) - 1}."
                              for index, member in enumerate(state.critic))
        actual_body = {key: value for key, value in state.critic.state_dict().items()
                       if not key.startswith(head_prefixes)}
        expected_body = {key: value for key, value in expected.items()
                         if not key.startswith(head_prefixes)}
        _assert_tree_equal(actual_body, expected_body)
        _assert_tree_equal(state.critic_target.state_dict(), state.critic.state_dict())
        _assert_tree_equal(state.actor.state_dict(), engine._actor_base.state_dict())
        for member in state.critic:
            head = member[-1]
            assert torch.count_nonzero(head.weight) > 0
            assert head.weight.abs().max() <= math.sqrt(6. / (head.in_features + head.out_features))
            assert torch.count_nonzero(head.bias) == 0
        assert state.critic_lifetime_steps == observed["solves"] * 2
        assert state.actor_lifetime_steps == state.temperature_lifetime_steps == 0
        assert state.critic_steps == state.actor_steps == state.temperature_steps == 0
        assert state.target_steps == state.critic_target_steps == state.replay.size == 0
        torch.testing.assert_close(engine.alpha, engine._initial_inner_alpha(), rtol=0, atol=0)
        assert_optimizer_reset(state.actor_optim, state.actor.parameters())
        assert_optimizer_reset(state.critic_optim, state.critic.parameters())
        assert_optimizer_reset(state.temperature_optim, [state.log_alpha])
        observed["allocations"].append(component_ids(state))
        observed["solves"] += 1

    engine._prepare_workspace = checked_prepare
    act = model.agent.act

    def checked_act(*args, **kwargs):
        action = act(*args, **kwargs)
        metrics = model.agent.last_inner_metrics
        assert metrics["inner_critic_head_reinitialized"] == metrics["inner_solve_performed"]
        return action

    model.agent.act = checked_act
    return observed


def _select_random_heads(monkeypatch):
    original_params = gate.critic_params
    monkeypatch.setattr(gate, "critic_params", lambda *args, **kwargs:
                        original_params(*args, inner_critic_transfer_head="random", **kwargs))
    monkeypatch.setattr(gate, "_instrument_boundaries", _instrument_hidden_boundaries)


def test_hidden_cuda_gate_fixture_contract_on_cpu(monkeypatch):
    """Exercise the reused gate and new instrumentation without a GPU compile."""
    _select_random_heads(monkeypatch)
    result = gate._run_fixture_gate("return", 3, device="cpu", compile_enabled=False)
    assert result["passed"] and result["allocation_reuse"]


@pytest.mark.skipif(os.environ.get("AMBI_RUN_CRITIC_HIDDEN_TRANSFER_CUDA_GATE") != "1",
                    reason="set AMBI_RUN_CRITIC_HIDDEN_TRANSFER_CUDA_GATE=1 on an allocated GPU")
@pytest.mark.parametrize("critic", ["soft", "return"])
@pytest.mark.parametrize("interval", [1, 3])
def test_cuda_hidden_transfer_strict_compile_parity_and_stochastic_lifecycle(monkeypatch, critic, interval):
    assert torch.cuda.is_available(), "Requested hidden-transfer CUDA gate requires a GPU."
    _select_random_heads(monkeypatch)
    deterministic = gate._run_fixture_gate(critic, interval, inner_critic_dropout_enabled=False)
    stochastic = gate._run_fixture_gate(critic, interval, reference_compile=True,
                                       inner_critic_dropout_enabled=True)
    report = {
        "passed": True, "critic": critic, "solve_interval": interval,
        "inner_critic_transfer_head": "random", "weight_initializer": "xavier_uniform",
        "bias_initializer": "zeros", "reset_timing": "every_solve_including_first",
        "deterministic_numerical_parity": deterministic,
        "compiled_stochastic_lifecycle": stochastic,
        "production_checkpoint_tested": False,
    }
    print("CRITIC_HIDDEN_TRANSFER_CUDA_GATE_REPORT " + json.dumps(report, sort_keys=True))
