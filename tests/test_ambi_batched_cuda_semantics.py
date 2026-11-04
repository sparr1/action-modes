"""CUDA compilation must preserve the independent-SAC reference equations."""

from copy import deepcopy

import pytest
import torch

from RL.tdmpc2_core.batched_teacher import BatchedTeacherCore
from RL.tdmpc2_core.common.compile_regions import _preserve_host_rng_state
from tests.test_ambi_batched_teacher import _agent, Tape
from tests.test_ambi_batched_optimizations import ALL, ReplayTape, assert_core_close
from tests.test_ambi_inner_decoupling import _assert_tree_equal
from tests.test_ambi_reanalyze import _globals, _assert_globals


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a scheduled CUDA GPU.")
def test_compiled_optimized_cuda_matches_eager_reference_full_sac_state_with_same_draws():
    """Compare different execution paths, rather than two compiled solvers.

    The tape fixes replay indices, root-specific head pairs, dropout and Gaussian
    noise. Both rounds cache the actor; the horizon tail evaluates the prior;
    automatic temperature learning exercises its gradient/Adam state as well.
    """
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    torch._dynamo.reset()
    try:
        torch.manual_seed(294)
        agent = _agent(
            device="cuda", q_representation="distributional",
            inner_rounds=2, inner_critic_updates_per_round=3,
            inner_actor_updates_per_round=2, inner_temperature_updates_per_round=2,
            inner_batch_size=8, inner_replay_capacity=16,
            inner_temperature_mode="auto", inner_actor_entropy_mode="tdmpc2_scaled",
            inner_log_std_mapping="tdmpc2_tanh", inner_finite_horizon=True,
            inner_q_actor_reduction="mean_pair", inner_q_target_reduction="min_pair",
            mppi_terminal_q_reduction="mean_pair", dropout=.2,
            inner_critic_dropout_enabled=True,
        )
        reference = BatchedTeacherCore(agent, 3, compiled=False, optimization_features=())
        optimized = BatchedTeacherCore(agent, 3, compiled=True, optimization_features=ALL)
        roots = torch.randn_like(reference.root)
        reference.root.copy_(roots)
        optimized.root.copy_(roots)
        # Pay for lazy compiler construction before checking RNG isolation. The
        # production solver uses this same host/tensor preservation discipline.
        device = torch.cuda.current_device()
        with _preserve_host_rng_state(), torch.random.fork_rng(devices=[device]):
            optimized.run()
        torch.cuda.synchronize()
        outer_before = deepcopy(agent.model.state_dict())
        for iteration in range(2):
            if iteration:
                with torch.no_grad():
                    for parameter in agent.model.parameters():
                        parameter.add_(.002)
                    if agent.log_ent_coef is not None:
                        agent.log_ent_coef.add_(.1)
                reference.refresh_inputs()
                optimized.refresh_inputs()
                outer_before = deepcopy(agent.model.state_dict())
            tape = Tape(reference)
            expected = reference.run(tape)
            # Explicit replay should consume no tensor, Python or NumPy draws.
            host_rng = _globals()
            cuda_rng = torch.cuda.get_rng_state().clone()
            reader = ReplayTape(tape.entries)
            actual = optimized.run(reader)
            torch.cuda.synchronize()
            assert reader.index == len(tape.entries)
            for value, target in zip(actual, expected):
                torch.testing.assert_close(value, target, rtol=5e-4, atol=1e-5)
            assert_core_close(optimized, reference, rtol=5e-4, atol=1e-5)
            _assert_tree_equal(agent.model.state_dict(), outer_before)
            assert all(parameter.grad is None for parameter in agent.model.parameters())
            _assert_globals(host_rng)
            torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng, rtol=0, atol=0)
    finally:
        torch._dynamo.reset()
        torch.set_float32_matmul_precision(previous_precision)
