"""Full-budget CUDA correctness gate for the measured reanalysis recipe.

These tests deliberately retain the production architecture, replay, dropout,
and all 128 critic / 32 actor updates. They compare optimization against the
previous arithmetic path using the same root seeds without a large RNG tape.
Run only on an allocated GPU; this module is collected but skipped on CPU.
"""

import gc

import pytest
import torch

from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
from RL.tdmpc2_core.batched_teacher import BatchedTeacherSolver
from RL.tdmpc2_core.common.math import diagonal_gaussian_reverse_kl
from RL.tdmpc2_core.reanalyze import _preserve_global_rng
from tests.benchmark_aux_return_joint_vmap import _build_cfg, _tiny_params
from tests.test_ambi_inner_decoupling import _assert_tree_equal, _clone_tree
from tests.test_ambi_reanalyze import _assert_globals, _globals


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Full reference recipe requires a scheduled CUDA GPU.",
)


def _reference_agent(optimized):
    params = _tiny_params(
        outer_actor_objective="distillation", inner_temperature_mode="inherit_outer",
        reanalyze_backend="batched_cuda_graph", reanalyze_parallel_roots=4,
        reanalyze_batched_workers=1, reanalyze_batched_optimizations=optimized,
        compile=True, compile_strict=True, wandb=False,
        enc_dim=256, mlp_dim=512, latent_dim=512, simnorm_dim=8,
        num_q=5, num_bins=101, q_num_bins=101,
        q_representation="distributional", q_vmin=-10, q_vmax=10,
        dropout=0.01, critic_ensemble_backend="vmap",
        train_unroll_horizon=3, outer_planning_horizon=3,
        inner_rounds=8, inner_rollouts_per_round=128,
        inner_rollout_horizon=3, inner_batch_size=256,
        inner_replay_capacity=3072,
        inner_critic_updates_per_round=16, inner_actor_updates_per_round=4,
        inner_temperature_updates_per_round=0,
        sac_actor_loss_scale_mode="none",
    )
    params.pop("inner_updates_per_round")
    cfg = _build_cfg(**params)
    cfg.device = "cuda"
    cfg.obs_shape = {"state": (67,)}
    cfg.obs_shapes = [cfg.obs_shape]
    cfg.action_dim = 21
    cfg.action_dims = [21]
    agent = AMBITDMPC2Agent(cfg)
    agent.model.eval()
    assert cfg.inner_critic_updates_per_action == 128
    assert cfg.inner_actor_updates_per_action == 32
    assert cfg.inner_critic_dropout_enabled is True
    assert cfg.inner_q_actor_reduction == cfg.inner_q_target_reduction == "min_pair"
    assert cfg.q_pair_size == 2
    return agent


def _snapshot_outer(agent):
    # Exercise restoration of a mixed train/eval module tree, including caller
    # gradients that an inference-only teacher must neither clear nor replace.
    agent.model.train()
    agent.model._pi[0].eval()
    for parameter in agent.model.parameters():
        parameter.grad = torch.full_like(parameter, 0.125)
    return {
        "state": _clone_tree(agent.training_state_dict()),
        "modes": tuple((module, module.training) for module in agent.modules()),
        "gradients": tuple(parameter.grad.clone() for parameter in agent.model.parameters()),
    }


def _assert_outer_unchanged(agent, snapshot):
    _assert_tree_equal(agent.training_state_dict(), snapshot["state"])
    assert all(module.training == training for module, training in snapshot["modes"])
    for parameter, gradient in zip(agent.model.parameters(), snapshot["gradients"]):
        torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)


def _assert_adapted_state_close(actual, expected):
    # Absolute tolerance protects small/near-zero Adam moments; relative
    # tolerance applies uniformly to both highest and production high precision.
    tolerance = dict(rtol=5e-4, atol=1e-5)
    for name in ("actor", "critic", "target", "prior_actor", "prior_critic"):
        for key, value in getattr(actual, name).items():
            torch.testing.assert_close(
                value, getattr(expected, name)[key], **tolerance,
                msg=lambda message, name=name, key=key: f"{name}.{key}: {message}",
            )
    for name in ("scale", "log_alpha", "alpha_fixed"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), **tolerance)
    assert (actual.size, actual.position, actual.critic_steps) == (
        expected.size, expected.position, expected.critic_steps,
    )
    torch.testing.assert_close(actual.replay[:, :actual.size], expected.replay[:, :expected.size], **tolerance)
    assert len(actual.optimizers) == len(expected.optimizers)
    for actual_optimizer, expected_optimizer in zip(actual.optimizers, expected.optimizers):
        assert len(actual_optimizer.state) == len(expected_optimizer.state)
        for actual_state, expected_state in zip(actual_optimizer.state.values(), expected_optimizer.state.values()):
            assert actual_state.keys() == expected_state.keys()
            for key in actual_state:
                torch.testing.assert_close(
                    actual_state[key], expected_state[key],
                    **(dict(rtol=0, atol=0) if key == "step" else tolerance),
                    msg=lambda message, key=key: f"Adam {key}: {message}",
                )


def _exercise_reference_recipe(precision, record_property):
    torch.manual_seed(193)
    reference_agent = _reference_agent(False)
    optimized_agent = _reference_agent(True)
    optimized_agent.load_state_dict(reference_agent.state_dict())
    with torch.no_grad():
        observations = torch.randn(4, 67, device="cuda")
        roots = reference_agent.model.encode(observations).detach()
    seeds = [173, 174, 175, 176]
    reference_outer = _snapshot_outer(reference_agent)
    optimized_outer = _snapshot_outer(optimized_agent)
    reference = BatchedTeacherSolver(reference_agent)
    optimized = BatchedTeacherSolver(optimized_agent)
    host_rng, cuda_rng = _globals(), torch.cuda.get_rng_state().clone()

    expected = reference.solve(roots, seeds)
    _assert_globals(host_rng)
    torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng, rtol=0, atol=0)
    actual = optimized.solve(roots, seeds)
    _assert_globals(host_rng)
    torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng, rtol=0, atol=0)
    _assert_outer_unchanged(reference_agent, reference_outer)
    _assert_outer_unchanged(optimized_agent, optimized_outer)

    for name, value, expected_value in zip(("mean", "log_std"), actual[:2], expected[:2]):
        assert torch.isfinite(value).all()
        error = (value - expected_value).abs().max().item()
        record_property(f"{precision}_teacher_{name}_max_abs_error", error)
        torch.testing.assert_close(value, expected_value, rtol=5e-4, atol=1e-5)
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
    torch.testing.assert_close(actual[2], roots.new_tensor([4 * 3072, 4 * 32, 4 * 128]), rtol=0, atol=0)

    # KL in double precision gives a direct distribution-level check separate
    # from tensor tolerances, including amplification by small teacher scales.
    actual_mean, actual_log_std = (value.double() for value in actual[:2])
    expected_mean, expected_log_std = (value.double() for value in expected[:2])
    for direction, current, teacher in (
        ("optimized_to_reference", (actual_mean, actual_log_std), (expected_mean, expected_log_std)),
        ("reference_to_optimized", (expected_mean, expected_log_std), (actual_mean, actual_log_std)),
    ):
        per_dimension_kl = diagonal_gaussian_reverse_kl(*current, *teacher) / 21
        maximum = per_dimension_kl.max().item()
        record_property(f"{precision}_{direction}_max_kl_per_dimension", maximum)
        assert torch.isfinite(per_dimension_kl).all()
        assert maximum <= 1e-6, f"{direction} KL/action dimension={maximum}"
    _assert_adapted_state_close(optimized.core, reference.core)


@pytest.mark.parametrize("precision", ["highest", "high"])
def test_full_reference_recipe_matches_previous_arithmetic(precision, record_property):
    previous_precision = torch.get_float32_matmul_precision()
    previous_cache_size = torch._dynamo.config.cache_size_limit
    previous_accumulated = torch._dynamo.config.accumulated_cache_size_limit
    torch._dynamo.reset()
    try:
        torch.set_float32_matmul_precision(precision)
        # Retain production compiler cache limits: this correctness gate must
        # expose exhausted compilation budgets rather than widen them.
        with _preserve_global_rng():
            _exercise_reference_recipe(precision, record_property)
    finally:
        torch._dynamo.reset()
        torch._dynamo.config.cache_size_limit = previous_cache_size
        torch._dynamo.config.accumulated_cache_size_limit = previous_accumulated
        torch.set_float32_matmul_precision(previous_precision)
        gc.collect()
        torch.cuda.empty_cache()


def test_full_reference_recipe_concurrent_workers_are_bitwise_identical_at_production_precision(record_property):
    """Concurrency preserves identical fixed-width kernels, seeds and budgets."""
    from RL.tdmpc2_core.batched_teacher_pool import ConcurrentBatchedTeacherSolver

    previous_precision = torch.get_float32_matmul_precision()
    previous_cache_size = torch._dynamo.config.cache_size_limit
    previous_accumulated = torch._dynamo.config.accumulated_cache_size_limit
    torch._dynamo.reset()
    try:
        torch.set_float32_matmul_precision("high")
        # Do not enlarge either compiler cache limit: the full eight-round
        # recipe must compile for every independent worker under normal limits.
        with _preserve_global_rng():
            torch.manual_seed(297)
            agent = _reference_agent(True)
            with torch.no_grad():
                observations = torch.randn(10, 67, device="cuda")
                roots = agent.model.encode(observations).detach()
            seeds = list(range(377, 387))
            snapshot = _snapshot_outer(agent)
            host_rng, cuda_rng = _globals(), torch.cuda.get_rng_state().clone()
            serial = BatchedTeacherSolver(agent)
            expected = serial.solve(roots, seeds)
            _assert_globals(host_rng)
            torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng, rtol=0, atol=0)
            _assert_outer_unchanged(agent, snapshot)
            torch.testing.assert_close(
                expected[2], roots.new_tensor([10 * 3072, 10 * 32, 10 * 128]), rtol=0, atol=0,
            )
            # Release graph-private pools between configurations, retaining the
            # compiler cache so repeated worker shapes still exercise its limit.
            del serial
            gc.collect()
            torch.cuda.empty_cache()
            for workers in (2, 4):
                agent.cfg.reanalyze_batched_workers = workers
                snapshot = _snapshot_outer(agent)
                pool = ConcurrentBatchedTeacherSolver(agent)
                for repetition in range(2):
                    actual = pool.solve(roots, seeds)
                    for name, value, target in zip(("mean", "log_std", "counts"), actual[:3], expected[:3]):
                        torch.testing.assert_close(
                            value, target, rtol=0, atol=0,
                            msg=lambda message, name=name: f"workers={workers}, {name}: {message}",
                        )
                    assert actual[3]["distillation_reanalyze_chunks"] == 3
                    assert actual[3]["distillation_reanalyze_padding_roots"] == 2
                    # Ten roots have three width-four chunks: a two-worker
                    # pool reuses its first workspace; a four-worker setting
                    # needs only three workers for this particular batch.
                    assert actual[3]["distillation_reanalyze_batched_workers"] == min(workers, 3)
                    _assert_outer_unchanged(agent, snapshot)
                    _assert_globals(host_rng)
                    torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng, rtol=0, atol=0)
                    record_property(f"high_workers_{workers}_repetition_{repetition}_bitwise_identical", True)
                del pool
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        torch._dynamo.reset()
        torch._dynamo.config.cache_size_limit = previous_cache_size
        torch._dynamo.config.accumulated_cache_size_limit = previous_accumulated
        torch.set_float32_matmul_precision(previous_precision)
        gc.collect()
        torch.cuda.empty_cache()
