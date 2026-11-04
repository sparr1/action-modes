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


def _reference_agent(optimized, first_layer_fusion=False):
    params = _tiny_params(
        outer_actor_objective="distillation", inner_temperature_mode="inherit_outer",
        reanalyze_backend="batched_cuda_graph", reanalyze_parallel_roots=4,
        reanalyze_batched_workers=1, reanalyze_batched_optimizations=optimized,
        reanalyze_batched_first_layer_fusion=first_layer_fusion,
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
    assert cfg.reanalyze_batched_first_layer_fusion is first_layer_fusion
    if not first_layer_fusion:
        assert "reanalyze_batched_first_layer_fusion" not in agent.distillation.specification()
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
    # tolerance applies to the strict, highest-precision arithmetic gate.
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


def _precision_state(core):
    """Retain CPU tensors so three full graph workspaces need not coexist."""
    clone = lambda value: value.detach().cpu().clone()
    families = {
        name: {key: clone(value) for key, value in getattr(core, name).items()}
        for name in ("actor", "critic", "target", "prior_actor", "prior_critic")
    }
    families["replay"] = {"rows": clone(core.replay[:, :core.size])}
    families["scalars"] = {
        name: clone(getattr(core, name)) for name in ("scale", "log_alpha", "alpha_fixed")
    }
    steps = {}
    for name, optimizer in (
        ("actor", core.actor_optim), ("critic", core.critic_optim),
        ("temperature", core.temperature_optim),
    ):
        if optimizer is None:
            continue
        for index, state in enumerate(optimizer.state.values()):
            assert state.keys() == {"step", "exp_avg", "exp_avg_sq"}
            steps[f"{name}_{index}"] = clone(state["step"])
            # First and second moments have different units and magnitudes;
            # neither can supply the numerical budget for the other.
            for moment in ("exp_avg", "exp_avg_sq"):
                families.setdefault(f"{name}_{moment}", {})[str(index)] = clone(state[moment])
    return {
        "families": families, "steps": steps,
        "structure": (core.width, core.size, core.position, core.critic_steps),
    }


def _assert_precision_budget(actual, reference, highest, *, name="state", record_property=None):
    """Bound each root's RMS drift by the reference's measured precision drift.

    Sixteen FP32 epsilons of reference RMS allow a small arithmetic-roundoff
    floor even when the reference high/highest paths happen to agree exactly.
    There is no absolute/data-dependent floor: an entirely zero reference family
    must stay zero. Double accumulation keeps the comparison itself precise.
    """
    assert actual.keys() == reference.keys() == highest.keys(), name
    assert actual, name
    roots = next(iter(reference.values())).shape[0]
    sums = torch.zeros(3, roots, dtype=torch.float64)
    elements = 0
    for key, value in actual.items():
        expected, baseline = reference[key], highest[key]
        assert value.shape == expected.shape == baseline.shape, f"{name}.{key} shape"
        assert value.dtype == expected.dtype == baseline.dtype == torch.float32, f"{name}.{key} dtype"
        assert value.shape[0] == roots, f"{name}.{key} roots"
        assert all(torch.isfinite(tensor).all() for tensor in (value, expected, baseline)), f"{name}.{key} nonfinite"
        value, expected, baseline = (tensor.detach().double().cpu().reshape(roots, -1)
                                     for tensor in (value, expected, baseline))
        sums[0] += (value - expected).square().sum(dim=1)
        sums[1] += (expected - baseline).square().sum(dim=1)
        sums[2] += baseline.square().sum(dim=1)
        elements += value.shape[1]
    assert elements > 0, name
    drift, precision_drift, magnitude = (sums / elements).sqrt()
    floor = 16 * torch.finfo(torch.float32).eps * magnitude
    budget = precision_drift + floor
    metrics = {
        "drift_rms_per_root": drift.tolist(),
        "reference_precision_drift_rms_per_root": precision_drift.tolist(),
        "roundoff_floor_per_root": floor.tolist(),
        "budget_rms_per_root": budget.tolist(),
    }
    if record_property is not None:
        for key, value in metrics.items():
            record_property(f"high_{name}_{key}", value)
    assert (drift <= budget).all(), f"{name} exceeds reference precision budget: {metrics}"
    return metrics


def _assert_precision_state_budget(actual, reference, highest, record_property):
    assert actual["structure"] == reference["structure"] == highest["structure"]
    assert actual["families"].keys() == reference["families"].keys() == highest["families"].keys()
    assert actual["steps"].keys() == reference["steps"].keys() == highest["steps"].keys()
    for key in actual["steps"]:
        torch.testing.assert_close(actual["steps"][key], reference["steps"][key], rtol=0, atol=0)
        torch.testing.assert_close(reference["steps"][key], highest["steps"][key], rtol=0, atol=0)
    for name, family in actual["families"].items():
        expected, baseline = reference["families"][name], highest["families"][name]
        assert family.keys() == expected.keys() == baseline.keys(), name
        if name in ("prior_actor", "prior_critic", "scalars"):
            for key, value in family.items():
                assert all(torch.isfinite(tensor).all() for tensor in (value, expected[key], baseline[key]))
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
                torch.testing.assert_close(expected[key], baseline[key], rtol=0, atol=0)
        else:
            _assert_precision_budget(family, expected, baseline, name=name, record_property=record_property)


def _exercise_reference_recipe(precision, record_property, first_layer_fusion=False):
    torch.manual_seed(193)
    reference_agent = _reference_agent(False)
    optimized_agent = _reference_agent(True, first_layer_fusion=first_layer_fusion)
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
        if precision == "highest":
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
    if precision == "highest":
        _assert_adapted_state_close(optimized.core, reference.core)
        return

    # TF32 kernels may change rounding when optimization changes operation
    # shapes. Measure its budget using the unoptimized reference itself, with
    # precisely the same starting priors, encoded roots and per-root seeds.
    actual_state = _precision_state(optimized.core)
    expected_state = _precision_state(reference.core)
    initial_state = {key: value.detach().cpu().clone() for key, value in reference_agent.state_dict().items()}
    actual = tuple(value.detach().cpu().clone() for value in actual[:3])
    expected = tuple(value.detach().cpu().clone() for value in expected[:3])
    del reference, optimized, reference_agent, optimized_agent, reference_outer, optimized_outer
    gc.collect()
    torch.cuda.empty_cache()
    torch._dynamo.reset()
    try:
        torch.set_float32_matmul_precision("highest")
        with _preserve_global_rng():
            baseline_agent = _reference_agent(False)
            baseline_agent.load_state_dict(initial_state)
            baseline_outer = _snapshot_outer(baseline_agent)
            baseline_solver = BatchedTeacherSolver(baseline_agent)
            baseline_host_rng, baseline_cuda_rng = _globals(), torch.cuda.get_rng_state().clone()
            baseline = baseline_solver.solve(roots, seeds)
            _assert_globals(baseline_host_rng)
            torch.testing.assert_close(torch.cuda.get_rng_state(), baseline_cuda_rng, rtol=0, atol=0)
            _assert_outer_unchanged(baseline_agent, baseline_outer)
            baseline_state = _precision_state(baseline_solver.core)
            baseline = tuple(value.detach().cpu().clone() for value in baseline[:3])
    finally:
        torch.set_float32_matmul_precision(precision)
    _assert_globals(host_rng)
    torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng, rtol=0, atol=0)
    torch.testing.assert_close(expected[2], baseline[2], rtol=0, atol=0)
    _assert_precision_state_budget(actual_state, expected_state, baseline_state, record_property)
    actual_policy, expected_policy, baseline_policy = (
        tuple(value.double() for value in result[:2]) for result in (actual, expected, baseline)
    )
    for direction, current, teacher, reference_current, reference_teacher in (
        ("optimized_to_reference", actual_policy, expected_policy, expected_policy, baseline_policy),
        ("reference_to_optimized", expected_policy, actual_policy, baseline_policy, expected_policy),
    ):
        drift = diagonal_gaussian_reverse_kl(*current, *teacher) / 21
        precision_drift = diagonal_gaussian_reverse_kl(*reference_current, *reference_teacher) / 21
        assert torch.isfinite(drift).all() and torch.isfinite(precision_drift).all()
        # Clamp mathematically nonnegative KL at zero to remove double-precision
        # cancellation; the 1e-12 floor is additional to the unchanged 1e-6 cap.
        budget = precision_drift.clamp_min(0) + 1e-12
        record_property(f"high_{direction}_kl_per_dimension_per_root", drift.flatten().tolist())
        record_property(f"high_{direction}_reference_precision_kl_per_root", precision_drift.flatten().tolist())
        assert (drift <= 1e-6).all()
        assert (drift <= budget).all(), f"{direction}: KL={drift.tolist()}, budget={budget.tolist()}"


@pytest.mark.parametrize("first_layer_fusion", [False, True], ids=["existing", "first_layer_fused"])
@pytest.mark.parametrize("precision", ["highest", "high"])
def test_full_reference_recipe_matches_previous_arithmetic(precision, first_layer_fusion, record_property):
    record_property("first_layer_fusion", first_layer_fusion)
    previous_precision = torch.get_float32_matmul_precision()
    previous_cache_size = torch._dynamo.config.cache_size_limit
    previous_accumulated = torch._dynamo.config.accumulated_cache_size_limit
    torch._dynamo.reset()
    try:
        torch.set_float32_matmul_precision(precision)
        # Retain production compiler cache limits: this correctness gate must
        # expose exhausted compilation budgets rather than widen them.
        with _preserve_global_rng():
            _exercise_reference_recipe(precision, record_property, first_layer_fusion=first_layer_fusion)
    finally:
        torch._dynamo.reset()
        torch._dynamo.config.cache_size_limit = previous_cache_size
        torch._dynamo.config.accumulated_cache_size_limit = previous_accumulated
        torch.set_float32_matmul_precision(previous_precision)
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.parametrize("first_layer_fusion", [False, True], ids=["existing", "first_layer_fused"])
def test_full_reference_recipe_concurrent_workers_are_bitwise_identical_at_production_precision(first_layer_fusion, record_property):
    """Concurrency preserves identical fixed-width kernels, seeds and budgets."""
    from RL.tdmpc2_core.batched_teacher_pool import ConcurrentBatchedTeacherSolver

    record_property("first_layer_fusion", first_layer_fusion)
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
            agent = _reference_agent(True, first_layer_fusion=first_layer_fusion)
            with torch.no_grad():
                observations = torch.randn(14, 67, device="cuda")
                roots = agent.model.encode(observations).detach()
            seeds = list(range(377, 391))
            snapshot = _snapshot_outer(agent)
            host_rng, cuda_rng = _globals(), torch.cuda.get_rng_state().clone()
            serial = BatchedTeacherSolver(agent)
            expected = serial.solve(roots, seeds)
            _assert_globals(host_rng)
            torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng, rtol=0, atol=0)
            _assert_outer_unchanged(agent, snapshot)
            torch.testing.assert_close(
                expected[2], roots.new_tensor([14 * 3072, 14 * 32, 14 * 128]), rtol=0, atol=0,
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
                    assert actual[3]["distillation_reanalyze_chunks"] == 4
                    assert actual[3]["distillation_reanalyze_padding_roots"] == 2
                    # Fourteen roots have four width-four chunks: two workers
                    # reuse their workspaces, while all four workers are active
                    # in the larger pool. The last chunk retains two pad rows.
                    assert actual[3]["distillation_reanalyze_batched_workers"] == min(workers, 4)
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
