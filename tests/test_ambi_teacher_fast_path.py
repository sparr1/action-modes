"""Teacher-only execution preserves native SAC learning and private RNG."""

from dataclasses import fields

import pytest
import torch

from RL.tdmpc2_core.inner_improvement import InnerImprovementEngine
from tests.test_ambi_inner_decoupling import _assert_tree_equal, _clone_tree
from tests.test_ambi_outer_replay import _buffer, _episode
from tests.test_ambi_root_local_sac import _tiny_model


@pytest.fixture
def models():
    opened = []

    def create(**overrides):
        options = {
            "inner_rounds": 2, "inner_updates_per_round": 1,
            "inner_replay_capacity": 8, "inner_diagnostic_rollouts": 0,
        }
        options.update(overrides)
        model = _tiny_model(**options)
        model.agent.model.eval()
        opened.append(model)
        return model

    yield create
    for model in opened:
        model.env.close()


def _workspace_state(workspace):
    result = {}
    for descriptor in fields(workspace):
        value = getattr(workspace, descriptor.name)
        if isinstance(value, torch.nn.Module):
            value = {
                "parameters": value.state_dict(),
                "modes": [module.training for module in value.modules()],
                "gradients": [parameter.grad for parameter in value.parameters()],
                "requires_grad": [parameter.requires_grad for parameter in value.parameters()],
            }
        elif hasattr(value, "state_dict"):
            value = value.state_dict()
        result[descriptor.name] = _clone_tree(value)
    return result


def _assert_engine_equivalent(actual, expected):
    _assert_tree_equal(_workspace_state(actual.state), _workspace_state(expected.state))
    _assert_tree_equal(_workspace_state(actual._action_pool), _workspace_state(expected._action_pool))
    _assert_tree_equal(actual.rng.training_state_dict(), expected.rng.training_state_dict())
    assert actual.action_index == expected.action_index
    assert actual.episode_index == expected.episode_index
    assert actual._collect_diagnostics == expected._collect_diagnostics


@pytest.mark.parametrize("adaptation", ["clone", "lora_rl"])
@pytest.mark.parametrize("entropy", ["reward_only", "entropy_augmented"])
@pytest.mark.parametrize("feature", ["ordinary", "finite_horizon", "real_replay"])
def test_teacher_fast_path_matches_native_fresh_and_pooled_solves(models, adaptation, entropy, feature):
    options = {
        "inner_critic_adaptation": adaptation,
        "outer_critic_target": entropy, "inner_sac_critic_target": entropy,
        "q_representation": "distributional", "num_q": 3, "dropout": 0.1,
        "inner_critic_dropout_enabled": True,
    }
    if adaptation == "lora_rl":
        options["inner_critic_lora_rank"] = 2
    if feature == "finite_horizon":
        options["inner_finite_horizon"] = True
    elif feature == "real_replay":
        options["inner_outer_replay_fraction"] = 0.5
    model = models(**options)
    reference, fast = (InnerImprovementEngine(model.agent) for _ in range(2))
    if feature == "real_replay":
        replay = _buffer()
        replay.add(_episode())
        reference.outer_replay_buffer = fast.outer_replay_buffer = replay
    first_actor_id = None
    for turn, seed in enumerate((81, 117)):
        with torch.no_grad():
            if turn:
                model.agent.model._pi[-1].bias.add_(0.03)
            root = model.agent.model.encode(torch.full((1, 3), 0.1 * (turn + 1))).detach()
        prior = _clone_tree(model.agent.model.state_dict())
        reference.reset_for_evaluation(seed, reuse_action_pool=True)
        fast.reset_for_evaluation(seed, reuse_action_pool=True)
        global_rng = torch.random.get_rng_state().clone()
        _, expected_counters, _ = reference.act(
            root, t0=True, eval_mode=True, collect_diagnostics=False,
            capture_teacher_policy=True,
        )
        with torch.no_grad():  # API still enables its local optimizer backwards.
            teacher, counters = fast.solve_teacher(root, t0=True)
        _assert_tree_equal(teacher, reference.last_teacher_policy)
        for key, value in counters.items():
            assert value == float(expected_counters[key])
        _assert_engine_equivalent(fast, reference)
        _assert_tree_equal(model.agent.model.state_dict(), prior)
        torch.testing.assert_close(torch.random.get_rng_state(), global_rng, rtol=0, atol=0)
        assert fast._pending_timers == {}
        assert not fast._teacher_only
        if first_actor_id is None:
            first_actor_id = id(fast._action_pool.actor)
        else:
            assert id(fast._action_pool.actor) == first_actor_id


@pytest.mark.parametrize("options", [
    {"inner_horizon_conditioning": "one_hot", "inner_finite_horizon": True},
    {"inner_update_timing": "step", "inner_steps_per_update": 2,
     "inner_updates_per_round": None},
    {"inner_actor_entropy_mode": "tdmpc2_scaled", "inner_temperature_mode": "inherit_outer",
     "ent_coef": 0.05, "sac_actor_loss_scale_mode": "tdmpc2_percentile_range",
     "inner_actor_loss_scale_update": "per_update"},
])
def test_teacher_fast_path_preserves_horizon_step_and_entropy_scaling(models, options):
    model = models(**options)
    root = model.agent.model.encode(torch.zeros(1, 3)).detach()
    reference, fast = (InnerImprovementEngine(model.agent) for _ in range(2))
    for engine in (reference, fast):
        engine.reset_for_evaluation(29, reuse_action_pool=True)
    _, expected_counters, _ = reference.act(root, t0=True, eval_mode=True,
                                           collect_diagnostics=False, capture_teacher_policy=True)
    teacher, counters = fast.solve_teacher(root)
    _assert_tree_equal(teacher, reference.last_teacher_policy)
    _assert_engine_equivalent(fast, reference)
    for key, value in counters.items():
        assert value == float(expected_counters[key])


def test_teacher_only_omits_execution_reporting_and_timers(models, monkeypatch):
    model = models()
    engine = InnerImprovementEngine(model.agent)
    root = model.agent.model.encode(torch.zeros(1, 3)).detach()

    def unused(*args, **kwargs):
        pytest.fail("Teacher-only execution performed unused action/reporting work")

    for name in ("_execute_policy", "_stats", "_average_update_metrics", "_base_metrics",
                 "_compile_fallback_metrics", "_source_td_metrics"):
        monkeypatch.setattr(engine, name, unused)
    import RL.tdmpc2_core.inner_improvement as inner_module
    monkeypatch.setattr(inner_module, "policy_sample_diagnostics", unused)
    monkeypatch.setattr(inner_module, "entropy_coefficient_diagnostics", unused)
    with monkeypatch.context() as timers:
        timers.setattr(inner_module.time, "perf_counter", unused)
        teacher, counters = engine.solve_teacher(root)
    assert set(teacher) == {"pre_tanh_mean", "log_std"}
    assert counters["inner_total_model_steps"] == 8
    assert engine._pending_timers == {}
    assert isinstance(engine._timer_start(), float)


def test_teacher_only_inactive_sac_and_failure_restore_flags(models, monkeypatch):
    model = models(inner_rounds=0)
    engine = InnerImprovementEngine(model.agent)
    root = model.agent.model.encode(torch.zeros(1, 3)).detach()
    teacher, counters = engine.solve_teacher(root)
    assert all(value == 0 for value in counters.values())
    assert all(not value.requires_grad for value in teacher.values())
    before = _clone_tree(engine.rng.training_state_dict())
    with engine.rng.action_fork():
        with pytest.raises(RuntimeError, match="not reentrant"):
            engine.solve_teacher(root)
    _assert_tree_equal(engine.rng.training_state_dict(), before)

    def fail(*args, **kwargs):
        raise RuntimeError("teacher failure")

    monkeypatch.setattr(engine, "_capture_teacher_policy", fail)
    with pytest.raises(RuntimeError, match="teacher failure"):
        engine.solve_teacher(root)
    assert not engine._teacher_only
    assert engine.last_teacher_policy is None
    assert engine._pending_timers == {}


def test_compiled_teacher_fast_path_matches_eager_with_reused_allocations(models, monkeypatch):
    """Exercise strict Dynamo graphs without requiring a local GPU compiler."""
    torch._dynamo.reset()
    original_compile = torch.compile
    compiled_regions, graphs = [], []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    def compile_graph(function, **kwargs):
        assert kwargs["fullgraph"] is True
        compiled_regions.append(function.__name__)
        return original_compile(function, backend=backend, **kwargs)

    monkeypatch.setattr(torch, "compile", compile_graph)
    options = {
        "inner_critic_adaptation": "clone",
        "critic_ensemble_backend": "loop", "q_representation": "distributional",
        "num_q": 3, "dropout": 0.1, "inner_critic_dropout_enabled": True,
    }
    eager_model = models(**options)
    compiled_model = models(**options, compile=True, compile_strict=True)
    eager = InnerImprovementEngine(eager_model.agent)
    compiled = InnerImprovementEngine(compiled_model.agent)
    global_rng = torch.random.get_rng_state().clone()
    try:
        for turn, seed in enumerate((101, 103, 107, 109)):
            with torch.no_grad():
                if turn:
                    for model in (eager_model, compiled_model):
                        for parameter in model.agent.model.parameters():
                            parameter.add_(0.0001)
                roots = [model.agent.model.encode(torch.full((1, 3), 0.1 * turn)).detach()
                         for model in (eager_model, compiled_model)]
            prior = _clone_tree(compiled_model.agent.model.state_dict())
            for engine in (eager, compiled):
                engine.reset_for_evaluation(seed, reuse_action_pool=True)
            expected = eager.solve_teacher(roots[0])
            actual = compiled.solve_teacher(roots[1])
            _assert_tree_equal(actual, expected)
            _assert_engine_equivalent(compiled, eager)
            _assert_tree_equal(compiled_model.agent.model.state_dict(), prior)
            assert all(parameter.grad is None for parameter in compiled_model.agent.model.parameters())
            assert all(not region.failed for region in compiled._compile_regions.values())
            torch.testing.assert_close(torch.random.get_rng_state(), global_rng, rtol=0, atol=0)
            if turn == 1:
                warm_graph_count = len(graphs)
                assert warm_graph_count > 0
            elif turn > 1:
                assert len(graphs) == warm_graph_count
        assert {"_dense_rollout_kernel", "_sac_critic_kernel", "_sac_actor_kernel"} <= set(compiled_regions)
    finally:
        torch._dynamo.reset()
