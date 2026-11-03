"""Static teacher-graph contracts and scheduled-CUDA parity checks."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
from RL.tdmpc2_core.inner_improvement import InnerImprovementEngine
from RL.tdmpc2_core.teacher_graph import (
    TEACHER_GRAPH_PROTOCOL, TeacherGraph, validate_teacher_graph_config,
)
from tests.test_ambi_root_local_sac import _build_cfg, _tiny_params


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a scheduled CUDA GPU.")


def _cfg(*, scaled=False, compiled=False, entropy="squashed"):
    params = _tiny_params(
        inner_rounds=2, inner_rollouts_per_round=4,
        inner_rollout_horizon=2, inner_updates_per_round=1,
        inner_batch_size=4, inner_replay_capacity=16,
        # Keep independent-head/dropout RNG branches represented in parity.
        q_representation="distributional", num_q=5, q_pair_size=2,
        q_num_bins=11, num_bins=11, dropout=0.01,
        inner_critic_dropout_enabled=True,
        outer_actor_entropy_mode=entropy, inner_actor_entropy_mode=entropy,
        target_entropy=-1.0, inner_target_entropy=-1.0,
        outer_critic_target="reward_only", inner_sac_critic_target="entropy_augmented",
        inner_finite_horizon=True, compile=compiled, compile_strict=compiled,
        sac_actor_loss_scale_mode="tdmpc2_percentile_range" if scaled else "none",
        inner_temperature_mode="inherit_outer" if scaled else "auto",
        ent_coef=0.01 if scaled else "auto_0.01",
    )
    return _build_cfg(**params)


@pytest.mark.parametrize("scaled", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("entropy", ["squashed", "tdmpc2_scaled"])
def test_teacher_graph_config_supports_fixed_static_recipes_on_cpu(scaled, compiled, entropy):
    validate_teacher_graph_config(_cfg(scaled=scaled, compiled=compiled, entropy=entropy))


@pytest.mark.parametrize("key,value", [
    ("obs", "rgb"), ("episodic", True), ("inner_operator", "td3"),
    ("inner_critic_adaptation", "lora_rl"), ("inner_actor_adaptation", "frozen"),
    ("inner_actor_initialization", "random"), ("inner_critic_initialization", "random"),
    ("inner_explorer_mode", "frozen_random"), ("inner_actor_scope", "episode"),
    ("inner_outer_replay_fraction", 0.5), ("inner_sac_return_estimator", "retrace"),
    ("aux_return_mode", "sac"), ("critic_value_mode", "return_entropy"),
    ("inner_horizon_conditioning", "one_hot"), ("inner_critic_bernoulli_p", 0.5),
    ("inner_actor_writeback_coef", 0.1), ("inner_steps_per_update", 2.0),
    ("inner_update_timing", "step"),
])
def test_teacher_graph_rejects_unsupported_protocols_without_allocating_cuda(key, value):
    cfg = _cfg()
    setattr(cfg, key, value)
    with pytest.raises(ValueError, match=key):
        validate_teacher_graph_config(cfg)


def test_teacher_graph_rejects_compile_fallback_and_inactive_solver():
    cfg = _cfg()
    cfg.compile = True
    with pytest.raises(ValueError, match="compile_strict"):
        validate_teacher_graph_config(cfg)
    cfg.compile = False
    cfg.inner_actor_updates_per_action = 0
    with pytest.raises(ValueError, match="positive rounds and actor updates"):
        validate_teacher_graph_config(cfg)


def test_teacher_graph_has_explicit_protocol_and_requires_cuda():
    agent = AMBITDMPC2Agent(_cfg())
    engine = InnerImprovementEngine(agent)
    root = torch.zeros(1, agent.cfg.latent_dim)
    assert TeacherGraph.protocol == TEACHER_GRAPH_PROTOCOL
    assert "per_root_seed" in TEACHER_GRAPH_PROTOCOL
    with pytest.raises(ValueError, match="CUDA engine and CUDA root"):
        TeacherGraph(engine, root)


@pytest.mark.parametrize("join_fails", [False, True])
def test_failed_constructor_joins_unreturned_slot_and_preserves_error(monkeypatch, join_fails):
    cfg = _cfg()
    device = torch.device("cuda:0")
    root = SimpleNamespace(ndim=2, shape=(1, cfg.latent_dim), device=device)
    engine = SimpleNamespace(cfg=cfg, device=device, model=SimpleNamespace(training=False))
    joined = []

    class Stream:
        def synchronize(self):
            joined.append(True)
            if join_fails:
                raise RuntimeError("cleanup failure")

    original_is_tensor = torch.is_tensor
    monkeypatch.setattr(torch, "is_tensor", lambda value: value is root or original_is_tensor(value))
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: Stream())

    def fail_setup(*args, **kwargs):
        raise ValueError("original capture failure")

    monkeypatch.setattr(TeacherGraph, "_initialize", fail_setup)
    with pytest.raises(ValueError, match="original capture failure"):
        TeacherGraph(engine, root)
    assert joined == [True]


def _agent(**kwargs):
    cfg = _cfg(**kwargs)
    cfg.device = "cuda"
    torch.manual_seed(17)
    agent = AMBITDMPC2Agent(cfg)
    agent.model.eval()
    return agent


def _snapshot(engine):
    pool = engine._action_pool
    result = {}
    for name in ("actor", "critic", "critic_target", "actor_target"):
        value = getattr(pool, name)
        result[name] = None if value is None else deepcopy(value.state_dict())
    for name in ("actor_optim", "critic_optim", "temperature_optim"):
        value = getattr(pool, name)
        result[name] = None if value is None else deepcopy(value.state_dict())
    result["log_alpha"] = None if pool.log_alpha is None else pool.log_alpha.detach().clone()
    result["alpha_fixed"] = None if pool.alpha_fixed is None else pool.alpha_fixed.detach().clone()
    result["replay"] = pool.replay._storage[:pool.replay.size].clone()
    return result


def _assert_close(actual, expected):
    if torch.is_tensor(actual):
        torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-7)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_close(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            _assert_close(a, b)
    else:
        assert actual == expected


def _copy_output(runner, output):
    # Follow the coordinator contract: copy before slot reuse, on its stream.
    teacher, counters = output
    with torch.cuda.stream(runner.stream):
        result = {key: value.clone() for key, value in teacher.items()}
    torch.cuda.current_stream(runner.device).wait_stream(runner.stream)
    return result, dict(counters)


@CUDA
@pytest.mark.parametrize("scaled,compiled,entropy", [
    (False, False, "squashed"),
    (False, True, "squashed"),
    (True, True, "squashed"),
    (True, True, "tdmpc2_scaled"),
])
def test_teacher_graph_matches_uncaptured_protocol_and_current_priors(scaled, compiled, entropy):
    agent = _agent(scaled=scaled, compiled=compiled, entropy=entropy)
    root = torch.zeros(1, agent.cfg.latent_dim, device="cuda")
    control = TeacherGraph(InnerImprovementEngine(agent), root, capture=False)
    graph = TeacherGraph(InnerImprovementEngine(agent), root)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    from torch._dynamo.utils import counters
    compile_count = int(counters["stats"]["unique_graphs"])
    for iteration, seed in enumerate((41, 53, 41)):
        with torch.no_grad():
            for parameter in agent.model.parameters():
                parameter.add_(0.0001)
            if agent.log_ent_coef is not None:
                agent.log_ent_coef.add_(0.1)
            else:
                agent.fixed_ent_coef.mul_(1.1)
            if scaled:
                agent.actor_loss_scale.fill_(2.0 + iteration)
        outer = deepcopy(agent.model.state_dict())
        next_root = root + iteration * 0.1
        expected = _copy_output(control, control(next_root, seed=seed))
        actual = _copy_output(graph, graph(next_root, seed=seed))
        torch.cuda.current_stream().synchronize()
        _assert_close(actual, expected)
        _assert_close(_snapshot(graph.engine), _snapshot(control.engine))
        _assert_close(agent.model.state_dict(), outer)
        assert all(parameter.grad is None for parameter in agent.model.parameters())
    assert int(counters["stats"]["unique_graphs"]) == compile_count
    torch.testing.assert_close(torch.get_rng_state(), cpu_rng, rtol=0, atol=0)
    torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng, rtol=0, atol=0)


@CUDA
def test_teacher_graph_slots_overlap_without_rng_or_workspace_aliasing(monkeypatch):
    agent = _agent()
    root = torch.zeros(1, agent.cfg.latent_dim, device="cuda")
    reference = TeacherGraph(InnerImprovementEngine(agent), root)
    workers = [TeacherGraph(InnerImprovementEngine(agent), root) for _ in range(2)]
    expected = [_copy_output(reference, reference(root + i * 0.1, seed=61 + i)) for i in range(2)]
    repeated = _copy_output(reference, reference(root, seed=61))
    torch.cuda.current_stream().synchronize()
    _assert_close(repeated, expected[0])
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    outputs = []
    with monkeypatch.context() as patch:
        def forbidden(*args, **kwargs):
            raise AssertionError("Teacher replay must not synchronize.")
        patch.setattr(torch.cuda, "synchronize", forbidden)
        patch.setattr(torch.cuda.Stream, "synchronize", forbidden)
        for i, runner in enumerate(workers):
            teacher, counts = runner(root + i * 0.1, seed=61 + i)
            with torch.cuda.stream(runner.stream):
                outputs.append(({key: value.clone() for key, value in teacher.items()}, dict(counts)))
        for runner in workers:
            torch.cuda.current_stream().wait_stream(runner.stream)
    torch.cuda.current_stream().synchronize()
    for actual, wanted in zip(outputs, expected):
        _assert_close(actual, wanted)
    for name in ("actor", "critic"):
        left = next(getattr(workers[0].engine._action_pool, name).parameters())
        right = next(getattr(workers[1].engine._action_pool, name).parameters())
        assert left.data_ptr() != right.data_ptr()
    torch.testing.assert_close(torch.get_rng_state(), cpu_rng, rtol=0, atol=0)
    torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng, rtol=0, atol=0)
