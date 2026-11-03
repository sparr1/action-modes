from copy import deepcopy

import pytest
import torch
from tensordict import TensorDict

from RL.tdmpc2_core.common.buffer import Buffer
from tests.test_ambi_inner_decoupling import _assert_tree_equal, _clone_tree, _model


def model(**overrides):
    return _model(**{"outer_actor_objective": "distillation", **overrides})


def teacher(agent, *, valid=True):
    shape = (agent.cfg.train_unroll_horizon, agent.cfg.batch_size, agent.cfg.action_dim)
    return {
        "mean": torch.zeros(shape),
        "log_std": torch.zeros(shape),
        "valid": torch.full((*shape[:-1], 1), valid, dtype=torch.bool),
        "version": torch.full((*shape[:-1], 1), agent.outer_version if valid else -1, dtype=torch.int64),
    }


def batch(agent):
    h, b, a = agent.cfg.train_unroll_horizon, agent.cfg.batch_size, agent.cfg.action_dim
    return (torch.randn(h + 1, b, 3), torch.randn(h, b, a).tanh(),
            torch.randn(h, b, 1), torch.zeros(h, b, 1))


def stub_policy(monkeypatch, agent, *, entropy=0.5):
    def pi(zs, **kwargs):
        theta = next(agent.model._pi.parameters()).reshape(-1)[0]
        shape = (*zs.shape[:-1], agent.cfg.action_dim)
        mean = torch.ones(shape) + theta - theta.detach()
        log_std = torch.zeros_like(mean) + theta * 0
        ent = mean[..., :1] * 0 + entropy
        info = {"pre_tanh_mean": mean, "log_std": log_std,
                "log_prob": -ent, "entropy": ent}
        if kwargs.get("include_scaled_entropy"):
            info["scaled_entropy"] = ent * 2
        return mean.tanh(), info
    monkeypatch.setattr(agent.model, "pi", pi)
    return torch.zeros(agent.cfg.train_unroll_horizon + 1, agent.cfg.batch_size, agent.cfg.latent_dim, requires_grad=True)


@pytest.mark.parametrize("entropy_mode", ["squashed", "tdmpc2_scaled"])
def test_distillation_analytic_loss_masking_and_no_q_gradients(monkeypatch, entropy_mode):
    agent = model(ent_coef=0.5, outer_actor_entropy_mode=entropy_mode).agent
    zs = stub_policy(monkeypatch, agent)
    labels = teacher(agent)
    labels["mean"].requires_grad_()
    labels["log_std"].requires_grad_()
    labels["valid"][1, 1:] = False
    labels["version"][1, 1:] = -1
    with torch.no_grad():
        labels["mean"][1, 1:] = float("nan")
    monkeypatch.setattr(agent.model, "Q", lambda *a, **kw: pytest.fail("Actor distillation evaluated Q"))
    before = _clone_tree(agent.model._Qs.state_dict())
    metrics = agent._update_actor(zs, teacher=labels)
    # KL(N(1,1)||N(0,1))=0.5; scale's unit floor remains active.
    selected_entropy = 1.0 if entropy_mode == "tdmpc2_scaled" else 0.5
    h = agent.cfg.train_unroll_horizon
    horizon_factor = sum(agent.cfg.rho ** t for t in range(h)) / h
    expected = torch.tensor(horizon_factor * (0.5 - 0.5 * selected_entropy))
    torch.testing.assert_close(metrics["actor_loss"], expected)
    torch.testing.assert_close(metrics["distillation_kl"], torch.tensor(0.5))
    assert zs.grad is None
    assert labels["mean"].grad is None and labels["log_std"].grad is None
    assert all(p.grad is None for p in agent.model._Qs.parameters())
    _assert_tree_equal(agent.model._Qs.state_dict(), before)
    assert agent.distillation.actor_updates == 1


@pytest.mark.parametrize("horizon", [1, 2, 4])
def test_distillation_constant_loss_does_not_shrink_with_unroll_horizon(monkeypatch, horizon):
    agent = model(ent_coef=0.5, rho=1.0, train_unroll_horizon=horizon).agent
    zs = stub_policy(monkeypatch, agent)
    metrics = agent._update_actor(zs, teacher=teacher(agent))
    torch.testing.assert_close(metrics["actor_loss"], torch.tensor(0.25))


def test_distillation_scale_uses_only_valid_depth_zero_and_is_independent(monkeypatch):
    agent = model(ent_coef=0.5).agent
    zs = stub_policy(monkeypatch, agent)
    labels = teacher(agent)
    labels["mean"][0, :, 0] = torch.tensor([1., 3., 5., 9.])
    labels["mean"][1] = 100.0
    # Depth-zero KLs are [0, 2, 8, 32], so P95-P5=28.1.
    metrics = agent._update_actor(zs, teacher=labels)
    torch.testing.assert_close(metrics["distillation_scale"], torch.tensor([1.271]))
    assert agent.actor_loss_scale is None
    previous = agent.distillation.scale.clone()
    labels["valid"][0] = False
    labels["version"][0] = -1
    agent._update_actor(zs, teacher=labels)
    torch.testing.assert_close(agent.distillation.scale, previous)


def test_all_invalid_skips_adam_momentum_and_temperature(monkeypatch):
    agent = model().agent
    zs = stub_policy(monkeypatch, agent)
    agent._update_actor(zs, teacher=teacher(agent))
    before = _clone_tree(agent.checkpoint_state())
    rng = torch.random.get_rng_state().clone()
    metrics = agent._update_actor(zs, teacher=teacher(agent, valid=False))
    _assert_tree_equal(agent.checkpoint_state(), before)
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    assert metrics["distillation_actor_skipped"] == 1


@pytest.mark.parametrize("scaled", [False, True])
def test_exact_resume_tracks_actual_actor_steps_and_rejects_corruption(scaled):
    options = dict(ent_coef=0.5, outer_critic_target="reward_only", inner_sac_critic_target="reward_only") if scaled else {}
    if scaled:
        options["sac_actor_loss_scale_mode"] = "tdmpc2_percentile_range"
    agent = model(**options).agent
    data = batch(agent)
    agent._update(*data, teacher=teacher(agent, valid=False))
    agent._update(*data, teacher=teacher(agent))
    assert agent.num_updates == 2 and agent.distillation.actor_updates == 1
    saved = _clone_tree(agent.training_state_dict())
    restored = model(**options).agent
    restored.load_training_state_dict(saved)
    _assert_tree_equal(restored.training_state_dict(), saved)
    for corrupt in ("scale", "count", "spec", "rng"):
        damaged = deepcopy(saved)
        if corrupt == "scale":
            damaged["outer"]["distillation_state"]["scale"].fill_(float("nan"))
        elif corrupt == "count":
            damaged["outer"]["distillation_state"]["actor_updates"] = 2
        elif corrupt == "spec":
            damaged["outer"]["distillation_spec"]["reanalyze_interval"] += 1
        else:
            damaged["outer"]["distillation_state"]["reanalyzer"]["rng_state"] = torch.zeros(2, dtype=torch.uint8)
        with pytest.raises((ValueError, RuntimeError, TypeError)):
            restored.load_training_state_dict(damaged)
        _assert_tree_equal(restored.training_state_dict(), saved)


def replay(agent):
    buffer = Buffer(agent.cfg, resumable=True)
    rows = 7
    buffer.add(TensorDict({
        "obs": torch.randn(rows, 3), "action": torch.zeros(rows, 1),
        "reward": torch.ones(rows), "terminated": torch.zeros(rows),
    }, batch_size=[rows]))
    return buffer


def test_refresh_after_nine_invalid_updates_exact_resume():
    source = model(reanalyze_batch_size=1).agent
    buffer = replay(source)
    for _ in range(9):
        metrics = source.update(buffer)
        assert metrics["distillation_actor_skipped"] == 1
        assert metrics["distillation_reanalyze_ran"] == 0
    saved = _clone_tree(source.training_state_dict())
    # Use the production replay shard protocol, including its sampler state.
    manifest = buffer.training_state_metadata()
    shards = _clone_tree(list(buffer.iter_training_state_shards(max_rows=3)))
    rng = torch.random.get_rng_state().clone()
    expected_metrics = source.update(buffer)
    expected = _clone_tree(source.training_state_dict())
    restored = model(reanalyze_batch_size=1).agent
    restored.load_training_state_dict(saved)
    restored_buffer = Buffer(restored.cfg, resumable=True)
    restored_buffer.load_training_state_shards(manifest, iter(shards))
    torch.random.set_rng_state(rng)
    actual_metrics = restored.update(restored_buffer)
    assert expected_metrics["distillation_reanalyze_roots"] > 0
    assert actual_metrics["distillation_actor_skipped"] == 0
    _assert_tree_equal(restored.training_state_dict(), expected)


def test_collection_and_training_end_to_end():
    wrapped = model(reanalyze_interval=2, reanalyze_batch_size=1)
    wrapped.learn(total_timesteps=12)
    assert wrapped.agent.distillation.actor_updates > 0
    assert wrapped.agent.distillation.reanalyzer.state_dict()["roots"] > 0
    wrapped.agent.prepare_training_resume_boundary()
    saved = _clone_tree(wrapped.agent.training_state_dict())
    restored = model(reanalyze_interval=2, reanalyze_batch_size=1)
    restored.agent.load_training_state_dict(saved)
    _assert_tree_equal(restored.agent.training_state_dict(), saved)


@pytest.mark.parametrize("options", [
    {"critic_value_mode": "return_entropy", "q_representation": "distributional", "ent_coef": 0.5,
     "sac_actor_loss_scale_mode": "tdmpc2_percentile_range"},
    {"q_representation": "distributional", "num_q": 5, "outer_q_target_reduction": "min_pair"},
    {"aux_return_mode": "return_actor", "aux_return_ent_coef": 0.1},
])
def test_distillation_retains_supported_q_critics_and_checkpoint_contract(options):
    from tests.test_ambi_root_local_sac import _tiny_model
    agent = _tiny_model(outer_actor_objective="distillation", **options).agent
    data = batch(agent)
    before = _clone_tree(agent.model._Qs.state_dict())
    metrics = agent._update(*data, teacher=teacher(agent))
    assert torch.isfinite(metrics["actor_loss"]).all()
    assert any(not torch.equal(value, before[key]) for key, value in agent.model._Qs.state_dict().items())
    saved = _clone_tree(agent.training_state_dict())
    restored = _tiny_model(outer_actor_objective="distillation", **options).agent
    restored.load_training_state_dict(saved)
    _assert_tree_equal(restored.training_state_dict(), saved)


def test_q_maximization_default_is_identical_to_explicit_selection():
    implicit = _model().agent
    explicit = _model(outer_actor_objective="q_maximization").agent
    explicit.load(_clone_tree(implicit.checkpoint_state()))
    data = batch(implicit)
    rng = torch.random.get_rng_state().clone()
    expected = implicit._update(*data)
    torch.random.set_rng_state(rng)
    actual = explicit._update(*data)
    _assert_tree_equal(actual, expected)
    _assert_tree_equal(explicit.training_state_dict(), implicit.training_state_dict())
    assert implicit.distillation is explicit.distillation is None


def test_distillation_compiled_tensor_learner_matches_eager(monkeypatch):
    original_compile = torch.compile
    graphs = []
    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward
    def compile_eager(function, **kwargs):
        return original_compile(function, backend=backend, **kwargs)
    torch._dynamo.reset()
    monkeypatch.setattr(torch, "compile", compile_eager)
    try:
        eager = model().agent
        compiled = model(compile=True, compile_strict=True).agent
        compiled.load(_clone_tree(eager.checkpoint_state()))
        data = batch(eager)
        for _ in range(2):
            labels = teacher(eager)
            rng = torch.random.get_rng_state().clone()
            expected = eager._update(*data, teacher=labels)
            torch.random.set_rng_state(rng)
            actual = compiled._update(*data, teacher=labels)
            assert actual["compile_fallback"] == 0
            torch.testing.assert_close(actual["actor_loss"], expected["actor_loss"], rtol=1e-5, atol=1e-7)
            for key, value in eager.model.state_dict().items():
                torch.testing.assert_close(compiled.model.state_dict()[key], value, rtol=1e-5, atol=1e-7)
        assert graphs
    finally:
        torch._dynamo.reset()


def test_distillation_full_trainer_resume_matches_continuous(monkeypatch, tmp_path):
    """Exercise persisted replay, collection labels, and refresh across segments."""
    import sys
    from RL.AMBITDMPC2 import AMBITDMPC2
    from tests.resume_test_support import (
        BoundaryEnv, _FakeWandb, _model as resume_model, _session,
        _replay_state, _assert_tree_equal_nan,
    )
    from tests.test_resume_split_matrix import _assert_trace_equal
    from utils.resume_training import RESUME_COMPLETE, RESUME_HANDOFF

    # The production publication protocol runs entirely against this in-memory
    # test double; no W&B service or account is contacted.
    monkeypatch.setitem(sys.modules, "wandb", _FakeWandb())
    def make(env):
        return resume_model(
            env, algorithm=AMBITDMPC2, inner_scope="action", total_steps=6,
            outer_actor_objective="distillation", reanalyze_interval=2,
            reanalyze_batch_size=1,
        )
    continuous_env = BoundaryEnv()
    continuous = make(continuous_env)
    session = _session(tmp_path / "continuous", mode="new", segment="all", total_steps=6)
    try:
        assert continuous.learn(total_timesteps=6, resume_session=session) == RESUME_COMPLETE
    finally:
        session.close()
        continuous._checkpoint_writer.shutdown()
        continuous.env.close()

    trace = []
    for part in range(3):
        holder = {}
        def drain():
            holder["session"]._drain_requested = True
        env = BoundaryEnv(on_first_done=drain if part < 2 else None)
        split = make(env)
        session = _session(
            tmp_path / "split", mode="new" if part == 0 else "required",
            segment=f"part-{part}", total_steps=6,
        )
        holder["session"] = session
        try:
            result = split.learn(total_timesteps=6, resume_session=session)
            assert result == (RESUME_HANDOFF if part < 2 else RESUME_COMPLETE)
            trace.extend(env.trace)
        finally:
            session.close()
            split._checkpoint_writer.shutdown()
            split.env.close()
    _assert_trace_equal(trace, continuous_env.trace)
    _assert_tree_equal(split.agent.training_state_dict(), continuous.agent.training_state_dict())
    _assert_tree_equal_nan(_replay_state(split.buffer), _replay_state(continuous.buffer))
    assert split.agent.distillation.reanalyzer.roots > 0
