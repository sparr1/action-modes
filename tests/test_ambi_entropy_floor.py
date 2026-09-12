import math
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from RL.tdmpc2_core.inner_improvement import (
    InnerImprovementEngine,
    InnerWorkspace,
)
from tests.test_ambi_inner_decoupling import (
    _assert_tree_equal,
    _clone_tree,
    _model,
)


ALPHA_FLOOR = 1e-8


def _engine(
    *,
    mode="auto",
    outer_alpha=0.2,
    initialization="inherit_outer",
    outer_auto=False,
):
    engine = object.__new__(InnerImprovementEngine)
    engine.cfg = SimpleNamespace(
        inner_operator="sac",
        inner_temperature_mode=mode,
        inner_temperature_initialization=initialization,
        inner_temperature=0.25,
    )
    engine.agent = SimpleNamespace(
        alpha=torch.tensor(float(outer_alpha)),
        log_ent_coef=(
            torch.nn.Parameter(torch.tensor(math.log(ALPHA_FLOOR)))
            if outer_auto
            else None
        ),
    )
    engine.device = torch.device("cpu")
    engine.state = InnerWorkspace()
    return engine


def test_learned_and_inherited_alpha_access_is_floored_but_fixed_is_unchanged():
    learned = _engine(mode="auto")
    learned.state.log_alpha = torch.nn.Parameter(torch.tensor(math.log(1e-30)))
    assert learned.alpha.item() == pytest.approx(ALPHA_FLOOR)

    inherited = _engine(mode="inherit_outer", outer_auto=True)
    inherited.state.alpha_fixed = torch.tensor(0.0)
    assert inherited.alpha.item() == pytest.approx(ALPHA_FLOOR)

    fixed = _engine(mode="fixed")
    fixed.state.alpha_fixed = torch.tensor(1e-12)
    torch.testing.assert_close(
        fixed.alpha,
        fixed.state.alpha_fixed,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("mode", ["auto", "inherit_outer"])
def test_zero_configured_inner_temperature_requires_fixed_mode(mode):
    from tests.test_ambi_root_local_sac import _build_cfg

    with pytest.raises(ValueError, match="zero with inner_temperature_mode='fixed'"):
        _build_cfg(inner_temperature=0.0, inner_temperature_mode=mode)


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), -float("inf"), True])
def test_fixed_inner_temperature_rejects_invalid_values(value):
    from tests.test_ambi_root_local_sac import _build_cfg

    with pytest.raises(ValueError, match="inner_temperature"):
        _build_cfg(inner_temperature=value, inner_temperature_mode="fixed")


def test_zero_fixed_scratch_solve_preserves_initialization_rng_and_update_counts():
    """A zero entropy coefficient changes gradients without changing the solve protocol."""
    from copy import deepcopy
    from pathlib import Path

    from RL.tdmpc2_core.inner_trace import InnerActionTrace
    from tests.test_ambi_root_local_sac import _tiny_component_model
    from utils.ambi_research import load_preset_matrix

    matrix = load_preset_matrix(
        Path(__file__).resolve().parents[1] / "configs/research/ambi_scratch_takeoff_h1.json"
    )
    params = {key: value for key, value in matrix["shared_alg_params"].items() if value is not None}
    params.update(sac_actor_loss_scale_mode="tdmpc2_percentile_range", ent_coef=1e-4,
                  outer_critic_target="reward_only", q_representation="distributional", num_q=5)
    baseline = _tiny_component_model(**params)
    zero = _tiny_component_model(**{**params, "inner_temperature": 0.0})
    before = deepcopy(zero.agent.model.state_dict())
    traces = [InnerActionTrace(probes=True, probe_mode="outer_tail", probe_rollouts=32,
                               probe_horizon=1, capture_actors=True) for _ in range(2)]
    try:
        for model, trace in zip((baseline, zero), traces):
            action = model.agent.act(torch.zeros(3), t0=True, collect_diagnostics=False, trace=trace)
            assert torch.isfinite(action).all()
            assert [s.actor_updates for s in trace.actor_snapshots] == [0, 4, 8, 12, 16]
            assert [s.critic_updates for s in trace.actor_snapshots] == [0, 32, 64, 96, 128]
            assert all(math.isfinite(float(value)) for event in trace.events
                       for value in event["metrics"].values() if isinstance(value, (int, float)))

        # Same actual random initialization and same draws throughout the solve.
        assert traces[0].actor_snapshots[0].sha256 == traces[1].actor_snapshots[0].sha256
        _assert_tree_equal(baseline.agent.inner_engine.rng.training_state_dict(),
                           zero.agent.inner_engine.rng.training_state_dict())
        _assert_tree_equal(zero.agent.model.state_dict(), before)

        updates = [event for event in traces[1].events if event["phase"] == "update"]
        assert len(updates) == 144
        assert all(event["metrics"]["alpha_used"] == 0.0 for event in updates)
        actor_updates = [event for event in updates if event["updated_actor"]]
        assert len(actor_updates) == 16
        assert all(event["metrics"]["actor_entropy_bonus"] == 0.0 for event in actor_updates)
        pool = zero.agent.inner_engine._action_pool
        assert pool.alpha_fixed.item() == 0.0
        assert pool.log_alpha is None and pool.temperature_optim is None
        assert zero.cfg.inner_temperature_updates_per_action == 0
    finally:
        baseline.env.close()
        zero.env.close()


def test_learned_inner_initialization_floors_an_underflowed_outer_alpha():
    engine = _engine(mode="auto", outer_alpha=0.0)

    assert engine._initial_inner_alpha().item() == pytest.approx(ALPHA_FLOOR)


class _RNG:
    @contextmanager
    def fork(self, _name):
        yield None


def _temperature_only_actor_region(z, *_args):
    log_prob = z.new_full((*z.shape[:-1], 1), -1.0)
    entropy = torch.zeros_like(log_prob)
    zero = z.new_zeros(())
    return log_prob, entropy, zero, zero, zero, zero, zero, zero


@pytest.mark.parametrize(
    ("initial_alpha", "learning_rate", "expected_log_alpha"),
    [
        (2e-8, 100.0, math.log(ALPHA_FLOOR)),
        (0.2, 0.01, math.log(0.2) - 0.01),
    ],
)
def test_learned_inner_log_alpha_is_clamped_only_when_an_update_crosses_floor(
    initial_alpha,
    learning_rate,
    expected_log_alpha,
):
    engine = _engine(mode="auto")
    engine.cfg.sac_actor_loss_scale_mode = "none"
    engine.cfg.inner_target_entropy = 0.0
    engine.cfg.action_dim = 1
    engine.cfg.inner_outer_policy_kl_coef = 0.0
    engine.cfg.inner_temperature_grad_clip_norm = 1e6
    engine.rng = _RNG()
    engine._compile_regions = {"actor": _temperature_only_actor_region}
    engine.state.log_alpha = torch.nn.Parameter(
        torch.tensor(math.log(initial_alpha))
    )
    engine.state.temperature_optim = torch.optim.SGD(
        [engine.state.log_alpha], lr=learning_rate
    )

    metrics = engine._sac_policy_step(
        {"z": torch.zeros(2, 3)},
        update_temperature=True,
        update_actor=False,
        alpha=engine.alpha.detach(),
    )

    assert engine.state.log_alpha.item() == pytest.approx(expected_log_alpha)
    assert engine.alpha.item() == pytest.approx(
        max(initial_alpha * math.exp(-learning_rate), ALPHA_FLOOR)
    )
    assert metrics["temperature_loss"].isfinite()
    assert engine.state.temperature_steps == 1
    assert engine.state.temperature_lifetime_steps == 1


def _stub_outer_actor(monkeypatch, agent, *, log_prob_value=-1.0):
    def policy(z):
        anchor = next(agent.model._pi.parameters()).reshape(-1)[0]
        action_shape = (*z.shape[:-1], agent.cfg.action_dim)
        pre_tanh = torch.zeros(action_shape, device=z.device, dtype=z.dtype)
        pre_tanh = pre_tanh + anchor * 0.0
        log_std = torch.zeros_like(pre_tanh)
        log_prob = torch.full(
            (*z.shape[:-1], 1),
            log_prob_value,
            device=z.device,
            dtype=z.dtype,
        )
        log_prob = log_prob + anchor * 0.0
        return torch.tanh(pre_tanh), {
            "pre_tanh_mean": pre_tanh,
            "pre_tanh_action": pre_tanh,
            "log_std": log_std,
            "log_prob": log_prob,
            "entropy": -log_prob,
        }

    def critic(z, action, **kwargs):
        values = action[..., :1] * 0.0
        if kwargs.get("reduction") == "all":
            return values.unsqueeze(0).expand(int(agent.cfg.num_q), *values.shape)
        return values

    monkeypatch.setattr(agent.model, "pi", policy)
    monkeypatch.setattr(agent.model, "Q", critic)
    return torch.zeros(
        int(agent.cfg.train_unroll_horizon) + 1,
        int(agent.cfg.batch_size),
        int(agent.cfg.latent_dim),
        device=agent.device,
    )


def test_outer_auto_alpha_is_floored_after_temperature_update(monkeypatch):
    agent = _model(
        ent_coef="auto",
        ent_coef_lr=100.0,
        target_entropy=0.0,
    ).agent
    agent.log_ent_coef.data.fill_(math.log(2e-8))
    zs = _stub_outer_actor(monkeypatch, agent)

    metrics = agent._update_actor(zs)

    assert agent.alpha.item() == pytest.approx(ALPHA_FLOOR)
    assert agent.log_ent_coef.item() == pytest.approx(math.log(ALPHA_FLOOR))
    assert metrics["ent_coef_floor_hit"] == 1.0


def test_outer_alpha_access_and_checkpoint_load_defensively_apply_floor():
    source = _model(ent_coef="auto").agent
    source.log_ent_coef.data.fill_(-1000.0)
    assert source.alpha.item() == pytest.approx(ALPHA_FLOOR)
    state = source.checkpoint_state()

    restored = _model(ent_coef="auto").agent
    restored.load(state)

    assert restored.alpha.item() == pytest.approx(ALPHA_FLOOR)
    assert restored.log_ent_coef.item() == pytest.approx(math.log(ALPHA_FLOOR))


def test_outer_auto_alpha_initialization_respects_floor():
    agent = _model(ent_coef="auto_1e-30").agent

    assert agent.alpha.item() == pytest.approx(ALPHA_FLOOR)
    assert agent.log_ent_coef.item() == pytest.approx(math.log(ALPHA_FLOOR))


def test_outer_fixed_alpha_below_floor_is_unchanged():
    agent = _model(ent_coef=1e-12).agent

    assert agent.log_ent_coef is None
    assert agent.alpha.item() == pytest.approx(1e-12)


def _run_scoped_temperature_state(**overrides):
    model = _model(inner_temperature_scope="run", **overrides)
    model.agent.act(torch.zeros(3), t0=True, collect_diagnostics=False)
    model.agent.prepare_training_resume_boundary()
    return _clone_tree(model.agent.training_state_dict())


def test_inner_exact_resume_clamps_a_legacy_learned_log_alpha():
    config = {
        "inner_temperature_mode": "auto",
        "inner_temperature_initialization": "fixed",
        "inner_temperature_updates_per_action": 1,
        "inner_temperature_optimizer_scope": "run",
    }
    state = _run_scoped_temperature_state(**config)
    state["inner"]["workspace"]["log_alpha"].fill_(-1000.0)

    restored = _model(inner_temperature_scope="run", **config).agent
    restored.load_training_state_dict(state)

    assert restored.inner_engine.state.log_alpha.item() == pytest.approx(
        math.log(ALPHA_FLOOR)
    )
    assert restored.inner_engine.alpha.item() == pytest.approx(ALPHA_FLOOR)


@pytest.mark.parametrize(
    ("mode", "field", "value", "message"),
    [
        ("auto", "log_alpha", float("nan"), "log_alpha must be finite"),
        ("inherit_outer", "alpha_fixed", float("nan"), "alpha_fixed must be finite"),
        (
            "inherit_outer",
            "alpha_fixed",
            -1.0,
            "alpha_fixed must be non-negative",
        ),
    ],
)
def test_inner_exact_resume_rejects_invalid_temperature_state_transactionally(
    mode,
    field,
    value,
    message,
):
    config = {"inner_temperature_mode": mode}
    if mode == "auto":
        config.update(
            inner_temperature_initialization="fixed",
            inner_temperature_updates_per_action=1,
            inner_temperature_optimizer_scope="run",
        )
    state = _run_scoped_temperature_state(**config)
    state["inner"]["workspace"][field].fill_(value)
    target = _model(inner_temperature_scope="run", **config).agent
    pristine = _clone_tree(target.training_state_dict())

    with pytest.raises(ValueError, match=message):
        target.load_training_state_dict(state)

    _assert_tree_equal(target.training_state_dict(), pristine)


def test_inherited_fixed_positive_alpha_below_floor_is_preserved_on_resume():
    config = {
        "ent_coef": 1e-12,
        "inner_temperature_mode": "inherit_outer",
    }
    state = _run_scoped_temperature_state(**config)
    saved_alpha = state["inner"]["workspace"]["alpha_fixed"]
    assert saved_alpha.item() == pytest.approx(1e-12)

    restored = _model(inner_temperature_scope="run", **config).agent
    restored.load_training_state_dict(state)

    assert restored.inner_engine.state.alpha_fixed.item() == pytest.approx(1e-12)
    assert restored.inner_engine.alpha.item() == pytest.approx(1e-12)
