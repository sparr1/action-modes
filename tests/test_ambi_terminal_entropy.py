"""Entropy at the frozen-prior boundary is distinct from inner actor entropy."""

from contextlib import contextmanager
from copy import deepcopy

import pytest
import torch

from RL.tdmpc2_core.inner_trace import InnerActionTrace
from tests.test_ambi_inner_decoupling import _assert_tree_equal
from tests.test_ambi_root_local_sac import _build_cfg, _tiny_model
from tests.test_aux_return_inner import _prepared as _aux_prepared
from utils.eval_series_data import planner_identity
from utils.resume_identity import scientific_trial_parameters


@contextmanager
def _prepared(**overrides):
    params = dict(
        inner_finite_horizon=True, inner_terminal_entropy="outer",
        ent_coef="auto_0.2", inner_temperature_mode="fixed", inner_temperature=.9,
        inner_rounds=1, inner_updates_per_round=1,
    )
    params.update(overrides)
    holder = _tiny_model(**params)
    try:
        engine = holder.agent.inner_engine
        with engine.rng.fork("initialization"):
            engine._prepare_workspace(t0=True)
        yield holder, engine
    finally:
        holder.close()


def test_default_and_case_normalization():
    assert _build_cfg().inner_terminal_entropy == "none"
    assert _build_cfg(inner_finite_horizon=True,
                      inner_terminal_entropy="OUTER").inner_terminal_entropy == "outer"
    # Initialization and actor entropy do not select the boundary objective.
    cfg = _build_cfg(aux_return_mode="sac", inner_critic_source="aux_return",
                     inner_entropy_enabled=False, inner_terminal_entropy="outer")
    assert cfg.inner_terminal_entropy == "outer"


@pytest.mark.parametrize("mode", [None, True, 1, "auto", "inner"])
def test_invalid_modes_are_rejected(mode):
    with pytest.raises(ValueError, match="inner_terminal_entropy"):
        _build_cfg(inner_terminal_entropy=mode)


@pytest.mark.parametrize("overrides", [
    {"inner_finite_horizon": False},
    {"outer_critic_target": "reward_only"},
    {"critic_value_mode": "return_entropy"},
    {"aux_return_mode": "sac", "inner_horizon_critic_source": "aux_return"},
    {"aux_return_mode": "return_actor", "inner_horizon_actor_source": "return_actor"},
    {"inner_operator": "none", "inner_finite_horizon": False},
])
def test_outer_mode_requires_a_finite_horizon_soft_sac_tail(overrides):
    params = {"inner_finite_horizon": True, "inner_terminal_entropy": "outer", **overrides}
    with pytest.raises(ValueError, match="inner_terminal_entropy"):
        _build_cfg(**params)


@pytest.mark.parametrize("conditioning", ["none", "one_hot"])
@pytest.mark.parametrize("horizon", [1, 2])
@pytest.mark.parametrize("mode", ["none", "outer"])
@pytest.mark.parametrize("target_mode", ["reward_only", "entropy_augmented"])
@pytest.mark.parametrize("auxiliary", [False, True])
def test_boundary_adds_outer_entropy_once_and_masks_true_terminals(
    monkeypatch, horizon, mode, target_mode, auxiliary, conditioning,
):
    with _prepared(
        inner_rollout_horizon=horizon, inner_terminal_entropy=mode,
        inner_horizon_conditioning=conditioning, inner_horizon_diagnostics=True,
        inner_sac_critic_target=target_mode, aux_return_mode="sac" if auxiliary else "off",
    ) as (holder, engine):
        calls = []

        def policy(z, **kwargs):
            outer = kwargs.get("policy") in (None, engine.model._pi)
            calls.append(outer)
            return z.new_full((len(z), 1), .75 if outer else -.5), {
                "log_prob": z.new_full((len(z), 1), -2. if outer else -8.),
            }

        def outer_q(z, action, **kwargs):
            torch.testing.assert_close(action, torch.full_like(action, .75))
            assert kwargs["reduction"] == holder.cfg.mppi_terminal_q_reduction
            assert kwargs.get("qs", engine.model._Qs) is engine.model._Qs
            assert not kwargs.get("target", False)
            # Unused continuation values must not poison truly terminal rows.
            return z.new_tensor([[11.], [11.], [float("nan")], [float("inf")]])

        monkeypatch.setattr(engine.model, "pi", policy)
        monkeypatch.setattr(engine.model, "Q", outer_q)
        monkeypatch.setattr(engine, "_bootstrap_q", lambda z, *args, **kwargs: z.new_full((len(z), 1), 5.))
        z = torch.zeros(4, holder.cfg.latent_dim)
        boundary = torch.ones(4, 1) if horizon == 1 else torch.tensor([[0.], [1.], [0.], [1.]])
        output = engine._sac_critic_kernel(
            z, torch.zeros(4, 1), torch.full((4, 1), 3.), z,
            torch.tensor([[0.], [0.], [1.], [1.]]), torch.tensor(.9),
            torch.zeros(4, 1), None, boundary, torch.zeros(4, 1),
            **engine._horizon_kwargs(torch.where(boundary.bool(), 1, horizon)),
        )
        interior = 5. + (.9 * 8. if target_mode == "entropy_augmented" else 0.)
        tail = 11. + (.2 * 2. if mode == "outer" else 0.)
        expected = 3. + holder.agent.discount * torch.tensor([
            [tail if horizon == 1 else interior], [tail], [0.], [0.],
        ])
        torch.testing.assert_close(output[2], expected)
        assert calls == [False, True]
        assert not output[2].requires_grad
        assert all(p.grad is None for p in engine.model.parameters())
        assert holder.agent.log_ent_coef.grad is None


@pytest.mark.parametrize("mode,entropy", [("squashed", 2.), ("tdmpc2_scaled", 3.)])
@pytest.mark.parametrize("scaled", [False, True])
def test_boundary_uses_outer_entropy_statistic_and_raw_q_coefficient(monkeypatch, mode, entropy, scaled):
    with _prepared(
        ent_coef=.2, outer_actor_entropy_mode=mode, inner_actor_entropy_mode="squashed",
        sac_actor_loss_scale_mode="tdmpc2_percentile_range" if scaled else "none",
    ) as (holder, engine):
        if scaled:
            holder.agent.actor_loss_scale.fill_(7.)

        def policy(z, **kwargs):
            assert kwargs.get("include_scaled_entropy", False) == (mode == "tdmpc2_scaled")
            assert "policy" not in kwargs
            return z.new_zeros(len(z), 1), {
                "log_prob": z.new_full((len(z), 1), -2.),
                "scaled_entropy": z.new_full((len(z), 1), 3.),
            }

        monkeypatch.setattr(engine.model, "pi", policy)
        monkeypatch.setattr(engine.model, "Q", lambda z, *args, **kwargs: z.new_full((len(z), 1), 11.))
        value = engine._prior_bootstrap(torch.zeros(2, holder.cfg.latent_dim), torch.zeros(2, 1))
        torch.testing.assert_close(value, torch.full_like(value, 11. + .2 * entropy * (7. if scaled else 1.)))


def test_boundary_with_return_initialization_and_actor_entropy_off():
    with _aux_prepared(
        inner_terminal_entropy="outer", inner_entropy_enabled=False,
    ) as (holder, engine):
        z = torch.zeros(2, holder.cfg.latent_dim)
        noise = torch.zeros(2, 1)
        action, info = engine.model.pi(z, noise=noise)
        expected = engine.model.Q(z, action, reduction="mean_pair") - .2 * info["log_prob"]
        torch.testing.assert_close(engine._prior_bootstrap(z, noise), expected)
        assert engine.alpha.item() == 0.


def test_corrected_boundary_compiles_without_graph_breaks_or_outer_mutation():
    with _prepared() as (holder, engine):
        before = deepcopy(holder.agent.checkpoint_state())
        z, noise = torch.zeros(2, holder.cfg.latent_dim), torch.zeros(2, 1)
        expected = engine._prior_bootstrap(z, noise)
        compiled = torch.compile(engine._prior_bootstrap, backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled(z, noise), expected)
        for name, value in holder.agent.model.state_dict().items():
            torch.testing.assert_close(value, before["model"][name], rtol=0, atol=0)
        torch.testing.assert_close(holder.agent.log_ent_coef, before["log_ent_coef"], rtol=0, atol=0)


@pytest.mark.parametrize("actor_entropy", [False, True])
def test_corrected_full_action_records_boundary_and_preserves_outer(actor_entropy):
    with _aux_prepared(
        inner_terminal_entropy="outer", inner_entropy_enabled=actor_entropy,
    ) as (holder, engine):
        before = deepcopy(holder.agent.checkpoint_state())
        trace = InnerActionTrace()
        assert torch.isfinite(holder.agent.act(torch.zeros(3), trace=trace)).all()
        assert trace.events
        assert all(event["value_routing"]["terminal_entropy_bonus"] for event in trace.events)
        _assert_tree_equal(holder.agent.checkpoint_state(), before)


def test_identity_preserves_old_runs_and_separates_corrected_runs():
    base = {"inner_operator": "sac", "inner_finite_horizon": True}
    explicit = {**base, "inner_terminal_entropy": "none"}
    corrected = {**base, "inner_terminal_entropy": "outer"}

    def training(config):
        return scientific_trial_parameters({"alg": "AMBITDMPC2/AMBITDMPC2", "alg_params": config})

    def curve(config):
        return planner_identity(config, {}, "AMBITDMPC2/AMBITDMPC2", "mean")

    assert training(base) == training(explicit)
    assert curve(base) == curve(explicit)
    assert training(base) != training(corrected)
    assert curve(base) != curve(corrected)
    assert curve(corrected)["settings"]["inner_terminal_entropy"] == "outer"


def test_old_weights_load_but_exact_resume_rejects_changed_boundary():
    with _prepared(inner_terminal_entropy="none") as (old, _), _prepared() as (new, _):
        checkpoint = deepcopy(old.agent.checkpoint_state())
        assert "terminal_entropy" not in checkpoint["critic_target_spec"]["inner_solve"]
        new.agent.load(checkpoint)
        assert new.agent._critic_target_spec()["inner_solve"]["terminal_entropy"] == "outer"
        with pytest.raises(ValueError, match="critic-target specification"):
            new.agent._preflight_outer_training_state(checkpoint)
