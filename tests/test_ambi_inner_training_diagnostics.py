"""Inner actor diagnostics describe the sample and scale used by the update."""

import math

import pytest
import torch

from RL.tdmpc2_core import inner_improvement
from tests.test_ambi_inner_decoupling import _assert_tree_equal, _fixed_reward_model
from tests.test_ambi_random_explorer_engine_invariants import _explorer_model
from tests.test_td_ambi_inner_objectives import _engine
from tests.test_tdambi_inner import _tdambi_model


@pytest.mark.parametrize("entropy_mode", ["squashed", "tdmpc2_scaled"])
@pytest.mark.parametrize("scale_update", ["per_action", "per_update"])
def test_inner_metrics_reuse_the_actor_sample_and_actual_divisor(
    monkeypatch, entropy_mode, scale_update,
):
    engine = _engine(
        inner_actor_entropy_mode=entropy_mode,
        inner_actor_loss_scale_update=scale_update,
    )
    engine._collect_diagnostics = False
    with torch.no_grad():
        engine.state.actor[-1].weight.zero_()
        engine.state.actor[-1].bias[:3].copy_(torch.tensor([20.0, 3.0, 0.0]))
    samples = []
    original = engine.model.pi

    def capture(*args, **kwargs):
        action, info = original(*args, **kwargs)
        samples.append((action.detach().clone(), {
            key: value.detach().clone() for key, value in info.items()
        }))
        return action, info

    monkeypatch.setattr(engine.model, "pi", capture)
    scale = engine.agent.actor_loss_scale.detach().clone()
    metrics = engine._sac_policy_step(
        {"z": torch.zeros(8, engine.cfg.latent_dim)},
        update_temperature=False, update_actor=True,
        alpha=torch.tensor(1e-4), actor_loss_scale=scale,
    )
    assert len(samples) == 1
    action, info = samples[0]
    torch.testing.assert_close(metrics["actor_pre_tanh_entropy"], (
        info["log_std"] + 0.5 * math.log(2 * math.pi * math.e)
    ).sum(-1).mean())
    torch.testing.assert_close(metrics["actor_post_tanh_entropy"], -info["log_prob"].mean())
    torch.testing.assert_close(metrics["actor_action_abs_ge_0p99_fraction"],
                               (action.abs() >= 0.99).float().mean())
    assert metrics["actor_mean_action_abs_ge_0p99_fraction"].item() == pytest.approx(2 / 3)
    assert metrics["actor_mean_action_exact_saturation_fraction"].item() == pytest.approx(1 / 3)
    torch.testing.assert_close(metrics["actor_q_scale_used"], scale.reshape(()))
    assert metrics["actor_entropy_coefficient_normalized"].item() == pytest.approx(1e-4)
    torch.testing.assert_close(metrics["actor_entropy_coefficient_return_units"], scale.reshape(()) * 1e-4)
    multiplier = 3 if entropy_mode == "tdmpc2_scaled" else 1
    torch.testing.assert_close(metrics["actor_entropy_coefficient_joint_units"], scale.reshape(()) * 1e-4 * multiplier)
    torch.testing.assert_close(metrics["actor_q_percentile_range_over_scale"],
                               metrics["actor_q_percentile_range"] / scale.reshape(()))
    assert all(not value.requires_grad for value in metrics.values() if torch.is_tensor(value))


def test_adapting_scale_history_owns_each_updates_snapshot():
    engine = _engine(sac_actor_loss_scale_tau=0.5)
    scale = engine.agent.actor_loss_scale.detach().clone()
    history = []
    for _ in range(2):
        history.append(engine._sac_policy_step(
            {"z": torch.zeros(8, engine.cfg.latent_dim)},
            update_temperature=False, update_actor=True,
            alpha=torch.tensor(1e-4), actor_loss_scale=scale,
        ))
        torch.testing.assert_close(history[-1]["actor_q_scale_used"], scale.reshape(()))
    assert history[0]["actor_q_scale_used"] != history[1]["actor_q_scale_used"]
    first = history[0]["actor_q_scale_used"].clone()
    coefficient = history[0]["actor_entropy_coefficient_return_units"].clone()
    scale.fill_(100)
    torch.testing.assert_close(history[0]["actor_q_scale_used"], first)
    torch.testing.assert_close(history[0]["actor_entropy_coefficient_return_units"], coefficient)
    averaged = engine._average_update_metrics(history)
    torch.testing.assert_close(averaged["inner_actor_q_scale_used"],
                               torch.stack([item["actor_q_scale_used"] for item in history]).mean())


@pytest.mark.parametrize("scale_update", ["per_action", "per_update"])
def test_each_decision_reports_scale_reference_and_boundaries(scale_update):
    model = _fixed_reward_model(
        sac_actor_loss_scale_mode="tdmpc2_percentile_range",
        inner_actor_loss_scale_update=scale_update,
        sac_actor_loss_scale_tau=0.5,
    )
    try:
        model.agent.actor_loss_scale.fill_(3.75)
        for _ in range(2):
            model.agent.act(torch.zeros(3), collect_diagnostics=False)
            metrics = model.agent.last_inner_metrics
            for key in ("outer", "initial"):
                assert metrics[f"inner_actor_loss_scale_{key}"] == pytest.approx(3.75)
            assert metrics["inner_actor_loss_scale_initial_over_outer"] == 1
            assert metrics["inner_actor_loss_scale_final"] == metrics["inner_actor_loss_scale"]
            assert metrics["inner_actor_loss_scale_final_over_outer"] == pytest.approx(
                metrics["inner_actor_loss_scale_final"] / 3.75
            )
            if scale_update == "per_action":
                assert metrics["inner_actor_loss_scale_final_over_outer"] == 1
            else:
                assert metrics["inner_actor_loss_scale_final_over_outer"] != 1
            # Historical name retains its normalized-alpha semantics.
            assert metrics["inner_effective_alpha"] == metrics["inner_alpha_final"]
            torch.testing.assert_close(model.agent.actor_loss_scale, torch.tensor([3.75]))
    finally:
        model.close()
        model.env.close()


def test_no_actor_update_omits_unmeasured_actor_diagnostics_and_keeps_scale():
    engine = _engine()
    scale = engine.agent.actor_loss_scale.detach().clone()
    metrics = engine._sac_policy_step(
        {"z": torch.zeros(8, engine.cfg.latent_dim)},
        update_temperature=False, update_actor=False,
        alpha=torch.tensor(0.0), actor_loss_scale=scale,
    )
    assert "actor_q_scale_used" not in metrics
    assert "actor_pre_tanh_entropy" not in metrics
    assert "actor_post_tanh_entropy" not in metrics
    assert "actor_entropy_coefficient_return_units" not in metrics
    torch.testing.assert_close(scale, engine.agent.actor_loss_scale)


def test_zero_entropy_coefficient_reports_zero_in_all_units():
    engine = _engine()
    metrics = engine._sac_policy_step(
        {"z": torch.zeros(8, engine.cfg.latent_dim)},
        update_temperature=False, update_actor=True,
        alpha=torch.tensor(0.0), actor_loss_scale=torch.tensor([3.75]),
    )
    for units in ("normalized", "return_units", "joint_units"):
        assert metrics[f"actor_entropy_coefficient_{units}"].item() == 0


def test_diagnostic_helpers_leave_actor_optimizer_and_rng_exactly_unchanged(monkeypatch):
    observed, baseline = _engine(), _engine()
    _assert_tree_equal(observed.state.actor.state_dict(), baseline.state.actor.state_dict())
    initial_rng = torch.random.get_rng_state().clone()
    scales, metrics = [], []
    for engine, enabled in ((observed, True), (baseline, False)):
        if not enabled:
            monkeypatch.setattr(inner_improvement, "policy_sample_diagnostics", lambda *a, **kw: {})
            monkeypatch.setattr(inner_improvement, "entropy_coefficient_diagnostics", lambda *a, **kw: {})
        engine._collect_diagnostics = False
        scales.append(engine.agent.actor_loss_scale.detach().clone())
        metrics.append(engine._sac_policy_step(
            {"z": torch.zeros(8, engine.cfg.latent_dim)},
            update_temperature=False, update_actor=True,
            alpha=engine.alpha.detach(), actor_loss_scale=scales[-1],
        ))
    assert "actor_pre_tanh_entropy" in metrics[0]
    assert "actor_pre_tanh_entropy" not in metrics[1]
    _assert_tree_equal(observed.state.actor.state_dict(), baseline.state.actor.state_dict())
    _assert_tree_equal(observed.state.actor_optim.state_dict(), baseline.state.actor_optim.state_dict())
    _assert_tree_equal(observed.rng.training_state_dict(), baseline.rng.training_state_dict())
    torch.testing.assert_close(torch.random.get_rng_state(), initial_rng, rtol=0, atol=0)
    torch.testing.assert_close(scales[0], scales[1], rtol=0, atol=0)
    torch.testing.assert_close(metrics[0]["actor_loss"], metrics[1]["actor_loss"], rtol=0, atol=0)
    for actual, expected in zip(observed.state.actor_params, baseline.state.actor_params):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)


def test_native_tdambi_reports_reused_policy_diagnostics_and_scale():
    model = _tdambi_model(inner_rounds=1, inner_updates_per_round=1)
    try:
        rng = torch.random.get_rng_state().clone()
        model.agent.act(torch.zeros(3), collect_diagnostics=False)
        metrics = model.agent.last_inner_metrics
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
        assert math.isfinite(metrics["inner_actor_pre_tanh_entropy"])
        assert math.isfinite(metrics["inner_actor_post_tanh_entropy"])
        assert metrics["inner_actor_q_scale_used"] == pytest.approx(metrics["inner_tdambi_q_scale_final"])
        assert metrics["inner_actor_entropy_coefficient_return_units"] == pytest.approx(
            model.cfg.tdambi_entropy_coef * metrics["inner_actor_q_scale_used"]
        )
    finally:
        model.close()
        model.env.close()


@pytest.mark.parametrize("entropy_mode", ["squashed", "tdmpc2_scaled"])
def test_separate_actors_reuse_samples_without_changing_updates(monkeypatch, entropy_mode):
    models = [_explorer_model(
        "separate_critics", inner_actor_entropy_mode=entropy_mode,
        inner_temperature_mode="fixed", inner_temperature=0.25,
    ) for _ in range(2)]
    engines = [model.agent.inner_engine for model in models]
    samples, calls = [], {"Q": 0}
    try:
        for engine in engines:
            engine._collect_diagnostics = False
            with engine.rng.fork("initialization"):
                engine._prepare_workspace(t0=True)
            with torch.no_grad():
                for actor, mean in ((engine.state.actor, 20.0), (engine.state.explorer_actor, 3.0)):
                    actor[-1].weight.zero_()
                    actor[-1].bias[0] = mean
        observed, baseline = engines
        original_pi, original_q = observed.model.pi, observed.model.Q

        def capture_pi(*args, **kwargs):
            action, info = original_pi(*args, **kwargs)
            samples.append((action.detach().clone(), {
                key: value.detach().clone() for key, value in info.items()
            }))
            return action, info

        def capture_q(*args, **kwargs):
            calls["Q"] += 1
            return original_q(*args, **kwargs)

        monkeypatch.setattr(observed.model, "pi", capture_pi)
        monkeypatch.setattr(observed.model, "Q", capture_q)
        rng = torch.random.get_rng_state().clone()
        results = []
        for engine in engines:
            if engine is baseline:
                monkeypatch.setattr(inner_improvement, "policy_sample_diagnostics", lambda *a, **kw: {})
                monkeypatch.setattr(inner_improvement, "entropy_coefficient_diagnostics", lambda *a, **kw: {})
            results.append(engine._separate_policy_step(
                {"z": torch.zeros(8, engine.cfg.latent_dim)},
                update_primary_actor=True, update_explorer_actor=True,
                update_primary_temperature=False, update_explorer_temperature=False,
            ))
        assert len(samples) == calls["Q"] == 2
        for prefix, (action, info) in zip(("actor_", "explorer_actor_"), samples):
            metrics = results[0]
            torch.testing.assert_close(metrics[f"{prefix}pre_tanh_entropy"], (
                info["log_std"] + 0.5 * math.log(2 * math.pi * math.e)
            ).sum(-1).mean())
            torch.testing.assert_close(metrics[f"{prefix}post_tanh_entropy"], -info["log_prob"].mean())
            torch.testing.assert_close(metrics[f"{prefix}action_abs_ge_0p99_fraction"],
                                       (action.abs() >= 0.99).float().mean())
            assert metrics[f"{prefix}mean_action_abs_ge_0p99_fraction"] == 1
            assert metrics[f"{prefix}mean_action_exact_saturation_fraction"] == (prefix == "actor_")
            assert metrics[f"{prefix}q_scale_used"] == 1
            assert metrics[f"{prefix}entropy_coefficient_return_units"] == pytest.approx(0.25)
            assert f"{prefix}pre_tanh_entropy" not in results[1]
        for name in ("actor", "explorer_actor", "actor_optim", "explorer_actor_optim"):
            _assert_tree_equal(getattr(observed.state, name).state_dict(),
                               getattr(baseline.state, name).state_dict())
        _assert_tree_equal(observed.rng.training_state_dict(), baseline.rng.training_state_dict())
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    finally:
        for model in models:
            model.close()
            model.env.close()
