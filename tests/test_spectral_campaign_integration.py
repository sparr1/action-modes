"""Real inner-SAC coverage for spectral initialization and sampled publication."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.test_ambi_inner_decoupling import _assert_tree_equal, _clone_tree
from tests.test_ambi_root_local_sac import _model_from_params
from tests.test_aux_critic_transfer import assert_optimizer_reset, critic_params
from utils.spectral_campaign import SpectralCampaignDiagnostics, verify_spectral_diagnostics
from utils.spectral_transfer_probes import build_spectral_context
from utils.transfer_campaign import arm_initialization, evaluate_episode, make_transfer_generators
from utils.transfer_campaign_diagnostics import verify_observational_isolation


PROBE = dict(state_count=4, action_count=2, mc_rollouts=2)
SPECTRAL = dict(enabled=True, decisions=[0, 1], ranks=[1, 2, 4], **PROBE)
BASIC = dict(decisions=[0, 1], stationary_decisions=[1], state_count=4,
             action_count=4, mc_rollouts=2, fit_steps=1)
OBSERVATION = np.array([1., .2, -.1], dtype=np.float32)


def model(horizon=2, **overrides):
    return _model_from_params(critic_params("return", inner_critic_scope="action",
        inner_rounds=1, inner_rollout_horizon=horizon, train_unroll_horizon=3, **overrides))


def spectral_arm(method, components, norm_matched=False):
    return dict(parameter_scope="matrices", **{f"{component}_spectral": dict(
        method=method, rank=None if method in {"gradient_projection", "gradient_gate"} else 2,
        strength=.5, norm_matched=norm_matched) for component in components})


@pytest.mark.parametrize("method", ["svd", "activation", "gradient", "gradient_projection", "gradient_gate"])
@pytest.mark.parametrize("components", [("actor",), ("critic",), ("actor", "critic")])
@pytest.mark.parametrize("norm_matched", [False, True])
def test_all_spectral_initializations_reset_nonmatrix_parameters_and_learning_state(method, components, norm_matched):
    wrapped = model()
    try:
        engine = wrapped.agent.inner_engine
        arm = spectral_arm(method, components, norm_matched)
        frozen = _clone_tree(wrapped.agent.model.state_dict())
        first = arm_initialization(engine, arm, None)
        assert "actor" not in first and "critic" not in first
        with engine.diagnostic_initialization(**first):
            wrapped.predict(OBSERVATION, deterministic=True, episode_start=True)
        donor = engine.export_diagnostic_state(include_optimizers=True, include_replay=True)
        for component in ("actor", "critic"):
            for value in donor["modules"][component].values():
                value.add_(.2)
        for value in donor["modules"]["critic_target"].values():
            value.add_(12.)
        unchanged = _clone_tree(donor)
        context = (build_spectral_context(wrapped, OBSERVATION, controller_seed=55,
            episode_seed=101, decision=1, settings=PROBE, components=components,
            compute_gradients=method in {"gradient", "gradient_projection", "gradient_gate"}) if method != "svd" else None)
        rng = _clone_tree(engine.rng.training_state_dict())
        options = arm_initialization(engine, arm, donor, spectral_context=context, transfer_metrics={})
        _assert_tree_equal(rng, engine.rng.training_state_dict())
        _assert_tree_equal(unchanged, donor)
        assert options["target"] == "online"
        assert not {"learner_state", "replay", "collection_actor"} & options.keys()
        with engine.diagnostic_initialization(**options):
            with engine.rng.fork("initialization"):
                engine._prepare_workspace(t0=False)
            engine._apply_diagnostic_initialization()
            for component in ("actor", "critic"):
                prior = getattr(engine, f"_{component}_base")
                selected = getattr(engine.state, component)
                expected = options.get(component, prior.state_dict())
                _assert_tree_equal(selected.state_dict(), expected)
                for name, value in selected.state_dict().items():
                    if value.ndim != 2 or component not in components:
                        torch.testing.assert_close(value, prior.state_dict()[name], rtol=0, atol=0)
                assert all(parameter.requires_grad for parameter in selected.parameters())
                assert_optimizer_reset(getattr(engine.state, f"{component}_optim"), selected.parameters())
                assert getattr(engine.state, f"{component}_steps") == 0
                assert getattr(engine.state, f"{component}_lifetime_steps") == 0
            _assert_tree_equal(engine.state.critic_target.state_dict(), engine.state.critic.state_dict())
            assert all(not p.requires_grad for p in engine.state.critic_target.parameters())
            assert_optimizer_reset(engine.state.temperature_optim, [engine.state.log_alpha])
            assert engine.state.replay.size == 0 and engine._diagnostic_previous_replay is None
            torch.testing.assert_close(engine.alpha, engine._initial_inner_alpha(), rtol=0, atol=0)
        _assert_tree_equal(frozen, wrapped.agent.model.state_dict())
    finally:
        wrapped.close()


@pytest.mark.parametrize("kind", ["rho", "bernoulli_p"])
def test_matrix_only_dense_and_bernoulli_controls_reset_bias_and_normalization(kind):
    wrapped = model()
    try:
        engine = wrapped.agent.inner_engine
        donor = dict(modules={component: {name: value + .5 for name, value in
            getattr(engine, f"_{component}_base").state_dict().items()} for component in ("actor", "critic")})
        arm = dict(parameter_scope="matrices", **{f"{component}_{kind}": 1. for component in ("actor", "critic")})
        generators = make_transfer_generators(engine, controller_seed=55, episode_seed=101)
        options = arm_initialization(engine, arm, donor, transfer_generators=generators)
        for component in ("actor", "critic"):
            prior = getattr(engine, f"_{component}_base").state_dict()
            for name, value in options[component].items():
                expected = donor["modules"][component][name] if value.ndim == 2 else prior[name]
                torch.testing.assert_close(value, expected, rtol=0, atol=0)
    finally:
        wrapped.close()


@pytest.mark.parametrize("method", ["gradient_projection", "gradient_gate"])
@pytest.mark.parametrize("components", [("actor",), ("critic",), ("actor", "critic")])
def test_gradient_coordinate_controller_uses_new_root_gradient_without_svd(monkeypatch, method, components):
    import utils.spectral_transfer as filters
    import utils.spectral_transfer_probes as probes
    original_context = probes.build_spectral_context
    calls = []

    def context(*args, **kwargs):
        calls.append((kwargs["decision"], kwargs["components"], kwargs["compute_gradients"]))
        return original_context(*args, **kwargs)

    def no_svd(*args, **kwargs):
        raise AssertionError("Gradient gate or line projection must not decompose donor matrices")

    monkeypatch.setattr(probes, "build_spectral_context", context)
    monkeypatch.setattr(filters, "_svd", no_svd)
    wrapped = model()
    try:
        rows = []
        output = evaluate_episode(wrapped, wrapped.env,
            spectral_arm(method, components),
            episode_seed=101, controller_seed=55, max_steps=3,
            on_step=rows.append, spectral_probe=PROBE)
        assert calls == [(1, components, True), (2, components, True)]
        assert output["spectral_probe_seconds"] > 0 and output["diagnostic_seconds"] == 0
        for row in rows[1:]:
            for component in components:
                prefix = f"inner_{component}_spectral_"
                quantity = "retained_fraction" if method == "gradient_gate" else "projection_coefficient"
                assert np.isfinite(row["metrics"][prefix + quantity])
                assert row["metrics"][prefix + "gradient_squared_norm"] >= 0
                if method == "gradient_gate":
                    assert 0 <= row["metrics"][prefix + "retained_fraction"] <= 1
                    assert row["metrics"][prefix + "predicted_benefit"] >= 0
                assert prefix + "selected_rank_sum" not in row["metrics"]
    finally:
        wrapped.close()


@pytest.mark.parametrize("horizon,method,components,norm_matched", [
    (1, "svd", ("actor",), False),
    (2, "activation", ("critic",), False),
    (3, "gradient", ("actor", "critic"), True),
    (2, "gradient_projection", ("actor", "critic"), False),
    (2, "gradient_projection", ("actor", "critic"), True),
    (1, "gradient_gate", ("actor",), False),
    (1, "gradient_gate", ("critic",), True),
    (1, "gradient_gate", ("actor", "critic"), False),
])
def test_sampled_spectral_and_basic_diagnostics_preserve_full_controller_and_account_for_costs(
        horizon, method, components, norm_matched):
    wrapped = model(horizon)
    try:
        frozen = _clone_tree(wrapped.agent.model.state_dict())
        arm = spectral_arm(method, components, norm_matched)
        outputs, trajectories, states = [], [], []
        for enabled in (False, True):
            diagnostic = (SpectralCampaignDiagnostics(SPECTRAL, basic_settings=BASIC,
                episode_seed=101, controller_seed=55, smoke=True) if enabled else None)
            rows = []
            before = torch.get_rng_state().clone()
            output = evaluate_episode(wrapped, wrapped.env, arm, episode_seed=101,
                controller_seed=55, max_steps=3, on_step=rows.append, smoke=True,
                diagnostics=diagnostic, spectral_probe=PROBE)
            torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
            trajectories.append(rows)
            outputs.append(output)
            states.append(dict(learner=wrapped.agent.inner_engine.export_diagnostic_state(),
                               rng=_clone_tree(wrapped.agent.inner_engine.rng.training_state_dict())))
            for name in ("prediction_seconds", "transfer_seconds", "control_seconds", "diagnostic_seconds",
                         "spectral_probe_seconds", "spectral_filter_seconds", "donor_export_seconds"):
                assert output[name] == pytest.approx(sum(row[name] for row in rows))
                assert output[name] >= 0
            assert output["control_seconds"] == pytest.approx(output["prediction_seconds"] + output["transfer_seconds"])
            assert rows[0]["spectral_filter_seconds"] == rows[0]["spectral_probe_seconds"] == 0
            assert output["spectral_filter_seconds"] > 0
            assert (output["spectral_probe_seconds"] > 0) == (method != "svd")
            for row in rows:
                # Timers share inclusive boundaries, with a few Python clock
                # operations outside the smaller named measurements.
                assert row["transfer_seconds"] + 1e-3 >= sum(row[name] for name in (
                    "spectral_probe_seconds", "spectral_filter_seconds", "donor_export_seconds"))
        verify_observational_isolation(trajectories[0], trajectories[1], states[0], states[1])
        _assert_tree_equal(frozen, wrapped.agent.model.state_dict())
        verify_spectral_diagnostics(outputs[1], SPECTRAL, smoke=True)
        assert outputs[0]["diagnostic_seconds"] == 0 < outputs[1]["diagnostic_seconds"]
        assert trajectories[1][2]["diagnostic_seconds"] == 0
        summary = outputs[1]["diagnostics"]["summary"]
        spectral_rows = [row["diagnostics"]["spectral"] for row in trajectories[1][:2]]
        assert not spectral_rows[0]["donor_available"] and spectral_rows[1]["donor_available"]
        if method in {"gradient", "gradient_projection", "gradient_gate"}:
            selection = outputs[0]["selection_diagnostics"]
            assert selection == outputs[1]["selection_diagnostics"]
            assert selection["summary"]
            for key, value in selection["summary"].items():
                # Three controller decisions have two donors, while the
                # sampled held-out roots above contain only one donor.
                assert selection["summary_counts"][key] == 2
                assert outputs[1]["diagnostics"]["summary_counts"][key] == 2
                assert summary[key] == value
                raw_key = key.replace("selection_", "inner_", 1).replace("_actor_", "_actor_spectral_").replace("_critic_", "_critic_spectral_")
                assert raw_key not in trajectories[0][0]["metrics"]
                assert value == pytest.approx(np.mean([row["metrics"][raw_key] for row in trajectories[0][1:]]))
            for component in components:
                assert f"selection_{component}_predicted_benefit" in selection["summary"]
                assert (f"selection_{component}_retained_fraction" in selection["summary"]) == (method == "gradient_gate")
                assert (f"selection_{component}_projection_coefficient" in selection["summary"]) == (method == "gradient_projection")
        else:
            assert "selection_diagnostics" not in outputs[0]
            assert "selection_diagnostics" not in outputs[1]
        for component in ("actor", "critic"):
            geometry = f"{component}_donor_squared_norm"
            assert summary[f"spectral_{geometry}"] == spectral_rows[1]["summary"][geometry]
            assert outputs[1]["diagnostics"]["summary_counts"][f"spectral_{geometry}"] == 1
            objective = f"initial_{component}_loss"
            assert summary[f"spectral_{objective}"] == pytest.approx(np.mean([
                row["summary"][objective] for row in spectral_rows]))
        invalid = deepcopy(outputs[1])
        invalid["diagnostics"]["summary"].pop("spectral_final_actor_loss")
        with pytest.raises(RuntimeError, match="objective"):
            verify_spectral_diagnostics(invalid, SPECTRAL, smoke=True)
        invalid = deepcopy(outputs[1])
        invalid["diagnostics"]["spectral"]["completed_decisions"] = [0]
        with pytest.raises(RuntimeError, match="coverage"):
            verify_spectral_diagnostics(invalid, SPECTRAL, smoke=True)
        for key in ("spectral_actor_mean_energy_at_rank_1", "spectral_critic_mean_effective_rank",
                    "spectral_actor_mean_positive_benefit_energy_fraction"):
            invalid = deepcopy(outputs[1])
            invalid["diagnostics"]["summary"].pop(key)
            with pytest.raises(RuntimeError, match="metric|geometry|spectral|Spectral"):
                verify_spectral_diagnostics(invalid, SPECTRAL, smoke=True)
    finally:
        wrapped.close()


@pytest.mark.parametrize("basic", [None, BASIC])
def test_fresh_control_observational_donor_does_not_change_controller_or_first_sample_geometry(basic):
    wrapped = model()
    try:
        results, rows, states = [], [], []
        for enabled in (False, True):
            diagnostic = (SpectralCampaignDiagnostics(SPECTRAL, basic_settings=basic,
                episode_seed=101, controller_seed=55) if enabled else None)
            records = []
            results.append(evaluate_episode(wrapped, wrapped.env, {}, episode_seed=101,
                controller_seed=55, max_steps=1, on_step=records.append, smoke=True,
                diagnostics=diagnostic, spectral_probe=PROBE))
            rows.append(records)
            states.append(dict(learner=wrapped.agent.inner_engine.export_diagnostic_state(),
                               rng=_clone_tree(wrapped.agent.inner_engine.rng.training_state_dict())))
        verify_observational_isolation(rows[0], rows[1], states[0], states[1])
        assert results[1]["donor_export_seconds"] == 0.
        summary = results[1]["diagnostics"]["summary"]
        assert "spectral_initial_actor_loss" in summary
        assert "spectral_actor_mean_effective_rank" not in summary
        assert results[1]["diagnostics"]["spectral"]["donor_samples"] == 0
        assert set(results[1]["diagnostics"]["spectral"]["stages"]) == {"prior", "initial", "final"}
    finally:
        wrapped.close()


@pytest.mark.parametrize("method", ["gradient", "gradient_projection", "gradient_gate"])
@pytest.mark.parametrize("strength,max_steps", [(0., 3), (1., 1)])
def test_uncomputed_gradient_selection_diagnostics_remain_absent(method, strength, max_steps):
    wrapped = model()
    try:
        arm = spectral_arm(method, ("actor", "critic"))
        for component in ("actor", "critic"):
            arm[f"{component}_spectral"]["strength"] = strength
        output = evaluate_episode(wrapped, wrapped.env, arm, episode_seed=101,
            controller_seed=55, max_steps=max_steps, spectral_probe=PROBE)
        assert "selection_diagnostics" not in output
    finally:
        wrapped.close()


def test_selection_diagnostic_counts_use_only_each_metrics_available_decisions(monkeypatch):
    import utils.transfer_campaign as campaign
    original = campaign.arm_initialization
    calls = []

    def missing_first_benefit(*args, **kwargs):
        options = original(*args, **kwargs)
        metrics = kwargs["transfer_metrics"]
        if "inner_actor_spectral_predicted_benefit" in metrics:
            calls.append(1)
            if len(calls) == 1:
                metrics.pop("inner_actor_spectral_predicted_benefit")
        return options

    monkeypatch.setattr(campaign, "arm_initialization", missing_first_benefit)
    wrapped = model()
    try:
        rows = []
        output = evaluate_episode(wrapped, wrapped.env, spectral_arm("gradient_gate", ("actor",)),
            episode_seed=101, controller_seed=55, max_steps=3, spectral_probe=PROBE, on_step=rows.append)
        selection = output["selection_diagnostics"]
        assert selection["summary_counts"]["selection_actor_predicted_benefit"] == 1
        assert selection["summary_counts"]["selection_actor_retained_fraction"] == 2
        assert selection["summary"]["selection_actor_predicted_benefit"] == rows[2]["metrics"]["inner_actor_spectral_predicted_benefit"]
    finally:
        wrapped.close()


@pytest.mark.parametrize("method", ["gradient", "gradient_projection", "gradient_gate"])
def test_compiled_and_eager_gradient_joint_transfer_preserve_exact_controller_rng(monkeypatch, method):
    torch._dynamo.reset()
    original_compile = torch.compile
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: original_compile(fn, backend="eager", **kwargs))
    eager = model()
    compiled = model(compile=True, compile_strict=True)
    try:
        compiled.agent.model.load_state_dict(eager.agent.model.state_dict())
        trajectories, states = [], []
        for wrapped in (eager, compiled):
            rows = []
            evaluate_episode(wrapped, wrapped.env, spectral_arm(method, ("actor", "critic")),
                episode_seed=101, controller_seed=55, max_steps=3, on_step=rows.append,
                smoke=True, spectral_probe=PROBE)
            trajectories.append(rows)
            states.append(dict(learner=wrapped.agent.inner_engine.export_diagnostic_state(),
                               rng=_clone_tree(wrapped.agent.inner_engine.rng.training_state_dict())))
        verify_observational_isolation(trajectories[0], trajectories[1], states[0], states[1])
        assert trajectories[1][1]["spectral_probe_seconds"] > 0
    finally:
        eager.close()
        compiled.close()
        torch._dynamo.reset()


@pytest.mark.parametrize("method", ["gradient", "gradient_projection", "gradient_gate"])
def test_smoke_evaluator_runs_real_spectral_episode_and_persists_complete_coverage(tmp_path, monkeypatch, method):
    import evaluate_ambi_transfer_campaign as evaluator
    from utils.transfer_campaign import load_campaign
    source = Path(__file__).resolve().parents[1] / "configs/research/ambi_transfer_discovery_575k.json"
    campaign = load_campaign(source)
    campaign.update(family="spectral_transfer", horizons=[2], rounds=[1], seeds=[101],
        arms={"spectral_joint": spectral_arm(method, ("actor", "critic"))},
        diagnostics=BASIC, spectral_diagnostics=SPECTRAL, spectral_probe=PROBE,
        critic_updates=2, actor_updates=2, rollouts=2, batch_size=4)
    wrapped = model()
    output = tmp_path / "output"
    args = evaluator.parser().parse_args(["--campaign", str(source), "--checkpoint", str(tmp_path / "checkpoint"),
        "--horizon", "2", "--rounds", "1", "--arm", "spectral_joint", "--output-dir", str(output),
        "--smoke", "--max-steps", "3", "--seeds", "101", "--no-compile", "--device", "cpu"])
    monkeypatch.setattr(evaluator, "load_campaign", lambda path: deepcopy(campaign))
    monkeypatch.setattr(evaluator, "load_preset_matrix", lambda path: {"checkpoint_contract": campaign["checkpoint_contract"]})
    monkeypatch.setattr(evaluator, "load_checkpoint_context", lambda *a, **k: SimpleNamespace(metadata={"checkpoint": {"step": 575000}}))
    monkeypatch.setattr(evaluator, "resolve_preset", lambda *a, **k: {"algorithm_config": {"alg_params": {}}, "selector": "test"})
    monkeypatch.setattr(evaluator, "_validate_checkpoint_contract", lambda *a: None)
    monkeypatch.setattr(evaluator, "_file_sha256", lambda path: "testhash")
    monkeypatch.setattr(evaluator, "source_identity", lambda: {"git_head": "test", "files": {}, "sha256": "test"})
    monkeypatch.setattr(evaluator, "_make_env", lambda *a: wrapped.env)
    monkeypatch.setattr(evaluator, "_initialize_frozen_model", lambda *a, **k: (wrapped, {}))
    monkeypatch.setattr(evaluator, "_close_resources", lambda *a: None)
    try:
        result = evaluator.run(args)
        assert result["complete"] and result["frozen_outer_verified"]
        assert result["episodes"][0]["diagnostic_isolation_verified"]
        assert result["episodes"][0]["diagnostics"]["spectral"]["samples"] == 2
        assert result["episodes"][0]["spectral_probe_seconds"] > 0
        manifest = json.loads((output / "manifest.json").read_text())
        assert "same bank and labels" in manifest["semantics"]["spectral_evaluation"]
        assert manifest["metadata_path"] == str(tmp_path / "checkpoint.metadata.json")
        assert manifest["metadata_sha256"] == "testhash"
        assert manifest["base_matrix_sha256"] == "testhash"
        decisions = [json.loads(line) for line in (output / "decisions-seed-101.jsonl").read_text().splitlines()]
        assert decisions[1]["diagnostics"]["spectral"]["donor_available"]
        assert json.loads((output / "progress.json").read_text())["status"] == "complete"
    finally:
        wrapped.close()
