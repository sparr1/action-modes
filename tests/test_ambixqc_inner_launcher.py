"""Runner preflight, reference reuse, acceptance, and durable result contracts."""

from copy import deepcopy
import gzip
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import run_ambixqc_inner_evaluation as runner
from test_ambixqc_mppi_launcher import checkpoint_manifest
from utils.ambi_benchmark import protocol_for, solver_seed


def _episodes(seeds, length, *, candidate=False):
    return [{"seed": seed, "solver_seed": solver_seed(12345, "episode", seed),
             "episode_id": f"seed-{seed}", "length": length, "return": float(seed) + (5 if candidate else 0),
             "status": "complete", "paired_return_delta": 5.0 if candidate else 0.0}
            for seed in seeds]


def _result(seeds, *, candidate=False):
    return {"outer_state_unchanged": True, "outer_updates_before": 99, "outer_updates_after": 99,
            "environment_seeds": seeds, "controller_seed": 12345, "seed_scheme": "sha256-v1",
            "resolved_config": dict(runner.INNER_SETTINGS) if candidate else {},
            "nonfinite_model_metrics": {}, "nonfinite_trace_metrics": {}}


def _reference(directory, row, protocol, seeds, length):
    directory.mkdir()
    prior = {"selector": "controller/prior", "config": {"alg": "AMBIXQC/AMBIXQC", "alg_params": {"inner_operator": "none"}},
             "status": "complete", "action_rule": "tanh_mean", "episodes": _episodes(seeds, length),
             "result": _result(seeds)}
    mppi = {"selector": "controller/mppi", "config": {"alg": "AMBIXQC/AMBIXQC", "alg_params": {"inner_operator": "none"},
             "evaluation_controller": {"type": "mppi", "params": {}}}, "status": "complete"}
    manifest = {"schema_version": 1, "status": "complete", "checkpoint": row,
                "protocol": protocol, "runs": [prior, mppi]}
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def _numerical_flags():
    import torch
    return (torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    manifest, rows = checkpoint_manifest(tmp_path)
    for row in rows:
        sidecar = Path(row["path"] + ".metadata.json")
        metadata = json.loads(sidecar.read_text())
        metadata["schema_version"] = 1
        metadata["checkpoint"].update(kind="periodic", episode=1, best_score=None, best_window=100)
        metadata["trial_run_params"]["alg_params"].update(
            inner_actor_lr=5e-5, inner_critic_lr=5e-5, xqc_policy_delay=3)
        sidecar.write_text(json.dumps(metadata))
        row["metadata_sha256"] = runner.file_sha256(sidecar)
    resolved = {"algorithm_config": {"alg_params": {"obs": "state"}},
                "environment": {"id": "DMControl-v0", "params": {"task": "humanoid-walk", "obs": "state"}}}
    production_protocol = protocol_for(resolved, 12345, 500)
    smoke_protocol = protocol_for(resolved, 12345, 3)
    production_path = _reference(tmp_path / "prior-production", rows[0], production_protocol, list(range(101, 106)), 500)
    smoke_path = _reference(tmp_path / "prior-smoke", rows[0], smoke_protocol, [101, 102], 3)
    rows[0].update(reference_bundle=str(production_path.parent), reference_manifest_sha256=runner.file_sha256(production_path))
    manifest.write_text(json.dumps({"source_run": runner.SOURCE_RUN, "checkpoints": rows}))
    return SimpleNamespace(manifest=manifest, row=rows[0], production=production_path,
                           smoke=smoke_path, production_protocol=production_protocol, smoke_protocol=smoke_protocol,
                           result_root=tmp_path / "result")


def _validated_reference(case, *, smoke=True):
    path = case.smoke if smoke else case.production
    seeds, length, protocol = ([101, 102], 3, case.smoke_protocol) if smoke else (list(range(101, 106)), 500, case.production_protocol)
    return runner.validate_reference_bundle(path.parent, checkpoint=case.row,
        expected_manifest_sha256=runner.file_sha256(path), seeds=seeds, max_steps=length, protocol=protocol)


def _candidate(directory, case, *, seeds, length, reference, variant="inner"):
    profile = runner.campaign_profile(variant)
    counts = profile["optimizer_steps"]
    directory.mkdir(parents=True)
    traces = []
    for seed in seeds:
        path = directory / f"seed-{seed}.jsonl.gz"
        events = [{"episode_id": f"seed-{seed}", "decision_index": index, "event_index": 0, "phase": "decision",
                   **{f"{component}_updates": count for component, count in counts.items()},
                   "metrics": {"decision/inner_model_steps": 9216,
                               **{f"decision/inner_{component}_optimizer_steps": count
                                  for component, count in counts.items()}}}
                  for index in range(length)]
        if profile["terminal_bootstrap"] == "outer":
            for event in events:
                event["metrics"].update({f"decision/{key}": value
                                         for key, value in runner.OUTER_TERMINAL_METRICS.items()})
                event["metrics"]["decision/inner_outer_terminal_bootstrap_rows"] = 3072
        if profile["update_timing"] == "step":
            for event in events:
                event["metrics"]["decision/inner_policy_delay"] = 1
                event["metrics"].update({f"decision/{key}": value
                                         for key, value in runner.STEP_UPDATE_METRICS.items()})
        with gzip.open(path, "wt") as stream:
            stream.write("\n".join(json.dumps(event) for event in events) + "\n")
        traces.append(path.name)
    run = {"selector": "controller/xqc", "config": {"alg": "AMBIXQC/AMBIXQC", "alg_params": {"inner_operator": "xqc"}},
           "status": "complete", "action_rule": "tanh_mean", "result": _result(seeds, candidate=True),
           "episodes": _episodes(seeds, length, candidate=True), "trace_files": traces}
    run["result"]["resolved_config"].update(profile["inner_settings"])
    if profile["terminal_bootstrap"] == "outer":
        run["config"]["alg_params"]["inner_terminal_bootstrap"] = "outer"
    if profile["update_timing"] == "step":
        run["config"]["alg_params"].update(inner_update_timing="step", inner_policy_delay=1)
    manifest = {"status": "complete", "checkpoint": case.row, "protocol": reference["manifest"]["protocol"],
                "reference": {"manifest_sha256": reference["manifest_sha256"]}, "runs": [run]}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def _stub_execution(monkeypatch, case):
    calls = {"evaluate": [], "reports": []}

    def evaluate(matrix, checkpoint, **kwargs):
        assert _numerical_flags() == (True, False, True, False)
        calls["evaluate"].append((matrix, checkpoint, kwargs))
        smoke = kwargs["max_steps"] == 3
        reference = _validated_reference(case, smoke=smoke)
        variant = next(name for name in runner.CAMPAIGN_VARIANTS
                       if runner.campaign_profile(name)["matrix"] == matrix)
        _candidate(kwargs["bundle_dir"], case, seeds=kwargs["seeds"], length=kwargs["max_steps"],
                   reference=reference, variant=variant)
        return {"checkpoint_sha256": case.row["sha256"], "results": [{"controller": "xqc"}]}

    def load(paths):
        calls["reports"].append(paths)
        return {"metric_catalog": {"unused-mppi": {}}, "runs": [
            {"controller_type": controller, "traces": []} for controller in ("prior", "mppi", "xqc")]}

    def write(data, output, **kwargs):
        calls["report_data"] = data
        calls["report_options"] = kwargs
        Path(output).write_text("<html>paired prior/XQC</html>")

    monkeypatch.setitem(sys.modules, "evaluate_ambi_checkpoint", SimpleNamespace(evaluate_matrix=evaluate))
    monkeypatch.setitem(sys.modules, "report_ambi_benchmark", SimpleNamespace(load_bundles=load, write_report=write))
    return calls


@pytest.mark.parametrize("variant", runner.CAMPAIGN_VARIANTS)
def test_production_reuses_five_full_prior_episodes_and_publishes_only_xqc(case, monkeypatch, variant):
    calls = _stub_execution(monkeypatch, case)
    source = json.loads(Path(case.row["path"] + ".metadata.json").read_text())
    assert "inner_reward_normalization" not in source["trial_run_params"]["alg_params"]
    previous = _numerical_flags()
    staged = []
    monkeypatch.setitem(sys.modules, "utils.eval_series", SimpleNamespace(
        load_run=lambda path: {"run_id": "selected"},
        stage_result=lambda run_dir, path, **kwargs: staged.append((run_dir, path, kwargs))))
    destination = runner.run(case.manifest, 0, case.result_root, variant=variant,
                             eval_run_dir=case.result_root.parent / "series")
    assert _numerical_flags() == previous
    assert len(calls["evaluate"]) == 1
    kwargs = calls["evaluate"][0][2]
    assert kwargs["selectors"] == ["controller/xqc"]
    assert kwargs["seeds"] == [101, 102, 103, 104, 105] and kwargs["max_steps"] == 500
    assert kwargs["reference_bundle"] == str(case.production.parent)
    assert kwargs.get("wandb_options") is None
    assert kwargs["eval_run_map"] == {"controller/xqc": str(case.result_root.parent / "series")}
    assert kwargs["stage_results"] is False
    assert kwargs["checkpoint_inventory"] == case.manifest
    assert len(staged) == 1 and staged[0][2]["selector"] == "controller/xqc"
    assert staged[0][2]["inventory_path"] == case.manifest
    provenance = json.loads((destination / "provenance.json").read_text())
    assert provenance["checkpoint"] == case.row
    assert provenance["reference_manifest_sha256"] == case.row["reference_manifest_sha256"]
    assert provenance["checkpoint_manifest_sha256"] == runner.file_sha256(case.manifest)
    assert provenance["matrix_sha256"] == runner.file_sha256(runner.campaign_profile(variant)["matrix"])
    assert provenance["variant"] == variant
    expected_numerics = {"device_type": "cuda", "deterministic_algorithms": True,
                        "deterministic_warn_only": False, "cudnn_deterministic": True,
                        "cudnn_benchmark": False, "cublas_workspace_config": ":4096:8"}
    assert provenance["numerical_settings"] == expected_numerics
    assert json.loads((destination / "paired.json").read_text())["numerical_settings"] == expected_numerics
    assert json.loads((destination / "validation.json").read_text())["decision_counts"] == {"controller/xqc": 2500}
    assert [item["controller_type"] for item in calls["report_data"]["runs"]] == ["prior", "xqc"]
    assert calls["report_data"]["metric_catalog"] == {}
    assert (destination / "paired.json").is_file()


def test_smoke_uses_explicit_short_reference_and_never_truncates_production_reference(case, monkeypatch):
    calls = _stub_execution(monkeypatch, case)
    smoke_sha = runner.file_sha256(case.smoke)
    destination = runner.run(case.manifest, 0, case.result_root, mode="smoke",
                             smoke_reference_bundle=case.smoke.parent,
                             smoke_reference_manifest_sha256=smoke_sha)
    kwargs = calls["evaluate"][0][2]
    assert kwargs["seeds"] == [101, 102] and kwargs["max_steps"] == 3
    assert kwargs["reference_bundle"] == str(case.smoke.parent) and kwargs.get("wandb_options") is None
    provenance = json.loads((destination / "provenance.json").read_text())
    assert provenance["reference_manifest_sha256"] == smoke_sha
    assert provenance["checkpoint"]["reference_manifest_sha256"] == case.row["reference_manifest_sha256"]


@pytest.mark.parametrize("options", [{"mode": "smoke"}, {"mode": "smoke", "wandb": True},
                                    {"smoke_reference_bundle": "/unwanted"}])
def test_mode_errors_precede_outputs(case, options):
    with pytest.raises(ValueError):
        runner.run(case.manifest, 0, case.result_root, **options)
    assert not case.result_root.exists()


@pytest.mark.parametrize("failure", ["hash", "protocol", "checkpoint", "metadata", "frozen", "seed", "length", "source_lr", "source_normalization"])
def test_reference_or_source_errors_precede_outputs_and_evaluation(case, monkeypatch, failure):
    calls = _stub_execution(monkeypatch, case)
    data = json.loads(case.production.read_text())
    if failure == "protocol":
        data["protocol"]["controller_seed"] = 55
    elif failure in {"checkpoint", "metadata"}:
        data["checkpoint"]["sha256" if failure == "checkpoint" else "metadata_sha256"] = "wrong"
    elif failure == "frozen":
        data["runs"][0]["result"]["outer_state_unchanged"] = False
    elif failure == "seed":
        data["runs"][0]["episodes"][0]["seed"] = 999
    elif failure == "length":
        data["runs"][0]["episodes"][0]["length"] = 3
    elif failure in {"source_lr", "source_normalization"}:
        sidecar = Path(case.row["path"] + ".metadata.json")
        metadata = json.loads(sidecar.read_text())
        key, value = ("inner_actor_lr", 0.2) if failure == "source_lr" else ("inner_reward_normalization", "action_local_imagined")
        metadata["trial_run_params"]["alg_params"][key] = value
        sidecar.write_text(json.dumps(metadata))
        case.row["metadata_sha256"] = runner.file_sha256(sidecar)
    case.production.write_text(json.dumps(data))
    if failure != "hash":
        case.row["reference_manifest_sha256"] = runner.file_sha256(case.production)
    else:
        case.row["reference_manifest_sha256"] = "0" * 64
    manifest = json.loads(case.manifest.read_text())
    manifest["checkpoints"][0] = case.row
    case.manifest.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        runner.run(case.manifest, 0, case.result_root)
    assert not case.result_root.exists() and calls["evaluate"] == []


@pytest.mark.parametrize("failure", ["counts", "model_steps", "duplicate", "missing", "delta", "extra_controller", "dose"])
def test_candidate_acceptance_rejects_wrong_work_or_pairing(case, tmp_path, failure):
    reference = _validated_reference(case)
    directory = tmp_path / "candidate"
    manifest = _candidate(directory, case, seeds=[101, 102], length=3, reference=reference)
    run = manifest["runs"][0]
    trace = directory / run["trace_files"][0]
    with gzip.open(trace, "rt") as stream:
        events = [json.loads(line) for line in stream]
    if failure == "counts":
        events[0]["actor_updates"] = 18
    elif failure == "model_steps":
        events[0]["metrics"]["decision/inner_model_steps"] = 192
    elif failure == "duplicate":
        events[-1] = events[0]
    elif failure == "missing":
        events.pop()
    elif failure == "delta":
        run["episodes"][0]["paired_return_delta"] = 100
    elif failure == "extra_controller":
        manifest["runs"].append(deepcopy(run))
    else:
        run["result"]["resolved_config"]["inner_batch_size"] = 64
    with gzip.open(trace, "wt") as stream:
        stream.write("\n".join(json.dumps(event) for event in events) + "\n")
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        runner.validate_bundle(directory, checkpoint=case.row, reference=reference,
                               protocol=case.smoke_protocol, seeds=[101, 102], max_steps=3)


def test_paired_results_survive_later_validation_failure(case, monkeypatch):
    _stub_execution(monkeypatch, case)
    monkeypatch.setattr(runner, "validate_bundle", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("acceptance failed")))
    with pytest.raises(ValueError, match="acceptance failed"):
        runner.run(case.manifest, 0, case.result_root, mode="smoke", smoke_reference_bundle=case.smoke.parent,
                   smoke_reference_manifest_sha256=runner.file_sha256(case.smoke))
    destination = case.result_root / "step_50000"
    assert (destination / "paired.json").is_file()
    assert (destination / "bundle" / "manifest.json").is_file()


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("original", [(False, True, False, True), (True, False, True, False)])
def test_cpu_deterministic_context_restores_all_flags(monkeypatch, fail, original):
    import torch
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    previous = _numerical_flags()
    try:
        torch.use_deterministic_algorithms(original[0], warn_only=original[1])
        torch.backends.cudnn.deterministic = original[2]
        torch.backends.cudnn.benchmark = original[3]
        try:
            with runner.deterministic_evaluation(device="cpu") as settings:
                assert _numerical_flags() == (True, False, True, False)
                assert settings["device_type"] == "cpu" and settings["cublas_workspace_config"] is None
                if fail:
                    raise RuntimeError("evaluation failed")
        except RuntimeError as error:
            assert fail and str(error) == "evaluation failed"
        assert _numerical_flags() == original
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
        torch.backends.cudnn.deterministic = previous[2]
        torch.backends.cudnn.benchmark = previous[3]


@pytest.mark.parametrize("workspace", [None, "", ":invalid"])
def test_cuda_workspace_preflight_precedes_outputs_and_evaluation(case, monkeypatch, workspace):
    if workspace is None:
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
    else:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", workspace)
    calls = _stub_execution(monkeypatch, case)
    previous = _numerical_flags()
    with pytest.raises(ValueError, match="set before starting Python/CUDA"):
        runner.run(case.manifest, 0, case.result_root)
    assert _numerical_flags() == previous
    assert not case.result_root.exists() and calls["evaluate"] == []


@pytest.mark.parametrize("workspace", [":4096:8", ":16:8"])
def test_cuda_context_accepts_supported_workspace_without_initializing_cuda(monkeypatch, workspace):
    import torch
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", workspace)
    initialized = torch.cuda.is_initialized()
    with runner.deterministic_evaluation(device="cuda:0") as settings:
        assert settings["cublas_workspace_config"] == workspace
        assert _numerical_flags() == (True, False, True, False)
    assert torch.cuda.is_initialized() == initialized


def test_runner_restores_numerics_when_evaluator_raises(case, monkeypatch):
    def fail(*args, **kwargs):
        assert _numerical_flags() == (True, False, True, False)
        raise RuntimeError("controller failed")

    monkeypatch.setitem(sys.modules, "evaluate_ambi_checkpoint", SimpleNamespace(evaluate_matrix=fail))
    previous = _numerical_flags()
    with pytest.raises(RuntimeError, match="controller failed"):
        runner.run(case.manifest, 0, case.result_root)
    assert _numerical_flags() == previous
    provenance = json.loads((case.result_root / "step_50000" / "provenance.json").read_text())
    assert provenance["numerical_settings"]["deterministic_algorithms"] is True


def test_outer_terminal_matrix_changes_only_terminal_bootstrap(case):
    from utils.ambi_research import load_preset_matrix, normalize_selectors, resolve_preset
    from utils.checkpoint_context import load_checkpoint_context

    inner = runner.campaign_profile()
    outer = runner.campaign_profile("outer_terminal")
    assert inner["matrix"] == runner.MATRIX and inner["inner_settings"] == runner.INNER_SETTINGS
    assert outer["inner_settings"] == {**runner.INNER_SETTINGS, "inner_terminal_bootstrap": "outer"}
    matrix = load_preset_matrix(outer["matrix"])
    baseline = load_preset_matrix(inner["matrix"])
    assert matrix["base_alg_config"] == "checkpoint"
    assert matrix["evaluation"] == baseline["evaluation"]
    assert matrix["budget_source"] == baseline["budget_source"]
    assert normalize_selectors(matrix) == ["controller/xqc"]
    context = load_checkpoint_context(case.row["path"])
    for selector in ("controller/prior", "controller/xqc"):
        previous = resolve_preset(inner["matrix"], selector, checkpoint_context=context)
        resolved = resolve_preset(outer["matrix"], selector, checkpoint_context=context)
        expected = deepcopy(previous["algorithm_config"])
        if selector == "controller/xqc":
            expected["alg_params"]["inner_terminal_bootstrap"] = "outer"
        assert resolved["algorithm_config"] == expected
        assert resolved["saved_algorithm_config"] == previous["saved_algorithm_config"]
        assert protocol_for(resolved, 12345, 500) == protocol_for(previous, 12345, 500)
    outer["inner_settings"]["inner_actor_lr"] = 1
    assert runner.campaign_profile("outer_terminal")["inner_settings"]["inner_actor_lr"] == 5e-5


@pytest.mark.parametrize("mode", ["production", "smoke"])
@pytest.mark.parametrize("variant", ["outer_terminal", "outer_terminal_step"])
def test_outer_terminal_runner_reuses_existing_priors_with_distinct_provenance(case, monkeypatch, mode, variant):
    calls = _stub_execution(monkeypatch, case)
    options = {"mode": mode, "variant": variant}
    if mode == "smoke":
        options.update(smoke_reference_bundle=case.smoke.parent,
                       smoke_reference_manifest_sha256=runner.file_sha256(case.smoke))
    destination = runner.run(case.manifest, 0, case.result_root, **options)
    profile = runner.campaign_profile(variant)
    matrix, checkpoint, kwargs = calls["evaluate"][0]
    assert matrix == profile["matrix"] and checkpoint == case.row["path"]
    assert kwargs["selectors"] == ["controller/xqc"]
    expected_reference = case.smoke if mode == "smoke" else case.production
    assert kwargs["reference_bundle"] == str(expected_reference.parent)
    for name in ("provenance.json", "paired.json", "validation.json"):
        assert json.loads((destination / name).read_text())["variant"] == variant
    provenance = json.loads((destination / "provenance.json").read_text())
    assert provenance["matrix_sha256"] == runner.file_sha256(profile["matrix"])
    assert provenance["checkpoint"] == case.row
    assert "outer-terminal" in calls["report_options"]["title"]
    assert [row["controller_type"] for row in calls["report_data"]["runs"]] == ["prior", "xqc"]


@pytest.mark.parametrize("key,value", [
    ("inner_terminal_bootstrap_outer", 0),
    ("inner_outer_terminal_boundary_rows", 9216),
    ("inner_outer_terminal_policy_evaluations", 3072),
    ("inner_outer_terminal_q_evaluations", 3072),
    ("inner_outer_terminal_bootstrap_rows", -1),
    ("inner_outer_terminal_bootstrap_rows", 9217),
    ("inner_outer_terminal_bootstrap_rows", 1.5),
    ("inner_outer_terminal_bootstrap_rows", None),
])
def test_outer_terminal_acceptance_requires_boundary_and_frozen_outer_work(case, tmp_path, key, value):
    reference = _validated_reference(case)
    directory = tmp_path / "candidate"
    manifest = _candidate(directory, case, seeds=[101, 102], length=3,
                          reference=reference, variant="outer_terminal")
    trace = directory / manifest["runs"][0]["trace_files"][0]
    with gzip.open(trace, "rt") as stream:
        events = [json.loads(line) for line in stream]
    if value is None:
        events[0]["metrics"].pop(f"decision/{key}")
    else:
        events[0]["metrics"][f"decision/{key}"] = value
    with gzip.open(trace, "wt") as stream:
        stream.write("\n".join(json.dumps(event) for event in events) + "\n")
    with pytest.raises(ValueError, match="Outer-terminal diagnostics"):
        runner.validate_bundle(directory, checkpoint=case.row, reference=reference,
                               protocol=case.smoke_protocol, seeds=[101, 102], max_steps=3,
                               variant="outer_terminal")


@pytest.mark.parametrize("sampled", [0, 9216])
def test_outer_terminal_sampling_accepts_valid_inclusive_row_bounds(case, tmp_path, sampled):
    reference = _validated_reference(case)
    directory = tmp_path / "candidate"
    manifest = _candidate(directory, case, seeds=[101, 102], length=3,
                          reference=reference, variant="outer_terminal")
    for relative in manifest["runs"][0]["trace_files"]:
        path = directory / relative
        with gzip.open(path, "rt") as stream:
            events = [json.loads(line) for line in stream]
        for event in events:
            event["metrics"]["decision/inner_outer_terminal_bootstrap_rows"] = sampled
        with gzip.open(path, "wt") as stream:
            stream.write("\n".join(json.dumps(event) for event in events) + "\n")
    result = runner.validate_bundle(directory, checkpoint=case.row, reference=reference,
                                    protocol=case.smoke_protocol, seeds=[101, 102], max_steps=3,
                                    variant="outer_terminal")
    assert result["decision_counts"] == {"controller/xqc": 6}


@pytest.mark.parametrize("actual,requested", [(actual, requested) for actual in runner.CAMPAIGN_VARIANTS
                                              for requested in runner.CAMPAIGN_VARIANTS if actual != requested])
def test_campaign_variants_reject_each_others_candidate(case, tmp_path, actual, requested):
    reference = _validated_reference(case)
    directory = tmp_path / "candidate"
    _candidate(directory, case, seeds=[101, 102], length=3, reference=reference, variant=actual)
    with pytest.raises(ValueError, match="Resolved inner-XQC settings"):
        runner.validate_bundle(directory, checkpoint=case.row, reference=reference,
                               protocol=case.smoke_protocol, seeds=[101, 102], max_steps=3, variant=requested)


def test_invalid_campaign_variant_fails_before_source_or_outputs(case):
    with pytest.raises(ValueError, match="Variant must"):
        runner.run("nonexistent", 0, case.result_root, variant="outer_target")
    assert not case.result_root.exists()


@pytest.mark.parametrize("variant", ["outer_terminal", "outer_terminal_step"])
def test_runner_cli_forwards_explicit_campaign_variant(monkeypatch, variant):
    calls = []
    monkeypatch.setattr(runner, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    runner.main(["--manifest", "/manifest.json", "--index", "0", "--result-root", "/results",
                 "--variant", variant])
    assert calls[0][1]["variant"] == variant


def test_step_matrix_preserves_outer_terminal_protocol_and_sets_one_actor_and_critic_per_depth(case):
    from utils.ambi_research import load_preset_matrix, resolve_preset
    from utils.checkpoint_context import load_checkpoint_context

    outer = runner.campaign_profile("outer_terminal")
    step = runner.campaign_profile("outer_terminal_step")
    assert step["terminal_bootstrap"] == "outer" and step["update_timing"] == "step"
    assert step["inner_policy_delay"] == 1 and outer["inner_policy_delay"] == 3
    assert outer["update_timing"] == "round"
    assert runner.campaign_profile()["terminal_bootstrap"] == "inner"
    assert step["inner_settings"] == {**outer["inner_settings"], "inner_update_timing": "step",
                                      "inner_policy_delay": 1}
    assert step["optimizer_steps"] == {"critic": 18, "actor": 18, "temperature": 18}
    assert outer["optimizer_steps"] == runner.OPTIMIZER_STEPS == {"critic": 18, "actor": 6, "temperature": 6}
    matrix, baseline = load_preset_matrix(step["matrix"]), load_preset_matrix(outer["matrix"])
    assert matrix["evaluation"] == baseline["evaluation"]
    assert matrix["budget_source"] == baseline["budget_source"]
    context = load_checkpoint_context(case.row["path"])
    for selector in ("controller/prior", "controller/xqc"):
        previous = resolve_preset(outer["matrix"], selector, checkpoint_context=context)
        resolved = resolve_preset(step["matrix"], selector, checkpoint_context=context)
        expected = deepcopy(previous["algorithm_config"])
        if selector == "controller/xqc":
            expected["alg_params"].update(inner_update_timing="step", inner_policy_delay=1)
        assert resolved["algorithm_config"] == expected
        assert resolved["saved_algorithm_config"] == previous["saved_algorithm_config"]
        assert protocol_for(resolved, 12345, 500) == protocol_for(previous, 12345, 500)
    assert expected["alg_params"]["inner_updates_per_round"] // expected["alg_params"]["inner_rollout_horizon"] == 1
    assert expected["alg_params"]["xqc_policy_delay"] == 3


@pytest.mark.parametrize("key,value", [
    ("inner_update_timing_step", 0), ("inner_update_timing_step", None),
    ("inner_updates_per_rollout_step", 3), ("inner_updates_per_rollout_step", None),
    ("inner_collection_steps", 6), ("inner_collection_steps", None),
])
def test_step_acceptance_requires_depth_timing_metrics(case, tmp_path, key, value):
    reference = _validated_reference(case)
    directory = tmp_path / "candidate"
    manifest = _candidate(directory, case, seeds=[101, 102], length=3,
                          reference=reference, variant="outer_terminal_step")
    path = directory / manifest["runs"][0]["trace_files"][0]
    with gzip.open(path, "rt") as stream:
        events = [json.loads(line) for line in stream]
    if value is None:
        events[0]["metrics"].pop(f"decision/{key}")
    else:
        events[0]["metrics"][f"decision/{key}"] = value
    with gzip.open(path, "wt") as stream:
        stream.write("\n".join(json.dumps(event) for event in events) + "\n")
    with pytest.raises(ValueError, match="Step-update diagnostics"):
        runner.validate_bundle(directory, checkpoint=case.row, reference=reference,
                               protocol=case.smoke_protocol, seeds=[101, 102], max_steps=3,
                               variant="outer_terminal_step")


def test_step_acceptance_rejects_delayed_actor_counts_even_with_correct_timing(case, tmp_path):
    reference = _validated_reference(case)
    directory = tmp_path / "candidate"
    manifest = _candidate(directory, case, seeds=[101, 102], length=3,
                          reference=reference, variant="outer_terminal_step")
    path = directory / manifest["runs"][0]["trace_files"][0]
    with gzip.open(path, "rt") as stream:
        events = [json.loads(line) for line in stream]
    events[0]["actor_updates"] = 6
    events[0]["metrics"]["decision/inner_actor_optimizer_steps"] = 6
    with gzip.open(path, "wt") as stream:
        stream.write("\n".join(json.dumps(event) for event in events) + "\n")
    with pytest.raises(ValueError, match="C18/A18/T18"):
        runner.validate_bundle(directory, checkpoint=case.row, reference=reference,
                               protocol=case.smoke_protocol, seeds=[101, 102], max_steps=3,
                               variant="outer_terminal_step")


@pytest.mark.parametrize("delay", [None, 3])
def test_step_acceptance_requires_inner_policy_delay_one(case, tmp_path, delay):
    reference = _validated_reference(case)
    directory = tmp_path / "candidate"
    manifest = _candidate(directory, case, seeds=[101, 102], length=3,
                          reference=reference, variant="outer_terminal_step")
    path = directory / manifest["runs"][0]["trace_files"][0]
    with gzip.open(path, "rt") as stream:
        events = [json.loads(line) for line in stream]
    if delay is None:
        events[0]["metrics"].pop("decision/inner_policy_delay")
    else:
        events[0]["metrics"]["decision/inner_policy_delay"] = delay
    with gzip.open(path, "wt") as stream:
        stream.write("\n".join(json.dumps(event) for event in events) + "\n")
    with pytest.raises(ValueError, match="Inner policy-delay diagnostic"):
        runner.validate_bundle(directory, checkpoint=case.row, reference=reference,
                               protocol=case.smoke_protocol, seeds=[101, 102], max_steps=3,
                               variant="outer_terminal_step")
