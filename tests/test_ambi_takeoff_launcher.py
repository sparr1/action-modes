"""Verify the campaign's workload mapping and publication boundary without Slurm."""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("takeoff_campaign", ROOT / "slurm/ambi_takeoff_campaign.py")
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


def inventory(tmp_path, steps=campaign.STEPS):
    rows = []
    for step in steps:
        checkpoint = tmp_path / f"model_{step}"
        checkpoint.write_text(str(step))
        metadata = Path(str(checkpoint) + ".metadata.json")
        metadata.write_text(json.dumps({"checkpoint": {"step": step}}))
        prior = tmp_path / f"prior_{step}"
        prior.mkdir()
        (prior / "manifest.json").write_text("{}")
        rows.append({"step": step, "path": str(checkpoint), "sha256": campaign._hash(checkpoint),
                     "metadata_sha256": campaign._hash(metadata), "prior_reference_bundle": str(prior),
                     "prior_reference_manifest_sha256": campaign._hash(prior / "manifest.json")})
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps({"source_run": campaign.SOURCE_RUN, "checkpoints": rows}))
    return path


def argument(command, key):
    return command[command.index(key) + 1]


@pytest.mark.parametrize("index", range(14))
def test_production_maps_each_checkpoint_to_separate_work_and_only_stages_curves(tmp_path, monkeypatch, index):
    manifest = inventory(tmp_path)
    run_map = tmp_path / "run-map.json"
    run_map.write_text("{}")
    calls = []
    monkeypatch.setattr(campaign, "_run", lambda args, **kwargs: calls.append(list(map(str, args))))
    campaign.run_worker(manifest, tmp_path / "results", "new-attempt", index, eval_run_map=run_map)
    mode, step_index = campaign.task_cell(index)
    evaluate = next(args for args in calls if args[0] in ("evaluate_ambi_checkpoint.py", "evaluate_ambi_calibration.py"))
    assert argument(evaluate, "--checkpoint").endswith(f"model_{campaign.STEPS[step_index]}")
    assert argument(evaluate, "--matrix") == campaign.MATRIX
    assert argument(evaluate, "--preset") == "initialization/scratch"
    assert argument(evaluate, "--device") == "cuda"
    assert all("--mode" not in command and "--wandb" not in command for command in calls)
    assert not any(command[:2] == ["-m", "pytest"] for command in calls)
    if mode == "episodes":
        assert argument(evaluate, "--eval-run-map") == str(run_map)
        assert argument(evaluate, "--checkpoint-inventory") == str(manifest)
        assert "--reference-bundle" in evaluate
        assert "--seeds" not in evaluate and "--max-steps" not in evaluate
        assert calls[-1][1] == "export-model"
    else:
        assert evaluate[1] == "run"
        assert argument(evaluate, "--benchmark-repetitions") == "7"
        assert "--seeds" not in evaluate and "--tail-steps" not in evaluate
        assert "--save-root-bank" in evaluate and "--reference-cache" in evaluate


def test_smoke_exercises_full_snapshot_batch_and_continuing_tail_without_publication(tmp_path, monkeypatch):
    manifest = inventory(tmp_path)
    calls = []
    monkeypatch.setattr(campaign, "_run", lambda args, **kwargs: calls.append((list(map(str, args)), kwargs)))
    campaign.run_worker(manifest, tmp_path / "results", "smoke-only", 10, smoke=True)
    assert "CUDA unavailable" in calls[0][0][-1]
    tests, options = calls[1]
    assert tests[:3] == ["-m", "pytest", "-q"]
    assert options["env"]["AMBI_RUN_REAL_DMCONTROL_TESTS"] == "1"
    evaluate = calls[-1][0]
    assert argument(evaluate, "--checkpoint").endswith("model_200000")
    for key, value in {"--rollout-repetitions": "4", "--solver-repetitions": "1",
                       "--tail-steps": "1000", "--max-steps": "500", "--seeds": "101",
                       "--decisions": "0", "--benchmark-repetitions": "7"}.items():
        assert argument(evaluate, key) == value
    assert "--rounds" not in evaluate  # All initialized/completed actor snapshots.
    assert "--eval-run-map" not in evaluate


@pytest.mark.parametrize("smoke", [False, True])
@pytest.mark.parametrize("index", [3, 10])
def test_selected_matrix_reaches_episode_and_real_workers(tmp_path, monkeypatch, smoke, index):
    manifest = inventory(tmp_path)
    matrix = tmp_path / "selected-matrix.json"
    matrix.write_text((ROOT / campaign.MATRIX).read_text())
    run_map = tmp_path / "run-map.json"
    run_map.write_text("{}")
    calls = []
    monkeypatch.setattr(campaign, "_run", lambda args, **kwargs: calls.append(list(map(str, args))))
    campaign.run_worker(manifest, tmp_path / "results", "matrix-attempt", index,
                        smoke=smoke, eval_run_map=run_map, matrix=matrix)
    evaluate = next(args for args in calls if args[0] in
                    ("evaluate_ambi_checkpoint.py", "evaluate_ambi_calibration.py"))
    assert argument(evaluate, "--matrix") == str(matrix)
    mode, _ = campaign.task_cell(index)
    output = tmp_path / "results" / ("smoke" if smoke else "production") / "step_200000" / mode
    assert json.loads((output / "worker-completion.json").read_text())["matrix"] == str(matrix)


@pytest.mark.parametrize("source", ["default", "environment", "explicit"])
def test_worker_cli_selects_matrix_with_explicit_argument_precedence(tmp_path, monkeypatch, source):
    calls = []
    monkeypatch.setattr(campaign, "run_worker", lambda **kwargs: calls.append(kwargs))
    monkeypatch.delenv("AMBI_TAKEOFF_MATRIX", raising=False)
    args = ["worker", "--inventory", str(tmp_path / "inventory.json"),
            "--output-root", str(tmp_path / "results"), "--attempt-label", "new-attempt",
            "--task-index", "3"]
    expected = Path(campaign.MATRIX)
    if source in ("environment", "explicit"):
        expected = tmp_path / "environment-matrix.json"
        monkeypatch.setenv("AMBI_TAKEOFF_MATRIX", str(expected))
    if source == "explicit":
        expected = tmp_path / "explicit-matrix.json"
        args += ["--matrix", str(expected)]
    assert campaign.main(args) == 0
    assert calls[0]["matrix"] == expected


def test_missing_matrix_is_rejected_before_compute_or_output_creation(tmp_path, monkeypatch):
    manifest = inventory(tmp_path)
    calls = []
    monkeypatch.setattr(campaign, "_run", lambda args, **kwargs: calls.append(args))
    with pytest.raises(ValueError, match="Research matrix must be an existing file"):
        campaign.run_worker(manifest, tmp_path / "results", "new-attempt", 10,
                            matrix=tmp_path / "missing-matrix.json")
    assert not calls
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("failure", ["checkpoint_hash", "sidecar_hash", "sidecar_step", "missing_reference", "reference_hash", "missing_reference_hash", "existing_output", "wrong_source", "wrong_panel"])
def test_preflight_rejects_invalid_work_before_any_compute(tmp_path, monkeypatch, failure):
    path = inventory(tmp_path)
    payload = json.loads(path.read_text())
    row = payload["checkpoints"][0]
    if failure == "checkpoint_hash":
        Path(row["path"]).write_text("changed")
    elif failure == "sidecar_hash":
        Path(row["path"] + ".metadata.json").write_text("{}")
    elif failure == "sidecar_step":
        metadata = Path(row["path"] + ".metadata.json")
        metadata.write_text(json.dumps({"checkpoint": {"step": 1}}))
        row["metadata_sha256"] = campaign._hash(metadata)
    elif failure == "missing_reference":
        row.pop("prior_reference_bundle")
    elif failure == "reference_hash":
        (Path(row["prior_reference_bundle"]) / "manifest.json").write_text('{"changed": true}')
    elif failure == "missing_reference_hash":
        row.pop("prior_reference_manifest_sha256")
    elif failure == "existing_output":
        (tmp_path / "results/production/step_100000/episodes").mkdir(parents=True)
    elif failure == "wrong_source":
        payload["source_run"] = "other"
    else:
        payload["checkpoints"].pop()
    path.write_text(json.dumps(payload))
    run_map = tmp_path / "run-map.json"
    run_map.write_text("{}")
    calls = []
    monkeypatch.setattr(campaign, "_run", lambda args, **kwargs: calls.append(args))
    with pytest.raises((ValueError, FileExistsError)):
        campaign.run_worker(path, tmp_path / "results", "new-attempt", 0, eval_run_map=run_map)
    assert not calls


def test_publisher_drains_successes_even_with_failed_or_missing_other_cells(tmp_path, monkeypatch):
    for mode, status in (("episodes", "complete"), ("real", "incomplete")):
        bundle = tmp_path / "production/step_100000" / mode / ("model-series" if mode == "episodes" else "bundle")
        bundle.mkdir(parents=True)
        (bundle / "manifest.json").write_text(json.dumps({"status": status}))
    calls = []
    monkeypatch.setattr(campaign, "_run", lambda args, **kwargs: calls.append(list(map(str, args))))
    assert campaign.publish_completed(tmp_path, entity="entity", project="project") == 1
    assert len(calls) == 1 and calls[0][:2] == ["evaluate_ambi_calibration.py", "publish"]
    assert argument(calls[0], "--mode") == "online"
    receipt = json.loads(next(tmp_path.glob("publication-summary-*.json")).read_text())
    assert len(receipt["results"]) == 14
    assert receipt["results"][0]["status"] == "complete"
    assert receipt["results"][7]["status"] == "skipped"


def test_shell_launchers_parse_and_keep_submission_concurrency_explicit():
    for name in ("run_ambi_takeoff_oscar.sbatch", "run_ambi_takeoff_publisher_oscar.sbatch"):
        path = ROOT / "slurm" / name
        subprocess.run(["bash", "-n", str(path)], check=True)
        text = path.read_text()
        assert "#SBATCH --array=" not in text
        assert "EXPECTED_ACTION_MODES_SHA" in text and "git status --porcelain" in text


PRIOR_MATRIX = ROOT / "configs/research/ambi_prior_refinement_h1_200k.json"


@pytest.mark.parametrize("smoke", [False, True])
@pytest.mark.parametrize("index,mode", [(0, "episodes"), (1, "real")])
def test_singleton_prior_matrix_limits_work_and_preserves_selector(tmp_path, monkeypatch, smoke, index, mode):
    manifest = inventory(tmp_path, steps=(200000,))
    run_map = tmp_path / "run-map.json"
    run_map.write_text("{}")
    calls = []
    monkeypatch.setattr(campaign, "_run", lambda args, **kwargs: calls.append(list(map(str, args))))
    campaign.run_worker(manifest, tmp_path / "results", "prior-attempt", index,
                        matrix=PRIOR_MATRIX, smoke=smoke, eval_run_map=run_map)
    evaluate = next(args for args in calls if args[0] in
                    ("evaluate_ambi_checkpoint.py", "evaluate_ambi_calibration.py"))
    assert argument(evaluate, "--checkpoint").endswith("model_200000")
    assert argument(evaluate, "--preset") == "initialization/inherited"
    if mode == "episodes":
        assert argument(calls[-1], "--selector") == "initialization/inherited"
        assert ("--max-steps" in evaluate) == smoke
    else:
        assert ("--solver-repetitions" in evaluate) == smoke
    output = tmp_path / "results" / ("smoke" if smoke else "production") / "step_200000" / mode
    receipt = json.loads((output / "worker-completion.json").read_text())
    assert receipt["checkpoint_steps"] == [200000]
    assert receipt["selector"] == "initialization/inherited"
    assert receipt["matrix_sha256"] == campaign._hash(PRIOR_MATRIX)


@pytest.mark.parametrize("index", [-1, 2, 3, 10, 13])
def test_singleton_rejects_unrequested_worker_indices_before_compute(tmp_path, monkeypatch, index):
    manifest = inventory(tmp_path, steps=(200000,))
    calls = []
    monkeypatch.setattr(campaign, "_run", lambda args, **kwargs: calls.append(args))
    with pytest.raises(ValueError, match="Task index must be between 0 and 1"):
        campaign.run_worker(manifest, tmp_path / "results", "prior-attempt", index,
                            matrix=PRIOR_MATRIX, smoke=True)
    assert not calls and not (tmp_path / "results").exists()


def test_singleton_rejects_seven_checkpoint_inventory(tmp_path, monkeypatch):
    manifest = inventory(tmp_path)
    calls = []
    monkeypatch.setattr(campaign, "_run", lambda args, **kwargs: calls.append(args))
    with pytest.raises(ValueError, match="exactly the matrix-selected"):
        campaign.run_worker(manifest, tmp_path / "results", "prior-attempt", 0,
                            matrix=PRIOR_MATRIX, smoke=True)
    assert not calls and not (tmp_path / "results").exists()


def reusable_real_inventory(tmp_path):
    manifest = inventory(tmp_path, steps=(200000,))
    payload = json.loads(manifest.read_text())
    bank = tmp_path / "roots.json"
    bank.write_text("{}")
    cache = tmp_path / "copied-reference-cache"
    cache.mkdir()
    payload["checkpoints"][0].update(real_root_bank=str(bank), real_root_bank_sha256=campaign._hash(bank),
                                     real_reference_cache=str(cache))
    manifest.write_text(json.dumps(payload))
    return manifest, payload


@pytest.mark.parametrize("smoke", [False, True])
def test_real_reference_reuse_is_explicit_and_production_only(tmp_path, monkeypatch, smoke):
    manifest, payload = reusable_real_inventory(tmp_path)
    row = payload["checkpoints"][0]
    calls = []
    monkeypatch.setattr(campaign, "_run", lambda args, **kwargs: calls.append(list(map(str, args))))
    campaign.run_worker(manifest, tmp_path / "results", "attempt", 1, matrix=PRIOR_MATRIX, smoke=smoke)
    evaluate = calls[-1]
    assert ("--save-root-bank" in evaluate) == smoke
    assert ("--root-bank" in evaluate) != smoke
    if not smoke:
        assert argument(evaluate, "--root-bank") == row["real_root_bank"]
        assert argument(evaluate, "--reference-cache") == row["real_reference_cache"]
        receipt = json.loads((tmp_path / "results/production/step_200000/real/worker-completion.json").read_text())
        assert receipt["real_root_bank_sha256"] == row["real_root_bank_sha256"]
    else:
        assert argument(evaluate, "--reference-cache") != row["real_reference_cache"]


@pytest.mark.parametrize("failure", ["missing_hash", "changed_bank", "relative_bank", "missing_bank",
                                     "missing_cache_key", "relative_cache", "missing_cache"])
def test_real_reference_reuse_is_verified_before_compute(tmp_path, monkeypatch, failure):
    manifest, payload = reusable_real_inventory(tmp_path)
    row = payload["checkpoints"][0]
    if failure == "missing_hash":
        row.pop("real_root_bank_sha256")
    elif failure == "changed_bank":
        Path(row["real_root_bank"]).write_text('{"changed":true}')
    elif failure == "relative_bank":
        row["real_root_bank"] = "roots.json"
    elif failure == "missing_bank":
        row["real_root_bank"] = str(tmp_path / "missing.json")
    elif failure == "missing_cache_key":
        row.pop("real_reference_cache")
    elif failure == "relative_cache":
        row["real_reference_cache"] = "cache"
    elif failure == "missing_cache":
        row["real_reference_cache"] = str(tmp_path / "missing-cache")
    manifest.write_text(json.dumps(payload))
    calls = []
    monkeypatch.setattr(campaign, "_run", lambda args, **kwargs: calls.append(args))
    with pytest.raises(ValueError):
        campaign.run_worker(manifest, tmp_path / "results", "attempt", 1, matrix=PRIOR_MATRIX)
    assert not calls and not (tmp_path / "results").exists()


@pytest.mark.parametrize("task_index,expected_modes", [(None, ["episodes", "real"]),
                                                       (0, ["episodes"]), (1, ["real"])])
def test_singleton_publisher_visits_only_selected_panel(tmp_path, monkeypatch, task_index, expected_modes):
    # Even when old checkpoint bundles exist, they are not selected for upload.
    for step in campaign.STEPS:
        for mode in ("episodes", "real"):
            output = tmp_path / "production" / f"step_{step}" / mode
            bundle = output / ("model-series" if mode == "episodes" else "bundle")
            bundle.mkdir(parents=True)
            (bundle / "manifest.json").write_text('{"status":"complete"}')
    calls = []
    monkeypatch.setattr(campaign, "_run", lambda args, **kwargs: calls.append(list(map(str, args))))
    assert campaign.publish_completed(tmp_path, entity="entity", project="project",
                                      task_index=task_index, matrix=PRIOR_MATRIX) == 0
    receipt = json.loads(next(tmp_path.glob("publication-summary-*.json")).read_text())
    assert [row["mode"] for row in receipt["results"]] == expected_modes
    assert all(row["step"] == 200000 for row in receipt["results"])
    assert len(calls) == len(expected_modes)
    assert all("step_200000" in argument(call, "--bundle") for call in calls)
    assert receipt["selector"] == "initialization/inherited"


def test_singleton_publisher_rejects_unrequested_index_before_publication(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(campaign, "_run", lambda args, **kwargs: calls.append(args))
    with pytest.raises(ValueError, match="Task index must be between 0 and 1"):
        campaign.publish_completed(tmp_path, entity="entity", project="project",
                                   task_index=2, matrix=PRIOR_MATRIX)
    assert not calls and not list(tmp_path.glob("publication-summary-*.json"))


@pytest.mark.parametrize("source", ["default", "environment", "explicit"])
def test_publisher_cli_matches_worker_matrix_selection(tmp_path, monkeypatch, source):
    calls = []
    monkeypatch.setattr(campaign, "publish_completed", lambda **kwargs: calls.append(kwargs) or 0)
    monkeypatch.delenv("AMBI_TAKEOFF_MATRIX", raising=False)
    args = ["publish", "--output-root", str(tmp_path)]
    expected = Path(campaign.MATRIX)
    if source in ("environment", "explicit"):
        expected = tmp_path / "environment-matrix.json"
        monkeypatch.setenv("AMBI_TAKEOFF_MATRIX", str(expected))
    if source == "explicit":
        expected = PRIOR_MATRIX
        args += ["--matrix", str(expected)]
    assert campaign.main(args) == 0
    assert calls[0]["matrix"] == expected


@pytest.mark.parametrize("failure", ["wrong_source", "missing_panel", "duplicate_step", "unknown_step",
                                     "missing_hash", "invalid_hash", "no_default", "multiple_defaults",
                                     "unknown_preset", "duplicate_json_key"])
def test_invalid_matrix_fails_before_compute_or_publication(tmp_path, monkeypatch, failure):
    config = json.loads(PRIOR_MATRIX.read_text())
    panel = config["checkpoint_contract"]["checkpoints"]
    if failure == "wrong_source":
        config["source_run"] = "other"
    elif failure == "missing_panel":
        config["checkpoint_contract"].pop("checkpoints")
    elif failure == "duplicate_step":
        panel.append(dict(panel[0]))
    elif failure == "unknown_step":
        panel[0]["step"] = 250000
    elif failure == "missing_hash":
        panel[0].pop("sha256")
    elif failure == "invalid_hash":
        panel[0]["sha256"] = "z" * 64
    elif failure == "no_default":
        config["evaluation"]["default_presets"] = []
    elif failure == "multiple_defaults":
        config["evaluation"]["default_presets"] = ["initialization/inherited", "initialization/prior"]
    elif failure == "unknown_preset":
        config["evaluation"]["default_presets"] = ["initialization/missing"]
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps(config) if failure != "duplicate_json_key" else
                      '{"schema_version": 1, "schema_version": 1}')
    calls = []
    monkeypatch.setattr(campaign, "_run", lambda args, **kwargs: calls.append(args))
    with pytest.raises(ValueError):
        campaign.run_worker(tmp_path / "missing-inventory", tmp_path / "results", "attempt", 0,
                            matrix=matrix)
    with pytest.raises(ValueError):
        campaign.publish_completed(tmp_path, entity="entity", project="project", matrix=matrix)
    assert not calls and not (tmp_path / "results").exists()
