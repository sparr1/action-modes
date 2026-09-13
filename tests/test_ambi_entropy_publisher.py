"""Publisher/worker-seal regressions; no W&B connection or scheduler calls."""
import copy
import gzip
import json
from pathlib import Path
import subprocess

import pytest

import evaluate_ambi_entropy as evaluator
from utils import ambi_entropy_reporting as reporting
from utils.ambi_benchmark import atomic_json, canonical_hash, read_json


def study_fixture():
    study = read_json(Path(__file__).resolve().parents[1] /
                      "configs/research/ambi_entropy_prior_h1.json")
    study.update(episode_seeds=[101, 102], decisions=[0], actor_updates=[0, 1],
                 solver_repetitions=1, rollout_repetitions=1)
    study["checkpoints"] = study["checkpoints"][:1]
    return study


def campaign_fixture(tmp_path):
    study = study_fixture()
    study_path = tmp_path / "study.json"
    atomic_json(study_path, study)
    tasks = []
    for seed in study["episode_seeds"]:
        directory = tmp_path / f"worker-{seed}"
        directory.mkdir()
        tasks.append({"checkpoint_index": 0, "episode_seed": seed, "output_dir": str(directory)})
    spec = {"study": str(study_path), "study_sha256": canonical_hash(study),
            "tasks": tasks, "compute_job_id": "12345", "wandb_run_id": "reserved-campaign-id",
            "attempt": "explicit-attempt", "commit": "a"*40,
            "entity": "test-entity", "project": "ambi-inner-bench", "name": "entropy-test"}
    campaign = tmp_path / "campaign.json"
    atomic_json(campaign, spec)
    return study, spec, campaign


class OfflineRun:
    id = "reserved-campaign-id"

    def __init__(self):
        self.summary, self.finished = {}, []

    def finish(self, **kwargs):
        self.finished.append(kwargs)


def test_publisher_updates_progress_twice_and_reports_incomplete_array(tmp_path, monkeypatch):
    _, spec, campaign = campaign_fixture(tmp_path)
    atomic_json(Path(spec["tasks"][0]["output_dir"]) / "status.json", {"status": "complete"})
    atomic_json(Path(spec["tasks"][1]["output_dir"]) / "status.json", {"status": "running"})
    run, starts, polls, sleeps = OfflineRun(), [], [], []
    def start(**kwargs):
        starts.append(kwargs)
        return run
    responses = iter(["12345_[1] RUNNING\n", ""])
    def queue(command, **kwargs):
        polls.append(command)
        return next(responses)
    monkeypatch.setattr(reporting, "start_entropy_wandb", start)
    monkeypatch.setattr(subprocess, "check_output", queue)
    monkeypatch.setattr(evaluator.time, "sleep", lambda seconds: sleeps.append(seconds))
    with pytest.raises(RuntimeError, match="incomplete coverage: 1/2"):
        evaluator.publish_campaign(campaign)
    assert len(polls) == 2 and sleeps == [15]
    assert starts[0]["run_id"] == spec["wandb_run_id"] and starts[0]["resume"] == "never"
    assert read_json(tmp_path / "analysis/progress.json")["completed_tasks"] == 1
    assert "incomplete coverage" in read_json(tmp_path / "analysis/publication-failed.json")["error"]
    assert run.summary["progress/complete"] == 0
    assert run.finished == [{"exit_code": 1}]
    assert not (tmp_path / "analysis/publication-complete.json").exists()


def test_resume_reuses_reserved_run_and_can_replace_old_failure_progress(tmp_path, monkeypatch):
    _, spec, campaign = campaign_fixture(tmp_path)
    receipt = {"run_id": spec["wandb_run_id"], "campaign": str(campaign)}
    atomic_json(tmp_path / "publication-started.json", receipt)
    atomic_json(tmp_path / "analysis/progress.json", {"completed_tasks": -1})
    atomic_json(tmp_path / "analysis/publication-failed.json", {"error": "previous interruption"})
    run, starts = OfflineRun(), []
    def start(**kwargs):
        starts.append(kwargs)
        return run
    monkeypatch.setattr(reporting, "start_entropy_wandb", start)
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "")
    with pytest.raises(RuntimeError, match="incomplete coverage: 0/2"):
        evaluator.publish_campaign(campaign)
    assert starts[0]["resume"] == "must"
    assert starts[0]["run_id"] == spec["wandb_run_id"]
    assert read_json(tmp_path / "publication-started.json") == receipt
    assert read_json(tmp_path / "analysis/progress.json")["completed_tasks"] == 0
    assert "incomplete coverage" in read_json(tmp_path / "analysis/publication-failed.json")["error"]


def sealed_worker(tmp_path, *, record_change=None, row_change=None):
    study = study_fixture()
    source = study["checkpoints"][0]
    directory = tmp_path / "sealed"
    directory.mkdir()
    arms = ["off", "prior_recipe", "squashed_matched"]
    rows = evaluator.expected_rows(study, source, [101], arms)
    for row in rows:
        row.update(mc_complete=True, truncated=False,
                   metrics={"real_mc_return": 10.0, "real_mc_gain_vs_prior": 0.0})
    record = {"schema_version": 1, "complete": True, "smoke": False,
              "outer_state_unchanged": True, "study_sha256": canonical_hash(study),
              "source": source, "episode_seed": 101, "arm_names": arms, "rows": len(rows)}
    if record_change:
        record_change(record)
    if row_change:
        row_change(rows)
    for name, data in (("results.json", record), ("study.json", study), ("matrix.json", {}),
                       ("checkpoint.metadata.json", {}), ("arms.json", {"arms": arms}),
                       ("root-bank.json", {"complete": True})):
        atomic_json(directory / name, data)
    with gzip.open(directory / "measurements.jsonl.gz", "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    atomic_json(directory / "checksums.json", {
        p.name: evaluator._file_sha256(p) for p in directory.iterdir() if p.is_file()})
    return directory, study, source, rows


def test_validate_worker_preserves_sealed_source_and_raw_pairing(tmp_path):
    directory, study, source, rows = sealed_worker(tmp_path)
    record, loaded = evaluator.validate_worker(directory, study=study, source=source, episode_seed=101)
    assert loaded == rows and record["rows"] == len(rows)
    report = reporting.aggregate_entropy_rows(
        loaded, expected_rows=evaluator.expected_rows(study, source, [101], record["arm_names"]))
    assert report["coverage"]["complete"]
    assert len(report["series"]) == 12


def test_worker_seal_rejects_modified_measurements_and_missing_required_file(tmp_path):
    directory, study, source, _ = sealed_worker(tmp_path)
    measurements = directory / "measurements.jsonl.gz"
    measurements.write_bytes(measurements.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        evaluator.validate_worker(directory, study=study, source=source, episode_seed=101)
    seal = read_json(directory / "checksums.json")
    seal.pop("arms.json")
    atomic_json(directory / "checksums.json", seal, overwrite=True)
    with pytest.raises(ValueError, match="missing required files"):
        evaluator.validate_worker(directory, study=study, source=source, episode_seed=101)


@pytest.mark.parametrize("change", [
    lambda r: r.update(smoke=True),
    lambda r: r.update(outer_state_unchanged=False),
    lambda r: r.update(episode_seed=102),
    lambda r: r.update(source={**r["source"], "sha256": "b"*64}),
    lambda r: r.update(study_sha256="wrong"),
    lambda r: r.update(arm_names=["off", "prior_recipe"]),
])
def test_worker_seal_does_not_substitute_for_protocol_validation(tmp_path, change):
    directory, study, source, _ = sealed_worker(tmp_path, record_change=change)
    with pytest.raises(ValueError, match="Worker source, protocol, coverage or frozen-state"):
        evaluator.validate_worker(directory, study=study, source=source, episode_seed=101)


@pytest.mark.parametrize("change", [
    lambda rows: rows.pop(),
    lambda rows: rows[0].update(mc_complete=False),
    lambda rows: rows[0].update(truncated=True),
])
def test_worker_rejects_incomplete_or_truncated_calibration(tmp_path, change):
    directory, study, source, _ = sealed_worker(tmp_path, row_change=change)
    with pytest.raises(ValueError, match="Incomplete worker measurement rows"):
        evaluator.validate_worker(directory, study=study, source=source, episode_seed=101)
