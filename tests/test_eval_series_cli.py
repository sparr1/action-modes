"""Explicit curve selection and accounting-aware CPU publication watching."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import eval_series as cli
from utils import eval_series as core


def normalized(tmp_path):
    result = tmp_path / "result.json"
    result.write_text('{"complete":true}\n')
    return {"identity": {"backbone": "entity/train/prior", "planner": {"operator": "prior"},
                         "protocol": {"seeds": [101], "max_steps": 500}, "science": {"evaluator": "v1"}},
            "checkpoint": {"step": 100_000, "sha256": "a" * 64},
            "metrics": {"eval/return_mean": 20}, "episodes": [{"seed": 101, "return": 20}],
            "artifact_files": {"result.json": str(result)}, "source_result_path": str(result),
            "provenance": {}, "label": "Prior | actor", "record_id": "result-100k",
            "selector": "prior/none", "controller": "prior"}


def run_from_stdout(capsys):
    return json.loads(capsys.readouterr().out)


def test_create_requires_explicit_template_and_attempt():
    with pytest.raises(SystemExit):
        cli.parser().parse_args(["create", "--root", "/tmp/registry", "--owner", "owner"])


def test_create_new_and_append_selected_run_are_distinct(tmp_path, monkeypatch, capsys):
    value = normalized(tmp_path)
    monkeypatch.setattr(cli, "load_records", lambda *args, **kwargs: [value])
    args = ["create", "--root", str(tmp_path / "registry"), "--record", value["source_result_path"],
            "--selector", "prior", "--attempt-label", "repeat", "--owner", "owner"]
    cli.main(args)
    first = run_from_stdout(capsys)
    cli.main(args)
    second = run_from_stdout(capsys)
    assert first["run_id"] != second["run_id"]
    assert json.loads((Path(first["run_dir"]) / "publication.json").read_text())["records"] == {}
    cli.main(["append", first["run_dir"], value["source_result_path"], "--selector", "prior"])
    assert run_from_stdout(capsys)["status"] == "staged"
    assert json.loads((Path(second["run_dir"]) / "publication.json").read_text())["records"] == {}
    cli.main(["append", first["run_dir"], value["source_result_path"], "--selector", "prior"])
    assert run_from_stdout(capsys)["status"] == "already_staged"


def test_create_from_identity_spec_before_results(tmp_path, capsys):
    record = normalized(tmp_path)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"identity": record["identity"], "label": record["label"]}))
    cli.main(["create", "--root", str(tmp_path / "registry"), "--spec", str(spec),
              "--attempt-label", "first", "--owner", "oscar"])
    created = run_from_stdout(capsys)
    assert core.load_run(created["run_dir"])["identity"] == record["identity"]


def test_append_incompatible_protocol_rejected(tmp_path, monkeypatch):
    value = normalized(tmp_path)
    run = core.create_run(tmp_path / "registry", value, "first", "project", "entity", "owner")
    value["identity"]["protocol"]["max_steps"] = 3
    monkeypatch.setattr(cli, "load_records", lambda *args, **kwargs: [value])
    with pytest.raises(core.SeriesError, match="Incompatible append: protocol"):
        cli.main(["append", run["run_dir"], value["source_result_path"]])


def test_append_prelaunch_spec_validates_without_staging_or_allocating(tmp_path, capsys):
    value = normalized(tmp_path)
    run = core.create_run(tmp_path / "registry", value, "first", "project", "entity", "owner")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"identity": value["identity"], "label": value["label"]}))
    before = {path: path.read_bytes() for path in Path(run["run_dir"]).rglob("*") if path.is_file()}
    cli.main(["append", run["run_dir"], "--spec", str(spec)])
    assert run_from_stdout(capsys)["run_id"] == run["run_id"]
    after = {path: path.read_bytes() for path in Path(run["run_dir"]).rglob("*") if path.is_file()}
    assert before == after
    value["identity"]["planner"] = {"operator": "sac"}
    spec.write_text(json.dumps({"identity": value["identity"]}))
    with pytest.raises(core.SeriesError, match="Incompatible append: planner"):
        cli.main(["append", run["run_dir"], "--spec", str(spec)])


@pytest.mark.parametrize("extra", [[], ["result.json", "--spec", "spec.json"]])
def test_append_requires_exactly_one_spec_or_result(extra):
    with pytest.raises(SystemExit):
        cli.main(["append", "/tmp/selected-run", *extra])


def test_selector_must_select_exactly_one_controller(tmp_path, monkeypatch):
    value = normalized(tmp_path)
    monkeypatch.setattr(cli, "load_records", lambda *args, **kwargs: [value, dict(value, selector="inner/sac", controller="sac")])
    with pytest.raises(ValueError, match="exactly one"):
        cli.select_records("bundle")
    assert cli.select_records("bundle", selector="prior") == value
    with pytest.raises(ValueError, match="exactly one"):
        cli.select_records("bundle", selector="not-present")


def scheduler(monkeypatch, active="", accounting=""):
    calls = []
    def output(command, **kwargs):
        calls.append(command)
        return active if command[0] == "squeue" else accounting
    monkeypatch.setattr(cli.subprocess, "check_output", output)
    return calls


@pytest.mark.parametrize("state", ["COMPLETED", "FAILED", "CANCELLED by 1001", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"])
def test_terminal_gpu_failures_allow_completed_results_to_publish(monkeypatch, state):
    calls = scheduler(monkeypatch, accounting=f"12345_0|{state}|\n12345_1|COMPLETED|\n")
    assert cli.scheduler_done(["12345"])
    assert "--array" in calls[1]
    assert calls[1][calls[1].index("-o") + 1].startswith("JobID%")
    assert not any("JobIDRaw" in part for part in calls[1])


@pytest.mark.parametrize("accounting", ["", "12345_0|RUNNING|\n", "99999|COMPLETED|\n"])
def test_missing_or_nonterminal_accounting_is_not_completion(monkeypatch, accounting):
    scheduler(monkeypatch, accounting=accounting)
    assert not cli.scheduler_done(["12345_0"])


def test_array_expansion_remembers_tasks_until_all_accounted(monkeypatch):
    expected = set()
    scheduler(monkeypatch, active="12345_0\n12345_1\n", accounting="")
    assert not cli.scheduler_done(["12345"], expected)
    assert expected == {"12345_0", "12345_1"}
    scheduler(monkeypatch, accounting="12345_0|COMPLETED|\n")
    assert not cli.scheduler_done(["12345"], expected)
    scheduler(monkeypatch, accounting="12345_0|COMPLETED|\n12345_1|FAILED|\n")
    assert cli.scheduler_done(["12345"], expected)


def test_job_task_prefix_does_not_match_other_task(monkeypatch):
    scheduler(monkeypatch, accounting="12345_10|COMPLETED|\n")
    assert not cli.scheduler_done(["12345_1"])


def test_accounting_ignores_step_records(monkeypatch):
    scheduler(monkeypatch, accounting="12345.batch|COMPLETED|\n12345.extern|COMPLETED|\n")
    assert not cli.scheduler_done(["12345"])


def test_purged_controller_job_can_complete_via_accounting(monkeypatch):
    def output(command, **kwargs):
        if command[0] == "squeue":
            raise subprocess.CalledProcessError(1, command, output="", stderr="slurm_load_jobs error: Invalid job id specified")
        return "12345|COMPLETED|\n"
    monkeypatch.setattr(cli.subprocess, "check_output", output)
    assert cli.scheduler_done(["12345"])


def test_scheduler_connection_failure_is_not_completion(monkeypatch):
    def output(command, **kwargs):
        raise subprocess.CalledProcessError(1, command, stderr="Unable to contact slurm controller")
    monkeypatch.setattr(cli.subprocess, "check_output", output)
    with pytest.raises(subprocess.CalledProcessError):
        cli.scheduler_done(["12345"])


@pytest.mark.parametrize("jobs", [[], ["123;bad"], ["123_[0-3]"], ["not-a-job"]])
def test_job_ids_are_explicit_and_unambiguous(jobs):
    with pytest.raises(ValueError, match="Slurm job IDs"):
        cli.scheduler_done(jobs)


class FakePublisher:
    def __init__(self, run_dir, owner=None):
        self.calls = 0
        self.exited = False
        self.exit_error = None

    def __enter__(self):
        return self

    def publish_pending(self):
        self.calls += 1
        return {"accepted": self.calls, "queued": self.calls, "published": 0}

    def __exit__(self, error_type, error, traceback):
        self.exited = True
        self.exit_error = error_type


def test_watch_uses_one_session_and_drains_after_terminal_tasks(tmp_path, monkeypatch):
    instance = FakePublisher(tmp_path)
    states = iter([False, True])
    monkeypatch.setattr(cli, "scheduler_done", lambda jobs, expected: next(states))
    sleeps = []
    monkeypatch.setattr(cli.time, "sleep", sleeps.append)
    result = cli.publish_watch(tmp_path, watch=True, jobs=["12345"], publisher_factory=lambda *args, **kwargs: instance)
    assert instance.calls == 3
    assert instance.exited and instance.exit_error is None
    assert sleeps == [15]
    assert result["accepted"] == 3


def test_timeout_flushes_publication_and_allows_retry(tmp_path, monkeypatch):
    instance = FakePublisher(tmp_path)
    clock = {"now": 0}
    monkeypatch.setattr(cli.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: clock.__setitem__("now", clock["now"] + seconds))
    with pytest.raises(TimeoutError, match="saved results remain available"):
        cli.publish_watch(tmp_path, watch=True, max_wait_seconds=20, publisher_factory=lambda *args, **kwargs: instance)
    assert clock["now"] == 20
    assert instance.exited and instance.exit_error is TimeoutError
    retry = FakePublisher(tmp_path)
    assert cli.publish_watch(tmp_path, publisher_factory=lambda *args, **kwargs: retry)["accepted"] == 1
    assert retry.exited


def test_watch_requires_a_bounded_completion_condition(tmp_path):
    with pytest.raises(ValueError, match="requires --jobs"):
        cli.publish_watch(tmp_path, watch=True)
    with pytest.raises(ValueError, match="positive"):
        cli.publish_watch(tmp_path, watch=True, max_wait_seconds=0)


def test_cpu_launcher_never_requests_gpu_and_passes_fixed_interval():
    script = Path("slurm/run_eval_series_publisher_oscar.sbatch").read_text()
    directives = [line for line in script.splitlines() if line.startswith("#SBATCH")]
    assert not any("gres" in line or "--gpus" in line for line in directives)
    assert "#SBATCH --signal=B:TERM@60" in script
    assert '--watch --poll-seconds 15 --jobs "${compute_jobs[@]}"' in script


@pytest.mark.parametrize("override_storage", [False, True])
def test_cpu_launcher_routes_artifact_storage_before_python(tmp_path, override_storage):
    launcher = Path("slurm/run_eval_series_publisher_oscar.sbatch").resolve()
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    run_dir = tmp_path / "authoritative results" / "curve"
    receipt = tmp_path / "launched.json"
    python = tmp_path / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "paths = {key: os.environ[key] for key in ['WANDB_CACHE_DIR', 'WANDB_DATA_DIR']}\n"
        "assert all(pathlib.Path(path).is_dir() for path in paths.values())\n"
        f"pathlib.Path({str(receipt)!r}).write_text(json.dumps(dict(paths=paths, args=sys.argv[1:])))\n"
    )
    python.chmod(0o755)
    environment = dict(os.environ)
    for key in ("WANDB_CACHE_DIR", "WANDB_DATA_DIR", "EXPECTED_ACTION_MODES_SHA"):
        environment.pop(key, None)
    environment.update(SLURM_SUBMIT_DIR=str(checkout), EVAL_RUN_DIR=str(run_dir),
                       EVAL_COMPUTE_JOBS="12345 12346", EVAL_PUBLICATION_OWNER="owner",
                       PYTHON_BIN=str(python))
    if override_storage:
        environment.update(WANDB_CACHE_DIR=str(tmp_path / "shared cache"),
                           WANDB_DATA_DIR=str(tmp_path / "scratch upload staging"))
    subprocess.run(["bash", str(launcher)], env=environment, check=True, capture_output=True, text=True)
    launched = json.loads(receipt.read_text())
    assert launched["paths"] == {
        "WANDB_CACHE_DIR": environment.get("WANDB_CACHE_DIR", str(run_dir / "wandb-cache")),
        "WANDB_DATA_DIR": environment.get("WANDB_DATA_DIR", str(run_dir / "wandb-data")),
    }
    assert launched["args"] == ["eval_series.py", "publish", str(run_dir), "--owner", "owner",
                                "--watch", "--poll-seconds", "15", "--jobs", "12345", "12346"]
