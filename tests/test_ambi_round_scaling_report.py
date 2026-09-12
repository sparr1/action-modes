"""The comparison publisher never turns partial seed cells into complete curves."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "slurm/ambi_round_scaling_report.py"
spec = importlib.util.spec_from_file_location("round_report", MODULE_PATH)
reporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporter)


def seal_bundle(path, manifest):
    path.mkdir(parents=True, exist_ok=True)
    receipt = {"schema_version": 1, "kind": "ambi_episode_seed_merge",
               "expected_seeds": manifest["runs"][0]["result"]["environment_seeds"], "sources": []}
    manifest["seed_shard_merge"] = receipt
    reporter.write(path / "manifest.json", manifest)
    reporter.write(path / "seed-shard-merge.json", receipt)
    seal = {"schema_version": 1, "kind": "ambi_episode_seed_shard", "files": {
        name: hashlib.sha256((path / name).read_bytes()).hexdigest()
        for name in ("manifest.json", "seed-shard-merge.json")}}
    seal["sha256"] = reporter.digest(seal)
    reporter.write(path / "seed-shard-checksums.json", seal)


def panel(tmp_path):
    campaign = {"attempt_label": "rounds-explicit-test", "cells": [], "worker_receipts": []}
    seeds = list(range(101, 121))
    for rounds in (1, 2, 4):
        for controller in (55, 56, 57):
            cell_id = f"j{rounds}-c{controller}"
            directory = tmp_path / cell_id
            cell = {"cell_id": cell_id, "rounds": rounds, "controller_seed": controller,
                    "seeds": seeds, "bundle": str(directory), "selector": "initialization/inherited",
                    "reused": rounds == 4 and controller == 55}
            params = {"inner_rounds": rounds, "inner_rollout_horizon": 1,
                      "inner_rollouts_per_round": 128, "inner_actor_updates_per_round": 4,
                      "inner_critic_updates_per_round": 32, "inner_batch_size": 256,
                      "inner_temperature": 0.0, "inner_actor_initialization": "prior",
                      "inner_critic_initialization": "prior"}
            runs = []
            for variant in ("prior", "inherited"):
                episodes = []
                for seed in seeds:
                    length = 100 if seed == 101 else 500
                    gain = 0 if variant == "prior" else rounds * 3 + (controller - 56)
                    episodes.append({"seed": seed, "solver_seed": seed + controller,
                                     "return": seed + gain, "length": length, "terminated": False,
                                     "truncated": True, "control_seconds": rounds * length * .1,
                                     "togo_probe_seconds": .05 * length, "togo_probe_model_steps": 64 * length,
                                     "model_metrics": {key: rounds * unit for key, unit in
                                         (("inner_actor_optimizer_steps", 4), ("inner_critic_optimizer_steps", 32),
                                          ("inner_optimization_model_steps", 128))}})
                config = {"alg_params": params}
                result = {"environment_seeds": seeds, "controller_seed": controller,
                          "outer_state_unchanged": True, "outer_updates_before": 20, "outer_updates_after": 20,
                          "alg_params": params, "episodes": episodes}
                runs.append({"selector": "initialization/" + variant, "status": "complete", "kind": "episodes",
                             "config": config, "config_hash": reporter.digest(config),
                             "episodes": copy.deepcopy(episodes), "result": result})
            manifest = {"status": "complete", "runs": runs,
                        "checkpoint": {"sha256": "same", "source_run": "mey3rxj8"},
                        "protocol": {"action_rule": "tanh_mean", "controller_seed": controller, "max_steps": 500},
                        "code": {"runtime": {"torch": "2.3.1"}, "commit": "old" if cell["reused"] else "new"}}
            seal_bundle(directory, manifest)
            campaign["cells"].append(cell)
    return campaign


def load_panel(campaign):
    return {c["cell_id"]: reporter.load_cell(c) for c in campaign["cells"]}


def test_weighting_pairs_actual_compute_and_reused_timing(tmp_path):
    campaign = panel(tmp_path)
    result = reporter.build_report(campaign, load_panel(campaign))
    assert len(result["paired_rows"]) == 180
    assert len(result["episode_averages"]) == 60
    for row in result["summaries"]:
        j = row["rounds"]
        assert row["metrics"]["return"]["mean"] == pytest.approx(110.5 + 3 * j)
        assert row["metrics"]["gain_vs_prior"]["mean"] == pytest.approx(3 * j)
        assert row["metrics"]["gain_vs_j1"]["mean"] == pytest.approx(3 * (j - 1))
        assert row["metrics"]["gain_vs_j1"]["ci95_low"] == pytest.approx(3 * (j - 1))
        assert row["metrics"]["actor_updates_per_decision"]["mean"] == 4 * j
        assert row["metrics"]["actor_updates_per_episode"]["mean"] == 480 * 4 * j
        assert row["metrics"]["control_seconds_per_decision"]["mean"] == pytest.approx(j * .1)
    timing = result["summaries"][-1]["timing_by_origin"]
    assert timing.keys() == {"new", "historical_reused"}
    assert timing["new"]["episode_count"] == 20
    assert timing["historical_reused"]["episode_count"] == 20
    assert [r["reused"] for r in result["paired_rows"]].count(True) == 20
    assert result["interval"]["resamples"] == 2000


def test_missing_cell_never_creates_science(tmp_path):
    campaign = panel(tmp_path)
    loaded = load_panel(campaign)
    loaded.pop("j2-c56")
    with pytest.raises(ValueError, match="incomplete"):
        reporter.build_report(campaign, loaded)


@pytest.mark.parametrize("mutate,message", [
    (lambda m: m["runs"][1]["result"].update(outer_state_unchanged=False), "Outer state"),
    (lambda m: m["runs"][1]["result"].update(controller_seed=54), "controller seed"),
    (lambda m: m["runs"][1]["result"]["episodes"].pop(), "result episodes"),
    (lambda m: m["runs"][1]["result"]["episodes"][0]["model_metrics"].update(inner_actor_optimizer_steps=0), "realized"),
    (lambda m: m["runs"][1]["result"]["alg_params"].update(inner_rollout_horizon=2), "schedule"),
])
def test_rejects_bad_complete_cells(tmp_path, mutate, message):
    campaign = panel(tmp_path)
    cell = campaign["cells"][0]
    directory = Path(cell["bundle"])
    manifest = reporter.read(directory / "manifest.json")
    mutate(manifest)
    seal_bundle(directory, manifest)
    with pytest.raises(ValueError, match=message):
        reporter.load_cell(cell)


def test_rejects_changed_manifest_hash(tmp_path):
    campaign = panel(tmp_path)
    cell = campaign["cells"][0]
    path = Path(cell["bundle"]) / "manifest.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="checksum mismatch"):
        reporter.load_cell(cell)


def test_cross_cell_science_and_prior_mismatch(tmp_path):
    campaign = panel(tmp_path)
    loaded = load_panel(campaign)
    loaded["j1-c55"]["compatibility"]["runtime"]["torch"] = "other"
    with pytest.raises(ValueError, match="configuration mismatch"):
        reporter.build_report(campaign, loaded)
    loaded = load_panel(campaign)
    loaded["j1-c55"]["prior"][0]["return"] += 1
    with pytest.raises(ValueError, match="prior returns differ"):
        reporter.build_report(campaign, loaded)


def test_report_html_embeds_raw_data_and_safely_handles_title(tmp_path):
    campaign = panel(tmp_path)
    campaign["attempt_label"] = "</script><script>alert(1)</script>"
    result = reporter.build_report(campaign, load_panel(campaign), resamples=20)
    output = reporter.render_html(result)
    assert campaign["attempt_label"] not in output
    embedded = output.split('<script id="report-data" type="application/json">')[1].split("</script>")[0]
    assert json.loads(embedded) == result
    assert "measured control seconds" in output


def test_scheduler_waits_pending_then_terminal_failure():
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="123_1|PENDING\n")
    assert reporter.job_state(["123"], run)["finished"] is False
    assert len(calls) == 1
    def done(command, **kwargs):
        return SimpleNamespace(stdout="" if command[0] == "squeue" else "123_0|COMPLETED|0:0\n123_1|FAILED|1:0\n")
    assert reporter.job_state(["123"], done)["finished"] is True
    assert reporter.job_state(["123", "456"], done)["finished"] is False


def test_cli_incomplete_preserved_without_importing_wandb(tmp_path):
    campaign = panel(tmp_path)
    (Path(campaign["cells"][0]["bundle"]) / "seed-shard-checksums.json").unlink()
    campaign_path = tmp_path / "campaign.json"
    reporter.write(campaign_path, campaign)
    output = tmp_path / "report"
    assert reporter.main(["--campaign", str(campaign_path), "--output", str(output)]) == 2
    result = reporter.read(output / "report.json")
    assert result["status"] == "incomplete" and result["complete_cells"] == 8
    assert "summaries" not in result
    assert "j1-c55" in result["missing"][0]["cell_id"]


class FakeRun:
    def __init__(self):
        self.logs, self.summary, self.finished = [], {}, []
    def log(self, value): self.logs.append(value)
    def define_metric(self, *args, **kwargs): pass
    def finish(self, **kwargs): self.finished.append(kwargs)
    def log_artifact(self, artifact): self.artifact = artifact


class FakeArtifact:
    def __init__(self, *args, **kwargs): self.files = []
    def add_file(self, path, **kwargs): self.files.append((path, kwargs))


def test_online_progress_appears_before_bundle_inspection_and_failure_finishes(tmp_path, monkeypatch):
    campaign = panel(tmp_path)
    campaign_path = tmp_path / "campaign.json"
    reporter.write(campaign_path, campaign)
    run = FakeRun()
    fake = SimpleNamespace(init=lambda **kwargs: run, Html=lambda *a, **k: "html", Artifact=FakeArtifact)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    original = reporter.inspect
    def inspect(*args):
        assert run.logs and run.logs[0]["progress/complete_cells"] == 0
        state = original(*args)
        state["missing"] = [{"cell_id": "missing", "reason": "failed"}]
        return state
    monkeypatch.setattr(reporter, "inspect", inspect)
    monkeypatch.setattr(reporter, "job_state", lambda jobs: {"finished": True, "accounting": []})
    assert reporter.main(["--campaign", str(campaign_path), "--mode", "online", "--wandb-run-id", "explicit",
                          "--watch", "--compute-jobs", "123"]) == 2
    assert run.summary["comparison/status"] == "incomplete"
    assert run.finished == [{"exit_code": 1}]
    assert not any(any(k.startswith("comparison_vs") for k in row) for row in run.logs)


def test_publisher_complete_panel_axes_artifact_and_reentry(tmp_path):
    campaign = panel(tmp_path)
    report = reporter.build_report(campaign, load_panel(campaign), resamples=20)
    run = FakeRun()
    wandb = SimpleNamespace(Html=lambda *a, **k: "html", Artifact=FakeArtifact)
    reporter.publish_science(run, wandb, report, tmp_path)
    assert len(run.logs) == 4
    assert len(run.artifact.files) == 3
    assert [r["compute/actor_updates_per_decision"] for r in run.logs[:3]] == [4, 8, 16]
    for row in run.logs[:3]:
        assert "comparison_vs_control_seconds_per_decision/return/ci95_high" in row
    reporter.publish_science(run, wandb, report, tmp_path)
    assert len(run.logs) == 4


def test_algorithm_learning_rate_difference_is_rejected(tmp_path):
    campaign = panel(tmp_path)
    loaded = load_panel(campaign)
    for cell in loaded.values():
        cell["compatibility"]["alg_params_except_rounds"]["inner_actor_lr"] = .0003
    loaded["j1-c55"]["compatibility"]["alg_params_except_rounds"]["inner_actor_lr"] = .0001
    with pytest.raises(ValueError, match="configuration mismatch"):
        reporter.build_report(campaign, loaded)


def test_real_round_matrices_compare_unresolved_params_without_derived_counts():
    # This is the same pre-model resolution used by result.alg_params. Derived
    # budgets belong to result.resolved_config and must not be compared blindly.
    from utils.ambi_research import resolve_preset
    root = MODULE_PATH.parents[1]
    context = SimpleNamespace(source="metadata", trial_run_params={
        "alg": "AMBITDMPC2/AMBITDMPC2", "env": "DMControl-v0", "seed": 55,
        "alg_params": {"inner_rounds": 0, "actor_lr": .0003}},
        experiment_params={"env_params": {"obs": "state", "render_mode": None, "task": "humanoid-walk"}})
    resolved = [resolve_preset(root / f"configs/research/round_scaling_j{j}_h1_200k.json",
                               "initialization/inherited", checkpoint_context=context)["algorithm_config"]
                for j in (1, 2, 4)]
    for j, config in zip((1, 2, 4), resolved):
        assert config["alg_params"]["inner_rounds"] == j
        assert "inner_actor_updates_per_action" not in config["alg_params"]
        assert "inner_expected_update_slots" not in config["alg_params"]
    normalized = [{k: v for k, v in r["alg_params"].items() if k != "inner_rounds"} for r in resolved]
    assert normalized[0] == normalized[1] == normalized[2]


def test_watch_stops_after_scheduler_errors_with_explicit_record(tmp_path, monkeypatch):
    campaign = panel(tmp_path)
    (Path(campaign["cells"][0]["bundle"]) / "seed-shard-checksums.json").unlink()
    campaign_path = tmp_path / "campaign.json"
    reporter.write(campaign_path, campaign)
    failures = []
    def fail(jobs):
        failures.append(jobs)
        raise OSError("scheduler unavailable")
    monkeypatch.setattr(reporter, "job_state", fail)
    monkeypatch.setattr(reporter.time, "sleep", lambda seconds: None)
    assert reporter.main(["--campaign", str(campaign_path), "--watch", "--compute-jobs", "123"]) == 2
    progress = reporter.read(tmp_path / "comparison" / "progress.json")
    assert len(failures) == 3
    assert progress["stop_reason"] == "Scheduler inspection failed three times"
    assert progress["scheduler"]["error"] == "scheduler unavailable"


def test_fatal_comparison_failure_is_durable(tmp_path, monkeypatch):
    campaign = panel(tmp_path)
    campaign_path = tmp_path / "campaign.json"
    reporter.write(campaign_path, campaign)
    def fail(*args): raise ValueError("science mismatch")
    monkeypatch.setattr(reporter, "build_report", fail)
    with pytest.raises(ValueError, match="science mismatch"):
        reporter.main(["--campaign", str(campaign_path)])
    saved = reporter.read(tmp_path / "comparison" / "failure.json")
    assert saved["status"] == "failed" and saved["error"] == "science mismatch"
