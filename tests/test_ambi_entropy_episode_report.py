"""Strict reuse, hierarchical pairing, and single-run complete-panel publication."""
import base64
import copy
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from slurm import ambi_entropy_episode_report as reporter


def seal_bundle(directory, manifest):
    directory.mkdir(parents=True, exist_ok=True)
    receipt = {"kind": "ambi_episode_seed_merge", "expected_seeds": reporter.SEEDS, "sources": []}
    manifest["seed_shard_merge"] = receipt
    reporter.write(directory / "manifest.json", manifest)
    reporter.write(directory / "seed-shard-merge.json", receipt)
    seal = {"kind": "ambi_episode_seed_shard", "files": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("manifest.json", "seed-shard-merge.json")}}
    seal["sha256"] = reporter.digest(seal)
    reporter.write(directory / "seed-shard-checksums.json", seal)
    return {"bundle_manifest_sha256": seal["files"]["manifest.json"],
            "bundle_seal_sha256": hashlib.sha256((directory / "seed-shard-checksums.json").read_bytes()).hexdigest()}


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    monkeypatch.setattr(reporter, "source_identity", lambda *args: {"source_sha256": "same"})
    campaign = {"attempt_label": "entropy-explicit-test", "output_root": str(tmp_path), "arms": copy.deepcopy(reporter.ARMS),
                "seeds": reporter.SEEDS, "controller_seeds": reporter.CONTROLLERS, "cells": [], "worker_receipts": []}
    for arm, entropy in reporter.ARMS.items():
        for rounds in reporter.ROUNDS:
            for controller in reporter.CONTROLLERS:
                cell_id = f"{arm}-j{rounds}-c{controller}"
                directory = tmp_path / cell_id
                cell = {"cell_id": cell_id, "arm": arm, "rounds": rounds, "controller_seed": controller,
                        "seeds": reporter.SEEDS, "bundle": str(directory), "selector": "initialization/inherited", "reused": arm == "off"}
                params = {"inner_rounds": rounds, "inner_rollout_horizon": 1, "inner_rollouts_per_round": 128,
                          "inner_actor_updates_per_round": 4, "inner_critic_updates_per_round": 32, "inner_batch_size": 256,
                          "inner_temperature": entropy["alpha"], "inner_actor_entropy_mode": entropy["mode"],
                          "inner_actor_initialization": "prior", "inner_critic_initialization": "prior",
                          "inner_temperature_mode": "fixed", "inner_temperature_initialization": "fixed",
                          "inner_execution_action": "mean", "inner_behavior_action": "policy_sample",
                          "inner_finite_horizon": True, "inner_sac_critic_target": "reward_only", "inner_actor_lr": .0003}
                runs = []
                for variant in ("prior", "inherited"):
                    rows = []
                    for seed in reporter.SEEDS:
                        length = 100 if seed == 101 else 500
                        gain = 0 if variant == "prior" else rounds * (3 + list(reporter.ARMS).index(arm)) + controller - 56
                        model = {key: rounds * unit for key, unit in (("inner_actor_optimizer_steps", 4),
                                 ("inner_critic_optimizer_steps", 32), ("inner_optimization_model_steps", 128))}
                        if arm != "squashed":
                            model["inner_actor_scaled_entropy"] = 10
                        probes = [{"round_index": j, "actor_updates": 4 * j, "critic_updates": 32 * j,
                                   "metrics": {"togo_return_gain_vs_outer": {"mean": j, "count": length, "sum": j * length}}}
                                  for j in range(rounds + 1)]
                        rows.append({"seed": seed, "solver_seed": seed + controller, "return": seed + gain,
                                     "length": length, "terminated": seed == 101, "truncated": seed != 101,
                                     "control_seconds": rounds * length * .1, "togo_probe_seconds": .05 * length,
                                     "togo_probe_model_steps": 64 * length, "model_metrics": model, "togo_round_summaries": probes})
                    config = {"alg": "AMBITDMPC2/AMBITDMPC2", "alg_params": copy.deepcopy(params)}
                    result = {"environment_seeds": reporter.SEEDS, "controller_seed": controller,
                              "outer_state_unchanged": True, "outer_updates_before": 20, "outer_updates_after": 20,
                              "alg_params": copy.deepcopy(params), "episodes": rows,
                              "togo_return_probe": {"rollouts": 32, "horizon": 1, "tail_actor": "outer",
                                "tail_critic": "outer_online", "tail_q_reduction": "mean_pair", "entropy_bonus": False}}
                    runs.append({"selector": "initialization/" + variant, "status": "complete", "kind": "episodes",
                                 "config": config, "config_hash": reporter.digest(config), "episodes": copy.deepcopy(rows), "result": result})
                manifest = {"status": "complete", "runs": runs,
                            "checkpoint": {"sha256": reporter.CHECKPOINT, "source_run": "rwgao_b-brown-university/ambi/mey3rxj8",
                                           "source_run_verified": False, "metadata": {"checkpoint": {"step": 200000}}},
                            "protocol": {"action_rule": "tanh_mean", "controller_seed": controller, "max_steps": 500},
                            "code": {"runtime": {"torch": "2.3.1"}, "commit": "historical" if arm == "off" else "new", "dirty": False}}
                hashes = seal_bundle(directory, manifest)
                if cell["reused"]:
                    cell.update(hashes)
                campaign["cells"].append(cell)
    return campaign


def load(campaign):
    return {c["cell_id"]: reporter.load_cell(c) for c in campaign["cells"]}


def test_complete_hierarchical_pairs_and_conditional_metrics(campaign):
    result = reporter.build_report(campaign, load(campaign), resamples=20)
    assert len(result["paired_rows"]) == 540 and len(result["episode_averages"]) == 180
    assert len(result["contrasts"]) == 18
    for row in result["summaries"]:
        j, arm = row["rounds"], row["arm"]
        multiplier = 3 + list(reporter.ARMS).index(arm)
        assert row["metrics"]["return"]["mean"] == pytest.approx(110.5 + j * multiplier)
        assert row["metrics"]["gain_vs_prior"]["mean"] == pytest.approx(j * multiplier)
        assert row["metrics"]["actor_updates_per_episode"]["mean"] == 480 * 4 * j
        assert row["metrics"]["control_seconds_per_decision"]["mean"] == pytest.approx(j * .1)
        assert ("inner_actor_scaled_entropy" in row["model_metrics"]) == (arm != "squashed")
    for contrast in result["contrasts"]:
        assert contrast["summary"]["episode_count"] == 20
        assert contrast["summary"]["ci95_low"] == contrast["summary"]["ci95_high"]
    assert sum(r["reused"] for r in result["paired_rows"]) == 180
    assert result["round_dynamics"][-1]["actor_updates"] == 16


def test_historical_pin_is_file_hash_and_required(campaign):
    cell = campaign["cells"][0]
    assert cell["bundle_seal_sha256"] != reporter.read(Path(cell["bundle"]) / "seed-shard-checksums.json")["sha256"]
    reporter.load_cell(cell)
    cell["bundle_seal_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="Historical bundle identity"):
        reporter.load_cell(cell)


@pytest.mark.parametrize("key,value,message", [("inner_temperature", .1, "schedule"),
 ("inner_actor_entropy_mode", "wrong", "schedule"), ("inner_rollouts_per_round", 256, "schedule")])
def test_bad_native_scientific_params(campaign, key, value, message):
    cell = campaign["cells"][9]
    directory = Path(cell["bundle"])
    manifest = reporter.read(directory / "manifest.json")
    manifest["runs"][1]["result"]["alg_params"][key] = value
    seal_bundle(directory, manifest)
    with pytest.raises(ValueError, match=message):
        reporter.load_cell(cell)


def test_incomplete_wrong_pair_and_learning_rate_rejected(campaign):
    values = load(campaign)
    missing = dict(values); missing.pop("native-j1-c55")
    with pytest.raises(ValueError, match="incomplete"):
        reporter.build_report(campaign, missing)
    values["native-j1-c55"]["rows"][0]["solver_seed"] += 1
    with pytest.raises(ValueError, match="RNG"):
        reporter.build_report(campaign, values)
    values = load(campaign)
    values["native-j1-c55"]["compatibility"]["alg_params_except_entropy_and_rounds"]["inner_actor_lr"] = .001
    with pytest.raises(ValueError, match="configuration mismatch"):
        reporter.build_report(campaign, values)


def test_wrong_source_and_prior_rejected(campaign):
    values = load(campaign)
    values["native-j1-c55"]["compatibility"]["scientific_source"] = {"source_sha256": "different"}
    with pytest.raises(ValueError, match="configuration mismatch"):
        reporter.build_report(campaign, values)
    values = load(campaign)
    values["native-j1-c55"]["prior"][0]["return"] += .01
    with pytest.raises(ValueError, match="prior results differ"):
        reporter.build_report(campaign, values)


def test_probe_counts_and_realized_work(campaign):
    cell = campaign["cells"][9]; directory = Path(cell["bundle"])
    manifest = reporter.read(directory / "manifest.json")
    manifest["runs"][1]["result"]["episodes"][0]["togo_round_summaries"][1]["actor_updates"] = 3
    seal_bundle(directory, manifest)
    with pytest.raises(ValueError, match="probe update counts"):
        reporter.load_cell(cell)


def test_html_roundtrip_is_portable_escaped_and_compact(campaign):
    campaign["attempt_label"] = "</script><script>bad</script>"
    result = reporter.build_report(campaign, load(campaign), resamples=20)
    html = reporter.render_html(result)
    assert campaign["attempt_label"] not in html
    raw = html.split('<script id="raw-gzip" type="application/octet-stream">')[1].split('</script>')[0]
    assert json.loads(gzip.decompress(base64.b64decode(raw))) == result
    assert "paired_rows" not in html.split('<script id="report-data" type="application/json">')[1].split('</script>')[0]


class FakeRun:
    def __init__(self):
        self.logs, self.summary, self.finished, self.axes = [], {}, [], []
    def log(self, value): self.logs.append(value)
    def define_metric(self, *args, **kwargs): self.axes.append((args, kwargs))
    def finish(self, **kwargs): self.finished.append(kwargs)
    def log_artifact(self, artifact): self.artifact = artifact


class FakeArtifact:
    def __init__(self, *args, **kwargs): self.files = []
    def add_file(self, path, **kwargs): self.files.append((path, kwargs))
    def add_dir(self, path, **kwargs): self.files.append((path, kwargs))


def test_publisher_distinct_arm_axes_and_idempotency(campaign, tmp_path):
    result = reporter.build_report(campaign, load(campaign), resamples=20)
    run = FakeRun(); fake = SimpleNamespace(Html=lambda *a, **k: "html", Artifact=FakeArtifact)
    reporter.publish_science(run, fake, result, tmp_path)
    count = len(run.logs)
    assert "episodes/squashed/vs_actor_updates_per_decision/gain_vs_native/mean" in run.logs[6]
    assert any("model_probes/squashed/j4/togo_return_gain_vs_outer/mean" in row for row in run.logs)
    reporter.publish_science(run, fake, result, tmp_path)
    assert len(run.logs) == count and len(run.artifact.files) == 3


def test_online_progress_before_inspection_failure_and_strict_resume(campaign, tmp_path, monkeypatch):
    campaign_path = tmp_path / "campaign.json"; reporter.write(campaign_path, campaign)
    runs, inits = [], []
    def init(**kwargs):
        inits.append(kwargs); run = FakeRun(); runs.append(run); return run
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    def inspect(*args):
        assert runs[-1].logs[0]["progress/complete_cells"] == 0
        return {"complete_cells": 9, "expected_cells": 27, "complete_workers": 0, "expected_workers": 42,
                "missing": [{"cell_id": "native-j1-c55", "reason": "absent"}]}
    monkeypatch.setattr(reporter, "inspect", inspect)
    monkeypatch.setattr(reporter, "job_state", lambda *a: {"finished": True})
    args = ["--campaign", str(campaign_path), "--mode", "online", "--wandb-run-id", "reserved", "--watch", "--compute-jobs", "123"]
    assert reporter.main(args) == 2
    assert inits[0]["resume"] == "never" and runs[0].finished == [{"exit_code": 1}]
    assert not any(any(k.startswith("episodes/") for k in row) for row in runs[0].logs)
    assert reporter.main(args) == 2
    assert inits[1]["resume"] == "must"
    with pytest.raises(ValueError, match="different run"):
        reporter.main([*args[:args.index("reserved")], "other", *args[args.index("reserved")+1:]])


def write_receipts(campaign, path):
    for cell in campaign["cells"]:
        receipt = {k: cell[k] for k in ("cell_id", "arm", "reused", "seeds")}
        receipt.update(status="complete", campaign_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        reporter.write(Path(campaign["output_root"]) / "production" / cell["cell_id"] / "merge-completion.json", receipt)


def test_requires_current_complete_merge_receipts_before_loading(campaign, tmp_path):
    path = tmp_path / "campaign.json"; reporter.write(path, campaign)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = {}
    assert reporter.inspect(campaign, loaded, checksum)["complete_cells"] == 0
    write_receipts(campaign, path)
    assert reporter.inspect(campaign, loaded, checksum)["complete_cells"] == 27
    assert reporter.inspect(campaign, {}, "bad-hash")["complete_cells"] == 0


def test_disabled_incomplete_has_no_science_or_wandb(campaign, tmp_path):
    (Path(campaign["cells"][9]["bundle"]) / "seed-shard-checksums.json").unlink()
    path = tmp_path / "campaign.json"; reporter.write(path, campaign)
    write_receipts(campaign, path)
    assert reporter.main(["--campaign", str(path)]) == 2
    result = reporter.read(tmp_path / "comparison/report.json")
    assert result["complete_cells"] == 26 and "summaries" not in result
