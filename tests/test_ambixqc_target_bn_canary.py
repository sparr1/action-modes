"""The default-mode guard rejects scientific drift before the ablation starts."""
from copy import deepcopy
import gzip
import json

import pytest

import run_ambixqc_bn_study as study
import run_ambixqc_target_bn_study as campaign


@pytest.fixture
def canary(tmp_path, monkeypatch):
    old_root = tmp_path / "old" / "smoke" / "bundle"
    new_root = tmp_path / "worker" / "default-compat" / "bundle"
    record = {"decision_index": 0, "event_index": 0, "phase": "decision", "round_index": 2,
        "critic_updates": 12, "actor_updates": 4, "temperature_updates": 4,
        "metrics": {"decision/inner_critic_loss": 5.4,
                    "decision/inner_reward_normalizer_count_final": 475000,
                    "decision/inner_reward_scale": 38.01174,
                    "decision/inner_outer_terminal_bootstrap_rows": 1513,
                    "decision/inner_action_seconds": 0.1}}
    for root, new in ((old_root, False), (new_root, True)):
        root.mkdir(parents=True)
        row = deepcopy(record)
        if new:
            row["metrics"]["decision/inner_critic_target_bn_running"] = 0
            row["metrics"]["decision/inner_action_seconds"] = 10.0
        with gzip.open(root / "trace.jsonl.gz", "wt") as stream:
            stream.write(json.dumps(row) + "\n")
        (root / "manifest.json").write_text(json.dumps({"runs": [{
            "episodes": [{"seed": 101, "length": 1, "return": 1.2}],
            "trace_files": ["trace.jsonl.gz"]}]}))
    calls = []
    monkeypatch.setattr(campaign, "validate_default_bundle", lambda *a: calls.append(a))
    plan = {"source_sha": "c" * 40, "reused": [{"result": {"output_path": str(tmp_path / "old")}}]}
    return plan, tmp_path / "worker", new_root, calls


def change_trace(root, transform):
    path = root / "trace.jsonl.gz"
    with gzip.open(path, "rt") as stream:
        row = json.loads(stream.readline())
    transform(row)
    with gzip.open(path, "wt") as stream:
        stream.write(json.dumps(row) + "\n")


def test_comparison_binds_both_sources_and_ignores_only_timing(canary):
    plan, root, _, calls = canary
    report = campaign.compare_default_canary(plan, root, study)
    assert [call[0]["source_sha"] for call in calls] == [plan["source_sha"], campaign.PARENT_SOURCE_SHA]
    assert report["action_comparison"] is False
    assert report["decisions"] == 1 and report["numeric_comparisons"] == 5
    assert report["max_absolute_difference"] == 0
    for key in ("old_manifest", "new_manifest"):
        study.check_binding(report[key])
    for key in ("old_traces", "new_traces"):
        for binding in report[key]:
            study.check_binding(binding)


@pytest.mark.parametrize("key,value", [
    ("decision/inner_reward_normalizer_count_final", 475001),
    ("decision/inner_reward_scale", 38.011741),
    ("decision/inner_outer_terminal_bootstrap_rows", 1514),
    ("decision/inner_critic_loss", 5.5),
    ("decision/inner_critic_target_bn_running", 1),
])
def test_scientific_drift_fails_before_full_run(canary, key, value):
    plan, root, traces, _ = canary
    change_trace(traces, lambda row: row["metrics"].update({key: value}))
    with pytest.raises(ValueError):
        campaign.compare_default_canary(plan, root, study)


def test_small_float_roundoff_allowed_with_reported_magnitude(canary):
    plan, root, traces, _ = canary
    change_trace(traces, lambda row: row["metrics"].update({"decision/inner_critic_loss": 5.400001}))
    report = campaign.compare_default_canary(plan, root, study)
    assert report["max_absolute_difference"] == pytest.approx(1e-6)


@pytest.mark.parametrize("problem", ["counter", "metric", "episode", "rows"])
def test_alignment_and_metric_schema_cannot_silently_change(canary, problem):
    plan, root, traces, _ = canary
    if problem == "counter":
        change_trace(traces, lambda row: row.update(critic_updates=13))
    elif problem == "metric":
        change_trace(traces, lambda row: row["metrics"].pop("decision/inner_critic_loss"))
    elif problem == "rows":
        with gzip.open(traces / "trace.jsonl.gz", "wt") as stream:
            stream.write("")
    else:
        path = traces / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["runs"][0]["episodes"][0]["seed"] = 102
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        campaign.compare_default_canary(plan, root, study)
