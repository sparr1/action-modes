import copy
import gzip
import json
import shutil
import subprocess
from pathlib import Path

import pytest

import report_ambi_benchmark as report


def _metric(definition="Critic minibatch loss", axis="critic_updates"):
    return {"definition": definition, "unit": "loss", "sampling_phase": "critic_update", "preferred_axis": axis}


def _event(run_id, decision, event, **updates):
    row = {"run_id": run_id, "episode_id": 0, "decision_index": decision,
           "event_index": event, "phase": "critic_update", "round_index": event // 3,
           "critic_updates": event + 1, "actor_updates": event // 3,
           "temperature_updates": 0, "metrics": {"critic_loss": 10.0 / (event + 1)}}
    row.update(updates)
    return row


def _bundle(path, *, evaluation="eval-1", run_id="sac", decisions=2, root_bank="bank-1", nonfinite=False):
    path.mkdir(parents=True)
    roots = [{"root_id": "root-7", "repeat": 0, "solver_seed": 789, "root_bank_id": root_bank,
              "probe_rollouts": 2, "probe_horizon": 3, "probe_seed": 91}]
    rows = [_event(run_id, decision, event) for decision in range(decisions) for event in range(6)]
    rows.append(_event(run_id, 0, 6, phase="actor_update", actor_updates=2, critic_updates=6,
                       metrics={"actor_loss": -2.0}))
    rows.append(_event(run_id, 0, 7, phase="decision", actor_updates=2, critic_updates=6,
                       metrics={"decision/reward": 0.25}))
    for event in range(6):
        root_row = _event(run_id, 0, event, root_id="root-7", repeat=0)
        root_row.pop("episode_id")
        rows.append(root_row)
    if nonfinite:
        rows[2]["metrics"]["critic_loss"] = None
        rows[2]["nonfinite"] = {"critic_loss": "nan"}
    with gzip.open(path / "trace.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    run = {"id": run_id, "selector": f"test/{run_id}", "config": {"inner_steps": 6},
           "config_hash": f"config-{run_id}", "kind": "both", "status": "complete",
           "episodes": [{"episode_id": 0, "seed": 101, "return": 10.0, "length": decisions,
                         "terminated": False, "truncated": False, "capped": True,
                         "control_seconds": 0.3, "paired_return_delta": 2.0}],
           "roots": roots, "trace_files": ["trace.jsonl.gz"]}
    manifest = {"schema_version": 1, "evaluation_id": evaluation,
                "checkpoint": {"sha256": "a" * 64, "path": "/saved/checkpoint.pt", "source_run": "ambi/source"},
                "code": {"commit": "example"}, "status": "complete",
                "protocol": {"environment": {"task": "humanoid-walk"}, "action_rule": "tanh_mean",
                             "max_steps": 500, "controller_seed": 18, "seed_scheme": "sha256-v1",
                             "root_bank_id": root_bank},
                "metric_catalog": {"critic_loss": _metric(), "actor_loss": _metric("Actor minibatch loss", "actor_updates"),
                                   "decision/reward": _metric("Real decision reward", "event_index")},
                "runs": [run]}
    _save_manifest(path, manifest)
    return manifest


def _save_manifest(path, manifest):
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _rewrite_rows(path, transform):
    with gzip.open(path / "trace.jsonl.gz", "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    rows = transform(rows)
    with gzip.open(path / "trace.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.writelines(json.dumps(row) + "\n" for row in rows)


def test_round_trip_preserves_sparse_counters_nonfinite_and_root_identity(tmp_path):
    _bundle(tmp_path / "first", nonfinite=True)
    data = report.load_bundles([tmp_path / "first"])
    run = data["runs"][0]
    episode = next(trace for trace in run["traces"] if trace["mode"] == "episodes" and trace["decision_index"] == 0)
    assert episode["event_index"] == list(range(8))
    assert episode["critic_updates"] == [1, 2, 3, 4, 5, 6, 6, 6]
    assert episode["metrics"]["critic_loss"] == [10, 5, None, 2.5, 2, 10 / 6, None, None]
    assert episode["nonfinite"]["critic_loss"][2] == "nan"
    assert episode["metrics"]["actor_loss"] == [None] * 6 + [-2.0, None]
    root = next(trace for trace in run["traces"] if trace["mode"] == "bank")
    assert json.loads(root["selection_key"]) == ["bank-1", "root-7", 0, 789]
    assert run["episodes"][0]["capped"]


def test_baseline_and_episode_only_bundle_can_compare_with_root_bank(tmp_path):
    baseline = _bundle(tmp_path / "baseline", evaluation="baseline", run_id="prior")
    baseline["protocol"].pop("root_bank_id")
    baseline["runs"][0].update({"kind": "episodes", "roots": [], "trace_files": []})
    _save_manifest(tmp_path / "baseline", baseline)
    _bundle(tmp_path / "candidate", evaluation="candidate")
    data = report.load_bundles([tmp_path / "baseline", tmp_path / "candidate"])
    assert data["runs"][0]["traces"] == []
    assert len(data["runs"]) == 2


@pytest.mark.parametrize("change,match", [
    (lambda manifest: manifest.update(schema_version=2), "unsupported schema_version"),
    (lambda manifest: manifest["checkpoint"].update(sha256="b" * 64), "incompatible checkpoint"),
    (lambda manifest: manifest["protocol"].update(action_rule="sample"), "incompatible checkpoint"),
    (lambda manifest: manifest["metric_catalog"]["critic_loss"].update(sampling_phase="after_update"), "incompatible semantics"),
    (lambda manifest: manifest["runs"][0]["roots"][0].update(probe_horizon=9), "incompatible probe protocol"),
    (lambda manifest: manifest["runs"][0]["roots"][0].update(probe_seed=92), "incompatible probe protocol"),
    (lambda manifest: manifest["runs"][0]["roots"][0].update(probe_rollouts=8), "incompatible probe protocol"),
])
def test_rejects_semantically_incompatible_comparisons(tmp_path, change, match):
    _bundle(tmp_path / "first")
    second = _bundle(tmp_path / "second", evaluation="eval-2")
    change(second)
    _save_manifest(tmp_path / "second", second)
    with pytest.raises(ValueError, match=match):
        report.load_bundles([tmp_path / "first", tmp_path / "second"])


def test_different_bank_ids_and_solver_seeds_do_not_join(tmp_path):
    _bundle(tmp_path / "first")
    second = _bundle(tmp_path / "second", evaluation="eval-2", root_bank="bank-2")
    second["runs"][0]["roots"][0]["solver_seed"] = 790
    _save_manifest(tmp_path / "second", second)
    data = report.load_bundles([tmp_path / "first", tmp_path / "second"])
    keys = [next(trace for trace in run["traces"] if trace["mode"] == "bank")["selection_key"] for run in data["runs"]]
    assert keys[0] != keys[1]


def test_matched_root_probes_allow_different_runtime_and_total_cost(tmp_path):
    for name, seconds, steps in (("first", 0.1, 192), ("second", 0.9, 384)):
        manifest = _bundle(tmp_path / name, evaluation=name)
        manifest["runs"][0]["roots"][0].update(probe_seconds=seconds, probe_model_steps=steps)
        _save_manifest(tmp_path / name, manifest)
    data = report.load_bundles([tmp_path / "first", tmp_path / "second"])
    roots = [run["roots"][0] for run in data["runs"]]
    assert [root["probe_seconds"] for root in roots] == [0.1, 0.9]
    assert [root["probe_model_steps"] for root in roots] == [192, 384]
    keys = [next(trace["selection_key"] for trace in run["traces"] if trace["mode"] == "bank")
            for run in data["runs"]]
    assert keys[0] == keys[1]


@pytest.mark.parametrize("transform,match", [
    (lambda rows: rows + [copy.deepcopy(rows[0])], "duplicate event_index"),
    (lambda rows: [{**rows[0], "run_id": "wrong"}, *rows[1:]], "run_id does not match"),
    (lambda rows: [{**rows[0], "episode_id": 42}, *rows[1:]], "unknown episode_id"),
    (lambda rows: [rows[0], {**rows[1], "critic_updates": 0}, *rows[2:]], "decreasing critic_updates"),
    (lambda rows: [{**rows[0], "metrics": {"unknown": 1}}, *rows[1:]], "absent from metric_catalog"),
    (lambda rows: [{**rows[0], "metrics": {"critic_loss": None}, "nonfinite": {"critic_loss": "oops"}}, *rows[1:]], "invalid nonfinite"),
])
def test_rejects_ambiguous_or_invalid_trace_rows(tmp_path, transform, match):
    _bundle(tmp_path / "bundle")
    _rewrite_rows(tmp_path / "bundle", transform)
    with pytest.raises(ValueError, match=match):
        report.load_bundles([tmp_path / "bundle"])


def test_partial_episode_keeps_failed_decision_trace(tmp_path):
    manifest = _bundle(tmp_path / "bundle")
    manifest["runs"][0]["status"] = "failed"
    manifest["runs"][0]["episodes"][0].update(length=1, capped=False)
    _save_manifest(tmp_path / "bundle", manifest)
    data = report.load_bundles([tmp_path / "bundle"])
    assert any(trace["decision_index"] == 1 for trace in data["runs"][0]["traces"])
    assert data["runs"][0]["status"] == "failed"


def test_episode_status_controls_partial_trace_and_bank_keeps_source_decision(tmp_path):
    manifest = _bundle(tmp_path / "bundle")
    manifest["runs"][0]["episodes"][0].update(length=1, status="partial")
    _save_manifest(tmp_path / "bundle", manifest)
    _rewrite_rows(tmp_path / "bundle", lambda rows: [{**row, "decision_index": 177} if "root_id" in row else row for row in rows])
    data = report.load_bundles([tmp_path / "bundle"])
    assert next(trace for trace in data["runs"][0]["traces"] if trace["mode"] == "bank")["decision_index"] == 177


def test_report_is_portable_escapes_labels_and_preserves_existing_file(tmp_path):
    manifest = _bundle(tmp_path / "bundle")
    manifest["runs"][0]["selector"] = "</script><script>alert(1)</script>"
    _save_manifest(tmp_path / "bundle", manifest)
    data = report.load_bundles([tmp_path / "bundle"])
    output = report.write_report(data, tmp_path / "report.html", title="<example>")
    rendered = output.read_text(encoding="utf-8")
    assert "<title>&lt;example&gt;</title>" in rendered
    assert "</script><script>alert(1)</script>" not in rendered
    assert "\\u003c/script\\u003e" in rendered
    assert "<script src=" not in rendered
    assert "<link " not in rendered
    assert "__REPORT_" not in rendered
    original = output.read_bytes()
    with pytest.raises(FileExistsError, match="--overwrite"):
        report.write_report(data, output)
    assert output.read_bytes() == original
    report.write_report(data, output, overwrite=True, title="Changed")
    assert "<title>Changed</title>" in output.read_text(encoding="utf-8")
    assert not list(tmp_path.glob("*.tmp"))


def test_cli_selects_named_metrics(tmp_path, capsys):
    _bundle(tmp_path / "bundle")
    assert report.main(["--bundle", str(tmp_path / "bundle"), "--output", str(tmp_path / "out.html"), "--metric", "actor_loss"]) == 0
    html = (tmp_path / "out.html").read_text(encoding="utf-8")
    encoded = html.split('<script id="report-data" type="application/json">', 1)[1].split("</script>", 1)[0]
    data = json.loads(encoded)
    assert set(data["metric_catalog"]) == {"actor_loss"}
    assert all(set(trace["metrics"]) <= {"actor_loss"} for trace in data["runs"][0]["traces"])
    assert json.loads(capsys.readouterr().out)["runs"] == 1


def test_template_placeholder_text_in_labels_is_not_recursively_expanded(tmp_path):
    _bundle(tmp_path / "bundle")
    data = report.load_bundles([tmp_path / "bundle"])
    data["runs"][0]["label"] = "__REPORT_SCRIPT__"
    rendered = report.render_html(data, title="__REPORT_DATA__")
    assert "<title>__REPORT_DATA__</title>" in rendered
    encoded = rendered.split('<script id="report-data" type="application/json">', 1)[1].split("</script>", 1)[0]
    assert json.loads(encoded)["runs"][0]["label"] == "__REPORT_SCRIPT__"


def test_trace_file_cannot_escape_bundle(tmp_path):
    manifest = _bundle(tmp_path / "bundle")
    manifest["runs"][0]["trace_files"] = ["../outside.jsonl"]
    _save_manifest(tmp_path / "bundle", manifest)
    with pytest.raises(ValueError, match="escapes its bundle"):
        report.load_bundles([tmp_path / "bundle"])


def test_node_selection_keeps_exact_counters_gaps_and_shared_root_alignment(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed; Python validation remains available.")
    _bundle(tmp_path / "first", nonfinite=True)
    second = _bundle(tmp_path / "second", evaluation="eval-2", run_id="other")
    # A different actor/critic dose ends early; the viewer must not extend it.
    _rewrite_rows(tmp_path / "second", lambda rows: [row for row in rows if row["event_index"] < 3])
    data_path = tmp_path / "data.json"
    data_path.write_text(json.dumps(report.load_bundles([tmp_path / "first", tmp_path / "second"])))
    js_path = Path(report.__file__).parent / "utils" / "ambi_benchmark_report.js"
    script = r'''
const assert = require("node:assert/strict");
const fs = require("node:fs");
const api = require(process.argv[1]);
const data = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const episode = api.selections(data.runs, "episodes")[0].key;
let curves = api.overlay(data.runs, "episodes", episode, 0, "critic_loss", "critic_updates");
assert.deepEqual(curves[0].points.map(p => p.x), [1,2,3,4,5,6]);
assert.equal(curves[0].points[2].nonfinite, "nan");
assert.deepEqual(curves[1].points.map(p => p.x), [1,2,3]);
assert.deepEqual(api.overlay(data.runs, "episodes", episode, 0, "actor_loss", "actor_updates")[0].points.map(p => p.x), [2]);
assert.equal(api.overlay(data.runs, "episodes", episode, 0, "actor_loss", "actor_updates")[1].points.length, 0);
curves = api.overlay(data.runs, "episodes", episode, 1, "critic_loss", "event_index");
assert.equal(curves[0].points.length, 6);
assert.equal(curves[0].points[2].nonfinite, null);
const roots = api.selections(data.runs, "bank");
assert.equal(roots.length, 1);
assert.deepEqual(JSON.parse(roots[0].key), ["bank-1","root-7",0,789]);
curves = api.overlay(data.runs, "bank", roots[0].key, 0, "critic_loss", "critic_updates");
assert.equal(curves[0].points.length, 6);
assert.equal(curves[1].points.length, 3);
const altered = structuredClone(data.runs[1]);
altered.traces.filter(t => t.mode === "bank").forEach(t => t.selection_key = '["other-bank","root-7",0,789]');
assert.equal(api.overlay([altered], "bank", roots[0].key, 0, "critic_loss", "critic_updates")[0].points.length, 0);
const traces = api.selectTraces(data.runs[0], "episodes", episode);
const cells = api.heatmapCells(traces, "critic_loss", "actor_updates");
const first = cells.find(cell => cell.decision === 0 && cell.x === 0);
assert.equal(first.count, 3);
assert.equal(first.nonfinite, "nan");
assert.equal(api.heatmapCells(traces, "actor_loss", "actor_updates").length, 1);
const summaryRun = structuredClone(data.runs[0]);
const summaryTraces = summaryRun.traces.filter(t => t.mode === "episodes");
summaryTraces.forEach(t => {
 t.metrics["decision/reward"] = t.event_index.map((_, index) => index === t.event_index.length - 1 ? 100 + t.decision_index : null);
});
curves = api.overlay([summaryRun], "episodes", episode, 1, "decision/reward", "decision_index");
assert.deepEqual(curves[0].points.map(p => [p.x, p.value]), [[0,100],[1,101]]);
assert.deepEqual(api.heatmapCells(summaryTraces, "decision/reward", "decision_index").map(p => [p.x, p.decision, p.value]), [[0,0,100],[1,1,101]]);
assert.equal(api.overlay([summaryRun], "episodes", '["unknown",999]', 0, "decision/reward", "decision_index")[0].points.length, 0);
assert.deepEqual(api.extent([null, NaN, Infinity, -2, 3]), [-2,3]);
console.log("selection/alignment passed");
'''
    result = subprocess.run([node, "-e", script, str(js_path), str(data_path)], text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "selection/alignment passed" in result.stdout
