import copy
import base64
import gzip
import html
import json
from pathlib import Path
import re
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from utils.ambi_entropy_reporting import (
    ROW_FIELDS, _REPORT_JS, aggregate_entropy_rows, publish_entropy_wandb,
    start_entropy_wandb, write_entropy_report,
)


def row(seed=101, root="r0", solver=0, rollout=0, arm="off", updates=0,
        value=0.0, source="entity/project/bank", rule="mean"):
    return {"source_run": source, "checkpoint_step": 200000,
            "checkpoint_sha256": "a"*64, "prefix_action_rule": rule,
            "episode_seed": seed, "root_id": root,
            "solver_repetition": solver, "rollout_repetition": rollout,
            "arm": arm, "actor_updates": updates,
            "metrics": {"return": value}, "policy_noise_seed": 1234}


def expected(rows):
    return [{key: r[key] for key in ROW_FIELDS} for r in rows]


def balanced_rows():
    return [row(seed=seed, arm=arm, updates=a,
                value=(seed-100)*10 + a*(offset+1))
            for seed in (101, 102, 103) for arm, offset in
            (("off", 0), ("prior_recipe", 2), ("squashed_matched", 5))
            for a in (0, 1, 4, 16)]


def test_hierarchy_weights_episodes_roots_solvers_and_rollouts_equally():
    # One root has unequal rollout counts across solvers; one episode has
    # unequal root counts; episodes contain unequal total measurement counts.
    rows = [row(seed=101, root="a", solver=0, rollout=i, value=0) for i in range(5)]
    rows += [row(seed=101, root="a", solver=1, value=20),
             row(seed=101, root="b", value=30), row(seed=102, value=100)]
    report = aggregate_entropy_rows(rows, expected_rows=expected(rows))
    stat = report["series"][0]["metrics"]["return"]
    assert stat["episode_means"] == {"101": 20.0, "102": 100.0}
    assert stat["mean"] == 60
    assert stat["n_episodes"] == 2
    assert stat["ci95"] == [20, 100]
    assert report["initialization_deltas"][0]["metrics"]["return"]["ci95"] == [0, 0]


def test_paired_contrasts_preserve_covariance_and_actual_initialization():
    rows = balanced_rows()
    before = copy.deepcopy(rows)
    report = aggregate_entropy_rows(rows, expected_rows=expected(rows))
    assert rows == before
    contrast = next(r for r in report["contrasts"] if r["arm"] == "squashed_matched"
                    and r["reference_arm"] == "prior_recipe" and r["actor_updates"] == 4)
    assert contrast["metrics"]["return"]["mean"] == 12
    assert contrast["metrics"]["return"]["ci95"] == [12, 12]
    for point in report["initialization_deltas"]:
        if point["actor_updates"] == 0:
            assert point["metrics"]["return"]["mean"] == 0
    assert report["raw_rows"][0]["policy_noise_seed"] == 1234
    assert report["uncertainty"]["resamples"] == 2000


def test_no_pooling_across_checkpoint_or_action_rule():
    rows = [row(seed=seed, value=value, source=source, rule=rule)
            for source, value in (("bank1", 1), ("bank2", 1000))
            for rule in ("mean", "sampled") for seed in (101, 102)]
    report = aggregate_entropy_rows(rows, expected_rows=expected(rows))
    assert len(report["series"]) == 4
    assert sorted(p["metrics"]["return"]["mean"] for p in report["series"]) == [1, 1, 1000, 1000]
    assert not report["uncertainty"]["across_bank_pooling"]


def test_partial_has_coverage_only_and_final_rejects_missing_rows():
    rows = balanced_rows()
    report = aggregate_entropy_rows(rows[:-1], expected_rows=expected(rows), final=False)
    assert report["status"] == "partial"
    assert len(report["coverage"]["missing_rows"]) == 1
    assert report["series"] == report["contrasts"] == report["initialization_deltas"] == []
    with pytest.raises(ValueError, match="Incomplete final panel"):
        aggregate_entropy_rows(rows[:-1], expected_rows=expected(rows))
    # A publisher must explicitly request a final report, not infer it from
    # a conveniently complete subset of currently available rows.
    assert aggregate_entropy_rows(rows, expected_rows=expected(rows), final=False)["series"] == []


@pytest.mark.parametrize("mutation,match", [
    (lambda rows: rows.append(copy.deepcopy(rows[0])), "Duplicate"),
    (lambda rows: rows[0]["metrics"].update({"return": float("nan")}), "finite"),
    (lambda rows: rows[0].update({"actor_updates": True}), "integer"),
    (lambda rows: rows[0].update({"prefix_action_rule": "stochastic"}), "prefix_action_rule"),
    (lambda rows: rows[0].update({"root_id": "unexpected"}), "Unexpected"),
])
def test_invalid_rows_rejected(mutation, match):
    rows = balanced_rows()
    identities = expected(rows)
    mutation(rows)
    with pytest.raises(ValueError, match=match):
        aggregate_entropy_rows(rows, expected_rows=identities)


def test_missing_metric_or_initialization_or_pair_rejected():
    rows = balanced_rows()
    with pytest.raises(ValueError, match="required metrics"):
        aggregate_entropy_rows(rows, expected_rows=expected(rows), required_metrics=("absent",))
    rows[0]["metrics"]["gradient"] = 1
    with pytest.raises(ValueError, match="Incomplete metric coverage"):
        aggregate_entropy_rows(rows, expected_rows=expected(rows))
    rows = [r for r in balanced_rows() if r["actor_updates"] != 0]
    with pytest.raises(ValueError, match="A0 initialization"):
        aggregate_entropy_rows(rows, expected_rows=expected(rows))
    rows = balanced_rows()
    for r in rows:
        if r["arm"] == "prior_recipe":
            r["rollout_repetition"] = 1
    with pytest.raises(ValueError, match="identical episode/root/solver/rollout"):
        aggregate_entropy_rows(rows, expected_rows=expected(rows))


def test_bootstrap_reproducible_and_single_episode_not_false_precision():
    rows = balanced_rows()
    a = aggregate_entropy_rows(rows, expected_rows=expected(rows), bootstrap_seed=7)
    b = aggregate_entropy_rows(reversed(rows), expected_rows=expected(rows), bootstrap_seed=7)
    assert a["series"] == b["series"]
    one = [row(value=12)]
    report = aggregate_entropy_rows(one, expected_rows=expected(one))
    assert report["series"][0]["metrics"]["return"]["ci95"] is None


class FakeRun:
    id = "explicit-new-campaign"

    def __init__(self):
        self.logs, self.definitions, self.artifacts, self.summary = [], [], [], {}

    def log(self, value):
        self.logs.append(value)

    def define_metric(self, name, **kwargs):
        self.definitions.append((name, kwargs))

    def log_artifact(self, value):
        self.artifacts.append(value)


class FakeArtifact:
    def __init__(self, name, **kwargs):
        self.name, self.files, self.metadata = name, [], kwargs["metadata"]

    def add_file(self, path, name):
        self.files.append((path, name))


def fake_wandb():
    run = FakeRun()
    calls = []
    def init(**kwargs):
        calls.append(kwargs)
        return run
    return SimpleNamespace(init=init, Artifact=FakeArtifact,
                           Html=lambda text, inject: {"html": text, "inject": inject},
                           run=run, calls=calls)


def test_bundle_html_wandb_round_trip_and_custom_axes(tmp_path):
    rows = balanced_rows()
    rows[0]["note"] = '</template><script>alert("unsafe")</script>'
    report = aggregate_entropy_rows(rows, expected_rows=expected(rows), metadata={"attempt": "new"})
    files = write_entropy_report(report, tmp_path)
    assert json.loads(Path(files["report"]).read_text()) == report
    assert [json.loads(line) for line in Path(files["rows"]).read_text().splitlines()] == rows
    document = Path(files["html"]).read_text()
    encoded = re.search(r'<template id="entropy-report-data" data-encoding="gzip-base64">(.*?)</template>', document).group(1)
    assert json.loads(gzip.decompress(base64.b64decode(encoded))) == report
    assert 'alert("unsafe")' not in document
    charts = re.search(r'<template id="entropy-chart-data">(.*?)</template>', document).group(1)
    assert 'raw_rows' not in json.loads(html.unescape(charts))
    assert 'id="metric"' in document and 'https://' not in document
    wb = fake_wandb()
    run = start_entropy_wandb(run_id="new-id", config={"protocol": "fixed-data"}, wandb_module=wb)
    assert wb.calls[0]["resume"] == "never"
    assert run.logs[0]["progress/observed_rows"] == 0
    publish_entropy_wandb(run, report, files, wandb_module=wb)
    assert run.summary["entropy_status"] == "complete"
    assert len(run.artifacts) == 1 and len(run.artifacts[0].files) == 3
    assert run.summary["entropy_report_html"]["inject"] is False
    axes = [name for name, _ in run.definitions if name.endswith("/actor_updates")]
    assert any("squashed_matched_minus_prior_recipe" in axis for axis in axes)
    for name, kwargs in run.definitions:
        if name.endswith("/mean"):
            assert kwargs["step_metric"].endswith("/actor_updates")


def test_partial_publication_has_no_scientific_points(tmp_path):
    rows = balanced_rows()
    report = aggregate_entropy_rows(rows[:1], expected_rows=expected(rows), final=False)
    files = write_entropy_report(report, tmp_path)
    run = FakeRun()
    wb = fake_wandb()
    publish_entropy_wandb(run, report, files, wandb_module=wb)
    assert all(key.startswith("progress/") for payload in run.logs for key in payload)
    assert 'INCOMPLETE' in Path(files["html"]).read_text()
    assert 'id="charts"' not in Path(files["html"]).read_text()
    assert run.artifacts[0].metadata["status"] == "partial"
    assert start_entropy_wandb(run_id="x", config={}, mode="disabled") is None
    publish_entropy_wandb(None, report)


def test_alias_provenance_is_preserved_without_collapsing_requested_arms():
    rows = balanced_rows()
    for r in rows:
        if r["arm"] == "squashed_matched":
            r["metrics"]["return"] = (r["episode_seed"]-100)*10 + r["actor_updates"]*3
            r["compute_alias_of"] = "prior_recipe"
    report = aggregate_entropy_rows(rows, expected_rows=expected(rows))
    assert all(p["metrics"]["return"]["mean"] == 0 for p in report["contrasts"]
               if p["arm"] == "squashed_matched" and p["reference_arm"] == "prior_recipe")
    assert any(r.get("compute_alias_of") == "prior_recipe" for r in report["raw_rows"])


def test_wandb_metric_names_do_not_collide_after_encoding():
    rows = [row(seed=seed) for seed in (101, 102)]
    for r in rows:
        r["metrics"].update({"q/gradient": 2, "q_gradient": 3})
    report = aggregate_entropy_rows(rows, expected_rows=expected(rows))
    run = FakeRun()
    publish_entropy_wandb(run, report)
    metric_names = {key for payload in run.logs for key in payload}
    assert any(key.endswith('/q%2Fgradient/mean') for key in metric_names)
    assert any(key.endswith('/q_gradient/mean') for key in metric_names)


@pytest.mark.skipif(shutil.which("node") is None, reason="Optional standalone HTML interaction check needs Node")
def test_lazy_html_metric_and_comparison_controls_execute():
    rows = balanced_rows()
    for r in rows:
        r["metrics"]["second_metric"] = -r["metrics"]["return"]
    report = aggregate_entropy_rows(rows, expected_rows=expected(rows))
    payload = {key: report[key] for key in ("status", "series", "contrasts", "initialization_deltas")}
    # Small DOM seam executes the actual shipped JavaScript, including SVG
    # generation and both dropdown callbacks, without a browser dependency.
    setup = r"""
const assert=require('node:assert/strict');
function el(value='') {return {value, handlers:{}, innerHTML:'',
  addEventListener(name,fn){this.handlers[name]=fn;},
  replaceChildren(...children){this.children=children;this.value=children[0]?.value||'';}};}
const elements={'family':el('series'),'metric':el(),'charts':el(),'download':el(),
  'entropy-chart-data':{content:{textContent:JSON.stringify(PAYLOAD)}}};
global.document={getElementById:id=>elements[id],createElement:()=>el()};
""".replace("PAYLOAD", json.dumps(payload))
    assertions = r"""
assert.ok(elements.charts.innerHTML.includes('<svg'));
assert.equal(elements.metric.children.length,2);
elements.metric.value='second_metric';elements.metric.handlers.change();
assert.ok(elements.charts.innerHTML.includes('second_metric versus actor updates'));
elements.family.value='contrasts';elements.family.handlers.change();
assert.ok(elements.charts.innerHTML.includes('squashed_matched − prior_recipe'));
assert.ok(!elements.charts.innerHTML.includes('NaN'));
"""
    result = subprocess.run([shutil.which("node"), "-"], input=setup+_REPORT_JS+assertions,
                            text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
