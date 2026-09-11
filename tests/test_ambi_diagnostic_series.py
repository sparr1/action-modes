import copy
import json
import random
from pathlib import Path

import pytest

from utils import ambi_diagnostic_series as diagnostics


def identity(scope="common_prior_roots"):
    return {"checkpoint": {"step": 2000000, "sha256": "a" * 64},
            "backbone": "rwgao_b-brown-university/ambi/mey3rxj8",
            "setting": {"inner_rounds": 5}, "protocol": {"gamma": .99, "horizon": 3},
            "code": {"commit": "tested-source"}, "scope": scope}


def panel():
    # Unequal root counts expose accidental flat averaging (expected mean 6.5,
    # flat mean 5). Two replicas and solvers expose their independent levels.
    roots = [{"episode_id": 0, "root_id": "zero"},
             {"episode_id": 0, "root_id": "two"},
             {"episode_id": 1, "root_id": "ten"}]
    expected = {"roots": roots, "solver_repeats": 2, "rollout_repeats": 2, "rounds": [0, 1]}
    rows = []
    for root, value in zip(roots, [0., 2., 10.]):
        for solver in range(2):
            for rollout in range(2):
                for round_index in (0, 1):
                    rows.append({**root, "solver_repeat": solver, "rollout_repeat": rollout,
                                 "round_index": round_index, "actor_updates": 3 * round_index,
                                 "critic_updates": 3 * round_index, "decision_index": 0,
                                 "metrics": {"real_return": value + solver + rollout + round_index,
                                             "prior_return": value + solver + rollout,
                                             "gain": float(round_index)}})
    return rows, expected


def record(**kwargs):
    rows, expected = panel()
    return diagnostics.build_diagnostic_record(identity(), rows, expected,
                                              attempt_label="scratch-1", **kwargs)


def test_nested_weighting_and_cluster_bootstrap_keep_sampling_units():
    rng = random.getstate()
    data = record(bootstrap_resamples=2000, bootstrap_seed=19)
    assert random.getstate() == rng
    summary = data["summaries"][0]["metrics"]["real_return"]
    assert summary["mean"] == 6.5  # episode means 2 and 11
    assert summary["mean"] != pytest.approx(5.)  # flat row mean
    assert summary["episode_std"] == pytest.approx(9 / 2 ** .5)
    assert summary["ci95_low"] == 2
    assert summary["ci95_high"] == 11
    assert summary["within_solver_rollout_std_mean"] == pytest.approx(1 / 2 ** .5)
    assert summary["within_root_solver_std_mean"] == pytest.approx(1 / 2 ** .5)
    assert summary["episodes"] == 2 and summary["roots"] == 3
    assert summary["solvers"] == 6 and summary["rollouts"] == 12
    gain = data["summaries"][1]["metrics"]["gain"]
    assert gain["mean"] == gain["ci95_low"] == gain["ci95_high"] == 1
    assert data == record(bootstrap_resamples=2000, bootstrap_seed=19)


def test_single_episode_interval_is_unavailable_not_false_precision():
    rows, expected = panel()
    expected["roots"] = expected["roots"][:2]
    rows = [row for row in rows if row["episode_id"] == 0]
    data = diagnostics.build_diagnostic_record(identity(), rows, expected, attempt_label="one")
    metric = data["summaries"][0]["metrics"]["real_return"]
    assert metric["ci95_low"] is metric["ci95_high"] is metric["episode_std"] is None


@pytest.mark.parametrize("mutation,match", [
    (lambda rows: rows.append(copy.deepcopy(rows[0])), "Duplicate diagnostic"),
    (lambda rows: rows[0].update(root_id="unknown"), "Unexpected diagnostic"),
    (lambda rows: rows[0].update(actor_updates=1), "Inconsistent update"),
    (lambda rows: rows[0].update(solver_repeat=True), "integer"),
    (lambda rows: rows[0]["metrics"].update(real_return=True), "finite numeric"),
])
def test_panel_rejects_ambiguous_measurements(mutation, match):
    rows, expected = panel()
    mutation(rows)
    with pytest.raises(ValueError, match=match):
        diagnostics.build_diagnostic_record(identity(), rows, expected, attempt_label="bad")


def test_missing_rows_and_missing_metrics_remain_inspectable_but_unpublishable():
    rows, expected = panel()
    removed = rows.pop()
    rows[0]["metrics"]["gain"] = None
    data = diagnostics.build_diagnostic_record(identity(), rows, expected, attempt_label="partial")
    assert data["status"] == "incomplete"
    assert data["missing_coordinates"] == [{key: removed[key] for key in diagnostics.COORDINATES}]
    assert data["missing_metrics"][0]["metrics"] == ["gain"]
    assert diagnostics.extract_diagnostic_html_data(diagnostics.render_diagnostic_html(data)) == data
    with pytest.raises(ValueError, match="Incomplete"):
        diagnostics.diagnostic_history(data)


def test_empty_observations_cannot_masquerade_as_complete():
    _, expected = panel()
    data = diagnostics.build_diagnostic_record(identity(), [], expected, attempt_label="empty")
    assert data["status"] == "incomplete"
    assert len(data["missing_coordinates"]) == 24
    assert all(summary["actor_updates"] is None for summary in data["summaries"])


def test_attempt_scope_and_checkpoint_are_part_of_series_identity():
    rows, expected = panel()
    base = record()
    for other_identity, attempt in [(identity(), "another"), (identity("controller_episode"), "scratch-1")]:
        other = diagnostics.build_diagnostic_record(other_identity, rows, expected, attempt_label=attempt)
        assert base["series_id"] != other["series_id"]
    with pytest.raises(ValueError, match="explicit nonempty attempt"):
        diagnostics.build_diagnostic_record(identity(), rows, expected)
    with pytest.raises(ValueError, match="Diagnostic scope"):
        diagnostics.build_diagnostic_record(identity("mixed"), rows, expected, attempt_label="x")


def test_bundle_html_artifacts_round_trip_and_immutability(tmp_path):
    data = record(timing={"optimization_seconds": 1.25, "simulator_seconds": 2.5})
    artifact = tmp_path / "reference.json"
    artifact.write_text('{"root":42}')
    bundle = diagnostics.write_diagnostic_bundle(tmp_path / "bundle", data,
                                                artifact_files={"references/prior.json": artifact})
    restored = diagnostics.read_diagnostic_bundle(bundle)
    assert restored == data
    assert diagnostics.extract_diagnostic_html_data((bundle / "report.html").read_text()) == data
    assert (bundle / "artifacts/references/prior.json").read_bytes() == artifact.read_bytes()
    with pytest.raises(FileExistsError):
        diagnostics.write_diagnostic_bundle(bundle, data)
    with pytest.raises(FileExistsError):
        diagnostics.render_diagnostic_html(data, bundle / "report.html")
    (bundle / "artifacts/references/prior.json").write_text("tampered")
    with pytest.raises(ValueError, match="artifact checksum"):
        diagnostics.read_diagnostic_bundle(bundle)


def test_html_embeds_only_inert_data_and_has_no_network_assets():
    rows, expected = panel()
    data = diagnostics.build_diagnostic_record(identity(), rows, expected,
                                              attempt_label="</script><script>alert(1)</script>")
    html = diagnostics.render_diagnostic_html(data, title="<safe>")
    assert "<title>&lt;safe&gt;</title>" in html
    assert "</script><script>alert(1)</script>" not in html
    assert "<script src=" not in html and "<link " not in html
    assert diagnostics.extract_diagnostic_html_data(html) == data


def test_bundle_completion_timing_is_saved_without_mutating_caller(tmp_path):
    data = record(timing={"optimization_seconds": 1.25})
    original = copy.deepcopy(data)
    directory = diagnostics.write_diagnostic_bundle(tmp_path / "bundle", data, elapsed_before_bundle=2.)
    saved = diagnostics.read_diagnostic_bundle(directory)
    assert data == original
    assert saved["timing"]["calibration_seconds_before_bundle"] == 2.
    assert saved["timing"]["bundle_serialization_seconds"] >= 0
    assert saved["timing"]["report_generation_seconds"] >= 0
    assert saved["timing"]["total_elapsed_seconds"] >= 2.
    assert saved["record_sha256"] != original["record_sha256"]
    assert diagnostics.extract_diagnostic_html_data((directory / "report.html").read_text()) == saved
    completion = json.loads((directory / "completion.json").read_text())
    assert completion["record_sha256"] == saved["record_sha256"]
    assert completion["total_elapsed_seconds"] >= saved["timing"]["total_elapsed_seconds"]
    assert completion["includes_final_metadata_persistence"]
    sdk = FakeWandb()
    receipt = diagnostics.publish_diagnostic_bundle(directory, entity="researcher", wandb_module=sdk)
    assert "completion.json" in sdk.run.artifacts[0].files
    assert sdk.run.summary["runtime/publication_seconds_before_finish"] >= 0
    assert receipt["publication_seconds"] >= sdk.run.summary["runtime/publication_seconds_before_finish"]


def test_row_and_record_corruption_is_detected(tmp_path):
    data = record()
    corrupt = copy.deepcopy(data)
    corrupt["rows"][0]["metrics"]["gain"] = 9
    with pytest.raises(ValueError, match="checksum"):
        diagnostics.write_diagnostic_bundle(tmp_path / "bad", corrupt)
    bundle = diagnostics.write_diagnostic_bundle(tmp_path / "bundle", data)
    with (bundle / "paired-rows.jsonl.gz").open("ab") as handle:
        handle.write(b"bad")
    with pytest.raises(ValueError, match="rows checksum"):
        diagnostics.read_diagnostic_bundle(bundle)


class FakeRun:
    def __init__(self, fail=False):
        self.history, self.axes, self.artifacts, self.summary = [], [], [], {}
        self.fail = fail

    def define_metric(self, *args, **kwargs):
        self.axes.append((args, kwargs))

    def log(self, row):
        self.history.append(row)
        if self.fail:
            raise RuntimeError("connection lost after write")

    def log_artifact(self, artifact):
        self.artifacts.append(artifact)

    def finish(self, **kwargs):
        pass


class FakeArtifact:
    def __init__(self, *args, **kwargs):
        self.files = {}

    def add_file(self, source, name):
        self.files[name] = Path(source).read_bytes()


class FakeWandb:
    Artifact = FakeArtifact

    def __init__(self, fail=False):
        self.run = FakeRun(fail)
        self.calls = []

    def init(self, **kwargs):
        self.calls.append(kwargs)
        return self.run


def test_wandb_roundtrip_preserves_full_rows_and_round_update_axes(tmp_path):
    data = record()
    directory = diagnostics.write_diagnostic_bundle(tmp_path / "bundle", data)
    sdk = FakeWandb()
    receipt = diagnostics.publish_diagnostic_bundle(directory, entity="researcher", wandb_module=sdk)
    assert receipt["status"] == "complete"
    assert sdk.calls[0]["project"] == "ambi-inner-bench"
    assert sdk.calls[0]["id"] == data["series_id"]
    assert sdk.run.history == diagnostics.diagnostic_history(data)
    assert [row["diagnostic/actor_updates"] for row in sdk.run.history] == [0, 3]
    assert all(not key.startswith("checkpoint/") for row in sdk.run.history for key in row)
    artifact = sdk.run.artifacts[0]
    assert diagnostics.extract_diagnostic_html_data(artifact.files["report.html"].decode()) == data
    assert "paired-rows.jsonl.gz" in artifact.files
    assert diagnostics.publish_diagnostic_bundle(directory, entity="researcher", wandb_module=sdk) == receipt
    assert len(sdk.calls) == 1
    with pytest.raises(ValueError, match="target differs"):
        diagnostics.publish_diagnostic_bundle(directory, entity="researcher", mode="online", wandb_module=sdk)


def test_uncertain_publication_is_not_blindly_retried(tmp_path):
    directory = diagnostics.write_diagnostic_bundle(tmp_path / "bundle", record())
    sdk = FakeWandb(fail=True)
    with pytest.raises(RuntimeError, match="connection lost"):
        diagnostics.publish_diagnostic_bundle(directory, entity="researcher", wandb_module=sdk)
    with pytest.raises(ValueError, match="uncertain"):
        diagnostics.publish_diagnostic_bundle(directory, entity="researcher", wandb_module=sdk)
    assert len(sdk.calls) == 1


def test_model_adapter_preserves_probe_identity_and_checks_missing_round(tmp_path):
    rows = [{"episode_id": 0, "root_id": "seed-101-decision-0", "decision_index": 0,
             "solver_repeat": 0, "rollout_repeat": 0, "round_index": j,
             "actor_updates": 3 * j, "critic_updates": 3 * j,
             "metrics": {"togo_return_mean": 12. + j}} for j in range(6)]
    manifest = {"checkpoint": identity()["checkpoint"], "code": identity()["code"],
                "protocol": {"seed_scheme": "paired"},
                "runs": [{"selector": "inner/scratch", "status": "complete", "config": {"alg_params": {"inner_rounds": 5}},
                          "togo_return_probe": {"version": 2, "rollouts": 32}, "togo_probe_rows": rows,
                          "episodes": [{"episode_id": 0, "seed": 101, "length": 1}]}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    data = diagnostics.record_from_model_bundle(tmp_path, "inner/scratch", "model-attempt")
    assert data["identity"]["scope"] == "controller_episode"
    assert data["status"] == "complete"
    assert [row["actor_updates"] for row in data["summaries"]] == [0, 3, 6, 9, 12, 15]
    manifest["runs"][0]["togo_probe_rows"].pop(2)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert diagnostics.record_from_model_bundle(tmp_path, "inner/scratch", "model-attempt")["status"] == "incomplete"
    manifest["runs"][0]["status"] = "failed"
    manifest["checkpoint"] = {"sha256": "a" * 64, "metadata": {"checkpoint": {"step": 2000000}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    failed = diagnostics.record_from_model_bundle(tmp_path, "inner/scratch", "model-attempt")
    assert failed["identity"]["checkpoint"]["step"] == 2000000
    assert failed["source_status"] == "failed"
    assert failed["status"] == "incomplete"
