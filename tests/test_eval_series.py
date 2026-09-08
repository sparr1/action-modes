"""Publication failures must never require repeating scientific evaluation."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from utils import eval_series as series


def record(tmp_path, step=100_000, **changes):
    source = tmp_path / (str(step) + ".json")
    if not source.exists():
        source.write_text(json.dumps({"step": step, "episodes": [1, 3]}))
    result = {
        "identity": {"backbone": "entity/train/prior55", "planner": {"operator": "sac", "critic_updates": 6}, "protocol": {"seeds": [101, 102], "max_steps": 500}, "science": {"evaluator": "frozen-v1"}},
        "checkpoint": {"step": step, "sha256": "a" * 64},
        "metrics": {"eval/return_mean": 2.0, "eval/return_sample_std": 2 ** 0.5, "eval/episode_count": 2, "runtime/control_seconds": 1.2},
        "episodes": [{"seed": 101, "return": 1}, {"seed": 102, "return": 3}],
        "artifact_files": {"result.json": str(source)}, "provenance": {"code_sha": "abc"},
        "record_id": "record-" + str(step), "source_result_path": str(source), "label": "Prior55 | SAC C6",
    }
    result.update(changes)
    return result


def create(tmp_path, value=None):
    return series.create_run(tmp_path / "registry", value or record(tmp_path), "attempt 1", "eval", "entity", "oscar-owner")


def index(registry):
    return json.loads((Path(registry["run_dir"]) / "publication.json").read_text())


class FakeArtifact:
    def __init__(self, name, **kwargs):
        self.name = name
        self.files = {}

    def add_file(self, path, name):
        self.files[name] = Path(path).read_bytes()

    def wait(self):
        return self


class FakeRun:
    def __init__(self, backend, run_id, config):
        self.backend = backend
        self.id = run_id
        self.config = config
        self.rows = []
        self.pending = []
        self.step = 0
        self.definitions = []
        self.artifacts = []
        self.used_artifacts = []
        self.settings = SimpleNamespace(mode="online")

    def define_metric(self, name, **kwargs):
        self.definitions.append((name, kwargs))

    def log_artifact(self, artifact, aliases):
        if self.backend.fail_artifact:
            raise RuntimeError("artifact network failure")
        self.artifacts.append((artifact, aliases))
        return artifact

    def use_artifact(self, name):
        self.used_artifacts.append(name)

    def log(self, row, step, commit):
        assert commit
        if self.backend.fail_before_log:
            raise RuntimeError("crash before SDK accepts row")
        assert step >= self.step
        self.pending.append(dict(row, _step=step))
        self.step = step + 1
        if self.backend.fail_after_log:
            raise RuntimeError("crash after SDK accepts row")

    def finish(self, exit_code=0):
        if self.backend.flush:
            self.rows.extend(self.pending)
            self.pending.clear()

    def scan_history(self, page_size):
        if self.backend.fail_history:
            raise RuntimeError("history temporarily unavailable")
        return iter(deepcopy(self.rows))


class FakeWandb:
    Artifact = FakeArtifact

    def __init__(self):
        self.runs = {}
        self.init_calls = []
        self.fail_artifact = self.fail_before_log = self.fail_after_log = self.fail_history = False
        self.flush = True

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        rid = kwargs["id"]
        if rid not in self.runs:
            assert kwargs["resume"] in ("never", "allow")
            self.runs[rid] = FakeRun(self, rid, kwargs["config"])
        else:
            assert kwargs["resume"] in ("must", "allow")
        return self.runs[rid]

    def Api(self, timeout):
        return SimpleNamespace(run=lambda path: self.runs[path.split("/")[-1]])


def publish(registry, backend):
    return series.publish_run(registry["run_dir"], owner="oscar-owner", wandb_module=backend, acknowledgement_timeout=0)


def test_new_attempt_always_allocates_distinct_id_before_results(tmp_path):
    value = record(tmp_path)
    template = {"identity": value["identity"], "label": value["label"]}
    first, second = create(tmp_path, template), create(tmp_path, template)
    assert first["run_id"] != second["run_id"]
    assert len(first["run_id"]) == 32
    assert index(first)["records"] == {}
    assert series.load_run(first["run_dir"]) == first


@pytest.mark.parametrize("part", ["backbone", "planner", "protocol", "science"])
def test_incompatible_append_rejected_before_sdk(tmp_path, part):
    registry = create(tmp_path)
    value = record(tmp_path)
    value["identity"][part] = "another" if part == "backbone" else {"changed": True}
    with pytest.raises(series.SeriesError, match="Incompatible append"):
        series.stage_record(registry["run_dir"], value)
    assert not index(registry)["records"]


def test_duplicate_retry_does_not_create_second_point(tmp_path):
    registry = create(tmp_path)
    value = record(tmp_path)
    assert series.stage_record(registry["run_dir"], value)["status"] == "staged"
    assert series.stage_record(registry["run_dir"], value)["status"] == "already_staged"
    backend = FakeWandb()
    assert publish(registry, backend)["published"] == 1
    assert series.stage_record(registry["run_dir"], value)["status"] == "published"
    publish(registry, backend)
    assert len(backend.runs[registry["run_id"]].rows) == 1


@pytest.mark.parametrize("change", ["record_id", "metrics", "checkpoint_hash", "artifact"])
def test_conflicting_checkpoint_requires_new_attempt(tmp_path, change):
    registry = create(tmp_path)
    value = record(tmp_path)
    series.stage_record(registry["run_dir"], value)
    if change == "record_id":
        value["record_id"] = "independent-repeat"
    elif change == "metrics":
        value["metrics"]["eval/return_mean"] = 9
    elif change == "checkpoint_hash":
        value["checkpoint"]["sha256"] = "b" * 64
    else:
        Path(value["source_result_path"]).write_text("changed")
    with pytest.raises(series.SeriesError, match="different accepted result"):
        series.stage_record(registry["run_dir"], value)


def test_out_of_order_checkpoint_axis_and_internal_steps(tmp_path):
    registry = create(tmp_path)
    for step in (300_000, 100_000, 200_000, 400_000):
        series.stage_record(registry["run_dir"], record(tmp_path, step))
    backend = FakeWandb()
    publish(registry, backend)
    remote = backend.runs[registry["run_id"]]
    assert [row[series.X_AXIS] for row in remote.rows] == [300_000, 100_000, 200_000, 400_000]
    assert [row["_step"] for row in remote.rows] == [0, 1, 2, 3]
    assert ("*", {"step_metric": series.X_AXIS, "step_sync": False, "hidden": True}) in remote.definitions
    assert ("eval/return_mean", {"step_metric": series.X_AXIS, "step_sync": False, "hidden": False}) in remote.definitions
    assert len(remote.artifacts) == 4
    assert all("evaluation-series-record.json" in item[0].files for item in remote.artifacts)


def test_open_publisher_reuses_sdk_and_does_not_resend_queued(tmp_path):
    registry = create(tmp_path)
    backend = FakeWandb()
    with series.Publisher(registry["run_dir"], wandb_module=backend) as publisher:
        series.stage_record(registry["run_dir"], record(tmp_path, 300_000))
        assert publisher.publish_pending()["queued"] == 1
        publisher.publish_pending()
        series.stage_record(registry["run_dir"], record(tmp_path, 100_000))
        assert publisher.publish_pending()["queued"] == 2
    assert len(backend.init_calls) == 1
    assert len(backend.runs[registry["run_id"]].rows) == 2


def test_crash_before_row_log_reuses_assigned_slot(tmp_path):
    registry = create(tmp_path)
    series.stage_record(registry["run_dir"], record(tmp_path))
    backend = FakeWandb()
    backend.fail_before_log = True
    with pytest.raises(RuntimeError, match="before SDK"):
        publish(registry, backend)
    assert index(registry)["records"]["record-100000"]["status"] == "row_inflight"
    backend.fail_before_log = False
    assert publish(registry, backend)["published"] == 1
    assert backend.runs[registry["run_id"]].rows[0]["_step"] == 0


def test_crash_after_row_log_reconciles_without_duplicate(tmp_path):
    registry = create(tmp_path)
    series.stage_record(registry["run_dir"], record(tmp_path))
    backend = FakeWandb()
    backend.fail_after_log = True
    with pytest.raises(RuntimeError, match="after SDK"):
        publish(registry, backend)
    backend.fail_after_log = False
    assert publish(registry, backend)["published"] == 1
    assert len(backend.runs[registry["run_id"]].rows) == 1


def test_crash_after_remote_receipt_before_local_index_recovers(tmp_path, monkeypatch):
    registry = create(tmp_path)
    series.stage_record(registry["run_dir"], record(tmp_path))
    backend = FakeWandb()
    original = series._atomic_json

    def fail_receipt(path, value):
        if Path(path).name == "publication.json" and any(entry["status"] == "published" for entry in value.get("records", {}).values()):
            raise OSError("local receipt disk failure")
        return original(path, value)

    monkeypatch.setattr(series, "_atomic_json", fail_receipt)
    with pytest.raises(OSError, match="receipt disk"):
        publish(registry, backend)
    assert index(registry)["records"]["record-100000"]["status"] == "queued"
    assert len(backend.runs[registry["run_id"]].rows) == 1
    monkeypatch.setattr(series, "_atomic_json", original)
    assert publish(registry, backend)["published"] == 1
    assert len(backend.runs[registry["run_id"]].rows) == 1


def test_uncertain_remote_visibility_fails_closed_then_recovers(tmp_path):
    registry = create(tmp_path)
    series.stage_record(registry["run_dir"], record(tmp_path))
    backend = FakeWandb()
    backend.flush = False
    with pytest.raises(series.PublicationUncertainError, match="acknowledged"):
        publish(registry, backend)
    with pytest.raises(series.PublicationUncertainError, match="slot may be occupied"):
        publish(registry, backend)
    remote = backend.runs[registry["run_id"]]
    assert len(remote.pending) == 1
    backend.flush = True
    remote.finish()
    assert publish(registry, backend)["published"] == 1
    assert len(remote.rows) == 1


def test_network_failure_does_not_modify_scientific_result(tmp_path):
    registry = create(tmp_path)
    value = record(tmp_path)
    series.stage_record(registry["run_dir"], value)
    before = Path(value["source_result_path"]).read_bytes()
    backend = FakeWandb()
    backend.fail_artifact = True
    with pytest.raises(RuntimeError, match="artifact network"):
        publish(registry, backend)
    assert Path(value["source_result_path"]).read_bytes() == before
    assert index(registry)["records"][value["record_id"]]["status"] == "staged"
    backend.fail_artifact = False
    assert publish(registry, backend)["published"] == 1


def test_remote_history_failure_never_retransmits(tmp_path):
    registry = create(tmp_path)
    series.stage_record(registry["run_dir"], record(tmp_path))
    backend = FakeWandb()
    publish(registry, backend)
    backend.fail_history = True
    with pytest.raises(series.PublicationUncertainError, match="verify remote history"):
        publish(registry, backend)
    assert len(backend.init_calls) == 1


@pytest.mark.parametrize("corruption", ["duplicate", "unknown", "changed_hash", "changed_identity", "removed"])
def test_remote_divergence_rejected(tmp_path, corruption):
    registry = create(tmp_path)
    series.stage_record(registry["run_dir"], record(tmp_path))
    backend = FakeWandb()
    publish(registry, backend)
    remote = backend.runs[registry["run_id"]]
    if corruption == "duplicate":
        remote.rows.append(dict(remote.rows[0]))
    elif corruption == "unknown":
        remote.rows[0][series.RECORD_KEY] = "unknown"
    elif corruption == "changed_hash":
        remote.rows[0][series.HASH_KEY] = "x"
    elif corruption == "changed_identity":
        remote.config["evaluation_identity_sha256"] = "x"
    else:
        remote.rows.clear()
    with pytest.raises(series.SeriesError):
        publish(registry, backend)


def test_exclusive_owner_and_shared_filesystem_lock(tmp_path):
    registry = create(tmp_path)
    with pytest.raises(series.SeriesError, match="registered publication owner"):
        series.Publisher(registry["run_dir"], owner="different-host")
    with series.Publisher(registry["run_dir"], wandb_module=FakeWandb()):
        with pytest.raises(series.SeriesError, match="already owns"):
            with series.Publisher(registry["run_dir"], wandb_module=FakeWandb()):
                pass


def test_copied_registry_is_not_a_second_publication_owner(tmp_path):
    registry = create(tmp_path)
    copied = tmp_path / "copied"
    copied.mkdir()
    (copied / "run.json").write_text(json.dumps(registry))
    with pytest.raises(series.SeriesError, match="authoritative owner directory"):
        series.load_run(copied)


def test_nonfinite_and_missing_remain_distinct(tmp_path):
    registry = create(tmp_path)
    value = record(tmp_path)
    value["metrics"].update({"diagnostic/missing": None, "diagnostic/nonfinite": {"nonfinite": "nan"}})
    series.stage_record(registry["run_dir"], value)
    backend = FakeWandb()
    publish(registry, backend)
    row = backend.runs[registry["run_id"]].rows[0]
    assert row["diagnostic/missing"] is None
    assert "diagnostic/nonfinite" not in row
    assert row["measurement_status/diagnostic/nonfinite"] == "nan"
    value["metrics"]["bad"] = float("nan")
    with pytest.raises(series.SeriesError, match="finite JSON"):
        series.validate_record(value)


def test_accepted_artifact_must_remain_immutable(tmp_path):
    registry = create(tmp_path)
    value = record(tmp_path)
    series.stage_record(registry["run_dir"], value)
    Path(value["source_result_path"]).write_text("changed")
    with pytest.raises(series.SeriesError, match="changed before publication"):
        publish(registry, FakeWandb())


def test_legacy_artifact_dependency_is_linked(tmp_path):
    registry = create(tmp_path)
    value = record(tmp_path)
    value["provenance"]["legacy_artifacts"] = ["entity/project/full-trace:v0"]
    series.stage_record(registry["run_dir"], value)
    backend = FakeWandb()
    publish(registry, backend)
    assert backend.runs[registry["run_id"]].used_artifacts == ["entity/project/full-trace:v0"]


def test_worker_pointer_is_atomic_idempotent_and_does_not_load_sdk(tmp_path, monkeypatch):
    registry = create(tmp_path)
    source = record(tmp_path)["source_result_path"]
    first = series.stage_result(registry["run_dir"], source, selector="prior")
    assert series.stage_result(registry["run_dir"], source, selector="prior") == first
    assert len(list((Path(registry["run_dir"]) / "incoming").glob("*.json"))) == 1
    assert index(registry)["records"] == {}


def test_cpu_publisher_resolves_pointer_and_selects_controller(tmp_path, monkeypatch):
    value = record(tmp_path)
    value["controller"] = "native_mppi"
    registry = create(tmp_path, value)
    series.stage_result(registry["run_dir"], value["source_result_path"], selector="native_mppi", format="tdmpc2-paired", source_run=value["identity"]["backbone"])
    calls = []

    def load_records(path, **kwargs):
        calls.append((path, kwargs))
        return [dict(value, controller="policy_prior", record_id="prior"), value]

    monkeypatch.setitem(sys.modules, "utils.eval_series_data", SimpleNamespace(load_records=load_records))
    backend = FakeWandb()
    assert publish(registry, backend)["published"] == 1
    assert calls[0][1]["source_run"] == value["identity"]["backbone"]
    assert list((Path(registry["run_dir"]) / "incoming").glob("*.json")) == []
    assert len(list((Path(registry["run_dir"]) / "accepted-pointers").glob("*.json"))) == 1


def test_copied_artifact_bytes_are_idempotent(tmp_path):
    registry = create(tmp_path)
    value = record(tmp_path)
    series.stage_record(registry["run_dir"], value)
    copied = tmp_path / "copied.json"
    copied.write_bytes(Path(value["source_result_path"]).read_bytes())
    value["source_result_path"] = str(copied)
    value["artifact_files"] = {"result.json": str(copied)}
    assert series.stage_record(registry["run_dir"], value)["status"] == "already_staged"


@pytest.mark.parametrize("name", ["../secret", "/absolute", "a/../../secret", "bad\\path", "evaluation-series-record.json"])
def test_artifact_paths_cannot_escape_bundle(tmp_path, name):
    value = record(tmp_path)
    value["artifact_files"] = {name: value["source_result_path"]}
    with pytest.raises(series.SeriesError, match="safe relative paths"):
        series.validate_record(value)


def label_registry(**settings):
    return {"identity": {"backbone": "rwgao_b-brown-university/ambi/u13m14st", "planner": {
        "type": "sac", "settings": {"inner_rounds": 6, "inner_actor_updates_per_action": 72,
        "inner_critic_updates_per_action": 36, "inner_temperature_updates_per_action": 18,
        "inner_actor_lr": 5e-5, "inner_bootstrap_source": "outer_target", **settings}}},
        "attempt_label": "actor-sweep-20260905", "run_id": "abcd0000111122223333444455556666"}


def test_curve_label_prioritizes_budgets_and_bootstrap():
    registry = label_registry()
    label = series.concise_curve_label(registry)
    assert label == "AMBI original · SAC C6/A12/T3 outer Q · #abcd"
    assert len(label) <= 60
    assert registry["attempt_label"] == "actor-sweep-20260905"


def test_curve_label_aliases_and_cosmetic_names_do_not_change_semantics():
    registry = label_registry()
    original = series.concise_curve_label(registry)
    registry.update(name="renamed arbitrary checkpoint campaign", label="obsolete alias")
    registry["identity"]["planner"]["selector"] = "a/different_alias"
    assert series.concise_curve_label(registry) == original


def test_curve_label_distinguishes_repeated_attempts_with_equal_names():
    first, second = label_registry(), label_registry()
    second["run_id"] = "efgh0000111122223333444455556666"
    assert series.concise_curve_label(first) != series.concise_curve_label(second)
    assert first["attempt_label"] == second["attempt_label"]


def test_curve_label_retains_interval_temperature_and_reduced_learning_rate():
    registry = label_registry(inner_critic_updates_per_action=72, inner_temperature_updates_per_action=72,
                              inner_steps_per_update=128, inner_actor_lr=2.5e-5)
    registry["attempt_label"] = "interval-sweep-20260906"
    label = series.concise_curve_label(registry)
    assert label.startswith("AMBI original · SAC C12/A12/T12 outer Q s128 aLR2.5e-5")
    assert len(label) <= 70


def test_curve_label_marks_xqc_terminal_and_nondefault_collection():
    registry = label_registry(inner_terminal_bootstrap="outer", inner_actor_updates_per_action=6,
                              inner_critic_updates_per_action=18, inner_temperature_updates_per_action=6,
                              inner_rollouts_per_round=256)
    registry["identity"]["backbone"] = "rwgao_b-brown-university/ambi/axqc-prior-92441d99-5959199"
    registry["identity"]["planner"]["type"] = "xqc"
    label = series.concise_curve_label(registry)
    assert label.startswith("AMBI-XQC · XQC C3/A1/T1 outer-term round N256")
    assert "92441d99" not in label


def test_xqc_step_curve_label_distinguishes_matched_optimizer_dose():
    registry = label_registry(inner_terminal_bootstrap="outer", inner_actor_updates_per_action=6,
                              inner_critic_updates_per_action=18, inner_temperature_updates_per_action=6)
    registry["identity"]["planner"]["type"] = "xqc"
    assert "outer-term round" in series.concise_curve_label(registry)
    registry["identity"]["planner"]["settings"].update(inner_update_timing="step",inner_policy_delay=1,
        inner_actor_updates_per_action=18,inner_temperature_updates_per_action=18)
    assert "C3/A3/T3 outer-term step" in series.concise_curve_label(registry)
    assert "step updates" in series.evaluation_run_name(registry)


def test_prior_label_excludes_inactive_settings_and_mppi_shows_changed_budget():
    registry = label_registry()
    registry["identity"]["planner"]["type"] = "prior"
    assert series.concise_curve_label(registry).startswith("AMBI original · Prior only (no planning)")
    registry["identity"]["planner"] = {"type": "mppi", "settings": {
        "horizon": 5, "effective_iterations": 12, "num_samples": 1024}}
    assert series.concise_curve_label(registry).startswith("AMBI original · MPPI H5 N1024 I12")


@pytest.mark.parametrize("source,expected", [
    ("u13m14st", "Original AMBI prior-only backbone"),
    ("axqc-prior-92441d99-5959199", "AMBI-XQC prior-only backbone"),
    ("xq3zva9u", "TD-MPC2 prior-only backbone"),
])
def test_run_name_separates_prior_controller_from_mppi_attempt(source, expected):
    registry = label_registry()
    registry["identity"]["backbone"] = "rwgao_b-brown-university/ambi/" + source
    registry["identity"]["planner"]["type"] = "prior"
    registry["attempt_label"] = "native-mppi-20260906"
    before = deepcopy(registry)
    assert series.evaluation_run_name(registry) == (
        expected + " | Prior only (no planning) | Attempt: MPPI comparison [abcd]")
    assert "MPPI" not in series.concise_curve_label(registry)
    assert registry == before


def test_unknown_source_never_uses_known_backbone_alias():
    first = label_registry()
    first["identity"]["backbone"] = "another-entity/ambi/u13m14st"
    second = deepcopy(first)
    second["identity"]["backbone"] = "another-entity/ambi/u13m14sx"
    assert "another-entity/ambi/u13m14st" in series.concise_curve_label(first)
    assert series.concise_curve_label(first) != series.concise_curve_label(second)
    assert "Original AMBI" not in series.evaluation_run_name(first)


def test_new_run_uses_resolved_controller_not_legacy_result_label(tmp_path):
    value = record(tmp_path)
    value["identity"]["backbone"] = "rwgao_b-brown-university/ambi/u13m14st"
    value["identity"]["planner"] = {"type": "prior", "settings": {"unused": 99}}
    value["label"] = "outdated MPPI alias"
    registry = series.create_run(tmp_path / "registry", value, "native-mppi-20260906", "eval", "entity", "oscar-owner")
    assert registry["name"] == series.evaluation_run_name(registry)
    assert registry["name"].startswith("Original AMBI prior-only backbone | Prior only (no planning) | Attempt:")
    assert registry["attempt_label"] == "native-mppi-20260906"
    assert registry["identity"] == value["identity"]


def test_resumed_publisher_refreshes_presentation_without_changing_identity(tmp_path):
    registry = create(tmp_path)
    registry["name"] = "obsolete alias"
    series._atomic_json(Path(registry["run_dir"]) / "run.json", registry)
    backend = FakeWandb()
    with series.Publisher(registry["run_dir"], wandb_module=backend):
        pass
    assert backend.init_calls[0]["name"] == series.evaluation_run_name(registry)
    assert backend.init_calls[0]["config"]["evaluation_identity_sha256"] == registry["identity_sha256"]
    assert backend.init_calls[0]["config"]["attempt_label"] == registry["attempt_label"]
