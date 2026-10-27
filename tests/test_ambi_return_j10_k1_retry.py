"""Delayed acknowledgment is reconciled without duplicate rows or new run IDs."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from slurm import ambi_return_j10_k1_publish as publish
from utils import eval_series as series
import test_ambi_return_prior_publication as original_test


class DelayedSDK:
    def __init__(self):
        self.visible = False
        self.rows, self.inits, self.artifacts = [], [], []
        self.config = {}

    def init(self, **kwargs):
        self.inits.append(kwargs)
        self.config.update(kwargs["config"])
        sdk = self
        class Run:
            config = sdk.config
            summary = {}
            step = len(sdk.rows)
            def define_metric(self, *args, **kwargs): pass
            def log_artifact(self, artifact, **kwargs):
                sdk.artifacts.append(artifact)
                return SimpleNamespace(wait=lambda: None)
            def log(self, row, step, commit):
                assert commit and step == len(sdk.rows)
                sdk.rows.append({**row, "_step": step})
            def finish(self, **kwargs): pass
        return Run()

    def Artifact(self, *args, **kwargs):
        return SimpleNamespace(add_file=lambda *args, **kwargs: None)

    def Api(self, **kwargs):
        return SimpleNamespace(run=lambda path: SimpleNamespace(config=self.config,
            scan_history=lambda **kwargs: list(self.rows) if self.visible else []))


def staged_row(tmp_path, monkeypatch):
    entry = {**original_test.inventory()["entries"][1], "J": 10, "K": 1}
    record = original_test.record(tmp_path, entry, "sac", [10, 20, 30, 40, 50])
    record["selector"] = publish.SELECTOR
    registry = series.create_run(tmp_path / "runs", record, publish.ATTEMPT,
        publish.prior.PROJECT, publish.prior.ENTITY, publish.prior.OWNER)
    series.stage_record(registry["run_dir"], record)
    # Exercise the real durable Publisher; replace only its wait budget in tests.
    original = series.Publisher
    class ImmediateTimeoutPublisher(original):
        def __init__(self, *args, **kwargs):
            assert kwargs.pop("acknowledgement_timeout") == 60
            super().__init__(*args, acknowledgement_timeout=0, **kwargs)
    monkeypatch.setattr(series, "Publisher", ImmediateTimeoutPublisher)
    row = {**registry, "name": "Train J8 K3 @ 500k | SAC eval J10 H3 K1", "source_J": 8,
        "index": 1, "complete": 1}
    return row, record


def test_retry_reconciles_delayed_history_with_same_identity_and_one_row(tmp_path, monkeypatch):
    row, record = staged_row(tmp_path, monkeypatch)
    sdk, delays = DelayedSDK(), []
    def sleep(delay):
        delays.append(delay)
        sdk.visible = True
    publish.publish_curve(row, False, wandb_module=sdk, sleep=sleep)
    assert delays == [15]
    assert len(sdk.rows) == len(sdk.artifacts) == 1
    assert [call["resume"] for call in sdk.inits] == ["never", "must"]
    assert {call["id"] for call in sdk.inits} == {row["run_id"]}
    journal = series._read(Path(row["run_dir"]) / "publication.json")
    assert journal["records"][record["record_id"]]["status"] == "published"
    assert sdk.rows[0][series.RECORD_KEY] == record["record_id"]


def test_three_attempts_exhaust_without_blind_relog_or_journal_reset(tmp_path, monkeypatch):
    row, record = staged_row(tmp_path, monkeypatch)
    sdk, delays = DelayedSDK(), []
    with pytest.raises(series.PublicationUncertainError, match="no duplicate was sent"):
        publish.publish_curve(row, False, wandb_module=sdk, sleep=delays.append)
    assert delays == [15, 30]
    assert len(sdk.inits) == 3
    assert len(sdk.rows) == len(sdk.artifacts) == 1
    journal = series._read(Path(row["run_dir"]) / "publication.json")
    assert journal["records"][record["record_id"]]["status"] == "queued"
    assert journal["records"][record["record_id"]]["wandb_step"] == 0


def test_incompatible_remote_identity_is_not_retried(tmp_path, monkeypatch):
    row, _ = staged_row(tmp_path, monkeypatch)
    sdk, delays = DelayedSDK(), []
    def sleep(delay):
        delays.append(delay)
        sdk.config["evaluation_identity_sha256"] = "incompatible"
    with pytest.raises(series.SeriesError, match="incompatible evaluation identity"):
        publish.publish_curve(row, False, wandb_module=sdk, sleep=sleep)
    assert delays == [15]
    assert len(sdk.inits) == len(sdk.rows) == 1


def test_nonpublication_failure_is_not_retried(tmp_path, monkeypatch):
    row, _ = staged_row(tmp_path, monkeypatch)
    sdk, delays = DelayedSDK(), []
    def broken_init(**kwargs):
        sdk.inits.append(kwargs)
        raise RuntimeError("bad credentials")
    sdk.init = broken_init
    with pytest.raises(RuntimeError, match="bad credentials"):
        publish.publish_curve(row, False, wandb_module=sdk, sleep=delays.append)
    assert not delays and len(sdk.inits) == 1 and not sdk.rows
