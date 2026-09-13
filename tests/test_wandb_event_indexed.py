"""One W&B training run preserves diagnostic events sharing an env step."""

import json
from pathlib import Path

import pytest

from utils.wandb_resume import CheckpointedWandbRun, WandbResumeContext
from utils.wandb_utils import (
    INTERNAL_EVENT_INDEX_KEY,
    OUTER_DIAGNOSTIC_UPDATE_KEY,
    EventIndexedWandbRun,
    finish_wandb,
    init_wandb,
    log_wandb,
    publish_outer_policy_diagnostics,
    wandb_enabled,
)


class _Run:
    id = "seed-55"

    def __init__(self, *, strict_steps=True):
        self.rows = []
        self.artifacts = []
        self.finished = False
        self.strict_steps = strict_steps

    def log(self, payload, *, step):
        if self.strict_steps and self.rows:
            assert step > self.rows[-1][0], "A committed W&B step cannot be reused."
        self.rows.append((step, dict(payload)))

    def log_artifact(self, artifact):
        self.artifacts.append(artifact)
        return artifact

    def finish(self):
        self.finished = True


class _Artifact:
    def __init__(self, name, *, type, metadata):
        self.name, self.type, self.metadata = name, type, metadata
        self.files = {}

    def add_dir(self, path):
        root = Path(path)
        self.files = {
            str(p.relative_to(root)): p.read_bytes()
            for p in root.rglob("*") if p.is_file()
        }


class _Wandb:
    Artifact = _Artifact

    def __init__(self, run=None):
        self.run = run or _Run()
        self.defined = []

    def init(self, *, id=None, resume=None, **kwargs):
        return self.run

    def define_metric(self, *args, **kwargs):
        self.defined.append((args, kwargs))

    def Api(self):
        raise AssertionError("These tests must never contact the W&B API.")


def _initialize(wandb, *, event_indexed=True, resume_context=None):
    return init_wandb(
        {
            "wandb": True,
            "wandb_mode": "online" if resume_context else "offline",
            "wandb_event_indexed": event_indexed,
        },
        default_project="ambi", run_name="test", wandb_module=wandb,
        resume_context=resume_context,
    )


def test_repeated_env_steps_keep_all_phases_and_existing_axes():
    wandb = _Wandb()
    run = _initialize(wandb)
    assert isinstance(run, EventIndexedWandbRun)
    for updates in (0, 1, 10, 2500):
        log_wandb(run, {OUTER_DIAGNOSTIC_UPDATE_KEY: updates}, step=2501)
    log_wandb(run, {"train/n_updates": 2999}, step=3000)
    assert [step for step, _ in wandb.run.rows] == list(range(5))
    assert [row["env_step"] for _, row in wandb.run.rows] == [2501] * 4 + [3000]
    assert [row[OUTER_DIAGNOSTIC_UPDATE_KEY] for _, row in wandb.run.rows[:4]] == [0, 1, 10, 2500]
    assert all(row[INTERNAL_EVENT_INDEX_KEY] == step for step, row in wandb.run.rows)
    for name in ("train/*", "eval/*", "episode/*"):
        assert ((name,), {"step_metric": "env_step"}) in wandb.defined
    assert (("outer_diag/*",), {"step_metric": OUTER_DIAGNOSTIC_UPDATE_KEY}) in wandb.defined
    finish_wandb(run)
    assert wandb.run.finished


def test_legacy_run_and_history_steps_are_unchanged():
    wandb = _Wandb(_Run(strict_steps=False))
    run = _initialize(wandb, event_indexed=False)
    assert run is wandb.run
    log_wandb(run, {"train/loss": 1}, step=2501)
    log_wandb(run, {"train/loss": 2}, step=2501)
    assert [step for step, _ in run.rows] == [2501, 2501]
    assert all(INTERNAL_EVENT_INDEX_KEY not in row for _, row in run.rows)
    assert not any(args == ("outer_diag/*",) for args, _ in wandb.defined)


def test_resume_retains_journal_ownership_before_and_after_commit():
    wandb = _Wandb()
    context = WandbResumeContext.new(run_id="seed-55")
    run = _initialize(wandb, resume_context=context)
    assert isinstance(run, CheckpointedWandbRun)
    assert not isinstance(run, EventIndexedWandbRun)
    for updates in (0, 2500):
        log_wandb(run, {OUTER_DIAGNOSTIC_UPDATE_KEY: updates}, step=2501)
    assert wandb.run.rows == []
    assert [event.env_step for event in context.buffer.events] == [2501, 2501]
    run.publish_committed(run.checkpoint_state())
    assert [step for step, _ in wandb.run.rows] == [0, 1]
    assert all(INTERNAL_EVENT_INDEX_KEY not in row for _, row in wandb.run.rows)
    assert (("outer_diag/*",), {"step_metric": OUTER_DIAGNOSTIC_UPDATE_KEY}) in wandb.defined


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_event_flag_rejects_nonbooleans_even_when_wandb_disabled(value):
    with pytest.raises(ValueError, match="wandb_event_indexed"):
        wandb_enabled({"wandb": False, "wandb_event_indexed": value})


def test_invalid_axis_payload_does_not_publish_or_consume_event():
    wandb = _Wandb()
    run = _initialize(wandb)
    for payload in ({"env_step": 99}, {INTERNAL_EVENT_INDEX_KEY: 4}):
        with pytest.raises(ValueError):
            log_wandb(run, payload, step=100)
    assert wandb.run.rows == []
    log_wandb(run, {}, step=100)
    assert wandb.run.rows[0][0] == 0


def _bundle(tmp_path):
    directory = tmp_path / "diagnostics"
    directory.mkdir()
    (directory / "manifest.json").write_text(json.dumps({"schema_version": 1}))
    (directory / "events.jsonl").write_text('{"updates_completed":0}\n')
    (directory / "probes").mkdir()
    (directory / "probes" / "bank.bin").write_bytes(b"fixed-observations")
    return directory


@pytest.mark.parametrize("kind", ["legacy", "event", "resume"])
def test_bundle_artifact_preserves_manifest_and_nested_files(tmp_path, kind):
    directory = _bundle(tmp_path)
    wandb = _Wandb()
    if kind == "resume":
        run = _initialize(wandb, resume_context=WandbResumeContext.new(run_id="seed-55"))
    else:
        run = _initialize(wandb, event_indexed=kind == "event")
    kwargs = {"wandb_module": wandb} if kind == "legacy" else {}
    artifact = publish_outer_policy_diagnostics(run, directory, **kwargs)
    assert artifact.name == "outer-policy-diagnostics-seed-55"
    assert artifact.type == "outer-policy-diagnostics"
    assert artifact.metadata == {"schema_version": 1}
    assert artifact.files["events.jsonl"] == b'{"updates_completed":0}\n'
    assert artifact.files["probes/bank.bin"] == b"fixed-observations"
    assert json.loads(artifact.files["manifest.json"]) == artifact.metadata
    assert wandb.run.artifacts == [artifact]
    assert wandb.run.rows == []
    assert not wandb.run.finished


def test_publication_failure_propagates_and_does_not_finish_run(tmp_path):
    directory = _bundle(tmp_path)
    wandb = _Wandb()
    run = _initialize(wandb)

    def fail(_artifact):
        raise OSError("injected artifact upload failure")

    wandb.run.log_artifact = fail
    with pytest.raises(OSError, match="artifact upload"):
        publish_outer_policy_diagnostics(run, directory)
    assert not wandb.run.finished


def test_bad_bundle_is_rejected_before_publication(tmp_path):
    directory = _bundle(tmp_path)
    (directory / "manifest.json").write_text("[]")
    wandb = _Wandb()
    run = _initialize(wandb)
    with pytest.raises(ValueError, match="JSON object"):
        publish_outer_policy_diagnostics(run, directory)
    assert wandb.run.artifacts == []
    assert publish_outer_policy_diagnostics(None, tmp_path / "absent") is None
