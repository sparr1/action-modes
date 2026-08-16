import copy
import json
import os
import signal
import sys
from pathlib import Path

import pytest
import torch

import utils.resume_training as resume_training
from tests.resume_test_support import BoundaryEnv, _FakeWandb, _model, _session
from utils.resume_lineage import GenerationFile, LineageStore
from utils.resume_training import (
    RESUME_COMPLETE,
    RESUME_HANDOFF,
    ResumeIncompatibilityError,
    ResumeStateCorruptionError,
    ResumeStorageError,
)
from utils.wandb_resume import WandbRemoteWriteError


def _close(model, session):
    if getattr(model, "_wandb_run", None) is not None:
        session.abort_wandb(model, RuntimeError("test cleanup"))
    session.close()
    model._checkpoint_writer.shutdown()


def _torch_writer(payload):
    def writer(path):
        with path.open("xb") as stream:
            torch.save(payload, stream)

    return writer


def _copy_writer(source):
    def writer(path):
        with source.open("rb") as incoming, path.open("xb") as outgoing:
            while block := incoming.read(1024 * 1024):
                outgoing.write(block)

    return writer


def test_segment_directory_entries_are_synced_inside_out(monkeypatch, tmp_path):
    synced = []
    monkeypatch.setattr(
        resume_training,
        "_fsync_directory",
        lambda path: synced.append(Path(path)),
    )
    lineage = tmp_path / "directory-order"
    session = _session(lineage, mode="new", segment="part-1")
    try:
        assert synced == [
            session.segment_dir,
            session.segment_dir.parent,
            lineage,
        ]
    finally:
        session.close()


def test_evaluation_flush_syncs_its_segment_directory(monkeypatch, tmp_path):
    model = _model(BoundaryEnv())
    session = _session(tmp_path / "eval-directory", mode="new", segment="part-1")
    session.eval_csv_path.write_text("step,reward\n")
    synced = []
    monkeypatch.setattr(
        resume_training,
        "_fsync_directory",
        lambda path: synced.append(Path(path)),
    )
    try:
        session._flush_outputs(model)
        assert synced == [session.segment_dir]
    finally:
        _close(model, session)


def test_evaluation_directory_sync_failure_is_typed_and_cannot_publish(
    monkeypatch, tmp_path
):
    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    lineage = tmp_path / "eval-directory-failure"
    model = _model(BoundaryEnv())
    session = _session(lineage, mode="new", segment="part-1")
    try:
        session.prepare_learner(model)
        durable = session.last_generation
        session.eval_csv_path.write_text("step,reward\n")

        def fail_directory_sync(path):
            assert Path(path) == session.segment_dir
            raise ResumeStorageError("injected evaluation directory sync failure")

        monkeypatch.setattr(
            resume_training, "_fsync_directory", fail_directory_sync
        )
        with pytest.raises(ResumeStorageError, match="injected evaluation"):
            session.publish(model, reason="must-not-publish")

        assert session.last_generation.generation_id == durable.generation_id
        assert (lineage / "LATEST").read_text().strip() == durable.generation_id
    finally:
        _close(model, session)


def test_normal_segments_share_one_physical_wandb_run_and_checkpoint_before_upload(
    monkeypatch, tmp_path
):
    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    lineage = tmp_path / "one-run"

    first_model = _model(BoundaryEnv())
    first = _session(lineage, mode="new", segment="part-1")
    try:
        first.prepare_learner(first_model)
        run_id = first_model._wandb_run.checkpoint_state()["run_id"]
        genesis = first.last_generation
        saved = torch.load(
            genesis.files_for_role("trainer")[0],
            map_location="cpu",
            weights_only=False,
        )["wandb"]
        assert saved == first_model._wandb_run.checkpoint_state()
        assert set(fake_wandb.runs) == {run_id}

        first_model._wandb_run.log({"train/manual": 1.0}, env_step=0)
        assert len(fake_wandb.runs[run_id].history) == len(saved["events"])
        committed = first.publish(first_model, reason="test-boundary")
        assert len(fake_wandb.runs[run_id].history) == len(
            first_model._wandb_run.checkpoint_state()["events"]
        )
        assert first.clean_handoff(first_model, committed) == RESUME_HANDOFF
    finally:
        _close(first_model, first)

    second_model = _model(BoundaryEnv())
    second = _session(lineage, mode="required", segment="part-2")
    try:
        second.prepare_learner(second_model)
        assert second_model._wandb_run.checkpoint_state()["run_id"] == run_id
        assert set(fake_wandb.runs) == {run_id}
    finally:
        _close(second_model, second)


def test_committed_upload_failure_keeps_latest_and_is_healed_on_required_resume(
    monkeypatch, tmp_path
):
    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    lineage = tmp_path / "upload-failure"
    model = _model(BoundaryEnv())
    session = _session(lineage, mode="new", segment="part-1")
    try:
        session.prepare_learner(model)
        run_id = model._wandb_run.checkpoint_state()["run_id"]
        remote = fake_wandb.runs[run_id]
        original_log = remote.log

        def fail_log(_payload, _step):
            raise OSError("injected upload failure")

        remote.log = fail_log
        model._wandb_run.log({"train/manual": 1.0}, env_step=0)
        with pytest.raises(WandbRemoteWriteError):
            session.publish(model, reason="locally-durable")
        durable = session.last_generation
        assert (lineage / "LATEST").read_text().strip() == durable.generation_id
        assert not (lineage / "HANDOFF.json").exists()
        remote.log = original_log
    finally:
        _close(model, session)

    restored_model = _model(BoundaryEnv())
    restored = _session(lineage, mode="required", segment="part-2")
    try:
        restored.prepare_learner(restored_model)
        state = restored_model._wandb_run.checkpoint_state()
        assert len(fake_wandb.runs[run_id].history) == len(state["events"])
    finally:
        _close(restored_model, restored)


def test_explicit_previous_generation_immediately_branches_with_a_fresh_wandb_id(
    monkeypatch, tmp_path
):
    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    lineage = tmp_path / "rollback"
    model = _model(BoundaryEnv())
    session = _session(lineage, mode="new", segment="source")
    try:
        session.prepare_learner(model)
        selected = session.last_generation
        original_id = model._wandb_run.checkpoint_state()["run_id"]
        latest = session.publish(model, reason="newer")
        session.clean_handoff(model, latest)
    finally:
        _close(model, session)

    branch_model = _model(BoundaryEnv())
    branch = _session(
        lineage,
        mode="required",
        segment="rollback-branch",
        generation=selected.generation_id,
    )
    try:
        branch.prepare_learner(branch_model)
        branched = branch.last_generation
        branch_id = branch_model._wandb_run.checkpoint_state()["run_id"]
        assert branch_id != original_id
        assert set(fake_wandb.runs) == {original_id, branch_id}
        assert branched.parent_generation == selected.generation_id
        assert branched.metadata["reason"] == "operator-rollback-branch"
        assert (lineage / "LATEST").read_text().strip() == branched.generation_id
    finally:
        _close(branch_model, branch)


def test_explicit_latest_is_rejected_before_wandb_can_create_a_branch(
    monkeypatch, tmp_path
):
    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    lineage = tmp_path / "not-a-rollback"
    model = _model(BoundaryEnv())
    session = _session(lineage, mode="new", segment="source")
    try:
        session.prepare_learner(model)
        latest = session.last_generation
        original_runs = set(fake_wandb.runs)
    finally:
        _close(model, session)

    with pytest.raises(ResumeIncompatibilityError, match="current LATEST"):
        _session(
            lineage,
            mode="required",
            segment="invalid-explicit-latest",
            generation=latest.generation_id,
        )
    assert set(fake_wandb.runs) == original_runs
    assert not (lineage / "segments" / "invalid-explicit-latest").exists()


def test_replay_progress_corruption_fails_before_wandb_resume(monkeypatch, tmp_path):
    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    lineage = tmp_path / "semantic-corruption"
    holder = {}

    def drain_after_episode():
        holder["session"]._drain_requested = True

    model = _model(BoundaryEnv(on_first_done=drain_after_episode))
    session = _session(lineage, mode="new", segment="source")
    holder["session"] = session
    try:
        assert model.learn(total_timesteps=4, resume_session=session) == RESUME_HANDOFF
    finally:
        _close(model, session)

    with LineageStore.open(lineage, mode="required") as store:
        source = store.load()
        child_id = "semantically-corrupt"
        trainer = torch.load(
            source.files_for_role("trainer")[0],
            map_location="cpu",
            weights_only=False,
        )
        trainer["generation_id"] = child_id
        replay_paths = source.files_for_role("replay")
        replay_metadata_path = next(path for path in replay_paths if path.name == "metadata.pt")
        replay_metadata = torch.load(
            replay_metadata_path, map_location="cpu", weights_only=False
        )
        replay_metadata["total_transitions"] += 1
        replay_metadata["torchrl"]["writer"]["_cursor"] = (
            replay_metadata["total_transitions"] + replay_metadata["num_eps"]
        ) % replay_metadata["signature"]["capacity"]
        files = []
        for record in source.files:
            path = source.file_path(record.path)
            if record.path == "trainer.pt":
                writer = _torch_writer(trainer)
            elif record.path == "replay/metadata.pt":
                writer = _torch_writer(replay_metadata)
            else:
                writer = _copy_writer(path)
            files.append(GenerationFile(record.path, record.role, writer))
        store.publish(
            child_id,
            files=files,
            metadata=copy.deepcopy(dict(source.metadata)),
            source_generation_id=source.generation_id,
        )

    restored_model = _model(BoundaryEnv())
    restored = _session(lineage, mode="required", segment="restore")
    try:
        with pytest.raises(ResumeStateCorruptionError, match="replay counters"):
            restored.prepare_learner(restored_model)
        assert len(fake_wandb.runs) == 1
    finally:
        _close(restored_model, restored)


def test_repeated_signals_wait_for_the_completed_episode_boundary(monkeypatch, tmp_path):
    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    def request_signals():
        for signum in (signal.SIGUSR1, signal.SIGUSR1, signal.SIGTERM):
            handler = signal.getsignal(signum)
            assert callable(handler)
            handler(signum, None)

    model = _model(BoundaryEnv(on_first_done=request_signals))
    session = _session(tmp_path / "signals", mode="new", segment="part-1")
    try:
        assert model.learn(total_timesteps=4, resume_session=session) == RESUME_HANDOFF
        assert model._global_step == 2
        assert model._episode_idx == 1
        assert session.last_generation.metadata["reason"] == "drain"
    finally:
        _close(model, session)


def test_signal_during_hourly_publish_forces_one_newer_drain_generation(
    monkeypatch, tmp_path
):
    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    model = _model(BoundaryEnv())
    session = _session(tmp_path / "publish-signal", mode="new", segment="part-1")
    monkeypatch.setattr(session, "checkpoint_due", lambda: True)
    original_publish = session.publish
    published = []

    def publish(learner, *, reason):
        generation = original_publish(learner, reason=reason)
        published.append(generation)
        if reason == "hourly":
            handler = signal.getsignal(signal.SIGUSR1)
            handler(signal.SIGUSR1, None)
        return generation

    monkeypatch.setattr(session, "publish", publish)
    try:
        assert model.learn(total_timesteps=4, resume_session=session) == RESUME_HANDOFF
        assert [item.metadata["reason"] for item in published[-2:]] == [
            "hourly",
            "drain-during-checkpoint",
        ]
        assert published[-1].parent_generation == published[-2].generation_id
    finally:
        _close(model, session)


def test_pending_evaluation_is_checkpointed_and_runs_once_after_resume(
    monkeypatch, tmp_path
):
    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    lineage = tmp_path / "pending-eval"
    holder = {}
    calls = []

    def drain_after_episode():
        holder["session"]._drain_requested = True

    def bind_fake_eval(model):
        def evaluate(step, **_kwargs):
            calls.append(int(step))
            model._record_evaluation(step, float(step))
            return float(step)

        model._evaluate_policy = evaluate

    first_model = _model(
        BoundaryEnv(on_first_done=drain_after_episode),
        eval_freq=2,
        eval_episodes=1,
    )
    bind_fake_eval(first_model)
    first = _session(lineage, mode="new", segment="part-1")
    holder["session"] = first
    try:
        assert first_model.learn(total_timesteps=4, resume_session=first) == RESUME_HANDOFF
        assert calls == [0]
        trainer = torch.load(
            first.last_generation.files_for_role("trainer")[0],
            map_location="cpu",
            weights_only=False,
        )["trainer"]
        assert trainer["eval_pending"] is True
    finally:
        _close(first_model, first)

    second_model = _model(BoundaryEnv(), eval_freq=2, eval_episodes=1)
    bind_fake_eval(second_model)
    second = _session(lineage, mode="required", segment="part-2")
    try:
        assert second_model.learn(total_timesteps=4, resume_session=second) == RESUME_COMPLETE
        assert calls == [0, 2, 4]
    finally:
        _close(second_model, second)


def test_target_recovery_publishes_current_segment_before_done(monkeypatch, tmp_path):
    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    lineage = tmp_path / "target-recovery"
    model = _model(BoundaryEnv())
    first = _session(lineage, mode="new", segment="part-1")

    def fail_after_finish(learner, _generation):
        first._finish_wandb(learner)
        raise OSError("injected DONE failure")

    monkeypatch.setattr(first, "complete", fail_after_finish)
    try:
        with pytest.raises(OSError, match="DONE failure"):
            model.learn(total_timesteps=4, resume_session=first)
        target_generation = first.last_generation
    finally:
        _close(model, first)

    restored_model = _model(BoundaryEnv())
    restored = _session(lineage, mode="required", segment="part-2")
    try:
        assert restored_model.learn(total_timesteps=4, resume_session=restored) == 0
        recovered = restored.last_generation
        assert recovered.parent_generation == target_generation.generation_id
        assert recovered.metadata["segment_id"] == "part-2"
        done = json.loads((lineage / "DONE").read_text())
        assert done["generation_id"] == recovered.generation_id
        assert done["segment_id"] == "part-2"
    finally:
        _close(restored_model, restored)


def test_trainer_timing_prefix_is_continuous_after_restore(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr("RL.TDMPC2.time.perf_counter", lambda: clock[0])
    source = _model(BoundaryEnv())
    source._reset_wandb_window()
    clock[0] = 16.0
    state = source.training_state_dict()
    assert state["wandb"]["elapsed_seconds"] == pytest.approx(6.0)

    restored = _model(BoundaryEnv())
    clock[0] = 100.0
    restored.load_training_state_dict(state)
    restored._commit_resume_timing_checkpoint(state["wandb"])
    clock[0] = 104.0
    assert restored._timing_wandb_payload(0)["time/time_elapsed"] == pytest.approx(10.0)
    source._checkpoint_writer.shutdown()
    restored._checkpoint_writer.shutdown()


def test_resume_rejects_preinitialized_evaluation_output(tmp_path):
    model = _model(BoundaryEnv())
    model._eval_csv_initialized = True
    session = _session(tmp_path / "bad-eval", mode="new", segment="part-1")
    try:
        with pytest.raises(ResumeIncompatibilityError, match="before resume setup"):
            session.prepare_learner(model)
    finally:
        _close(model, session)
