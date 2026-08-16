from types import SimpleNamespace

import pytest

from utils.wandb_resume import (
    ENV_STEP_KEY,
    EVENT_INDEX_KEY,
    CheckpointedWandbRun,
    WandbCapabilityError,
    WandbFinishError,
    WandbInitializationError,
    WandbRemoteAheadError,
    WandbRemoteDivergenceError,
    WandbRemoteHistoryError,
    WandbRemoteNetworkError,
    WandbRemoteVerificationTimeout,
    WandbRemoteWriteError,
    WandbResumeConfigurationError,
    WandbResumeContext,
    WandbStateError,
    fetch_remote_history,
    validate_remote_prefix,
    verify_remote_checkpoint,
)
from utils.wandb_utils import abort_wandb, finish_wandb, init_wandb, log_wandb


class _RemoteStore:
    def __init__(self, rows=()):
        self.rows = [dict(row) for row in rows]


class _RawRun:
    def __init__(self, store, *, fail_log=False, fail_finish=False):
        self.store = store
        self.logged = []
        self.finished = False
        self.fail_log = fail_log
        self.fail_finish = fail_finish

    def log(self, payload, *, step):
        if self.fail_log:
            raise OSError("injected log failure")
        self.logged.append((dict(payload), step))

    def finish(self):
        self.finished = True
        if self.fail_finish:
            raise OSError("injected finish failure")
        for payload, step in self.logged:
            self.store.rows.append({"_step": step, **payload})
        self.logged.clear()


class _Api:
    def __init__(self, store, *, fail=False):
        self.store = store
        self.fail = fail

    def run(self, path):
        assert path.startswith("brown/ambi/")
        if self.fail:
            raise OSError("injected API failure")
        store = self.store

        class RemoteRun:
            def scan_history(self, *, page_size):
                assert page_size == 1000
                return tuple(store.rows)

        return RemoteRun()


class _FakeWandb:
    # Deliberately not 0.17.4: exact resume depends on capabilities, not a pin.
    __version__ = "0.99.0"

    def __init__(self, store=None, raw_run=None):
        self.store = store or _RemoteStore()
        self.raw_run = raw_run or _RawRun(self.store)
        self.init_calls = []
        self.defined = []

    def init(
        self,
        *,
        project=None,
        entity=None,
        name=None,
        config=None,
        mode=None,
        dir=None,
        tags=None,
        group=None,
        id=None,
        resume=None,
    ):
        values = {
            "project": project,
            "entity": entity,
            "name": name,
            "config": config,
            "mode": mode,
            "dir": dir,
            "tags": tags,
            "group": group,
            "id": id,
            "resume": resume,
        }
        self.init_calls.append(
            {key: value for key, value in values.items() if value is not None}
        )
        return self.raw_run

    def define_metric(self, *args, **kwargs):
        self.defined.append((args, kwargs))

    def Api(self, *, timeout=None):
        assert timeout == 20
        return _Api(self.store)


def _params(**overrides):
    return {
        "wandb": True,
        "wandb_project": "ambi",
        "wandb_entity": "brown",
        "wandb_mode": "online",
        **overrides,
    }


def _rows(context, count=None):
    events = context.buffer.events
    if count is not None:
        events = events[:count]
    return tuple(
        {"_step": event.event_index, **event.wandb_payload()} for event in events
    )


def _new_context_with_events(*steps):
    context = WandbResumeContext.new(run_id="stable-run")
    for index, step in enumerate(steps):
        context.buffer.append({"train/loss": index + 0.5}, env_step=step)
    return context


def test_buffer_state_is_self_contained_and_round_trips_without_files():
    original = _new_context_with_events(10, 10, 20)
    state = original.checkpoint_state()
    restored = WandbResumeContext.resume(
        run_id="stable-run", checkpoint_state=state, remote_rows=()
    )

    assert restored.checkpoint_state() == state
    assert [event.event_index for event in restored.buffer.events] == [0, 1, 2]
    assert [event.env_step for event in restored.buffer.events] == [10, 10, 20]


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda state: state.update(schema_version=99), "schema version"),
        (lambda state: state.update(run_id="different"), "does not match"),
        (lambda state: state["events"][0].update(event_index=2), "contiguous"),
        (lambda state: state["events"][1].update(env_step=0), "nondecreasing"),
        (lambda state: state["events"][0]["payload"].update(loss=float("nan")), "finite"),
    ],
)
def test_checkpoint_state_validation_is_strict(mutation, message):
    context = _new_context_with_events(1, 2)
    state = context.checkpoint_state()
    mutation(state)
    with pytest.raises(WandbStateError, match=message):
        WandbResumeContext.resume(
            run_id="stable-run", checkpoint_state=state, remote_rows=()
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"loss": float("inf")},
        {EVENT_INDEX_KEY: 1},
        {ENV_STEP_KEY: 2},
        {"_private": 1},
        {1: "non-string"},
        {"loss": object()},
    ],
)
def test_buffer_rejects_ambiguous_or_nonfinite_payloads(payload):
    context = WandbResumeContext.new(run_id="stable-run")
    with pytest.raises(WandbStateError):
        context.buffer.append(payload, env_step=1)
    assert len(context.buffer) == 0


def test_new_run_buffers_until_durable_checkpoint_is_explicitly_published(tmp_path):
    context = WandbResumeContext.new(run_id="stable-run", directory=tmp_path)
    wandb = _FakeWandb()
    run = init_wandb(
        _params(),
        default_project="unused",
        run_name="seed-1",
        config={"seed": 1},
        resume_context=context,
        wandb_module=wandb,
    )

    assert isinstance(run, CheckpointedWandbRun)
    assert wandb.init_calls == [
        {
            "project": "ambi",
            "entity": "brown",
            "name": "seed-1",
            "config": {"seed": 1},
            "mode": "online",
            "dir": str(tmp_path),
            "id": "stable-run",
            "resume": "never",
        }
    ]
    log_wandb(run, {"train/loss": 1.0}, step=10)
    log_wandb(run, {"train/loss": 2.0}, step=10)
    assert wandb.raw_run.logged == []
    assert wandb.store.rows == []

    durable_state = run.checkpoint_state()
    run.publish_committed(durable_state)
    run.publish_committed(durable_state)
    assert [step for _, step in wandb.raw_run.logged] == [0, 1]
    assert [payload[ENV_STEP_KEY] for payload, _ in wandb.raw_run.logged] == [10, 10]
    assert [payload[EVENT_INDEX_KEY] for payload, _ in wandb.raw_run.logged] == [0, 1]

    finish_wandb(run)
    assert wandb.raw_run.finished
    assert len(wandb.store.rows) == 2


def test_uncommitted_crash_tail_is_not_part_of_restored_history():
    before_crash = _new_context_with_events(10)
    durable_state = before_crash.checkpoint_state()
    before_crash.buffer.append({"train/loss": 99.0}, env_step=20)

    restored = WandbResumeContext.resume(
        run_id="stable-run",
        checkpoint_state=durable_state,
        remote_rows=_rows(before_crash, count=1),
    )
    assert len(restored.buffer) == 1
    assert restored.checkpoint_state() == durable_state


def test_resume_uses_same_physical_run_and_replays_only_missing_committed_suffix():
    checkpoint = _new_context_with_events(10, 20, 30)
    store = _RemoteStore(_rows(checkpoint, count=1))
    wandb = _FakeWandb(store=store)
    context = WandbResumeContext.resume(
        run_id="stable-run",
        checkpoint_state=checkpoint.checkpoint_state(),
        remote_rows=tuple(store.rows),
    )
    run = init_wandb(
        _params(),
        default_project="unused",
        run_name="seed-1",
        resume_context=context,
        wandb_module=wandb,
    )

    assert wandb.init_calls[0]["id"] == "stable-run"
    assert wandb.init_calls[0]["resume"] == "must"
    assert [step for _, step in wandb.raw_run.logged] == [1, 2]
    finish_wandb(run)
    assert [row["_step"] for row in store.rows] == [0, 1, 2]


def test_exact_remote_prefix_requires_every_metric_and_index():
    context = _new_context_with_events(10, 20)
    exact = list(_rows(context))
    assert validate_remote_prefix(context.buffer.events, exact[:1]) == 1

    divergent = [dict(row) for row in exact]
    divergent[1]["train/loss"] = 99.0
    with pytest.raises(WandbRemoteDivergenceError, match="event 1"):
        validate_remote_prefix(context.buffer.events, divergent)

    missing = [dict(row) for row in exact]
    missing[0].pop(EVENT_INDEX_KEY)
    with pytest.raises(WandbRemoteHistoryError, match="lacks"):
        validate_remote_prefix(context.buffer.events, missing)

    gap = [dict(row) for row in exact]
    gap[1]["_step"] = 3
    with pytest.raises(WandbRemoteHistoryError, match="contiguous"):
        validate_remote_prefix(context.buffer.events, gap)

    foreign = [dict(row) for row in exact]
    foreign[0]["_foreign"] = 1
    with pytest.raises(WandbRemoteHistoryError, match="reserved"):
        validate_remote_prefix(context.buffer.events, foreign)


def test_remote_ahead_makes_same_id_rollback_a_hard_error_but_branch_is_explicit():
    old = _new_context_with_events(10)
    later = _new_context_with_events(10, 20)
    with pytest.raises(WandbRemoteAheadError, match="cannot reuse this run ID"):
        WandbResumeContext.resume(
            run_id="stable-run",
            checkpoint_state=old.checkpoint_state(),
            remote_rows=_rows(later),
        )

    branch = WandbResumeContext.branch(
        run_id="new-rollback-experiment", checkpoint_state=old.checkpoint_state()
    )
    assert branch.new_run
    assert branch.run_id == "new-rollback-experiment"
    assert branch.remote_event_count == branch.committed_event_count == 0
    assert len(branch.buffer) == 1

    wandb = _FakeWandb()
    run = init_wandb(
        _params(),
        default_project="unused",
        run_name="rollback",
        resume_context=branch,
        wandb_module=wandb,
    )
    assert wandb.init_calls[0]["id"] == "new-rollback-experiment"
    assert wandb.init_calls[0]["resume"] == "never"
    assert wandb.raw_run.logged == []
    state = run.checkpoint_state()
    assert state["run_id"] == "new-rollback-experiment"
    run.publish_committed(state)
    finish_wandb(run)
    assert [row["_step"] for row in wandb.store.rows] == [0]


def test_committed_upload_failure_is_typed_and_poisoned():
    store = _RemoteStore()
    raw = _RawRun(store, fail_log=True)
    wandb = _FakeWandb(store=store, raw_run=raw)
    run = init_wandb(
        _params(),
        default_project="unused",
        run_name=None,
        resume_context=WandbResumeContext.new(run_id="stable-run"),
        wandb_module=wandb,
    )
    log_wandb(run, {"loss": 1.0}, step=1)
    state = run.checkpoint_state()
    with pytest.raises(WandbRemoteWriteError, match="event 0"):
        run.publish_committed(state)
    with pytest.raises(WandbRemoteWriteError, match="disabled"):
        run.publish_committed(state)
    with pytest.raises(WandbRemoteWriteError, match="disabled"):
        log_wandb(run, {"loss": 2.0}, step=2)
    abort_wandb(run)
    assert raw.finished


def test_finish_refuses_uncommitted_events_and_wraps_sdk_failure():
    wandb = _FakeWandb()
    run = init_wandb(
        _params(),
        default_project="unused",
        run_name=None,
        resume_context=WandbResumeContext.new(run_id="stable-run"),
        wandb_module=wandb,
    )
    log_wandb(run, {"loss": 1.0}, step=1)
    with pytest.raises(WandbStateError, match="uncommitted"):
        finish_wandb(run)
    assert not wandb.raw_run.finished

    run.publish_committed(run.checkpoint_state())
    wandb.raw_run.fail_finish = True
    with pytest.raises(WandbFinishError, match="finish"):
        finish_wandb(run)


class _Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class _SnapshotApi:
    def __init__(self, snapshots):
        self.snapshots = list(snapshots)
        self.calls = 0

    def run(self, _path):
        index = min(self.calls, len(self.snapshots) - 1)
        self.calls += 1
        rows = self.snapshots[index]

        class RemoteRun:
            def scan_history(self, *, page_size):
                assert page_size == 1000
                return rows

        return RemoteRun()


def test_post_finish_verification_polls_only_a_valid_remote_prefix():
    context = _new_context_with_events(10, 20)
    behind = _rows(context, count=1)
    exact = _rows(context)
    api = _SnapshotApi((behind, behind, exact))
    clock = _Clock()

    assert verify_remote_checkpoint(
        _FakeWandb(),
        entity="brown",
        project="ambi",
        checkpoint_state=context.checkpoint_state(),
        timeout_seconds=2,
        poll_interval_seconds=1,
        api=api,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    ) == exact
    assert api.calls == 3
    assert clock.sleeps == [1, 1]


def test_post_finish_verification_timeout_and_network_failures_are_typed():
    context = _new_context_with_events(10, 20)
    clock = _Clock()
    with pytest.raises(WandbRemoteVerificationTimeout, match="only 1 of 2"):
        verify_remote_checkpoint(
            _FakeWandb(),
            entity="brown",
            project="ambi",
            checkpoint_state=context.checkpoint_state(),
            timeout_seconds=2,
            poll_interval_seconds=1,
            api=_SnapshotApi((_rows(context, count=1),)),
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )

    with pytest.raises(WandbRemoteNetworkError, match="Could not fetch"):
        fetch_remote_history(
            _FakeWandb(),
            entity="brown",
            project="ambi",
            run_id="stable-run",
            api=_Api(_RemoteStore(), fail=True),
        )


def test_resume_mode_rejects_offline_and_missing_capabilities_but_allows_one_run_group():
    context = WandbResumeContext.new(run_id="stable-run")
    with pytest.raises(WandbResumeConfigurationError, match="online"):
        init_wandb(
            _params(wandb_mode="offline"),
            default_project="ambi",
            run_name=None,
            resume_context=context,
            wandb_module=_FakeWandb(),
        )

    grouped = _FakeWandb()
    run = init_wandb(
        _params(wandb_group="one-experiment-suite"),
        default_project="ambi",
        run_name=None,
        resume_context=context,
        wandb_module=grouped,
    )
    assert isinstance(run, CheckpointedWandbRun)
    assert grouped.init_calls == [
        {
            "project": "ambi",
            "entity": "brown",
            "config": {},
            "mode": "online",
            "group": "one-experiment-suite",
            "id": "stable-run",
            "resume": "never",
        }
    ]

    incomplete = SimpleNamespace(
        init=lambda **_kwargs: object(), define_metric=lambda *_a, **_k: None
    )
    with pytest.raises(WandbCapabilityError, match="Api"):
        init_wandb(
            _params(),
            default_project="ambi",
            run_name=None,
            resume_context=context,
            wandb_module=incomplete,
        )


def test_resume_initialization_failure_closes_the_partially_opened_run():
    wandb = _FakeWandb()

    def fail_define(*_args, **_kwargs):
        raise OSError("injected metric definition failure")

    wandb.define_metric = fail_define
    with pytest.raises(
        WandbInitializationError,
        match="Could not initialize prepared online W&B run",
    ):
        init_wandb(
            _params(),
            default_project="ambi",
            run_name=None,
            resume_context=WandbResumeContext.new(run_id="stable-run"),
            wandb_module=wandb,
        )
    assert wandb.raw_run.finished


def test_legacy_wandb_log_and_finish_are_unchanged():
    class LegacyRun:
        def __init__(self):
            self.logged = []
            self.finished = False

        def log(self, payload, *, step):
            self.logged.append((dict(payload), step))

        def finish(self):
            self.finished = True

    run = LegacyRun()
    payload = {"loss": 1.0}
    log_wandb(run, payload, step=7)
    finish_wandb(run)
    assert payload == {"loss": 1.0}
    assert run.logged == [({"loss": 1.0, "env_step": 7}, 7)]
    assert run.finished
