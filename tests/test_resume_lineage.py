import errno
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import utils.resume_lineage as resume_lineage
from utils.resume_lineage import (
    GENERATION_MANIFEST,
    GenerationFile,
    LineageCompatibilityError,
    LineageConcurrentWriterError,
    LineageCorruptionError,
    LineageGenerationNotFoundError,
    LineageModeError,
    LineageStorageError,
    LineageStore,
    write_bytes,
)


def _files(label):
    return [
        GenerationFile("trainer.pt", "trainer", write_bytes(f"trainer-{label}".encode())),
        GenerationFile("replay/meta.pt", "replay", write_bytes(f"replay-{label}".encode())),
    ]


def _publish(store, label, *, source=None):
    return store.publish(
        label,
        files=_files(label),
        metadata={"global_step": int(label.removeprefix("step-").split("-", 1)[0])},
        source_generation_id=source,
    )


def test_new_publish_required_round_trip_and_retention(tmp_path):
    root = tmp_path / "lineage"
    metadata = {"fingerprint": "abc", "total_steps": 30}
    with LineageStore.open(root, mode="new", lineage_metadata=metadata) as store:
        first = _publish(store, "step-10")
        second = _publish(store, "step-20")
        third = _publish(store, "step-30")
        assert third.parent_generation == second.generation_id
        assert not first.path.exists()
        assert second.path.exists() and third.path.exists()
        assert (root / "LATEST").read_text() == "step-30\n"

    with LineageStore.open(
        root, mode="required", expected_lineage=metadata
    ) as store:
        latest = store.load()
        previous = store.load(latest.parent_generation)
        assert latest.generation_id == "step-30"
        assert previous.generation_id == "step-20"
        assert latest.file_path("trainer.pt").read_bytes() == b"trainer-step-30"
        assert latest.files_for_role("replay") == (latest.path / "replay/meta.pt",)
        manifest = json.loads((latest.path / GENERATION_MANIFEST).read_text())
        assert manifest["metadata"] == {"global_step": 30}


def test_modes_and_immutable_identity_fail_closed(tmp_path):
    root = tmp_path / "lineage"
    with pytest.raises(LineageModeError):
        LineageStore.open(root, mode="required")
    with LineageStore.open(
        root, mode="new", lineage_metadata={"fingerprint": "one"}
    ):
        pass
    with pytest.raises(LineageModeError):
        LineageStore.open(root, mode="new", lineage_metadata={"fingerprint": "one"})
    with pytest.raises(LineageCompatibilityError):
        LineageStore.open(
            root, mode="required", expected_lineage={"fingerprint": "two"}
        )


def test_new_lineage_durably_creates_missing_parent_chain(tmp_path, monkeypatch):
    synced = []
    real_sync = resume_lineage._fsync_directory

    def observed(path):
        synced.append(Path(path))
        real_sync(path)

    monkeypatch.setattr(resume_lineage, "_fsync_directory", observed)
    root = tmp_path / "allocations" / "experiment" / "lineage"
    with LineageStore.open(root, mode="new", lineage_metadata={"id": 1}):
        pass
    assert tmp_path in synced
    assert root.parent in synced


def test_sibling_lineages_can_create_their_shared_parent_concurrently(
    tmp_path, monkeypatch
):
    shared = tmp_path / "allocation" / "sweep"
    barrier = threading.Barrier(2)
    real_mkdir = Path.mkdir

    def synchronized_mkdir(path, *args, **kwargs):
        if path == shared:
            barrier.wait(timeout=5)
        return real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", synchronized_mkdir)

    def create(label):
        root = shared / label
        with LineageStore.open(
            root, mode="new", lineage_metadata={"id": label}
        ):
            pass
        return root

    with ThreadPoolExecutor(max_workers=2) as executor:
        roots = tuple(executor.map(create, ("g2", "g4")))

    assert all((root / "LINEAGE.json").is_file() for root in roots)


def test_process_lifetime_lock_rejects_second_writer(tmp_path):
    root = tmp_path / "lineage"
    first = LineageStore.open(root, mode="new", lineage_metadata={"id": 1})
    try:
        with pytest.raises(LineageConcurrentWriterError):
            LineageStore.open(root, mode="required")
    finally:
        first.close()
    with LineageStore.open(root, mode="required"):
        pass


@pytest.mark.parametrize(
    ("stage", "expected_latest"),
    [
        ("after_staging_created", "step-10"),
        ("after_payloads_written", "step-10"),
        ("before_generation_rename", "step-10"),
        ("after_generation_rename", "step-10"),
        ("before_latest_replace", "step-10"),
        ("after_latest_replace", "step-20"),
    ],
)
def test_publication_failure_exposes_old_or_new_never_mixed(
    tmp_path, stage, expected_latest
):
    root = tmp_path / stage
    with LineageStore.open(root, mode="new", lineage_metadata={"id": stage}) as store:
        _publish(store, "step-10")

    def fail(selected, _context):
        if selected == stage:
            raise RuntimeError(stage)

    store = LineageStore.open(root, mode="required", fault_injector=fail)
    try:
        store.load()
        with pytest.raises(RuntimeError, match=stage):
            _publish(store, "step-20")
    finally:
        store.close()

    with LineageStore.open(root, mode="required") as reopened:
        generation = reopened.load()
        assert generation.generation_id == expected_latest
        assert generation.file_path("trainer.pt").read_bytes() == (
            f"trainer-{expected_latest}".encode()
        )
    assert not any(path.name.endswith(".tmp") for path in (root / "generations").iterdir())


def test_payload_corruption_and_missing_latest_are_not_auto_recovered(tmp_path):
    root = tmp_path / "lineage"
    with LineageStore.open(root, mode="new", lineage_metadata={"id": 1}) as store:
        first = _publish(store, "step-10")
        _publish(store, "step-20")
    first.file_path("trainer.pt").write_bytes(b"corrupt")
    with LineageStore.open(root, mode="required") as store:
        with pytest.raises(LineageCorruptionError, match="checksum"):
            store.load("step-10")

    (root / "LATEST").unlink()
    with LineageStore.open(root, mode="required") as store:
        with pytest.raises(LineageGenerationNotFoundError):
            store.load()


def test_manifest_corruption_and_unselected_orphans_are_never_loadable(tmp_path):
    root = tmp_path / "lineage"
    with LineageStore.open(root, mode="new", lineage_metadata={"id": 1}) as store:
        latest = _publish(store, "step-10")

        def fail(stage, _context):
            if stage == "after_generation_rename":
                raise RuntimeError("crash")

        store._fault_injector = fail
        with pytest.raises(RuntimeError, match="crash"):
            _publish(store, "step-20")

    with LineageStore.open(root, mode="required") as store:
        with pytest.raises(LineageGenerationNotFoundError, match="retained predecessor"):
            store.load("step-20")

    manifest = latest.path / GENERATION_MANIFEST
    document = json.loads(manifest.read_text())
    document["metadata"]["global_step"] = 11
    manifest.write_text(json.dumps(document))
    with LineageStore.open(root, mode="required") as store:
        with pytest.raises(LineageCorruptionError, match="manifest checksum"):
            store.load()


def test_writer_storage_error_preserves_latest(tmp_path):
    root = tmp_path / "lineage"
    with LineageStore.open(root, mode="new", lineage_metadata={"id": 1}) as store:
        _publish(store, "step-10")

        def no_space(_path: Path):
            raise OSError(errno.ENOSPC, "full")

        with pytest.raises(LineageStorageError):
            store.publish(
                "step-20",
                files=[GenerationFile("trainer.pt", "trainer", no_space)],
                metadata={"global_step": 20},
            )
        assert store.load().generation_id == "step-10"


def test_explicit_previous_generation_branches_without_selecting_an_orphan(tmp_path):
    root = tmp_path / "lineage"
    with LineageStore.open(root, mode="new", lineage_metadata={"id": 1}) as store:
        first = _publish(store, "step-10")
        _publish(store, "step-20")
        branch = _publish(store, "step-15-branch", source=first.generation_id)
        assert branch.parent_generation == first.generation_id
        assert store.load().generation_id == branch.generation_id


def test_payload_paths_and_generation_ids_have_basic_safety_checks(tmp_path):
    root = tmp_path / "lineage"
    with LineageStore.open(root, mode="new", lineage_metadata={"id": 1}) as store:
        with pytest.raises(LineageCompatibilityError):
            store.publish(
                "../escape",
                files=_files("step-1"),
                metadata={},
            )
        with pytest.raises(LineageCompatibilityError):
            store.publish(
                "step-1",
                files=[GenerationFile("../escape", "trainer", write_bytes(b"x"))],
                metadata={},
            )
