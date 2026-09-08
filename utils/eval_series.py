"""Durable, single-writer publication of frozen evaluation curves.

Evaluation workers only stage immutable records. A CPU publisher owns a run's
directory and its W&B connection. Scientific results survive publication errors;
an uncertain history write is never blindly retried.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import time
import uuid


SCHEMA_VERSION = 1
X_AXIS = "checkpoint/training_decisions"
RECORD_KEY = "publication/record_id"
HASH_KEY = "publication/record_sha256"
VISIBLE_METRICS = ("eval/return_mean", "eval/paired_gain_mean", "runtime/control_seconds")


class SeriesError(ValueError):
    """A result cannot safely belong to, or be published to, this curve."""


class PublicationUncertainError(SeriesError):
    """Remote acknowledgement is unresolved; retain the local journal."""


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
            name = handle.name
            handle.write(_json(value) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        name = None
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if name is not None:
            os.unlink(name)


def _read(path):
    with open(path) as handle:
        return json.load(handle)


@contextmanager
def _lock(path, *, blocking=True):
    with open(path, "a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise SeriesError("Another publisher already owns this evaluation run") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def validate_record(record, identity=None):
    """Validate the adapter contract without importing a model or the SDK."""
    try:
        result = json.loads(_json(record))
    except (TypeError, ValueError) as exc:
        raise SeriesError("Records must be finite JSON; encode nonfinite measurements explicitly") from exc
    required = {"identity", "checkpoint", "metrics", "episodes", "artifact_files", "provenance", "record_id", "source_result_path", "label"}
    if not isinstance(result, dict) or not required.issubset(result):
        raise SeriesError("Incomplete normalized evaluation record")
    ident = result["identity"]
    if not isinstance(ident, dict) or set(ident) != {"backbone", "planner", "protocol", "science"}:
        raise SeriesError("Identity requires backbone, planner, protocol, and science")
    if not isinstance(ident["backbone"], str) or not ident["backbone"].strip():
        raise SeriesError("A verified backbone identity is required")
    if any(not isinstance(ident[key], dict) or not ident[key] for key in ("planner", "protocol", "science")):
        raise SeriesError("Planner, protocol, and scientific identity must be nonempty mappings")
    if identity is not None and _json(ident) != _json(identity):
        changed = [key for key in ident if ident[key] != identity.get(key)]
        raise SeriesError("Incompatible append: " + ", ".join(changed) + "; create a new evaluation run")
    checkpoint = result["checkpoint"]
    if not isinstance(checkpoint, dict) or type(checkpoint.get("step")) is not int or checkpoint["step"] < 0:
        raise SeriesError("Checkpoint step must be a nonnegative integer")
    if not re.fullmatch(r"[0-9a-f]{64}", str(checkpoint.get("sha256", ""))):
        raise SeriesError("Checkpoint requires its content SHA256")
    if not isinstance(result["record_id"], str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,200}", result["record_id"]) or result["record_id"] in (".", ".."):
        raise SeriesError("Invalid stable record ID")
    if not isinstance(result["episodes"], list) or not isinstance(result["metrics"], dict) or not isinstance(result["provenance"], dict):
        raise SeriesError("Invalid metrics, episodes, or provenance")
    for key, value in result["metrics"].items():
        if key.startswith(("publication/", "checkpoint/")):
            raise SeriesError("Normalized metrics must not overwrite publication identity")
        if not (value is None or isinstance(value, (int, float, bool)) or isinstance(value, dict) and set(value) == {"nonfinite"} and value["nonfinite"] in ("nan", "positive_infinity", "negative_infinity")):
            raise SeriesError("Normalized metrics must be scalar or explicit nonfinite measurements")
    if not isinstance(result["label"], str) or not result["label"].strip():
        raise SeriesError("A descriptive planner label is required")
    if not isinstance(result["source_result_path"], str) or not Path(result["source_result_path"]).is_absolute():
        raise SeriesError("Source result path must be absolute")
    files = result["artifact_files"]
    if not isinstance(files, dict) or not files:
        raise SeriesError("Complete result artifact files are required")
    for name, path in files.items():
        relative = PurePosixPath(name)
        if not name or relative.is_absolute() or ".." in relative.parts or "\\" in name or name in (".", "evaluation-series-record.json"):
            raise SeriesError("Artifact names must be safe relative paths")
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise SeriesError("Artifact source paths must be absolute")
    return result


def load_run(run_dir):
    """Read a registry selected by an explicit directory/run ID."""
    run_dir = Path(run_dir).resolve()
    registry = _read(run_dir / "run.json")
    if registry.get("schema_version") != SCHEMA_VERSION or registry.get("run_dir") != str(run_dir):
        raise SeriesError("Registry must be used in its authoritative owner directory")
    if registry.get("identity_sha256") != _digest(registry["identity"]):
        raise SeriesError("Registry identity was changed")
    return registry


def validate_identity(registry, identity):
    """Read-only launch preflight for a deliberately selected New/Append run."""
    if not isinstance(registry, dict):
        registry = load_run(registry)
    expected = registry["identity"]
    if _json(identity) != _json(expected):
        changed = sorted(key for key in set(expected) | set(identity) if expected.get(key) != identity.get(key))
        raise SeriesError("Incompatible append: " + ", ".join(changed) + "; create a new evaluation run")
    return registry


def backbone_display_label(identity, *, compact=False):
    """Readable aliases for verified source IDs, never scientific identity."""
    known = {
        "rwgao_b-brown-university/ambi/u13m14st": ("Original AMBI prior-only backbone", "AMBI original"),
        "rwgao_b-brown-university/ambi/axqc-prior-92441d99-5959199": ("AMBI-XQC prior-only backbone", "AMBI-XQC"),
        "rwgao_b-brown-university/ambi/xq3zva9u": ("TD-MPC2 prior-only backbone", "TD-MPC2"),
    }
    source = identity["backbone"]
    if source in known:
        return known[source][bool(compact)]
    # Unknown sources must not acquire a misleading alias or collapse into a
    # known backbone merely because their last path component happens to match.
    return source if compact else "Backbone " + source


def _planner_display_label(planner, *, compact=False):
    """Describe executed behavior rather than the campaign that measured it."""
    settings = planner.get("settings", {})
    kind = planner.get("type", planner.get("operator", "planner"))
    number = lambda value: format(value, "g").replace("e-0", "e-").replace("e+0", "e+")
    parts = ["Prior only (no planning)" if kind == "prior" else str(kind).upper()]
    if kind in ("sac", "xqc"):
        rounds = settings.get("inner_rounds")
        doses = []
        for component in ("critic", "actor", "temperature"):
            dose = settings.get(f"inner_{component}_updates_per_round")
            total = settings.get(f"inner_{component}_updates_per_action")
            if dose is None and isinstance(total, (int, float)) and rounds:
                dose = total / rounds
            doses.append("?" if dose is None else number(dose))
        parts.append("/".join(prefix + dose for prefix, dose in zip("CAT", doses)))
        if settings.get("inner_terminal_bootstrap") == "outer":
            parts.append("outer-term" if compact else "outer terminal Q")
        else:
            bootstrap = settings.get("inner_bootstrap_source", "inner_target")
            parts.append("outer Q" if bootstrap.startswith("outer") else "inner Q")
        interval = settings.get("inner_steps_per_update")
        if interval:
            parts.append(("s" if compact else "interval") + number(interval))
        if kind == "xqc":
            step_timing = settings.get("inner_update_timing", "round") == "step"
            parts.append(("step" if step_timing else "round") + ("" if compact else " updates"))
        if settings.get("inner_component_update_schedule"):
            parts.append("split")
        rate = settings.get("inner_actor_lr")
        if rate is not None and (not compact or rate != 5e-5):
            parts.append("aLR" + number(rate))
        for key, prefix, default in (("inner_rounds", "J", 6), ("inner_rollouts_per_round", "N", 512), ("inner_rollout_horizon", "H", 3)):
            value = settings.get(key)
            if value is not None and (not compact or value != default):
                parts.append(prefix + number(value))
    elif kind == "mppi":
        for value, prefix in ((settings.get("planning_horizon", settings.get("horizon")), "H"),
                              (settings.get("num_samples"), "N"),
                              (settings.get("num_elites"), "E"),
                              (settings.get("num_pi_trajs"), "pi"),
                              (settings.get("effective_iterations"), "I")):
            if value is not None and (not compact or prefix in ("H", "N", "I")):
                parts.append(prefix + number(value))
        if not compact:
            backend = planner.get("backend")
            if backend == "tdmpc2_mppi_over_frozen_xqc":
                parts.append("online XQC Q × frozen scale")
            elif backend == "native_tdmpc2":
                parts.append("online TD-MPC2 Q")
    return " ".join(parts)


def _attempt_display_label(registry):
    attempt = re.sub(r"[-_ ]?20\d{2}[-_]?\d{2}[-_]?\d{2}", "", registry["attempt_label"]).strip("-_ ")
    labels = {"baseline": "baseline comparison", "critic-sweep": "critic sweep",
              "actor-sweep": "actor sweep", "interval-sweep": "interval sweep",
              "native-mppi": "MPPI comparison", "paper-mppi": "MPPI comparison",
              "inner-j6": "inner-Q comparison", "outer-terminal-j6": "outer-terminal comparison"}
    return labels.get(attempt, attempt.replace("_", " ").replace("-", " ")) or "evaluation"


def evaluation_run_name(registry):
    """Descriptive name with campaign context explicitly separated as an attempt."""
    identity = registry["identity"]
    return (backbone_display_label(identity) + " | " + _planner_display_label(identity["planner"]) +
            " | Attempt: " + _attempt_display_label(registry) + " [" + registry["run_id"][:4] + "]")


def concise_curve_label(registry):
    """Readable backbone and actual controller, with a suffix for repetitions."""
    identity = registry["identity"]
    return (backbone_display_label(identity, compact=True) + " · " +
            _planner_display_label(identity["planner"], compact=True) + " · #" + registry["run_id"][:4])


def create_run(registry_root, record, attempt_label, project, entity, owner):
    """Explicitly allocate a new attempt before launching its checkpoint jobs.

    The representative record establishes identity; it is not automatically
    staged or uploaded. Repeating this operation intentionally creates a new ID.
    """
    # Launchers allocate the identity before any GPU result exists. Reuse the
    # identity validator with an otherwise harmless template; never stage it.
    template = dict(record)
    template.update(checkpoint={"step": 0, "sha256": "0" * 64}, metrics={}, episodes=[], artifact_files={"template.json": "/template.json"}, provenance={}, record_id="template", source_result_path="/template.json")
    validate_record(template)
    for label, value in (("attempt label", attempt_label), ("project", project), ("entity", entity), ("owner", owner)):
        if not isinstance(value, str) or not value.strip():
            raise SeriesError("An explicit " + label + " is required")
    run_id = uuid.uuid4().hex
    run_dir = Path(registry_root).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "records").mkdir()
    registry = {
        "schema_version": SCHEMA_VERSION, "run_id": run_id, "run_dir": str(run_dir),
        "identity": record["identity"], "identity_sha256": _digest(record["identity"]),
        "attempt_label": attempt_label,
        "project": project, "entity": entity, "owner": owner,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    registry["name"] = evaluation_run_name(registry)
    _atomic_json(run_dir / "run.json", registry)
    _atomic_json(run_dir / "publication.json", {"schema_version": SCHEMA_VERSION, "records": {}, "next_step": 0, "remote_initialized": False})
    return registry


def _record_fingerprint(record, files):
    # Absolute storage locations are not scientific identity, nor are they part
    # of retry identity when the same immutable files are copied between hosts.
    payload = {key: value for key, value in record.items() if key not in ("artifact_files", "source_result_path")}
    return _digest({"record": payload, "artifact_sha256": files})


def stage_record(run_dir, record):
    """Append a completed checkpoint to an explicitly selected attempt locally."""
    run_dir = Path(run_dir).resolve()
    registry = load_run(run_dir)
    record = validate_record(record, registry["identity"])
    file_hashes = {name: _file_digest(path) for name, path in record["artifact_files"].items()}
    fingerprint = _record_fingerprint(record, file_hashes)
    record_id = record["record_id"]
    with _lock(run_dir / ".records.lock"):
        index = _read(run_dir / "publication.json")
        for other_id, entry in index["records"].items():
            if entry["checkpoint_step"] == record["checkpoint"]["step"]:
                if other_id != record_id or entry["record_sha256"] != fingerprint:
                    raise SeriesError("Checkpoint already has a different accepted result; create a new run for a repeated evaluation")
                # A byte-identical result may have moved; refresh artifact paths.
                _atomic_json(run_dir / "records" / (record_id + ".json"), record)
                return {"status": "published" if entry["status"] == "published" else "already_staged", "record_id": record_id}
        if record_id in index["records"]:
            raise SeriesError("Record ID was reused for a different checkpoint")
        _atomic_json(run_dir / "records" / (record_id + ".json"), record)
        index["records"][record_id] = {"status": "staged", "accepted_order": len(index["records"]), "checkpoint_step": record["checkpoint"]["step"], "checkpoint_sha256": record["checkpoint"]["sha256"], "record_sha256": fingerprint, "artifact_sha256": file_hashes}
        _atomic_json(run_dir / "publication.json", index)
    return {"status": "staged", "record_id": record_id}


def stage_result(run_dir, result_path, selector=None, format="ambi-bundle", source_run=None, inventory_path=None):
    """Queue an atomic result pointer; safe for GPU jobs without SDK imports."""
    run_dir = Path(run_dir).resolve()
    load_run(run_dir)
    if format not in ("ambi-bundle", "tdmpc2-paired"):
        raise SeriesError("Unsupported evaluation result format")
    path = Path(result_path).resolve()
    if not path.exists():
        raise SeriesError("Result pointer target does not exist")
    pointer = {"result_path": str(path), "selector": selector, "format": format, "source_run": source_run, "inventory_path": str(Path(inventory_path).resolve()) if inventory_path else None}
    pointer_id = _digest(pointer)
    _atomic_json(run_dir / "incoming" / (pointer_id + ".json"), pointer)
    return {"status": "staged", "pointer_id": pointer_id, "run_id": load_run(run_dir)["run_id"]}


class Publisher:
    """One serial SDK session, reusable by a CPU-only polling command.

    ``queued`` is deliberately distinct from ``published``. ``finish`` verifies
    the remote history after SDK shutdown. An interrupted uncertain write that
    is still absent remotely needs reconciliation, never blind retransmission.
    """

    def __init__(self, run_dir, owner=None, wandb_module=None, *, acknowledgement_timeout=30, poll_interval=1):
        self.run_dir = Path(run_dir).resolve()
        self.registry = load_run(self.run_dir)
        if owner is not None and owner != self.registry["owner"]:
            raise SeriesError("Only the registered publication owner may publish this run")
        self.wandb = wandb_module
        self.run = None
        self._guard = None
        self.acknowledgement_timeout = acknowledgement_timeout
        self.poll_interval = poll_interval

    def __enter__(self):
        self._guard = _lock(self.run_dir / ".publisher.lock", blocking=False)
        self._guard.__enter__()
        try:
            if self.wandb is None:
                import wandb
                self.wandb = wandb
            index = _read(self.run_dir / "publication.json")
            if index["remote_initialized"]:
                self._reconcile()
                index = _read(self.run_dir / "publication.json")
            # Record init intent before touching the network: a timeout can still
            # have created the remote run, so its next opening must resume.
            with _lock(self.run_dir / ".records.lock"):
                index = _read(self.run_dir / "publication.json")
                resume = "must" if index["remote_initialized"] else ("allow" if index.get("init_started") else "never")
                index["init_started"] = True
                _atomic_json(self.run_dir / "publication.json", index)
            cfg = {"eval_series_schema": SCHEMA_VERSION, "evaluation_identity": self.registry["identity"], "evaluation_identity_sha256": self.registry["identity_sha256"], "attempt_label": self.registry["attempt_label"], "publication_owner": self.registry["owner"], "source_run": self.registry["identity"]["backbone"], "backbone_id": self.registry["identity"]["backbone"], "planner": self.registry["identity"]["planner"], "curve_label": concise_curve_label(self.registry)}
            self.run = self.wandb.init(id=self.registry["run_id"], entity=self.registry["entity"], project=self.registry["project"], name=evaluation_run_name(self.registry), config=cfg, tags=["evaluation-curve", "eval-series-v1"], job_type="evaluation-curve", resume=resume, dir=str(self.run_dir), reinit=True)
            if getattr(self.run, "disabled", False) or getattr(getattr(self.run, "settings", None), "mode", "online") != "online":
                raise SeriesError("Durable publication requires online W&B; use stage-only validation for offline work")
            with _lock(self.run_dir / ".records.lock"):
                index = _read(self.run_dir / "publication.json")
                index["remote_initialized"] = True
                _atomic_json(self.run_dir / "publication.json", index)
            # Resume returns the SDK's authoritative next internal history step.
            # Resend a missing interrupted row only into its original unused
            # slot. Never allocate a different step for the same result.
            for entry in index["records"].values():
                if entry["status"] in ("row_inflight", "queued"):
                    sdk_step = getattr(self.run, "step", None)
                    if type(sdk_step) is not int or sdk_step > entry["wandb_step"]:
                        raise PublicationUncertainError("An interrupted row is not visible but its SDK slot may be occupied. Retry after remote history becomes consistent; no duplicate was sent")
            self.run.define_metric("*", step_metric=X_AXIS, step_sync=False, hidden=True)
            self.run.define_metric(X_AXIS, hidden=True)
            for key in VISIBLE_METRICS:
                self.run.define_metric(key, step_metric=X_AXIS, step_sync=False, hidden=False)
            return self
        except BaseException:
            try:
                if self.run is not None:
                    self.run.finish(exit_code=1)
                    self.run = None
            finally:
                self._guard.__exit__(None, None, None)
                self._guard = None
            raise

    def _reconcile(self):
        registry = self.registry
        try:
            remote = self.wandb.Api(timeout=30).run(f"{registry['entity']}/{registry['project']}/{registry['run_id']}")
            if remote.config.get("evaluation_identity_sha256") != registry["identity_sha256"]:
                raise SeriesError("Remote run has an incompatible evaluation identity")
            rows = list(remote.scan_history(page_size=1000))
        except SeriesError:
            raise
        except Exception as exc:
            raise PublicationUncertainError("Could not verify remote history; publication stopped without retransmitting") from exc
        with _lock(self.run_dir / ".records.lock"):
            index = _read(self.run_dir / "publication.json")
            seen = set()
            next_step = index["next_step"]
            for row in rows:
                rid = row.get(RECORD_KEY)
                if rid is None:
                    # Artifacts and summaries do not create history rows. Any
                    # other writer would invalidate the exclusive-owner rule.
                    raise SeriesError("Remote history contains an unrecognized row")
                if rid in seen or rid not in index["records"]:
                    raise SeriesError("Remote history contains duplicate or unknown results")
                seen.add(rid)
                entry = index["records"][rid]
                if row.get(HASH_KEY) != entry["record_sha256"] or row.get(X_AXIS) != entry["checkpoint_step"]:
                    raise SeriesError("Remote row differs from the accepted immutable result")
                internal_step = row.get("_step")
                if type(internal_step) is not int or internal_step < 0:
                    raise SeriesError("Remote row has an invalid internal history step")
                entry.update(status="published", wandb_step=internal_step)
                entry.pop("error", None)
                next_step = max(next_step, internal_step + 1)
            for rid, entry in index["records"].items():
                if entry["status"] == "published" and rid not in seen:
                    raise PublicationUncertainError("A previously acknowledged result is missing from remote history")
            index["next_step"] = next_step
            _atomic_json(self.run_dir / "publication.json", index)

    def publish_pending(self):
        if self.run is None:
            raise SeriesError("Use Publisher as a context manager")
        self._load_incoming()
        # Preserve actual arrival order. Plotting uses checkpoint position, not
        # publication order; later polling can append an earlier checkpoint.
        with _lock(self.run_dir / ".records.lock"):
            index = _read(self.run_dir / "publication.json")
            pending = [rid for rid, entry in index["records"].items() if entry["status"] in ("staged", "artifact_ready", "row_inflight", "queued")]
            # Queued rows from this open session must not be resent. Interrupted
            # rows still own their earlier explicit SDK slot and go first.
            pending = [rid for rid in pending if rid not in getattr(self, "_sent", set())]
            pending.sort(key=lambda rid: (index["records"][rid].get("wandb_step", float("inf")), index["records"][rid]["accepted_order"]))
        for rid in pending:
            try:
                self._publish_one(rid)
            except Exception as exc:
                with _lock(self.run_dir / ".records.lock"):
                    index = _read(self.run_dir / "publication.json")
                    index["records"][rid]["error"] = f"{type(exc).__name__}: {exc}"
                    _atomic_json(self.run_dir / "publication.json", index)
                raise
        index = _read(self.run_dir / "publication.json")
        return {"run_id": self.registry["run_id"], "queued": sum(e["status"] == "queued" for e in index["records"].values()), "published": sum(e["status"] == "published" for e in index["records"].values()), "accepted": len(index["records"])}

    def _load_incoming(self):
        incoming = self.run_dir / "incoming"
        if not incoming.exists():
            return
        from utils.eval_series_data import load_records
        for path in sorted(incoming.glob("*.json")):
            pointer = _read(path)
            records = load_records(pointer["result_path"], source_run=pointer["source_run"], inventory_path=pointer["inventory_path"])
            selector = pointer["selector"]
            if selector is not None:
                records = [record for record in records if record.get("selector") == selector or record.get("controller") == selector or record.get("provenance", {}).get("selector") == selector or record.get("provenance", {}).get("controller") == selector]
            if len(records) != 1:
                raise SeriesError("Result pointer must select exactly one planning configuration")
            stage_record(self.run_dir, records[0])
            completed = self.run_dir / "accepted-pointers"
            completed.mkdir(exist_ok=True)
            os.replace(path, completed / path.name)

    def _publish_one(self, rid):
        record = validate_record(_read(self.run_dir / "records" / (rid + ".json")), self.registry["identity"])
        with _lock(self.run_dir / ".records.lock"):
            entry = _read(self.run_dir / "publication.json")["records"][rid]
        files = {name: _file_digest(path) for name, path in record["artifact_files"].items()}
        if _record_fingerprint(record, files) != entry["record_sha256"]:
            raise SeriesError("Accepted result or artifact changed before publication")
        if entry["status"] == "staged":
            for name in record["provenance"].get("legacy_artifacts", []):
                self.run.use_artifact(name)
            artifact = self.wandb.Artifact("eval-" + self.registry["run_id"] + "-" + str(entry["checkpoint_step"]), type="evaluation-checkpoint", metadata={"record_id": rid, "record_sha256": entry["record_sha256"], "checkpoint_sha256": entry["checkpoint_sha256"]})
            for name, path in record["artifact_files"].items():
                artifact.add_file(path, name=name)
            artifact.add_file(str(self.run_dir / "records" / (rid + ".json")), name="evaluation-series-record.json")
            logged = self.run.log_artifact(artifact, aliases=["checkpoint-" + str(entry["checkpoint_step"]), "record-" + hashlib.sha256(rid.encode()).hexdigest()[:20]])
            logged.wait()
            with _lock(self.run_dir / ".records.lock"):
                index = _read(self.run_dir / "publication.json")
                index["records"][rid]["status"] = "artifact_ready"
                _atomic_json(self.run_dir / "publication.json", index)
        row = {X_AXIS: entry["checkpoint_step"], RECORD_KEY: rid, HASH_KEY: entry["record_sha256"], "checkpoint/sha256": entry["checkpoint_sha256"]}
        for key, value in record["metrics"].items():
            if key in row or key.startswith("publication/") or key.startswith("checkpoint/"):
                raise SeriesError("Normalized metrics must not overwrite publication identity")
            if isinstance(value, dict) and "nonfinite" in value:
                row["measurement_status/" + key] = value["nonfinite"]
            elif value is None or isinstance(value, (int, float, bool)):
                row[key] = value
            else:
                raise SeriesError("Normalized metrics must be scalar or explicit nonfinite measurements")
        with _lock(self.run_dir / ".records.lock"):
            index = _read(self.run_dir / "publication.json")
            step = index["records"][rid].get("wandb_step", index["next_step"])
            index["records"][rid].update(status="row_inflight", wandb_step=step)
            index["next_step"] = max(index["next_step"], step + 1)
            _atomic_json(self.run_dir / "publication.json", index)
        self.run.log(row, step=step, commit=True)
        if not hasattr(self, "_sent"):
            self._sent = set()
        self._sent.add(rid)
        with _lock(self.run_dir / ".records.lock"):
            index = _read(self.run_dir / "publication.json")
            index["records"][rid]["status"] = "queued"
            index["records"][rid].pop("error", None)
            _atomic_json(self.run_dir / "publication.json", index)

    def finish(self, exit_code=0):
        if self.run is None:
            return
        run, self.run = self.run, None
        run.finish(exit_code=exit_code)
        deadline = time.monotonic() + self.acknowledgement_timeout
        while True:
            try:
                self._reconcile()
                pending = [e for e in _read(self.run_dir / "publication.json")["records"].values() if e["status"] in ("row_inflight", "queued")]
                if not pending:
                    return
                error = PublicationUncertainError("W&B finished but has not acknowledged all rows; retained journal requires reconciliation")
            except PublicationUncertainError as exc:
                error = exc
            if time.monotonic() >= deadline:
                raise error
            time.sleep(min(self.poll_interval, max(0, deadline - time.monotonic())))

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.finish(exit_code=1 if exc_type else 0)
        except Exception:
            if exc_type is None:
                raise
        finally:
            if self._guard is not None:
                self._guard.__exit__(exc_type, exc, traceback)
                self._guard = None
        return False


def publish_run(run_dir, owner=None, wandb_module=None, **kwargs):
    """Publish a staged batch, flush, and verify its remote receipts."""
    with Publisher(run_dir, owner=owner, wandb_module=wandb_module, **kwargs) as publisher:
        publisher.publish_pending()
    index = _read(Path(run_dir) / "publication.json")
    return {"run_id": load_run(run_dir)["run_id"], "published": sum(e["status"] == "published" for e in index["records"].values()), "accepted": len(index["records"])}
