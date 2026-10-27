"""Derive one longer, state-preserving continuation from a completed lineage.

Run this offline tool from the tooling checkout. Training continues with the
original clean source checkout and its original environment. No resume checks
are relaxed, and this module never initializes W&B or a learner.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import tempfile
import uuid

from utils.oscar_resume_launcher import verify_done
from utils.resume_identity import (
    canonical_json,
    dependency_identity,
    lineage_identity,
    source_identity,
)
from utils.resume_lineage import GenerationFile, LineageStore
from utils.resume_training import (
    _copy_writer,
    _atomic_json,
    _envelope,
    _fsync_directory,
    _load_torch,
    _torch_writer,
)


class ExtensionError(ValueError):
    """The source cannot safely be extended by this narrow migration."""


def _positive_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ExtensionError(f"{label} must be a positive integer.")
    return value


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)


def _extended_configuration(identity, *, target, capacity, training_repo):
    """Allow only a larger target and an unchanged effective replay capacity."""
    scientific = identity.get("scientific", {})
    if identity.get("schema_version") != 1 or identity.get("fingerprint") != (
        hashlib.sha256(canonical_json(scientific).encode()).hexdigest()
    ):
        raise ExtensionError("Source scientific identity has an invalid fingerprint.")
    if scientific.get("source") != source_identity(training_repo):
        raise ExtensionError("Training checkout must exactly match the original clean source.")
    if scientific["source"].get("dirty"):
        raise ExtensionError("Extending a dirty source identity is unsupported.")
    if scientific.get("dependencies") != dependency_identity():
        raise ExtensionError("Use the original training Python environment and dependencies.")

    run = copy.deepcopy(scientific["trial_run_params"])
    experiment = copy.deepcopy(scientific["experiment_params"])
    old_target = _positive_int(run.get("total_steps"), "Original target")
    if target <= old_target:
        raise ExtensionError("New target must exceed the completed target.")
    horizon = _positive_int(run.get("resume_environment", {}).get("episode_steps"), "Episode length")
    if target % horizon:
        raise ExtensionError("New target must end at an episode boundary.")
    configs = experiment.get("configs")
    if experiment.get("trials") != 1 or not isinstance(configs, list) or len(configs) != 1:
        raise ExtensionError("Extension requires one algorithm and one trial.")
    name = configs[0]
    if not isinstance(name, str) or not name or Path(name).name != name or name in {".", ".."}:
        raise ExtensionError("Invalid algorithm config basename.")
    algorithm = run.get("alg_params", {})
    old_buffer_size = _positive_int(algorithm.get("buffer_size"), "Original buffer size")
    if min(old_buffer_size, old_target) != capacity:
        raise ExtensionError("Saved replay capacity disagrees with the original configuration.")
    run["total_steps"] = target
    algorithm["buffer_size"] = capacity
    overrides = experiment.get("overrides_alg", {})
    if "total_steps" in overrides:
        overrides["total_steps"] = target
    if "alg_params" in overrides:
        # main.py shallow-replaces alg_params, so keep the complete old mapping.
        overrides["alg_params"]["buffer_size"] = capacity

    # Re-resolve the emitted full algorithm plus manifest exactly as main does.
    resolved = {"name": name, **copy.deepcopy(run), **copy.deepcopy(overrides)}
    if resolved != run:
        raise ExtensionError("Manifest overrides do not reproduce the saved resolved configuration.")
    new_identity = lineage_identity(
        trial_run_params=resolved, experiment_params=experiment, repo_root=training_repo
    )
    expected = copy.deepcopy(scientific)
    expected["trial_run_params"]["total_steps"] = target
    expected["trial_run_params"]["alg_params"]["buffer_size"] = capacity
    old_overrides = expected["experiment_params"].get("overrides_alg", {})
    if "total_steps" in old_overrides:
        old_overrides["total_steps"] = target
    if "alg_params" in old_overrides:
        old_overrides["alg_params"]["buffer_size"] = capacity
    if new_identity["scientific"] != expected:
        raise ExtensionError("Configuration normalization would change scientific settings.")
    return new_identity, run, experiment, name


def extend_completed_training(*, source_lineage, total_steps, training_repo, destination):
    """Publish one derived lineage; never alter source checkpoint artifacts.

    A durable sibling claim plus the source's exclusive lease prevent multiple
    destinations from continuing one W&B identity. Same-request retries recover
    interrupted staging. Further extensions use the completed child as source.
    """
    target = _positive_int(total_steps, "New target")
    source = Path(source_lineage).resolve(strict=True)
    training_repo = Path(training_repo).resolve(strict=True)
    destination = Path(destination).resolve()
    if destination == source or source in destination.parents or destination in source.parents:
        raise ExtensionError("Destination must be separate from the source lineage.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    claim_path = source.with_name(source.name + ".extension-claim.json")
    staging = None
    with LineageStore.open(source, mode="required") as original:
        source_generation_id = verify_done(lineage_dir=source)
        generation = original.load()  # Verify every payload checksum before copying.
        if generation.generation_id != source_generation_id:
            raise ExtensionError("Completed generation changed during preflight.")
        metadata = original.lineage_metadata
        old_target = _positive_int(metadata.get("total_steps"), "Original target")
        trainers = generation.files_for_role("trainer")
        replay = [path for path in generation.files_for_role("replay") if path.name == "metadata.pt"]
        if len(trainers) != 1 or len(replay) != 1:
            raise ExtensionError("Source needs exactly one trainer and replay metadata file.")
        envelope = _envelope(_load_torch(trainers[0], "trainer"))
        trainer = envelope["trainer"]
        if (
            envelope["generation_id"] != generation.generation_id
            or trainer.get("total_steps") != old_target
            or trainer.get("global_step") != old_target
            or trainer.get("phase") != "between_episodes_before_reset"
            or trainer.get("eval_pending") is not False
            or generation.metadata.get("total_steps") != old_target
        ):
            raise ExtensionError("Source is not a fully completed episode-boundary checkpoint.")
        wandb_run_id = envelope["wandb"].get("run_id")
        if not wandb_run_id or generation.metadata.get("wandb_run_id") != wandb_run_id:
            raise ExtensionError("Source W&B identity is inconsistent.")
        replay_metadata = _load_torch(replay[0], "replay metadata")
        capacity = _positive_int(replay_metadata.get("signature", {}).get("capacity"), "Replay capacity")
        identity, run, experiment, name = _extended_configuration(
            metadata["scientific_identity"], target=target, capacity=capacity,
            training_repo=training_repo,
        )
        if metadata["scientific_identity"]["scientific"]["trial_run_params"]["total_steps"] != old_target:
            raise ExtensionError("Scientific identity and completed target disagree.")
        # These must be the entire resident replay, not a weights-only export.
        if (replay_metadata.get("total_transitions") != old_target
                or replay_metadata.get("num_eps") != trainer.get("completed_episodes")
                or replay_metadata.get("storage_rows") != generation.metadata.get("replay_rows")):
            raise ExtensionError("Trainer and replay progress disagree.")

        new_generation_id = f"extend-{old_target:012d}-to-{target:012d}-{uuid.uuid4().hex[:12]}"
        provenance = {
            "schema_version": 1,
            "operation": "extend-completed-training",
            "source_lineage": str(source),
            "source_generation": generation.generation_id,
            "source_manifest_sha256": generation.manifest_sha256,
            "source_identity_fingerprint": metadata["scientific_identity"]["fingerprint"],
            "training_source": identity["scientific"]["source"],
            "from_total_steps": old_target,
            "to_total_steps": target,
            "replay_capacity": capacity,
            "wandb_run_id": wandb_run_id,
        }
        request = {"destination": str(destination), **provenance}
        claim = None
        if claim_path.exists() or claim_path.is_symlink():
            if claim_path.is_symlink():
                raise ExtensionError("Extension claim must not be a symlink.")
            claim = json.loads(claim_path.read_text())
            if claim.get("request") != request:
                raise ExtensionError(f"Source already has a different continuation claim: {claim_path}")
        if destination.exists() or destination.is_symlink():
            if claim is None or destination.is_symlink():
                raise ExtensionError(f"Unclaimed continuation destination already exists: {destination}")
            # An interrupted post-rename bookkeeping write is recoverable.
            with LineageStore.open(destination, mode="required") as existing:
                if existing.lineage_metadata.get("extension") != provenance:
                    raise ExtensionError("Existing destination does not match the continuation claim.")
                current = existing.load()
            result = _result(destination, training_repo, old_target, target, capacity, wandb_run_id,
                             claim.get("generation_id") or current.generation_id)
            _atomic_json(claim_path, {**claim, "status": "committed", "staging": None,
                                      "generation_id": result["generation_id"]})
            return result
        if claim is not None and claim.get("status") == "committed":
            raise ExtensionError("Committed continuation is missing; refusing a duplicate continuation.")
        if claim is not None and claim.get("staging"):
            previous = Path(claim["staging"])
            owner = previous / "OWNER.json"
            if (previous.parent == destination.parent and not previous.is_symlink()
                    and previous.name.startswith(f".{destination.name}.extension-")
                    and owner.is_file() and json.loads(owner.read_text()) == request):
                shutil.rmtree(previous)
            elif previous.exists():
                raise ExtensionError("Interrupted staging ownership cannot be verified.")
        claim = {"schema_version": 1, "status": "reserved", "request": request,
                 "staging": None, "generation_id": new_generation_id}
        _atomic_json(claim_path, claim)
        migrated = copy.deepcopy(envelope)
        migrated["generation_id"] = new_generation_id
        migrated["trainer"]["total_steps"] = target
        new_metadata = copy.deepcopy(metadata)
        new_metadata.update(scientific_identity=identity, total_steps=target, extension=provenance)
        generation_metadata = copy.deepcopy(generation.metadata)
        generation_metadata.update(total_steps=target, reason="completed-target-extension", extension=provenance)
        try:
            # The visible destination appears only after all state/configs are durable.
            staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.extension-", dir=destination.parent))
            _write_json(staging / "OWNER.json", request)
            claim["staging"] = str(staging)
            _atomic_json(claim_path, claim)
            lineage_staging = staging / "lineage"
            with LineageStore.open(lineage_staging, mode="new", lineage_metadata=new_metadata) as derived:
                files = [GenerationFile(
                    record.path, record.role,
                    _torch_writer(migrated) if record.role == "trainer"
                    else _copy_writer(generation.file_path(record.path)),
                ) for record in generation.files]
                derived.publish(new_generation_id, files=files, metadata=generation_metadata)
                config_root = lineage_staging / "configs"
                _write_json(config_root / "experiment.json", experiment)
                _write_json(config_root / "algs" / f"{name}.json", run)
                _fsync_directory(config_root)
                _write_json(lineage_staging / "EXTENSION.json", provenance)
                _fsync_directory(lineage_staging)
                # Validate the committed destination payloads before exposing it.
                derived.load()
                os.rename(lineage_staging, destination)
                _fsync_directory(destination.parent)
            claim.update(status="committed", staging=None)
            _atomic_json(claim_path, claim)
            return _result(destination, training_repo, old_target, target, capacity,
                           wandb_run_id, new_generation_id)
        finally:
            if staging is not None:
                shutil.rmtree(staging)


def _result(destination, training_repo, old_target, target, capacity, run_id, generation_id):
    return {
        "lineage_dir": str(destination),
        "run_config": str(destination / "configs" / "experiment.json"),
        "alg_dir": str(destination / "configs" / "algs"),
        "training_repo": str(training_repo),
        "global_step": old_target,
        "total_steps": target,
        "replay_capacity": capacity,
        "wandb_run_id": run_id,
        "generation_id": generation_id,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-lineage", required=True)
    parser.add_argument("--total-steps", type=int, required=True)
    parser.add_argument("--training-repo", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args(argv)
    result = extend_completed_training(**vars(args))
    result["resume_command"] = shlex.join([
        sys.executable, "main.py", "--run", result["run_config"],
        "--alg-dir", result["alg_dir"], "--num-runs", "1",
        "--lineage-dir", result["lineage_dir"], "--resume-mode", "required",
        "--resume-wandb-mode", "online",
    ])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
