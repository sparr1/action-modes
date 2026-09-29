"""Bind six successful short GPU MPPI evaluations to their immutable inputs.

This gate certifies six representative checkpoints, not all 120 production
checkpoints. It reads JSON and diagnostic traces only; it never loads a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_ambixqc_mppi_evaluation import MATRIX, validate_bundle

SMOKE_INDICES = (0, 39, 40, 79, 80, 119)
SCHEMA = "ambixqc-backbone-mppi-smoke-gate-v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON in {path}: {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject_constant)


def _identity(manifest, source_sha):
    if not isinstance(source_sha, str) or re.fullmatch(r"[0-9a-f]{40}", source_sha) is None:
        raise ValueError("Expected a full lowercase evaluation source SHA.")
    manifest = Path(manifest).resolve(strict=True)
    inventory = read_json(manifest)
    rows = inventory.get("checkpoints", [])
    if (inventory.get("schema") != "ambixqc-backbone-mppi-inventory-v1"
            or len(rows) != 120):
        raise ValueError("Expected the complete six-bank, 120-checkpoint inventory.")
    return {
        "schema": SCHEMA,
        "source_sha": source_sha,
        "manifest_sha256": sha256(manifest),
        "matrix_sha256": sha256(MATRIX),
        "smoke_indices": list(SMOKE_INDICES),
        "checkpoint_count_validated": len(SMOKE_INDICES),
        "production_checkpoint_count": 120,
    }, inventory


def _receipt(validation_path, identity, inventory):
    path = Path(validation_path).resolve(strict=True)
    validation = read_json(path)
    index = validation.get("index")
    if type(index) is not int or index not in SMOKE_INDICES:
        raise ValueError("Validation is not one of the six representative smoke indices.")
    row = inventory["checkpoints"][index]
    expected = {
        "source_sha": identity["source_sha"],
        "manifest_sha256": identity["manifest_sha256"],
        "matrix_sha256": identity["matrix_sha256"],
        "training_source_sha": inventory["training_source_sha"],
        "mode": "smoke", "index": index, "cell": row["cell"], "step": row["step"],
        "outer_state_unchanged": True, "optimizer_updates": 0,
        "decision_counts": {"controller/prior": 6, "controller/mppi": 6},
        "mppi_model_steps_per_decision": 12_336,
    }
    if any(validation.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Smoke index {index} validation does not match this source and protocol.")
    if validation.get("outer_state_unchanged") is not True or type(validation.get("optimizer_updates")) is not int:
        raise ValueError("Smoke frozen-state checks have invalid types.")
    destination = path.parent
    if destination.name != f"step_{row['step']}" or destination.parent.name != row["cell"]:
        raise ValueError("Smoke output path does not identify its checkpoint.")
    job_root = destination.parent.parent
    for marker in (destination / "PASS", job_root / "PASS"):
        if marker.read_text().strip() != "PASS":
            raise ValueError(f"Missing successful smoke completion marker: {marker}")
    if (job_root / "FAILED").exists():
        raise ValueError("Smoke job has a failure marker.")
    manifest = read_json(destination / "bundle/manifest.json")
    code = manifest.get("code", {})
    if code.get("commit") != identity["source_sha"] or code.get("dirty") is not False:
        raise ValueError("Smoke bundle was not evaluated with the clean expected source.")
    checkpoint = manifest.get("checkpoint", {})
    if checkpoint.get("sha256") != row["sha256"] or checkpoint.get("metadata_sha256") != row["metadata_sha256"]:
        raise ValueError("Smoke bundle checkpoint hashes do not match the source inventory.")
    checked = validate_bundle(destination / "bundle", seeds=[101, 102], max_steps=3)
    if any(validation.get(key) != value for key, value in checked.items()):
        raise ValueError("Stored smoke validation differs from its diagnostic traces.")
    files = sorted(file for file in destination.rglob("*") if file.is_file())
    files += [job_root / "PASS", job_root / "runtime.json", job_root / "gpu.txt"]
    if index == 0:
        log = job_root / "pytest.log"
        if not log.is_file() or re.search(r"\b[1-9][0-9]* passed\b", log.read_text()) is None:
            raise ValueError("The first smoke task must pass the GPU regression suite.")
        files.append(log)
    hashes = {}
    for file in files:
        resolved = file.resolve(strict=True)
        if job_root not in resolved.parents:
            raise ValueError("Smoke artifact escapes its owning job directory.")
        hashes[str(file.relative_to(job_root))] = sha256(file)
    return {"index": index, "cell": row["cell"], "step": row["step"],
            "validation_path": str(path), "file_sha256": hashes}


def create_gate(smoke_root, manifest, source_sha, output):
    identity, inventory = _identity(manifest, source_sha)
    paths = list(Path(smoke_root).resolve(strict=True).rglob("validation.json"))
    receipts = [_receipt(path, identity, inventory) for path in paths]
    receipts.sort(key=lambda item: item["index"])
    if [receipt["index"] for receipt in receipts] != list(SMOKE_INDICES):
        raise ValueError("Require exactly six successful, unique smoke validations at indices 0,39,40,79,80,119.")
    payload = {**identity, "receipts": receipts}
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to replace an existing smoke gate: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".mppi-smoke-gate-", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Publish once, without a race that could overwrite an existing gate.
        os.link(temporary, output)
    finally:
        os.unlink(temporary)
    return payload


def verify_gate(gate_path, manifest, source_sha):
    identity, inventory = _identity(manifest, source_sha)
    gate = read_json(gate_path)
    if any(gate.get(key) != value for key, value in identity.items()):
        raise ValueError("Smoke gate belongs to a different source, inventory or protocol.")
    receipts = gate.get("receipts", [])
    if [receipt.get("index") for receipt in receipts] != list(SMOKE_INDICES):
        raise ValueError("Smoke gate does not contain exactly the six required indices.")
    for receipt in receipts:
        current = _receipt(receipt["validation_path"], identity, inventory)
        if current != receipt:
            raise ValueError(f"Smoke artifacts changed for checkpoint index {receipt['index']}.")
    return gate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("create", "verify"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--smoke-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gate", type=Path)
    args = parser.parse_args(argv)
    if args.command == "create":
        if args.smoke_root is None or args.output is None:
            parser.error("create requires --smoke-root and --output")
        result = create_gate(args.smoke_root, args.manifest, args.source_sha, args.output)
    else:
        if args.gate is None:
            parser.error("verify requires --gate")
        result = verify_gate(args.gate, args.manifest, args.source_sha)
    print(json.dumps({key: result[key] for key in (
        "source_sha", "manifest_sha256", "smoke_indices", "checkpoint_count_validated",
        "production_checkpoint_count",
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
