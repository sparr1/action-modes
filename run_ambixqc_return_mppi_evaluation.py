"""Return-only MPPI on four auxiliary-enabled 500k banks, reusing prior episodes."""
from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
import json

import run_ambixqc_backbone_mppi_evaluation as banks
from run_ambixqc_mppi_evaluation import file_sha256, validate_bundle

ROOT = Path(__file__).resolve().parent
MATRIX = ROOT / "configs/research/ambixqc_humanoid_return_mppi_benchmark.json"
CELLS = tuple(cell for cell in banks.CELLS if cell.startswith("aux_"))
INDICES = tuple(index for bank_index, cell in enumerate(banks.CELLS) if cell in CELLS
                for index in range(bank_index * 20, (bank_index + 1) * 20))
SELECTOR = "controller/mppi_return"
TERMINAL_SOURCE = "online_aux_return_twin_mean"
MAP_SCHEMA = "ambixqc-return-mppi-run-map-v1"
REFERENCE_SCHEMA = "ambixqc-return-mppi-prior-references-v1"
SMOKE_INDICES = (0, 39, 40, 79)
validate_return_bundle = partial(validate_bundle, mppi_selector=SELECTOR,
                                 terminal_value_source=TERMINAL_SOURCE)


def select_checkpoint(manifest, index, *, checkpoint_root=None):
    if type(index) is not int or not 0 <= index < len(INDICES):
        raise ValueError("Return-only checkpoint index must be 0 through 79.")
    return banks.select_checkpoint(manifest, INDICES[index], checkpoint_root=checkpoint_root)


def select_reference(path, row, manifest):
    if path is None:
        raise ValueError("Production requires a pinned prior-reference index.")
    data = banks.read_json(path)
    refs = data.get("references", [])
    if (data.get("schema") != REFERENCE_SCHEMA
            or data.get("checkpoint_manifest_sha256") != file_sha256(manifest)
            or [(r.get("cell"), r.get("step")) for r in refs]
            != [(cell, step) for cell in CELLS for step in banks.STEPS]):
        raise ValueError("Prior-reference index must pin the same 80 auxiliary checkpoints.")
    record = next(r for r in refs if (r["cell"], r["step"]) == (row["cell"], row["step"]))
    bundle = Path(record["bundle_path"])
    if not bundle.is_absolute() or not (bundle / "manifest.json").is_file():
        raise ValueError("Prior reference must be an existing absolute bundle directory.")
    if (record.get("checkpoint_sha256") != row["sha256"]
            or file_sha256(bundle / "manifest.json") != record.get("manifest_sha256")):
        raise ValueError("Prior reference identity or manifest hash differs.")
    saved = banks.read_json(bundle / "manifest.json")
    if saved.get("checkpoint", {}).get("sha256") != row["sha256"]:
        raise ValueError("Prior reference checkpoint association differs.")
    return bundle


def resolve_run_map(path, cell, mode):
    if mode == "smoke":
        if path is not None:
            raise ValueError("Smoke must not assign evaluation curves.")
        return {}
    if path is None:
        raise ValueError("Production requires four explicit New evaluation curves.")
    data = banks.read_json(path)
    if data.get("schema") != MAP_SCHEMA or set(data.get("cells", {})) != set(CELLS):
        raise ValueError("Return MPPI map must contain exactly the four auxiliary cells.")
    paths = []
    for entries in data["cells"].values():
        if not isinstance(entries, dict) or set(entries) != {SELECTOR}:
            raise ValueError("Each cell must assign only its return-only MPPI curve.")
        value = entries[SELECTOR]
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError("Evaluation curve directories must be absolute.")
        paths.append(str(Path(value).resolve()))
    if len(set(paths)) != len(CELLS):
        raise ValueError("Each auxiliary cell requires a distinct curve.")
    from utils.ambi_benchmark import resolve_eval_run_map
    return resolve_eval_run_map([SELECTOR], run_map=data["cells"][cell])


def prepare_specs(manifest, index, directory, *, checkpoint_root=None):
    row = select_checkpoint(manifest, index, checkpoint_root=checkpoint_root)
    from evaluate_ambi_checkpoint import evaluate_matrix
    return evaluate_matrix(MATRIX, row["path"], selectors=[SELECTOR],
                           checkpoint_inventory=manifest, source_run=row["source_run"],
                           eval_series_spec_dir=directory)


def run(manifest, index, result_root, *, mode="production", device="cuda",
        eval_run_map=None, checkpoint_root=None, reference_index=None):
    if mode not in ("smoke", "production"):
        raise ValueError("Mode must be smoke or production.")
    row = select_checkpoint(manifest, index, checkpoint_root=checkpoint_root)
    assigned = resolve_run_map(eval_run_map, row["cell"], mode)
    reference = select_reference(reference_index, row, manifest) if mode == "production" else None
    from evaluate_ambi_checkpoint import evaluate_matrix
    from utils.ambi_benchmark import atomic_json, code_identity, stage_completed_bundle
    from report_ambi_benchmark import load_bundles, write_report
    code = code_identity()
    if code.get("dirty") is not False or not code.get("commit"):
        raise ValueError("Evaluation requires a clean committed checkout.")
    output = Path(result_root).resolve() / row["cell"] / f"step_{row['step']}"
    if output == ROOT or ROOT in output.parents:
        raise ValueError("Results must be outside the source checkout.")
    output.mkdir(parents=True, exist_ok=False)
    seeds, steps = ([101, 102], 3) if mode == "smoke" else (list(range(101, 106)), 500)
    selectors = ["controller/prior", SELECTOR] if mode == "smoke" else [SELECTOR]
    provenance = {"source_sha": code["commit"], "training_source_sha": banks.TRAINING_SOURCE_SHA,
                  "manifest_sha256": file_sha256(manifest), "matrix_sha256": file_sha256(MATRIX),
                  "index": index, "inventory_index": INDICES[index], "cell": row["cell"],
                  "step": row["step"], "checkpoint": row, "mode": mode,
                  "seeds": seeds, "max_steps": steps, "controller_seed": 12345,
                  "terminal_value_source": TERMINAL_SOURCE, "replay_used": False,
                  "source_run": row["source_run"], "eval_run_map": assigned,
                  "reference_bundle": str(reference) if reference is not None else None,
                  "reference_index_sha256": file_sha256(reference_index) if reference is not None else None}
    atomic_json(output / "provenance.json", provenance)
    payload = evaluate_matrix(MATRIX, row["path"], selectors=selectors, seeds=seeds,
                              controller_seed=12345, max_steps=steps, device=device,
                              bundle_dir=output / "bundle", reference_bundle=reference,
                              checkpoint_inventory=manifest, source_run=row["source_run"],
                              eval_run_map=assigned or None, stage_results=False)
    atomic_json(output / "results.json", payload)
    if payload["checkpoint_sha256"] != row["sha256"]:
        raise ValueError("Evaluated checkpoint differs from inventory.")
    validation = validate_return_bundle(output / "bundle", seeds=seeds, max_steps=steps,
                                         reference_bundle=reference)
    atomic_json(output / "validation.json", {**provenance, **validation})
    write_report(load_bundles([output / "bundle"]), output / "comparison.html",
                 title=f"AMBI-XQC {row['cell']} return-only MPPI at {row['step']:,} decisions")
    if assigned:
        staged = stage_completed_bundle(output / "bundle", assigned,
                                        source_run=row["source_run"], inventory_path=manifest)
        if set(staged) != {SELECTOR} or staged[SELECTOR]["status"] != "queued":
            raise RuntimeError("Failed to stage the return-only curve; preserve results for upload recovery.")
    (output / "PASS").write_text("PASS\n")
    print(json.dumps({"cell": row["cell"], "step": row["step"], "output": str(output), **validation}))
    return output


def verify_smokes(smoke_root, manifest, source_sha):
    paths = sorted(Path(smoke_root).glob("job*-task*/*/step_*/validation.json"))
    if len(paths) != len(SMOKE_INDICES):
        raise ValueError("Require exactly four successful return-only smoke evaluations.")
    seen = []
    for path in paths:
        saved = banks.read_json(path)
        index = saved["index"]
        if type(index) is not int or index not in SMOKE_INDICES:
            raise ValueError("Unexpected return-only smoke checkpoint.")
        row = banks.load_inventory(manifest)[0]["checkpoints"][INDICES[index]]
        job = path.parent.parent.parent
        expected = {"source_sha": source_sha, "manifest_sha256": file_sha256(manifest),
                    "matrix_sha256": file_sha256(MATRIX), "mode": "smoke", "cell": row["cell"],
                    "step": row["step"], "terminal_value_source": TERMINAL_SOURCE}
        if any(saved.get(key) != value for key, value in expected.items()):
            raise ValueError("Smoke source, checkpoint or terminal semantics differ.")
        if saved["checkpoint"]["sha256"] != row["sha256"]:
            raise ValueError("Smoke checkpoint hash differs.")
        bundle = banks.read_json(path.parent / "bundle/manifest.json")
        if bundle["code"].get("commit") != source_sha or bundle["code"].get("dirty") is not False:
            raise ValueError("Smoke source is not the expected clean checkout.")
        for marker in (job / "PASS", path.parent / "PASS"):
            if not marker.is_file() or marker.read_text().strip() != "PASS":
                raise ValueError("Smoke job is incomplete.")
        if (job / "FAILED").exists():
            raise ValueError("Smoke job failed.")
        actual = validate_return_bundle(path.parent / "bundle", seeds=[101, 102], max_steps=3)
        if any(saved.get(key) != value for key, value in actual.items()):
            raise ValueError("Smoke validation differs from its diagnostics.")
        if index == 0:
            import re
            if not re.search(r"\b[1-9][0-9]* passed\b", (job / "pytest.log").read_text()):
                raise ValueError("GPU regression tests did not pass.")
        seen.append(index)
    if sorted(seen) != list(SMOKE_INDICES):
        raise ValueError("Smoke checkpoints are duplicated or missing.")
    return {"validated_indices": sorted(seen), "source_sha": source_sha}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--checkpoint-root", type=Path)
    p.add_argument("--index", type=int)
    p.add_argument("--result-root", type=Path)
    p.add_argument("--mode", choices=("smoke", "production"), default="production")
    p.add_argument("--device", default="cuda")
    p.add_argument("--eval-run-map", type=Path)
    p.add_argument("--reference-index", type=Path)
    p.add_argument("--eval-series-spec-dir", type=Path)
    p.add_argument("--verify-smoke-root", type=Path)
    p.add_argument("--source-sha")
    args = p.parse_args(argv)
    if args.verify_smoke_root:
        print(json.dumps(verify_smokes(args.verify_smoke_root, args.manifest, args.source_sha)))
    elif args.index is None:
        p.error("--index is required")
    elif args.eval_series_spec_dir:
        print(json.dumps(prepare_specs(args.manifest, args.index, args.eval_series_spec_dir,
                                        checkpoint_root=args.checkpoint_root)))
    elif args.result_root is None:
        p.error("--result-root is required")
    else:
        run(args.manifest, args.index, args.result_root, mode=args.mode, device=args.device,
            eval_run_map=args.eval_run_map, checkpoint_root=args.checkpoint_root,
            reference_index=args.reference_index)


if __name__ == "__main__":
    main()
