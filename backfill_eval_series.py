"""Validate, then optionally publish paired prior gains to an existing curve."""
import argparse
import json
from pathlib import Path

from utils import eval_series as series
from utils.eval_series_data import load_records
from utils.eval_series_paired import prepare_paired_reference, stage_paired_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--prior-bundles", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint-inventory", type=Path, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--publish", action="store_true", help="Otherwise perform read-only validation")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    registry = series.load_run(args.run_dir)
    if args.owner != registry["owner"]:
        raise series.SeriesError("Only the registered owner may backfill")
    index = series._read(args.run_dir / "publication.json")
    originals = {e["checkpoint_step"]: rid for rid, e in index["records"].items()
                 if not e.get("record_kind")}
    priors = {}
    for path in args.prior_bundles:
        records = load_records(path, inventory_path=args.checkpoint_inventory)
        if len(records) != 1 or records[0]["checkpoint"]["step"] in priors:
            raise series.SeriesError("Supply one prior bundle per checkpoint")
        priors[records[0]["checkpoint"]["step"]] = records[0]
    if set(priors) != set(originals):
        raise series.SeriesError("Prior bundle grid must match the complete original curve")
    prepared = [prepare_paired_reference(args.run_dir, originals[step], priors[step])
                for step in sorted(priors)]
    receipt = {"run_id": registry["run_id"], "published": False,
               "points": [{"checkpoint": r["checkpoint"], "metrics": r["metrics"],
                           "record_id": r["record_id"], "provenance": r["provenance"]}
                          for r in prepared]}
    if args.publish:
        # Exclusive lock and remote reconciliation precede journal mutation.
        with series.Publisher(args.run_dir, owner=args.owner) as publisher:
            for step in sorted(priors):
                stage_paired_reference(args.run_dir, originals[step], priors[step])
            publisher.publish_pending()
        final = series._read(args.run_dir / "publication.json")
        if any(final["records"][r["record_id"]]["status"] != "published" for r in prepared):
            raise series.PublicationUncertainError("Backfill acknowledgement incomplete")
        receipt["published"] = True
    if args.receipt:
        series._atomic_json(args.receipt, receipt)
    print(json.dumps({"run_id": receipt["run_id"], "published": receipt["published"],
                      "paired_checkpoints": len(prepared)}, indent=2))


if __name__ == "__main__":
    main()
