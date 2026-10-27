"""Evaluate J10 at three selected 500k checkpoints, retaining original H3/K1 or K3."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slurm import ambi_return_cadence_eval as cadence

MATRIX = cadence.ROOT / "configs/research/return_prior_j10.json"
CAMPAIGN_ID = "return-prior-j10-20261026"


def selection(index):
    cadence.require(index in (0, 1), "Only the two selected J8 checkpoints may use this J10 evaluation")
    return "rounds/j10", (1, 3)[index]


def validate_results(payload, entry, index, *, smoke=False, smoke_steps=6):
    return cadence.validate_results(payload, entry, index, smoke=smoke, smoke_steps=smoke_steps,
                                    selected=selection(index), horizon=3, rounds=10, replay_capacity=3840)


def validate_third_results(payload, entry, *, smoke=False, smoke_steps=6):
    counts = cadence.original.validate_results(payload, entry, smoke=smoke, smoke_steps=smoke_steps)
    sac = next(row for row in payload["results"] if row["selector"] == "comparison/sac")
    work = cadence.validate_results({**payload, "results": [sac]}, entry, 2,
        smoke=smoke, smoke_steps=smoke_steps, selected=("comparison/sac", 3),
        horizon=3, rounds=10, replay_capacity=3840)
    return {**counts, "solves": work["solves"]}


def run(args):
    inventory = cadence.read_json(args.inventory)
    cadence.require(inventory.get("campaign_id") == CAMPAIGN_ID, "Expected the frozen J10 campaign inventory")
    if args.index == 2:
        entry = cadence.verify_entry(inventory, 2)
        saved = cadence.read_json(entry["metadata"])["trial_run_params"]["alg_params"]
        cadence.require((entry["step"], entry["J"], entry["K"], saved.get("inner_replay_capacity")) ==
                        (500000, 10, 3, 3840), "Third reference requires the J10 K3 500k checkpoint with replay3840")
        return cadence.original.run(args, validator=validate_third_results, receipt_fields={
            "campaign_id": CAMPAIGN_ID, "source_J": 10, "eval_J": 10,
            "source_K": 3, "eval_K": 3, "source_H": 3, "eval_H": 3})
    return cadence.run(args, matrix=MATRIX, select=selection, validator=validate_results,
                       campaign_id=CAMPAIGN_ID, rounds=10,
                       receipt_fields={"source_J": 8, "eval_J": 10, "source_H": 3, "eval_H": 3})


if __name__ == "__main__":
    cadence.main(run_fn=run, description=__doc__)
