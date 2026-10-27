"""Evaluate the two missing J10/H3/K1 arms, reusing each checkpoint's prior."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slurm import ambi_return_cadence_eval as cadence

MATRIX = cadence.ROOT / "configs/research/return_prior_j10_k1.json"
CAMPAIGN_ID = "return-prior-j10-k1-20261026"


def selection(index):
    cadence.require(index in (1, 2), "Only indices1/2 are missing J10 K1; reuse index0")
    return "rounds/j10_k1", 1


def validate_results(payload, entry, index, *, smoke=False, smoke_steps=6):
    return cadence.validate_results(payload, entry, index, smoke=smoke, smoke_steps=smoke_steps,
                                    selected=selection(index), horizon=3, rounds=10, replay_capacity=3840)


def run(args):
    selection(args.index)
    inventory = cadence.read_json(args.inventory)
    cadence.require(inventory.get("campaign_id") == CAMPAIGN_ID, "Expected the frozen J10 K1 campaign inventory")
    source_rounds = 8 if args.index == 1 else 10
    return cadence.run(args, matrix=MATRIX, select=selection, validator=validate_results,
                       campaign_id=CAMPAIGN_ID, rounds=10, source_rounds=source_rounds,
                       receipt_fields={"source_J": source_rounds, "eval_J": 10, "source_H": 3, "eval_H": 3})


if __name__ == "__main__":
    cadence.main(run_fn=run, description=__doc__)
