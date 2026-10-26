"""Evaluate H4 K1 on both selected J8 checkpoints without rerunning controls."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slurm import ambi_return_cadence_eval as cadence

MATRIX = cadence.ROOT / "configs/research/return_prior_h4_k1.json"
CAMPAIGN_ID = "return-prior-h4-k1-20261026"


def selection(index):
    cadence.require(index in (0, 1), "Only the two selected J8 checkpoints may use H4")
    return "horizon/h4_k1", 1


def validate_results(payload, entry, index, *, smoke=False, smoke_steps=6):
    return cadence.validate_results(payload, entry, index, smoke=smoke, smoke_steps=smoke_steps,
                                    selected=selection(index), horizon=4, replay_capacity=4096)


def run(args):
    return cadence.run(args, matrix=MATRIX, select=selection, validator=validate_results,
                       campaign_id=CAMPAIGN_ID, receipt_fields={"source_H": 3, "eval_H": 4})


if __name__ == "__main__":
    cadence.main(run_fn=run, description=__doc__)
