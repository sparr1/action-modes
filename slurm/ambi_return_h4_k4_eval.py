"""Evaluate H4 K4 on both selected J8 checkpoints without rerunning controls."""

import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slurm import ambi_return_cadence_eval as cadence

MATRIX = cadence.ROOT / "configs/research/return_prior_h4_k4.json"
CAMPAIGN_ID = "return-prior-h4-k4-20261026"


def selection(index):
    cadence.require(index in (0, 1), "Only the two selected J8 checkpoints may use H4 K4")
    return "horizon/h4_k4", 4


def validate_results(payload, entry, index, *, smoke=False, smoke_steps=6):
    counts = cadence.validate_results(payload, entry, index, smoke=smoke, smoke_steps=smoke_steps,
                                      selected=selection(index), horizon=4, replay_capacity=4096)
    result = payload["results"][0]
    for episode in result["episodes"]:
        steps = episode["length"]
        held = episode["model_metrics"].get("inner_policy_held")
        cadence.require(isinstance(held, (int, float)) and
                        math.isclose(held, (steps - math.ceil(steps / 4)) / steps, rel_tol=1e-9, abs_tol=1e-9),
                        "Incorrect held-policy cadence within an episode")
    held = result["model_metrics"].get("inner_policy_held", {})
    counts["held_decisions"] = counts["decisions"] - counts["solves"]
    cadence.require(held.get("count") == counts["decisions"] and held.get("sum") == counts["held_decisions"],
                    "Incorrect aggregate held-policy cadence")
    return counts


def run(args):
    return cadence.run(args, matrix=MATRIX, select=selection, validator=validate_results,
                       campaign_id=CAMPAIGN_ID, receipt_fields={"source_H": 3, "eval_H": 4})


if __name__ == "__main__":
    cadence.main(run_fn=run, description=__doc__)
