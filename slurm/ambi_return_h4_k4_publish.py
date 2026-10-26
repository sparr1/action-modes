"""Two H4/K4 runs, paired priors, and independently completing H4/K1 context."""
from functools import partial
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slurm import ambi_return_h4_publish as workflow

ATTEMPT = "return-prior-h4-k4-20261026"
SELECTOR = "horizon/h4_k4"
METRICS = (
    ("Episode return: mean of five seeds", "eval/return_mean"),
    ("Episode return: sample standard deviation", "eval/return_sample_std"),
    ("Paired improvement over the same checkpoint's prior", "eval/paired_gain_mean"),
    ("Paired prior gain: sample standard deviation", "eval/paired_gain_sample_std"),
    ("Fraction of seeds improved over the prior", "eval/paired_win_fraction"),
    ("Controller seconds per decision", "runtime/control_seconds_per_decision"),
)
CAMPAIGN = workflow.Campaign(attempt=ATTEMPT, selector=SELECTOR, eval_K=4, context_h4=True,
    root_key="h4_root", prefix="Frozen H4 K4 / ", title="Frozen J8 H4 checkpoints: evaluation K4 versus K1", metrics=METRICS,
    description=("**Frozen J8 checkpoints, H4: evaluation K4 versus K1.** Train K identifies each 500k checkpoint's "
        "training cadence. The two H4/K4 runs are new; both priors and both H4/K1 runs reuse their existing identities. "
        "K4 solves afresh every four real decisions, holding its adapted policy and solve-time encoder between solves. "
        "K1 solves at every decision. All arms use mean actions, seeds 101–105, controller seed 55, and complete "
        "500-decision episodes. Prior gains pair the same checkpoint and environment seed. Missing bars mean "
        "pending, never zero return. The return and cost panels compare K4 with K1. "
        "Sample SD measures episode variability, not training-seed uncertainty."))

validate_spec = partial(workflow.validate_spec, campaign=CAMPAIGN)
prepare = partial(workflow.prepare, campaign=CAMPAIGN)
completed_record = partial(workflow.completed_record, campaign=CAMPAIGN)
install_workspace = partial(workflow.install_workspace, campaign=CAMPAIGN)
verify_workspace = partial(workflow.verify_workspace, campaign=CAMPAIGN)
publish_once = partial(workflow.publish_once, campaign=CAMPAIGN)

if __name__ == "__main__":
    workflow.main(campaign=CAMPAIGN)
