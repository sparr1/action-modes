# Bernoulli weight transfer at the frozen 575k checkpoint

[The versioned matrix](ambi_bernoulli_transfer_575k.json) tests actor-only,
critic-only and joint random weight transfer at **p = 0.5**, H = 1, 2, 3 and
J = 1, 2, 4, 6. These are 36 new cells, each with three paired development
seeds 101–103 and up to 500 real decisions (108 new episodes). The frozen
575k checkpoint, return/return objective, every-decision mean-action execution,
C16/A4/N128/B256 and controller seed 55 match the earlier discovery campaign.

At each decision after the first, independently for each scalar parameter,

```text
next = prior + Bernoulli(p) * (previous_final - prior)
```

Thus p means **probability of retaining the adapted value**. Masks include
weights, biases and normalization affine parameters; nonparameter buffers
retain the prior. Actor and critic masks use separate private streams, and
mask draws do not advance the learner RNG streams. The first decision is
fresh, and episode reset discards the donor. Adam, imagined replay and learned
temperature reset at each decision; the target critic is copied from the
resulting online critic. This is not deterministic 50% parameter blending.

Diagnostics sample zero-based decisions 0, 1, 25, 100, 250 and 499. They compare
prior, previous donor, actual masked initialization and final learner on
common fixed model targets, actions and latent-state banks. Recorded families
cover critic target error and action ranking, critic action-gradient utility,
policy shift and first-action model utility, feature activity/rank, and
stationary fixed-target learning. Disposable four-step critic fitting runs at
decisions 25 and 250; smoke tests force one at decision 1. Missing donor
information at the first decision is structural, not a zero measurement.
Model-based measurements are diagnostics, not environment-return ground truth.
See [the measurement definitions and isolation checks](BERNOULLI_TRANSFER_DIAGNOSTICS.md)
for the exact probe and summary semantics.

Diagnostic snapshots and probes use isolated state/RNG and are timed
separately from controller prediction plus transfer construction/export.
Full-episode wall time includes both. Smoke and production completion gates
verify expected diagnostic decisions, families, stages and finite summaries;
a missing diagnostic family cannot silently become a completed result.

Preparation and workers reuse the discovery launcher:

```bash
python slurm/ambi_transfer_discovery_campaign.py prepare \
  --root /path/to/new/campaign --checkpoint /path/to/575k.pt \
  --matrix configs/research/ambi_bernoulli_transfer_575k.json \
  --reference-root /path/to/completed/historical/discovery/campaign
```

For the sbatch entry point, pass `CAMPAIGN_MATRIX` and
`REFERENCE_CAMPAIGN_ROOT` alongside the existing pinned-commit/campaign-root
variables. Preparation binds the matrix hash, diagnostic settings and each
historical fresh-control result/manifest/receipt hash. Six smoke cells cover
all mechanisms and horizons, two handoffs, and J6. Every production worker
requires all same-commit smoke receipts. Scheduler concurrency is selected
from current Oscar account limits; there is no scientific array throttle.

Publication creates a new run and saved workspace in `ambi-inner-bench`,
with a `bernoulli575` view prefix, progress and result tables, separate
controller/diagnostic timing, and explicit blue/orange/green mechanism colors.
The historical fresh baseline is black and labeled historical. Its original
source and timing are retained, and its cells do not count toward the 36 new
cells. Paired return gains use matching environment and solver seeds; different
controllers visit different states. Historical full carry and deterministic
50% results remain in the original discovery campaign and are not rerun.
Only the historical fresh controls are embedded in this initial dashboard;
they have no new diagnostic measurements.

The publisher registers a content-addressed custom chart, verifies it and the
saved workspace through readback, and fails explicitly on mismatched or
unavailable definitions. Existing runs and personal/saved views are preserved.
Browser rendering must also be checked for the exact returned dashboard URL.
