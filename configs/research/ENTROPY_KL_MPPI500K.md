# MPPI500k sampled entropy and prior-KL comparison

This campaign uses only the 500,000-decision auxiliary-return MPPI checkpoint
from `ambi/mppi7530927x0`. The profile pins the checkpoint and adjacent sidecar
by SHA256. Its learned outer temperature is **0.023066775873303413** and its
squashed-action target entropy is **−10.5**. No outer parameters or optimizers
are updated.

Four independent raw evaluation records share environment seeds 101–105,
controller seed55, sampled actions, and all500 decisions of each episode:

| Mode | Actor objective | Temperature updates per decision |
|---|---|---:|
| prior | Frozen original actor; no inner optimization | 0 |
| entropy | −Q − α H | 32 |
| kl | −Q + β KL(inner ∥ frozen prior) | 0 |
| both | −Q − α H + β KL(inner ∥ frozen prior) | 32 |

β is the saved outer temperature, held fixed. Automatic α is freshly initialized
from that temperature at every real decision and uses learning rate0.0003.
KL-only disables the entropy term and temperature learner entirely; its actual
inner α diagnostics are0. Prior α diagnostics report the unchanged outer α and
do not describe a temperature update. All three refinement methods use fresh
prior-initialized H3/J8 solves,128 trajectories per round, batch256,16 critic
and4 actor updates per round, reward-only auxiliary-Q targets and terminal
bootstrap, no actor/Q normalization, no transfer, and no writeback.

The matrix retains the existing separate-noise to-go return diagnostic:32
rollouts at decisions0/100/200/300/400, evaluated at initialization and each
round. The prior has only the initialization boundary. Probe model work and
probe timing are recorded separately from controller timing. Per-decision and
optimizer-event traces stay on Oscar; publication uploads bounded summaries.

Run `slurm/ambi_entropy_kl_mppi500k.py prepare --root <scratch>/episodes` from a
clean committed checkout. Preparation validates the saved checkpoint α and
allocates all four identities using specification-only evaluation without an
unfinished prior reference. Run index0 for the full prior and isolated
`--smoke-steps 32` jobs for indices1–3. After those checks pass, production
indices1–3 require and revalidate the prior receipt, result, manifest and all
trace hashes before and after evaluation. Their paired gains use the actual
same-seed prior episode returns. Each worker owns all five seeds and exclusive
output paths; reruns cannot overwrite saved results.

The launcher requires explicit Slurm resources and the exact source SHA.
Workers do not initialize W&B. The separate publisher verifies complete
receipts and appends each method to its persistent checkpoint curve, including
compatible historical results. See
[ENTROPY_KL_CHECKPOINT_CURVES.md](ENTROPY_KL_CHECKPOINT_CURVES.md) for the catalog,
layout and future-append contract. The four methods have distinct native runs,
colors and legends; each line connects only the same method across checkpoints.
