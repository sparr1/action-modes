# Entropy in the established full-episode controller

This experiment retains the full-episode evaluation that previously showed a
benefit from prior-initialized adaptation. The frozen backbone is
`rwgao_b-brown-university/ambi/mey3rxj8` at 200,000 training decisions, checkpoint
SHA256 `909e5c1d125aecc952e0544d802b4b946ae5089909c7a55885f272c32ed3e12f`.
It does not train or modify the backbone.

At each of 500 real decisions, create a fresh inherited actor and critic,
optimizers and replay; execute J1, J2 or J4 rounds; return the adapted actor's
mean action. Each round collects N128/H1 imagined transitions, then performs
32 critic and four actor updates with batch size 256. Additional rounds retain
the established collection and critic-fitting schedule. All previous optimizer,
replay, Q-reduction, policy parameterization, checkpoint compilation, saved
Q-scale and target-network settings remain unchanged.

The only differences within each J setting are:

| Arm | Inner entropy objective | Fixed coefficient |
|---|---|---:|
| off | Existing TD-MPC2 statistic, multiplied by zero | 0 |
| native | Prior's TD-MPC2 entropy statistic | 0.0001 |
| squashed | True squashed-action entropy | 0.0021 |

The squashed coefficient equals 21 times the saved prior coefficient, matching
the Gaussian coefficient in the native objective. This is not a claim that the
resulting gradient norms or Adam steps are matched, or that this strength is
optimal. The checkpoint's saved Q divisor stays fixed. Critics continue to use
reward-only targets with the existing frozen outer terminal bootstrap. Entropy
is an actor objective; reported environmental returns contain no entropy bonus.

## Pairing and primary outcome

Each arm/J combination covers environment seeds 101–120 and controller seeds
55–57. This yields 540 refined episodes in 27 cells. The nine existing alpha-zero
cells (180 episodes) can be reused only after verifying their checkpoint,
scientific implementation, resolved protocol, complete seed coverage and
immutable bundle hashes. The other 360 episodes are evaluated on Oscar.
Historical references, source identities and timing remain explicitly marked.

The primary outcome is undiscounted raw environment reward over the ordinary
500-decision episode. Form entropy-arm differences at matched J, environment
seed and controller seed; average the three controller repetitions within each
environment seed; then weight the 20 environment seeds equally. Report paired
95% percentile bootstrap intervals using 2,000 episode-cluster resamples.
Also report gains over the frozen prior and compute scaling within each arm.
The intervals are exploratory and not adjusted for multiple comparisons.

Episode seeds are paired. States and imagined data can diverge after different
policies act; this is part of the end-to-end controller comparison. Do not
claim fixed states, identical minibatches throughout the episode, or independent
backbone training seeds. J varies imagined data, critic updates and actor
updates jointly. J1/J2/J4 must not be relabeled A1/A2/A4: they perform 4/8/16
actor updates per decision.

## Diagnostics and publication

Retain the existing 32-sample model-return probes at initialization and each
completed round, raw inner traces and ordinary per-episode metrics. Label these
as diagnostics along each controller's own trajectories. They are separate from
the earlier shared-prior-root, fixed-critic, single-action entropy diagnostic.
No 1,000-step continuing-calibration branches replace episode evaluation here.
Ordinary Gym and DMControl time limits stay enabled.

The immutable campaign assigns output ownership and weighted seed shards.
GPU workers do not initialize W&B. CPU mergers validate complete seed cells,
including realized model/optimizer work and frozen outer state. One explicit
new W&B comparison run owns progress, return/entropy/compute curves and complete
paired report artifacts. Existing historical evaluation runs are unchanged.
Incomplete panels must be labeled incomplete and cannot publish final science.

The campaign code and Slurm entry points are
`slurm/ambi_entropy_episode_campaign.py`,
`slurm/ambi_entropy_episode_report.py`, and the corresponding
`run_ambi_entropy_episodes*_oscar.sbatch` scripts. Commit, scheduler IDs, absolute
runtime/output paths, immutable campaign and acceptance receipts belong in the
external experiment output directory, not this source tree.
