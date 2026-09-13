# Testing J2 with C64

This follow-up asks whether the C64 critic recipe continues to improve real
control when the inner solve runs for more rounds. It uses
[closed-loop refinement eval](CLOSED_LOOP_REFINEMENT_EVAL.md) on frozen
`rwgao_b-brown-university/ambi/mey3rxj8`, checkpoint 200,000, SHA256
`909e5c1d125aecc952e0544d802b4b946ae5089909c7a55885f272c32ed3e12f`.

## Fixed recipe and intervention

Keep H1, N128 per round, B256, four actor updates per round, inherited actor and
online critic, fixed alpha zero, and the saved Q divisor 11.182598114013672.
Discount is 0.99. The sole change from the C64/J1 recipe is `inner_rounds`.
Each round collects from the current actor, performs 64 critic updates, then
four actor updates. Actors, critics, target state, replay, and optimizers carry
across rounds within a solve; they start fresh from the frozen prior at the
next real decision. Replay capacity remains 2,048, so J2 does not evict any of
its 256 collected transitions. Real execution uses the final actor mean.

| C64 budget | Imagined transitions per decision | Critic updates | Actor updates |
|---|---:|---:|---:|
| J1 | 128 | 64 | 4 |
| J2 | 256 | 128 | 8 |

Increasing J scales collection, critic fitting, and actor learning together.
It measures the complete iterative solve rather than isolating critic compute.
At H1, all imagined transitions start from the current real root; later rounds
add actions from the adapted actor. The finite-horizon outer terminal function
stays frozen, but sampled terminal prior actions and critic pairs are resampled
during fitting. No horizon, entropy, learning-rate, dropout, Q-reduction, policy
parameterization, or outer-learning change accompanies this experiment.

## Paired panel and reuse

Evaluate ordinary 500-decision Humanoid Walk state episodes with environment
seeds 101–120 and controller seeds 55–57. Retain both Gym and raw DMControl time
limits. Each C/J setting has 60 episodes. The primary new evaluations are
**C64/J2 only**, totaling 60 refined episodes. Reuse the verified C64/J1
result and C32/J1/J2 results, totaling 180 historical refined episodes.
These four settings form 12 controller cells and 240 paired refined episodes.
Historical C32 curves provide context for whether the larger critic budget
changes round scaling without requiring another evaluation.

Verify checkpoint/runtime/scientific source, resolved configuration, seed and
episode coverage, and immutable raw manifest/seal hashes before reuse. Keep
raw seal-file hashes distinct from the canonical seal-content digest. No
historical adaptive episode is rerun. The unchanged sealed merger requires
inline prior provenance, so new workers additionally run 60 inexpensive
prior-only episodes; their returns must match the reusable prior exactly.

Average controller repetitions within environment seed and weight the twenty
seeds equally. Use 2,000 paired environment-seed percentile bootstrap resamples
with seed 20260912. Report full-episode return, gains over the prior and C64/J1,
round-to-round contrasts, C64 versus C32 at the same J, and, when available,
the paired change in round scaling between C64 and C32. Intervals are
exploratory and unadjusted for multiplicity; these are not independent
backbone-training repetitions.

## Diagnostics and interpretation

Retain 32 sampled model-return probes at actual actor initialization and after
every completed round. J2 has three probe boundaries. For C64/J2,
the critic coordinate is 0/64/128 and actor coordinate 0/4/8. Preserve predicted reward, discounted
outer-Q bootstrap, and paired gains. All reported probe returns use raw
discounted reward units, without entropy or Q normalization.

Stream full traces into per-round and per-critic-step training summaries.
Distinguish round-local update index from cumulative completed critic updates,
and retain the critic/actor interleaving. Each training error is measured on
its minibatch before the corresponding update, with resampled bootstrap
targets; it is not held-out calibration. Different controllers visit different
later real states, so these diagnostic curves are not a shared-root comparison.
Sampled actor-training saturation is distinct from executed mean-action
saturation; the unchanged historical trace does not contain executed vectors.

The primary scaling conclusion must come from real episode returns. Increasing
model-probe return alone does not establish better real control. A flat C64
round curve would motivate separating actor optimization, replay collection,
and critic fit in a subsequent study; it does not establish that added compute
can never help. C32 historical curves are context, not additional selected
checkpoints. No new horizon or learning-rate sweep is included.

## Execution and publication

Use `slurm/ambi_c64_rounds_campaign.py`, its Oscar worker and merger launchers,
and the CPU-only `slurm/ambi_c64_rounds_report.py`. The new matrix is
`c64_rounds_j2_h1_200k.json`. Historical settings
use their unchanged matrices. Do not change the learner, evaluator, simulator,
or sealed-episode merger to launch this follow-up.

Each new controller cell has four five-seed shards. The 12 independent GPU
tasks evaluate only J2/C64, using the available 12-GPU allowance. Recheck live
account resources and other allocations at submission. Merge all shards into
one complete cell before publication. Smoke J2, including a multi-shard
merge; validate all nine historical cells with the current trace parser.

One explicit new W&B comparison attempt in `ambi-inner-bench` contains all four
C/J settings, round/update/work axes, model and training diagnostics, full
paired summaries, and standalone HTML. Preserve raw trace bundles and their
paths/hashes. Publish final scientific comparisons only after all 12 cells
validate. GPU workers and mergers remain offline. C32 and C64/J1 control
timings are historical; do not claim a controlled simultaneous runtime ratio.

Production submission follows local tests, exact tested Git synchronization,
short GPU smoke, and historical-reference validation. Existing results and
checkpoint-axis curves remain intact.
