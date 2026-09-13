# Critic updates in closed-loop prior refinement

This experiment varies the number of critic updates while holding the successful
H1, alpha-zero, prior-initialized controller fixed. It uses
[closed-loop refinement eval](CLOSED_LOOP_REFINEMENT_EVAL.md): a fresh solve and
mean-action execution at every real decision throughout a normal episode.

## Experimental question and controls

Does fitting the inherited critic more extensively on a fixed amount of imagined
data improve real full-episode return? Can the controller achieve the C32
reference's performance with fewer updates, and does additional fitting help or
hurt? Horizon is held at one for this experiment; horizon comparisons are a
separate intervention.

The frozen backbone is `rwgao_b-brown-university/ambi/mey3rxj8`, checkpoint
200,000, SHA256
`909e5c1d125aecc952e0544d802b4b946ae5089909c7a55885f272c32ed3e12f`.
The saved Q divisor 11.182598114013672 and gamma 0.99 remain fixed. The actor,
critic, target convention, policy parameterization and all optimizer settings
match the established reference. There is no outer learning or writeback.

Each real decision starts an inherited actor, online critic and fresh replay
and optimizers. It performs **one round (J1)**: collect **N128/H1**, fit the critic
for **C** updates, then perform **A4** actor updates with **B256**. Replay capacity
is 2,048 and sampling is with replacement. Alpha stays zero. The sole changed
learner parameter is `inner_critic_updates_per_round`.

| C | Critic minibatch rows drawn | Draws per collected transition | Actor updates | New episodes |
|---|---:|---:|---:|---:|
| 0 | 0 | 0 | 4 | 60 |
| 1 | 256 | 2 | 4 | 60 |
| 4 | 1,024 | 8 | 4 | 60 |
| 8 | 2,048 | 16 | 4 | 60 |
| 16 | 4,096 | 32 | 4 | 60 |
| 32 | 8,192 | 64 | 4 | 0; reuse the existing 60 |
| 64 | 16,384 | 128 | 4 | 60 |

These ratios count sampled minibatch rows, not independent data. Every condition
collects exactly 128 imagined transitions and performs exactly four actor
updates per real decision. C0 still adapts the actor against the inherited
critic; it is not a prior-only controller, random critic, or zero-actor-update
condition. C0 has no critic optimizer updates and no associated critic-training
measurements.

At H1 the outer terminal function is frozen, but its sampled prior action and
critic pair are resampled during critic fitting. The targets are not one
unchanging scalar label. Training loss/TD error are measurements on resampled
training minibatches, not held-out calibration errors. Inherited policy
saturation and a learned model may limit how critic fitting translates to
useful actor gradients; the real return remains the primary outcome.

## Pairing and outcomes

Use environment seeds 101–120 and controller seeds 55–57, with ordinary
500-decision episodes and mean-action execution. The full panel has 21 cells
and 420 refined episodes: 360 new and 60 historical C32 references. Do not
rerun C32. The current sealed-episode merger requires inline prior references
and rejects external-reference provenance. To preserve that tested path, new
workers also run 360 cheap prior-only episodes, whose paired values are checked
against the historical references. These have no inner optimization and do not
rerun C32. A two-decision smoke must not reuse a 500-decision performance result
as its outcome.

Form paired differences at matched environment/controller seeds. Average
controller repetitions within environment seed, then give each of the 20
environment seeds equal weight. Report 95% paired environment-seed percentile
bootstrap intervals using 2,000 resamples (bootstrap seed 20260912). Report
return gains against the frozen prior, C0 and C32. These are exploratory
pointwise intervals, not multiplicity-adjusted or backbone-training intervals.
All conditions start from paired episode seeds; later real states and replay
can diverge after the controllers choose different actions.

The primary plots show undiscounted full-episode return and paired gains versus
C. Include measured control time and total optimizer updates C+4. Actor updates
and imagined transitions are constant across this sweep; they are workload
checks, not increasing-compute axes. Historical C32 timing is not a simultaneous
hardware-load control.

Retain the 32-sample reward-plus-outer-Q probes before learning and after the
round, with actor-update coordinates 0/4 and critic coordinates 0/C. C0 has
two distinct probe boundaries at critic coordinate zero. Preserve the round or
actor coordinate so one point does not overwrite the other.

Stream the saved critic-update traces into per-update training-loss, decoded
TD-error and Q/gradient diagnostics. Average decisions within episode,
controllers within environment seed, then environment seeds equally. Mark the
measurements as pre-update training minibatches on controller-specific states.
Do not fill structurally missing C0 critic losses with zero. Retain complete
paired summaries, source trace hashes and locations, and the original sealed
trace bundles.

## Execution and reporting

The matrices `critic_steps_c{0,1,4,8,16,64}_j1_h1_200k.json` derive from
`round_scaling_j1_h1_200k.json`, changing only the critic-update count in the
learner. C32 uses that unchanged historical matrix. The campaign validates
resolved configurations, immutable checkpoint/reference hashes, expected work,
episode coverage and output ownership. It preserves the existing learner,
simulator and diagnostic implementation.

Use `slurm/ambi_critic_steps_campaign.py` and its Oscar worker/merge launchers
from a tested, clean Git checkout. One CPU publisher,
`slurm/ambi_critic_steps_report.py`, owns the explicit new comparison attempt in
`ambi-inner-bench`. GPU workers and mergers remain offline. The complete panel,
critic-step curves, paired data and standalone HTML belong in that comparison;
existing checkpoint-axis runs and historical results remain intact. A partial
panel must remain visibly incomplete.

Independent controller/seed tasks are scheduled in parallel. C64 uses two
10-seed shards per controller; each other new cell uses one 20-seed task. This
gives 21 new GPU tasks, with expensive tasks first. Select actual concurrency
from live account resources. Validate every new dose with short GPU smoke
episodes and verify the reused C32 bundles before production submission.

## Interpretation

If C0 performs comparably to C32, inherited-critic actor adaptation may already
explain much of the gain in this regime. If return improves from low C toward
C32, critic fitting contributes. A plateau or decline at C64 motivates examining
training fit, model-probe gains and policy changes before simply adding more
updates. Similar means with wide intervals do not establish equivalence.

This experiment alone does not separate the earlier horizon reduction from the
critic-update change. It isolates critic count at H1; hold an explicitly chosen
critic recipe fixed when studying horizon afterward.
