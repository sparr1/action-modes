# Remaining-horizon SAC: conditioning and training diagnostics

`ambi_aux_horizon_conditioning_625k.json` prepares eight conditioned soft/soft settings and two new J1 controls on
`rwgao_b-brown-university/ambi/aux6434715x3`, checkpoint 625,000 (seed 55).
It does not submit jobs. The default selectors are:

| H | J | Conditioning | Imagined rows per decision | Critic / actor updates |
|---|---|---|---|---|
| 2 | 1 | `none`, `one_hot` | 256 | 16 / 4 |
| 2 | 2 | `one_hot` | 512 | 32 / 8 |
| 2 | 4 | `one_hot` | 1,024 | 64 / 16 |
| 2 | 8 | `one_hot` | 2,048 | 128 / 32 |
| 3 | 1 | `none`, `one_hot` | 384 | 16 / 4 |
| 3 | 2 | `one_hot` | 768 | 32 / 8 |
| 3 | 4 | `one_hot` | 1,536 | 64 / 16 |
| 3 | 8 | `one_hot` | 3,072 | 128 / 32 |

Reuse the six completed unconditioned C16 settings from
`ambi_aux_soft_critic_budget_625k.json` (source `fe87ae07`), matched by H and J.
Their checkpoint, seeds, budgets, entropy settings and full-replay protocol
match this comparison. Compare paired real returns and the existing aggregate
training metrics/model probes. Historical traces do not contain the new
per-horizon minibatch sums, so those curves cannot be recovered by republication.
Before publication, validate the immutable baseline bundles and record their
original source/run identities; do not append new conditioned results to them.

This adds 50 full episodes: eight conditioned settings and the two missing
J1/C16 unconditioned controls, with five seeds each. Existing J2/J4/J8 controls
are reused; H1/J4 and other unconditioned variants remain explicit selectors for
validation and are excluded from the production selection. H1 performance
assessment is deferred.

`slurm/launch_ambi_aux_horizon_oscar.sh` prepares explicit new curve identities,
queues a three-case H2/J1 and H3/J8 smoke, then ten independent GPU workers gated
on its success, plus one CPU publisher with four publication subprocesses.
Longer J panels are first. The selected ten-way concurrency uses 40 GPU-worker
CPUs/320 GiB, plus 8 CPUs/64 GiB for publication. Check live quota before use.
The launcher reuses the existing frozen-prior reference. Comparison publication
revalidates baseline identity and immutable manifest/trace hashes, and waits for
the new J1 controls when needed. Training runs and the overview include
`comparison/unconditioned_gain_*` with paired seed bootstrap intervals, in
addition to the unchanged paired-prior improvement metrics. Historical controls
retain their existing run identities; no historical results are republished.

All arms enable `inner_horizon_diagnostics`. N128/B256, C16/A4, learning rates
3e-4, tau .01, inherited adaptive alpha, critic-first updates and capacity 3072
retain every imagined transition within a real decision. Actor, critic, target,
temperature, replay and optimizers reset at the next real decision. Execution
uses the final mean action. Full comparisons use five paired environment seeds,
500 decisions, and the existing 32-rollout frozen-model probes.

## Conditioning and checkpoint compatibility

`inner_horizon_conditioning` defaults to `none`. `one_hot` appends a one-hot
remaining-horizon vector to the private inner actor and each inner critic head.
Initial horizon columns are zero; existing first-layer weights are copied from
the checkpoint before optimizers are constructed. All inner weights can adapt.
Fresh solves restore the prior and zero those columns in place, preserving
Parameter identities and compiled graphs. The learned input h is H at the
root, then H-1 through 1; there is no learned h=0 value. H1 is an initialization
parity check, not an exact optimization null: its constant column can learn.

Supported configuration: canonical action-local finite-horizon SAC, cloned
prior-initialized actor/critic, inner-target bootstrap, single value, no explorer,
outer replay mixing, writeback, value-equivalence extension, or legacy diagnostic
rollouts. Conditioning and diagnostics both accept `aux_return_mode="off"` or
`"sac"`, provided inner actor/critic and terminal actor/critic sources are all
`"sac"`. Auxiliary checkpoint modules and their state remain frozen and intact.
Other source combinations remain rejected.

For h>1 the critic target is the existing interior SAC Bellman target. At h=1,
`inner_terminal_entropy="outer"` uses the frozen outer-policy sample and online
soft critic:

```
y_h = r + gamma * (1 - terminated) * V_{h-1}
V_{h-1} = Q_inner_target(z', a'; h-1) + alpha_inner * scale_inner * E_inner(a')  [h>1]
V_0     = Q_outer(z', a_outer) + alpha_outer * scale_outer * E_outer(a_outer) [h=1]
```

Here E is the configured entropy statistic (`-log pi` for squashed entropy);
scale is 1 when Q scaling is disabled. Inner and outer statistics/coefficients
follow their existing independent configuration. The boundary uses the saved
outer alpha and scale, not the adapted inner alpha. True termination selects
zero continuation, including when unused continuation values are nonfinite.

## Observational interface

`inner_horizon_diagnostics` is a strict Boolean, default `false`. It appears in
resolved operational configuration but is excluded from scientific planner and
resume identities. Omitted/false preserves the existing actor-kernel output
contract and unconditioned replay format. It never enables extra model probes.

Replay stores `remaining_horizon` whenever conditioning **or** diagnostics needs
it. `remaining_horizon_max` describes replay storage independently of network
conditioning. Unconditioned networks receive no h argument. Labelled replay
uses metadata version 4; legacy unlabelled versions 1–3 remain unchanged.
Appends/restores validate h in [1,H], integer values, and
`horizon_end == (h == 1)`; restoration validates metadata before mutation.
Trusted collectors skip host-side value checks on internally generated labels.
Labelled samples make one device-local copy of the learner fields to preserve
the original packed minibatch strides/offsets. Otherwise the extra label column
can change compiled reduction schedules and accumulate floating-point drift in
an unconditioned control, despite identical samples and RNG states.
Diagnostic-only labels are attached outside the unconditioned compiled rollout,
preserving that kernel's original output layout as well.
Conditioned exact inner state uses version 7 and, for auxiliary backbones,
includes and validates both conditioning and source metadata.

## Raw trace metrics

All values below describe **pre-update training minibatches**, using the exact
samples already consumed by the optimizer. They are not held-out evaluations,
real Monte Carlo returns, or post-update measurements. Duplicated replay draws
count as separate sampled rows. Critic heads are averaged, not counted as rows.

| Raw metric suffix | Definition |
|---|---|
| `critic_horizon_{h}_sample_count` | Rows at h in this critic minibatch |
| `critic_horizon_{h}_td_error_abs_sum` | Sum over rows of head-mean absolute TD error |
| `critic_horizon_{h}_predicted_q_sum` | Sum over rows of head-mean decoded Q |
| `critic_horizon_{h}_target_q_sum` | Sum of Bellman target Q |
| `actor_horizon_{h}_sample_count` | Rows at h in this actor minibatch |
| `actor_horizon_{h}_entropy_sum` | Sum of the actor objective's entropy statistic |
| `actor_horizon_{h}_objective_q_sum` | Sum of its reduced Q after configured Q scaling |

The actor entropy statistic is in nats for squashed entropy and uses the existing
TD-MPC2 scaled statistic when selected. It does not include alpha. Objective Q
uses the exact divisor used by that update, including per-update scale proposals.
Critic Q is in decoded value units; it is not divided by the actor's scale.

The actor kernel optionally returns detached per-row objective Q before its
existing optional scale proposal. Entropy comes from its existing output.
No policy/Q forward, RNG draw, backward pass, or per-update CPU synchronization
is added. Fixed-size reductions stay on the device until normal trace flushing.

## Summary and W&B aggregation

Raw traces retain counts and sums. `training_summary` publishes each `_sum` as
a `_mean`: sum/count within an evaluation seed, then equal weight across seeds
with positive coverage. Update curves pool decision roots at the same optimizer
index within each seed. Decision curves pool optimizer updates within that
seed/decision. Empty groups retain counts but have no mean entry (no zero/NaN).

Each mean's `count` is the number of contributing seeds; `sample_count` is the
total contributing replay draws. Standalone `*_sample_count` curves also retain
zero-coverage seeds. The W&B publisher emits these through the existing
`critic/*` / critic-update, `actor/*` / actor-update, and `episode/*` / real-decision
axes. The training-summary artifact includes metric descriptions and coverage.
Legacy aggregate metrics, real episode returns, paired improvement and frozen
outer-tail probes retain their existing definitions and aggregation.

## Short smoke, separately from production

`slurm/run_ambi_horizon_diagnostics_smoke_oscar.sbatch` runs an explicitly selected
array index: 0 = H1/J4, 1 = H3/J8. Each index checks conditioning none/one_hot
and diagnostics false/true independently: four cases, three real decisions each,
one seed, no online publication or full evaluations. Source must be a clean Git
checkout matching `EXPECTED_ACTION_MODES_SHA`.

The runner verifies the actual checkpoint hash, full replay, every work count,
saved alpha, frozen outer/auxiliary state, unchanged model-probe semantics,
zero compile fallback and no additional graphs on subsequent decisions.
Diagnostics on/off pairs check learner RNG equality and executed actions under
the existing numerical tolerances. It saves bundles, training summaries and
`validation.json` receipts. Constant-code controls, actor-only/critic-only
conditioning and additional diagnostic rollouts are outside this comparison.
