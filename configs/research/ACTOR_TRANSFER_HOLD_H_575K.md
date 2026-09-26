# Actor transfer with H-decision feedback holds

`actor-transfer-hold-h-v1` changes the solve cadence of the
[uniform-J actor-transfer study](ACTOR_TRANSFER_575K.md). It uses the same frozen
575k auxiliary-return checkpoint, H={1,2,3}, J={1,2,4,6,8,10}, cold/warm actor
initialization, five paired seeds 101–105, controller seed 55, and 500 real
decisions. Every solve performs the selected J rounds at C16/A4/N128/B256.
The prior reference remains hash-pinned and unchanged.

A solve occurs at real decision t=0,H,2H,… in each episode. Between solves,
the adapted actor's weights stay fixed, but it evaluates each **fresh observed
state** and executes its mean action `tanh(mu)`. No action sequence is cached.
No imagined collection, learning, temperature update, replay mutation, or
model probe occurs on held decisions. The final H3 block starts at decision 498
and executes only decisions 498 and 499; no extra step is added.

| H | Solves per episode | Held decisions per episode | Final block length |
|---|---:|---:|---:|
| 1 | 500 | 0 | 1 |
| 2 | 250 | 250 | 2 |
| 3 | 167 | 333 | 2 |

Cold resets actor weights from the checkpoint at each solve. Warm retains
only actor weights from the previous solve within the episode. Critic and
target critic, replay, alpha, optimizers, and optimizer moments are fresh at
each solve. Both modes execute their fixed feedback actor between solves.
Every episode starts from the checkpoint priors, and the outer model remains
frozen with no writeback. Replay capacity remains `max(3072,128*H*J)`.

The configuration key is `inner_solve_interval=H`. This cadence is distinct
from rollout horizon despite using the same numerical value in this study.
There is no special first-solve budget. The ordinary uniform-J study remains
`actor-transfer-v2`; existing outputs and their identities are preserved.

## Diagnostics and comparison

Each actual solve retains the J+3 model/probe stages: initialization, before
and after the first actor block, then after each round. Held decisions contain
only real-decision records. Validation checks the exact solve mask, solve
index, actor age, episode index, zero held-decision update/model-step counters,
missing held-decision model probes, and actor lifetime measured in prior solves.

The overview publishes full-episode returns versus J **per solve**, paired
warm-minus-cold effects, solve/held counts, total updates, and controller
seconds **per all 500 real decisions**. Solve and held prediction/control/probe
latencies remain separate metrics. Controller time subtracts measured probe
duration while retaining host tracing overhead; it is not uninstrumented
production latency. Initialization and compilation are separate overheads.
Model predictions are not measured value error. The experiment has one trained
checkpoint and five environment seeds; paired intervals are exploratory.

The CPU publisher installs explicit results panels using the existing personal
project workspace and preserves unrelated sections, settings, and filters.
Layout failure is visible but never triggers a GPU reevaluation. Progress and
measurement tables expose cadence and reuse provenance before all cells finish.

## Preparation, reuse, and launch

Use `slurm/ambi_actor_transfer_hold_h_campaign.py prepare` or
`slurm/run_ambi_actor_transfer_hold_h_oscar.sbatch` with the same required
variables as the uniform-J launcher: `CAMPAIGN_ROOT`, `EXPECTED_ACTION_MODES_SHA`,
`EVAL_MODE`, checkpoint inventory/reference paths, and the locked `PYTHON_BIN`.
The modes remain CPU `prepare`, GPU `worker`, and CPU `watch`. Workers require
L40S for timing comparisons. Submit the exact indices in `campaign.json`.

H1 has no held decisions. An explicit `--reuse-h1-from` corrected v2 campaign
can reuse its twelve H1 cells, preserving their original bundles, traces,
scientific identities, publication IDs, and URLs. The source may itself retain
original J10 artifacts; that complete provenance chain is validated. H2/H3
cells, including J10, require new evaluations.

H1 reuse additionally requires `--h1-compatibility-proof`: a reviewed
cross-revision action/state/RNG audit covering all six J values and both modes,
at least two episodes and three decisions per episode. The receipt must pin
both scientific implementation fingerprints, declare no differences in actions,
actor/critic/optimizer/scalar states and RNG, and be hash-pinned in each reused
cell. This finite fixture audit is supporting compatibility evidence, not a
new identity for old measurements. Requested/resolved configs must differ only
by default/explicit interval 1, and all source full-episode traces, hashes,
checkpoint/runtime/hardware and completed publications are revalidated.

The corresponding launcher environment variables are `EVAL_REUSE_H1_FROM`
and `EVAL_H1_COMPATIBILITY_PROOF`. Without them, all 36 settings are new. With
strict H1 reuse, 24 settings (120 episodes) are new and 12 (60 episodes) reused.
Six seven-decision/two-seed GPU smoke cells cover H2/J4, H3/J8, and H3/J10 in
both modes, including held decisions, later solves, episode resets, and partial
final blocks. Without reuse, smokes instead cover H1/J1, H2/J4, and H3/J8.
GPU workers never publish directly; one CPU watcher owns result publication.
