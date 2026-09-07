# Frozen-checkpoint AMBI research

## One W&B run per checkpoint curve

A run contains one backbone and one planning configuration across checkpoints.
Before submitting workers, explicitly choose `eval_series.py create` for a new
attempt or `eval_series.py append` with the selected existing run directory.
Create/append checks the scientific identity; repeating a study means creating
a separate attempt rather than appending duplicate checkpoint measurements.
Generate a new planner template first with the ordinary matrix/checkpoint/preset
arguments plus `--eval-series-spec-dir /absolute/specs --checkpoint-inventory
/absolute/checkpoint-manifest.json`. This resolves the saved settings without
constructing a model or running episodes. Then use `eval_series.py create --root
/absolute/registry --spec /absolute/specs/SELECTOR.json --attempt-label LABEL
--owner oscar-rgao48`, or `eval_series.py append RUN_DIR --spec SPEC` to validate
a deliberate extension. The emitted registry gives the run directory to assign.

Pass `--eval-run-dir /absolute/run-directory` for a single selected planner, or
`--eval-run-map /absolute/run-map.json` for several planners. The map is an object
from exact preset selectors to existing run directories, and can include other
selectors needed by the same array. Set `EVAL_RUN_MAP` to that map when using the
cluster launchers, and set `CHECKPOINT_MANIFEST` to the verified checkpoint
source inventory for SAC launchers. Assignment is prepared once before submission; checkpoint
workers never create W&B runs. Legacy `--wandb` also requires this assignment.
Local-only commands omit both the assignment and `--wandb`.

Run names identify the verified backbone and executed planner. The campaign label
is shown separately as `Attempt: ...`; a prior arm of an MPPI comparison is
explicitly named `Prior only (no planning)`. Chart legends use the same backbone
and planner with a short run suffix to distinguish repetitions. Original source
IDs and attempt labels remain unchanged in the run configuration.

GPU workers preserve the existing manifests, episode records and diagnostic
traces, then atomically queue completed results. A separate CPU process on the
authoritative result owner runs `eval_series.py publish RUN_DIR --watch --jobs ARRAY_ID` and
checks for new results every 15 seconds. This is the only W&B writer for the run.
Failed publication never changes evaluation completion or requires repeating
episodes. Keep the run directory and bundles available to the owner; stage
cross-cluster results there instead of resuming the same run from two hosts.
Checkpoint return, paired improvement and runtime use
`checkpoint/training_decisions` as their x-axis. Detailed diagnostics remain in
the portable artifact and existing HTML report.


## Humanoid inner-SAC benchmark

`ambi_humanoid_inner_benchmark.json` evaluates a frozen checkpoint using its
adjacent `.metadata.json` as the source of algorithm, observation, and environment
settings. The matrix declares source run `rwgao_b-brown-university/ambi/u13m14st`;
this is provenance supplied by the experiment configuration, not proof that an
arbitrary checkpoint came from that W&B run. Pin the intended checkpoint file
and keep its sidecar. Every result records the actual checkpoint SHA256, saved
settings, code fingerprint (including uncommitted source), and runtime versions.

The workflow has three compute stages. Each can run independently; report
generation uses saved data and never invokes a model or environment.

```bash
AMBI_EVAL_PY=environments/dmcontrol/.venv/bin/python
AMBI_INNER_MATRIX=configs/research/ambi_humanoid_inner_benchmark.json
AMBI_CHECKPOINT=/absolute/path/to/pinned-checkpoint.pt
```

**1. Evaluate the prior and save shared observations.** The default selection
is prior-only: five episodes, seeds 101–105, at most 500 decisions each. Save
the observation immediately before decisions 0, 100, 200, 300, and 400; an early
termination simply yields fewer observations. Choose fresh output paths.

```bash
"$AMBI_EVAL_PY" evaluate_ambi_checkpoint.py \
  --matrix "$AMBI_INNER_MATRIX" --checkpoint "$AMBI_CHECKPOINT" \
  --preset inner_budget/prior --device cuda \
  --bundle-dir results/inner-bench/prior \
  --save-root-bank results/inner-bench/roots.json --eval-run-map "$EVAL_RUN_MAP"
```

**2. Screen explicitly selected SAC budgets on those identical observations.**
Each observation uses three independent, reproducible solver seeds. This does
not step a simulator: the normal prediction path re-encodes the saved state and
performs a fresh inner solve. Select only the budgets to screen.

```bash
"$AMBI_EVAL_PY" evaluate_ambi_checkpoint.py \
  --matrix "$AMBI_INNER_MATRIX" --checkpoint "$AMBI_CHECKPOINT" \
  --preset inner_budget/sac_1x --preset inner_budget/sac_2x --device cuda \
  --root-bank results/inner-bench/roots.json --bank-only \
  --bundle-dir results/inner-bench/screen
```

The SAC presets use eight rounds, 512 rollouts per round, horizon three, batch
size 512, replay capacity 12,288, cloned action-local learners, and an adaptive
local temperature initialized from the frozen outer temperature. Learning rates
come from the checkpoint settings. Per-round critic/actor updates are 3/1 for
`sac_1x`, 6/2 for `sac_2x`, and 12/4 for `sac_4x`; temperature updates follow actor
updates. Collection counts are held fixed, but the generated transitions can
differ as the adapted policies change. Larger budgets are never defaulted in.

**3. Confirm promising configurations with complete episodes.** Inner SAC runs
at every real decision; matched prior returns come from the saved reference.

```bash
"$AMBI_EVAL_PY" evaluate_ambi_checkpoint.py \
  --matrix "$AMBI_INNER_MATRIX" --checkpoint "$AMBI_CHECKPOINT" \
  --preset inner_budget/sac_1x --device cuda \
  --reference-bundle results/inner-bench/prior \
  --bundle-dir results/inner-bench/confirmation --eval-run-map "$EVAL_RUN_MAP"
```

Both controllers execute `tanh(mu)` in the real environment. Inner imagined
actions, SAC actor/target sampling, and minibatch selection remain stochastic.
The evaluator never calls outer learning or applies prior writeback. Episode
and bank seeds are independent of execution/configuration order. Evaluation
resets reuse eligible action-local allocations to retain compiled graphs, while
resetting all scientific state as in a fresh solve.

Five episodes and one trained checkpoint are an exploratory screen. Inspect
paired per-seed deltas and cost, then expand seeds/checkpoints for stronger
claims. Bank roots share trajectory context and are not independent episode
return measurements.

### D512-4-J6 checkpoint progression on Hydra

`named_run/d512_4_j6` reproduces the inner settings from
[the D512-4-J6 run](https://wandb.ai/rwgao_b-brown-university/ambi/runs/09fdc28b8d304f2f8667d6d10799a792):
six rounds, 512 rollouts per round, horizon three, batch 512, replay 9,216,
and three **joint** updates per round (18 critic, actor, and temperature updates).
The actor/critic/temperature learning rates are explicitly 5e-5/1e-4/3e-4.
Finite-horizon handoff, transition-based scheduling, and real-replay mixing stay
disabled to match that run. Real evaluation actions remain `tanh(mu)`.

`slurm/run_ambi_inner_benchmark_hydra.sbatch` evaluates checkpoints 100k through
500k in increments of 100k, one GPU per array task. Each task saves the five-seed
prior baseline and shared bank, evaluates the named inner configuration for the
same five episodes, then generates a report for that checkpoint. It does not
run extra bank solves. W&B outputs go to `ambi-inner-bench`.

Submit from the clean, synchronized checkout. Export `EXPECTED_ACTION_MODES_SHA`,
`AMBI_CHECKPOINT_PREFIX` (the absolute filename prefix ending before the step
number), and a fresh `AMBI_BENCHMARK_OUTPUT_ROOT`. Pass durable `--output` and
`--error` paths to `sbatch`. A preliminary `--array=1 --time=00:30:00` submission
with script argument `--smoke` tests three decisions per controller, omits W&B,
and must use a separate output root. Default submissions run five episodes of
up to 500 decisions per controller.

For the fixed outer-Q comparison, set
`AMBI_BENCHMARK_PRESET=named_run/d512_4_j6_outer_target` and use a fresh output
root. This changes only the bootstrap critic to the frozen outer target Q at
every transition; the next action and entropy term still use the adapting inner
actor and local temperature. It does not enable finite-horizon handoff.
Set `AMBI_BENCHMARK_REFERENCE_ROOT` to the previous campaign's `evaluation`
directory to reuse its five completed prior bundles. A smoke invocation needs
the corresponding previous `smoke` directory instead, because the three-decision
protocol differs from the full evaluation protocol. Pairing is validated by the
evaluator before results are accepted.

### Higher-critic full-episode evaluation

For full return evaluation of the higher-critic settings, use
`ambi_humanoid_inner_critic_sweep.json` with the two desired presets and omit
the launcher's bank flags. Each checkpoint task evaluates C6 followed by C12
sequentially on one GPU. Each configuration runs five episodes with seeds
101–105 and at most 500 decisions per episode, with inner SAC at every decision.
The shared bundle contains separate configuration results and prepared evaluation-run assignments; its
report compares both configurations with the saved prior. No bank solves run.

Set `AMBI_BENCHMARK_REFERENCE_ROOT` to the completed original campaign's
`evaluation` directory, containing `step_100000/prior/manifest.json` through
`step_500000/prior/manifest.json`. The existing prior episodes are reused;
three-update SAC baselines are not selected or rerun. Export the checkpoint
prefix and exact pushed commit as above, then choose a fresh output root:

```bash
export AMBI_BENCHMARK_MATRIX=configs/research/ambi_humanoid_inner_critic_sweep.json
export AMBI_BENCHMARK_PRESETS='critic_budget/inner_target_c6 critic_budget/inner_target_c12'
export AMBI_BENCHMARK_OUTPUT_ROOT=/absolute/new/inner-target-c6-c12
sbatch --array=1-5%3 --time=06:00:00 \
  --output=/absolute/log/path/full-inner-%A_%a.out \
  --error=/absolute/log/path/full-inner-%A_%a.err \
  slurm/run_ambi_inner_benchmark_hydra.sbatch
```

Submit a second array with presets `critic_budget/outer_target_c6` and
`critic_budget/outer_target_c12`, a different fresh output root, and the same
prior reference. A concurrency cap of three per array permits six GPUs total.
`AMBI_BENCHMARK_PRESETS` preserves the listed evaluation order; the existing
single `AMBI_BENCHMARK_PRESET` remains the fallback when the list is omitted.
`--smoke` keeps its separate seed-101, three-decision protocol and omits W&B;
it requires a matching three-decision prior reference, not the full-episode
reference above.

### The same checkpoint evaluation on Oscar

`slurm/run_ambi_inner_benchmark_oscar.sbatch` supplies Oscar's resource and
runtime settings, then runs the same launcher body as Hydra. Each array task
requests one L40S, six CPUs, 32 GB, and six hours in partition `gpu`; the account
and QoS use the cluster defaults. The array still selects checkpoints 100k
through 500k. The default interpreter is the existing locked environment at
`/oscar/home/rgao48/action-modes/environments/dmcontrol/.venv/bin/python`;
override it with `AMBI_DMC_PYTHON` if needed. It imports code from the submitted
checkout, not from the environment's source directory.

Synchronize the tested commit through Git and submit from that clean checkout.
Checkpoint weights and adjacent sidecars, plus prior reference bundles, must be
accessible on Oscar. Copying them preserves their checkpoint and protocol
identities, which the evaluator checks before pairing returns. For the current
C6/C12 full-episode comparison, export these paths using the Oscar copies:

```bash
export EXPECTED_ACTION_MODES_SHA="$(git rev-parse HEAD)"
export AMBI_CHECKPOINT_PREFIX=/absolute/Oscar/path/to/checkpoint_
export AMBI_BENCHMARK_REFERENCE_ROOT=/absolute/Oscar/path/to/original/evaluation
export AMBI_BENCHMARK_MATRIX=configs/research/ambi_humanoid_inner_critic_sweep.json
export AMBI_BENCHMARK_PRESETS='critic_budget/inner_target_c6 critic_budget/inner_target_c12'
export AMBI_BENCHMARK_OUTPUT_ROOT=/oscar/scratch/rgao48/ambi/inner-benchmark/new-campaign/inner-target
mkdir -p /oscar/scratch/rgao48/ambi/inner-benchmark/new-campaign/slurm
sbatch --array=1-5%2 \
  --output=/oscar/scratch/rgao48/ambi/inner-benchmark/new-campaign/slurm/inner-%A_%a.out \
  --error=/oscar/scratch/rgao48/ambi/inner-benchmark/new-campaign/slurm/inner-%A_%a.err \
  slurm/run_ambi_inner_benchmark_oscar.sbatch
```

For fixed outer-Q bootstraps, select `critic_budget/outer_target_c6` and
`critic_budget/outer_target_c12`, change the output root and log paths, and
submit separately. Choose concurrency against the account's available GPU
slots; the example permits two simultaneous tasks per array. Each configuration
still evaluates seeds 101–105 for up to 500 decisions, with full traces and
offline HTML reports. W&B uses the same `ambi-inner-bench` project, informative
configuration/checkpoint names and tags, and completion summaries as Hydra.

The critic-sweep matrix requires a prior reference: it contains only critic
variants, not the `named_run/prior` preset used when the launcher creates a new
baseline. If needed, create prior bundles first with
`ambi_humanoid_inner_benchmark.json` and `named_run/prior`. For a scheduled
`--array=1 --time=00:30:00` smoke, pass `--smoke` and a fresh output root; use a
seed-101, three-decision reference or the original benchmark matrix without a
reference. Smoke runs omit W&B. Full prior references cannot pair with the
short smoke protocol.

### More actor updates for C6/C12 on Oscar

`ambi_humanoid_inner_actor_sweep.json` extends the full-episode comparison with
six and twelve actor updates per round for each C6/C12 critic budget and each
inner/outer target-Q bootstrap. Only actor dose changes from the corresponding
C6/A3 or C12/A3 run. Six rounds, 512 rollouts per round, horizon three, batch
512, learning rates, and adaptive temperature with **three** updates per round
stay fixed. The six-round critic/actor/temperature totals are C36 or C72,
A36 or A72, and T18; collection remains 9,216 imagined transitions per decision.

The existing total-budget schedule keeps each component active in its first
allocated slots. The first three slots update critic, actor, and temperature;
later slots update critic and actor until their budgets are exhausted. C6/A12
ends with six actor-only slots; C12/A6 ends with six critic-only slots. Within
each shared slot, the critic precedes the actor, then temperature. Explicit
per-round component scheduling is not enabled because it would change this
ordering. Temperature remains adaptive even though its update count is held
fixed; changed actor distributions can still change its fitted value.

Use the same checkpoint prefix and completed five-seed prior reference as
above. Split the campaign into four arrays: A6/inner Q, A6/outer Q, A12/inner Q,
and A12/outer Q. Each array covers all five checkpoints, with C6 and C12 run
sequentially in each task. This is 20 GPU tasks producing 40 configuration runs
and 200 full episodes, up to 100,000 decisions. It reuses prior results and
does not run shared-observation banks or repeat A3 controls.

For example, the A6/inner-Q array is:

```bash
export AMBI_BENCHMARK_MATRIX=configs/research/ambi_humanoid_inner_actor_sweep.json
export AMBI_BENCHMARK_PRESETS='actor_budget/inner_target_c6_a6 actor_budget/inner_target_c12_a6'
export AMBI_BENCHMARK_OUTPUT_ROOT=/oscar/scratch/rgao48/ambi/inner-benchmark/new-actor-campaign/a6-inner-target
sbatch --array=1-5%3 \
  --output=/absolute/log/path/a6-inner-%A_%a.out \
  --error=/absolute/log/path/a6-inner-%A_%a.err \
  slurm/run_ambi_inner_benchmark_oscar.sbatch
```

For the other arrays substitute `outer_target` and/or `a12` in both selectors
and use separate fresh output roots and log paths. A cap of three per array
permits twelve GPUs and 72 CPUs total; lower it when other jobs share the quota.
The evaluator also accepts `--comparison actor_budget` to select all eight
settings in one invocation, although the split arrays provide more parallelism.
For a short scheduled smoke, select all eight presets through
`AMBI_BENCHMARK_PRESETS`, use array index 3, and pass `--smoke` with a matching
seed-101, three-decision prior reference and a fresh output root.

W&B names and tags include critic, actor, and temperature counts, bootstrap
source, checkpoint, and episode mode. Existing checkpoint curves derive actor
and critic counts from completed results and distinguish configuration selectors,
so the new settings appear as they finish. Full inner traces and per-decision
aggregates remain in the portable bundles and generated HTML reports.

### Joint update intervals without smaller collection rounds

`ambi_humanoid_inner_interval_sweep.json` keeps the existing six collection
rounds, 512 rollouts per round, horizon three, batch size 512, and cumulative
replay capacity 9,216. It varies the existing `inner_steps_per_update` control
and adds one smaller actor-learning-rate setting, mirrored for inner and
frozen outer target-Q bootstraps:

| Variant suffix | Imagined transitions per joint update | Actor LR | C / A / T per full round | C / A / T per decision |
| --- | ---: | ---: | --- | --- |
| `s512` | 512 | 5e-5 | 3 / 3 / 3 | 18 / 18 / 18 |
| `s256` | 256 | 5e-5 | 6 / 6 / 6 | 36 / 36 / 36 |
| `s128` | 128 | 5e-5 | 12 / 12 / 12 | 72 / 72 / 72 |
| `s128_half_actor_lr` | 128 | 2.5e-5 | 12 / 12 / 12 | 72 / 72 / 72 |

Selectors are `interval_budget/{inner_target|outer_target}_<suffix>`. Critic LR
stays 1e-4 and temperature LR stays 3e-4. Every joint update runs the critic,
then actor, then automatic temperature fitting. This varies temperature dose
together with critic and actor dose; it does not hold T3 fixed as the preceding
actor sweep does. Interval 512 matches the original joint-G3 update count when
every rollout reaches H3, while interval 128 plus half actor LR isolates the
actor step size at the same high critic/actor/temperature count.

Updates are released after each complete collection round, using the cumulative
number of new imagined transitions. Collection is not split into smaller
blocks. The table assumes full-length imagined rollouts; actual transitions
determine counts if a rollout terminates early. All inherited legacy aliases,
model-step/per-action budgets, and shared/component per-round update controls
are explicitly removed so the canonical interval scheduler is selected.
Actor, critic, target, temperature, optimizers, and replay remain fresh per
decision. The frozen evaluator executes `tanh(mu)`, disables outer learning and
writeback, and preserves full inner traces and per-decision aggregates.

Use checkpoints 100k, 200k, 300k, 400k, and 500k from source `u13m14st`, paired
seeds 101–105, and up to 500 decisions per episode. Reuse the five completed
prior reference bundles; this matrix contains no prior preset and does not
request shared-observation banks. Eight presets across five checkpoints produce
40 configuration runs and 200 full episodes. Use four five-checkpoint arrays:
`s512` then `s256`, and `s128` then `s128_half_actor_lr`, for each bootstrap
source. Each task runs its two presets sequentially with fresh output paths.
For example, the lower-dose inner-target array is:

```bash
export AMBI_BENCHMARK_MATRIX=configs/research/ambi_humanoid_inner_interval_sweep.json
export AMBI_BENCHMARK_PRESETS='interval_budget/inner_target_s512 interval_budget/inner_target_s256'
export AMBI_BENCHMARK_OUTPUT_ROOT=/oscar/scratch/rgao48/ambi/inner-benchmark/new-interval-campaign/inner-target-s512-s256
sbatch --array=1-5%3 \
  --output=/absolute/log/path/inner-s512-s256-%A_%a.out \
  --error=/absolute/log/path/inner-s512-s256-%A_%a.err \
  slurm/run_ambi_inner_benchmark_oscar.sbatch
```

Keep the checkpoint prefix, matching prior reference root, and exact tested
commit exported as in the Oscar instructions above. Repeat for the other three
pairs with distinct output roots and log paths. A cap of three per array permits
twelve GPUs and 72 CPUs; reduce concurrency when other jobs share the quota.
First run a scheduled
array-index-3 `--smoke` with all eight selectors, a matching seed-101,
three-decision prior reference, and a new output root; smoke omits W&B.
Production runs publish to `ambi-inner-bench`, with the interval, actor LR,
bootstrap, and checkpoint identifying each setting.

### One update per parallel timestep, varying inner rounds

`ambi_humanoid_inner_step_rounds.json` is a small exploratory screen on the
original AMBI prior-only backbone `ambi/u13m14st`. Its three presets are
`step_rounds/j1`, `step_rounds/j3`, and `step_rounds/j6`. All use
`inner_update_timing="step"`, `inner_steps_per_update=512`, 512 parallel
trajectories, horizon 3, and minibatch 512. After each parallel timestep, one
joint critic/actor/automatic-temperature update runs before the next timestep.

| Inner rounds | Imagined transitions per real action | Updates per component per real action |
| --- | --- | --- |
| 1 | 1,536 | 3 |
| 3 | 4,608 | 9 |
| 6 | 9,216 | 18 |

These are nominal counts; terminated branches earn credit only for actual
transitions. Increasing rounds increases both collection and optimizer work.
The comparison does not hold total compute fixed. Replay capacity stays at
9,216 for every setting, with cumulative sampling with replacement. Actor,
critic, temperature, replay, and optimizers reset for each real decision.
Actor/critic/temperature learning rates are 5e-5/1e-4/3e-4. Inner-target Q,
ordinary discounted targets (`inner_finite_horizon=false`), and zero real-replay
mixing remain fixed. Frozen evaluation executes `tanh(mu)` and checks that outer
learning state remains unchanged.

`slurm/run_ambi_inner_step_rounds_oscar.sbatch` maps array indices 1–10 to
50k, 100k, ..., 500k checkpoints. Production uses controller seed 55,
environment seeds 101–105, and 500 decisions per episode: 150 new SAC episodes.
It requests one L40S, six CPUs, 32 GiB and one hour per task, with at most three
tasks active by default. Each checkpoint task evaluates all three presets.
Supply `AMBI_CHECKPOINT_PREFIX`, `AMBI_BENCHMARK_OUTPUT_ROOT`,
`EXPECTED_ACTION_MODES_SHA`, and `CHECKPOINT_MANIFEST` before submission.
Prepare three explicit **New** evaluation-series assignments and export
`EVAL_RUN_MAP`; run one CPU publisher per assigned run. Curve names distinguish
step timing and include every round count, including J6.

Set `AMBI_BENCHMARK_REFERENCE_ROOT` to a directory containing verified
`step_N/prior` reference bundles (symlinks to existing bundles are sufficient).
The launcher reuses those complete references. Only a missing 50k reference is
generated locally; it does not resume or overwrite an existing prior curve.
A fresh `--smoke` output uses seeds 101/102 and three decisions for all selected
settings and its own short prior reference, without curve publication. Override
`--array=6` for a single 300k smoke task and supply durable Slurm output paths.

### Matched round sweep with a frozen prior policy/value tail

`ambi_humanoid_inner_prior_tail_step_rounds.json` repeats the preceding
J1/J3/J6 step-update experiment with exactly one algorithmic change:
`inner_finite_horizon=true`. Select `prior_tail_rounds/j1`,
`prior_tail_rounds/j3`, and `prior_tail_rounds/j6`. The final imagined transition
at H=3 bootstraps from a sampled frozen outer policy action and frozen outer
online Q, using the checkpoint's `mppi_terminal_q_reduction` (mean_pair for
the original AMBI backbone). No additional entropy term is added at this
boundary; the learned outer critic keeps its original soft-Q semantics.
Interior transitions retain the adapting inner actor, inner target critic,
and entropy-augmented SAC target. See the [finite-horizon convention](../../RL/tdmpc2_core/README.md#finite-horizon-inner-sac-and-real-replay-mixing).

N512/H3/B512, one joint update per parallel timestep, 1/3/6 rounds, learning
rates, replay capacity, action-local resets, controller seed 55, environment
seeds 101–105, and 500 real decisions remain matched to `step_rounds`.
The nominal update totals remain 3/9/18 per component; horizon-tail computation
adds policy/Q overhead and consumes additional bootstrap randomness. The
first boundary rows become available at the third parallel step, so J1 gets
its first boundary-informed update at its final update.

Use `slurm/run_ambi_inner_prior_tail_step_rounds_oscar.sbatch` with the same
environment variables, checkpoint grid, resource limits, and explicit New
curve/publisher workflow described above. This launcher requires complete
prior references for **every** production checkpoint, including 50k, and never
reevaluates a full prior episode. `--smoke` still creates a separate three-step
prior for seeds 101/102. All production SAC evaluations are fresh; the original
step-round curves remain available for paired comparison. Curve labels identify
the prior policy and Q tail.

### More critic updates on shared observations

`ambi_humanoid_inner_critic_sweep.json` holds the D512-4-J6 collection and
actor/temperature budgets fixed while increasing critic fitting. Select
`critic_budget/inner_target_c6` and `critic_budget/inner_target_c12`, or their
`outer_target_c6` / `outer_target_c12` counterparts. Each round performs the
original three joint critic/actor/temperature updates, then three or nine
additional critic-only updates. The six-round totals are 36 or 72 critic updates
and 18 actor/temperature updates, with 9,216 imagined transitions. Learning
rates, replay capacity, and bootstrap semantics match the named-run benchmark.
Extra critic updates in rounds one through five can affect subsequent actor
updates. The final round's critic-only tail measures fitting but cannot change
the already-updated actor or its executed action.

This uses the existing total-budget scheduler because explicit per-round
component budgets would change the update order to all critics before actors.
The three-update control is available for explicit selection; it is not launched
by default. Prior episode traces are not substitutes for shared-observation
controls: only bank solves with matching observation and repetition IDs support
fixed-input comparisons.

Reuse the original campaign's `evaluation` directory as
`AMBI_BENCHMARK_REFERENCE_ROOT`; it must contain `step_<step>/roots.json`.
Export the checkpoint prefix, exact pushed commit, and a fresh output root as
above. For the inner-target pair:

```bash
export AMBI_BENCHMARK_MATRIX=configs/research/ambi_humanoid_inner_critic_sweep.json
export AMBI_BENCHMARK_PRESETS='critic_budget/inner_target_c6 critic_budget/inner_target_c12'
sbatch --array=3,4 --time=01:00:00 \
  --output=/absolute/log/path/bank-%A_%a.out \
  --error=/absolute/log/path/bank-%A_%a.err \
  slurm/run_ambi_inner_benchmark_hydra.sbatch --bank-only
```

Submit the outer-target pair separately with its own output root. Each task
runs the two selected budgets sequentially on one GPU, producing local
diagnostic bundles and a combined offline HTML report. Observation-bank solves
are not checkpoint episode-return curves and do not create W&B runs. Each configuration evaluates 25 saved
observations with three solver seeds: 75 solves and no environment episodes.
Initial and per-round probes use eight fixed-noise rollouts of horizon three;
probe model steps are recorded separately from optimization model steps.
`--bank-smoke` uses one repetition per observation and omits W&B, while keeping
the bank's original 500-decision protocol. It also requires a fresh output root.

### Traces and portable report

```bash
"$AMBI_EVAL_PY" report_ambi_benchmark.py \
  --bundle results/inner-bench/prior \
  --bundle results/inner-bench/screen \
  --bundle results/inner-bench/confirmation \
  --output results/inner-bench/comparison.html
```

Open the resulting HTML locally in a browser. It contains its scripts and data;
no server, CDN, plotting package, or W&B login is needed. Rebuild with additional
`--bundle` inputs as experiments finish. Repeat `--metric` to produce a smaller
report containing only selected metrics. Existing reports require `--overwrite`.

Each bundle contains a manifest, compressed JSONL traces per episode or bank
solve, and the observation bank when applicable. Completed shards survive later
failures. Failed/partial episodes and missing/nonfinite measurements remain
explicit. A new evaluation requires a new bundle directory; `--overwrite`
retains its existing meaning for the evaluator's optional legacy `--output`.

The report offers decision/update heatmaps and selected-solve overlays. Critic
metrics use critic updates; actor metrics use actor updates; fixed policy probes
use round boundaries. Update losses measure the minibatch *before* that update,
although the counters name the completed update. Missing metrics and unequal
budget endpoints are never interpolated. `decision/` metrics retain the existing
per-solve aggregate statistics and real rewards/timing.

Trajectory positions across configurations can represent different states.
Shared-root mode instead joins the exact bank, observation, repetition, solver
seed, and probe protocol. Selecting the same observation allows a controlled
comparison of inner learning. Incompatible checkpoints, metric meanings, and
episode protocols are rejected instead of silently overlaid.

Raw traces reuse existing optimizer metrics, including losses, Q/target values,
entropy, temperature, gradients, and available saturation/KL statistics. They
add no model calls, random draws, or per-update device synchronization. Detached
scalars travel with the existing action/metrics CPU transfer; serialization and
W&B work happen outside the solve.

Extra probes run only on the bank, initially and after every round. They use
eight fixed-noise rollouts over three model steps and a fixed outer temperature.
The report separates predicted reward, terminal frozen target-Q bootstrap, and
entropy contribution, alongside fixed target-Q action gain, action displacement,
and KL. These are model predictions; a soft-Q bootstrap is not a measured reward
return. Full-episode returns remain the performance test.

### W&B and timing

`--wandb` explicitly enables summaries and artifacts in project `ambi-inner-bench`
under `rwgao_b-brown-university`. Omit it for local-only evaluation, or use
`--wandb-mode offline` to test SDK publication without network access. Each
selected configuration gets a separate run, grouped by checkpoint/configuration
and tagged as episodes, bank, or both. Episode histories and bank summaries use
stable metric names; full traces live in the portable artifact. Credentials are
not fetched by the evaluator. Use the execution environment's normal W&B setup.

Each episode row is an independent seeded evaluation observation from the same
frozen checkpoint. Its return can vary between seeds without any training
occurring. `episode/index` identifies evaluation order; `env_step` counts
cumulative environment decisions evaluated so far. Neither measures training
progress. Raw episode rows and paired return deltas remain in W&B history, but
their automatic plots are hidden; their declared x-axis is `episode/index`.
The cumulative `env_step` metric is also hidden from automatic plots.

Run names include the task, checkpoint step, controller/update schedule, and
bootstrap critic. Tags expose the source run, checkpoint, controller, bootstrap,
and collection/update settings for filtering. Names and tags are saved in the
bundle manifest; prior-only runs carry no active SAC schedule labels.

Bank repetitions and probe settings live under the matrix's `evaluation` block;
`--bank-repetitions` overrides its repetition count. Model initialization, one
unscored warmup including lazy compilation, control, probe, serialization, and
publication costs are recorded separately. Total bundle elapsed time includes
all stages and W&B finish. Probe model steps are additional to optimizer model
steps. CUDA event timing is resolved only after the normal action transfer.

The raw-trace overhead acceptance target is at most 5% median warmed CUDA time.
It requires a measurement on the actual checkpoint/hardware; CPU correctness
tests do not establish it. The opt-in test in `tests/test_ambi_inner_trace.py`
supports an actual checkpoint and emits matched trace-off/on timings. Run GPU
validation through the scheduler, not a login-node shell.

```bash
AMBI_RUN_TRACE_CUDA_BENCHMARK=1 \
AMBI_TRACE_BENCHMARK_CHECKPOINT="$AMBI_CHECKPOINT" \
AMBI_TRACE_BENCHMARK_OUTPUT=/tmp/ambi-trace-overhead.json \
"$AMBI_EVAL_PY" -m pytest -s -q \
  tests/test_ambi_inner_trace.py::test_cuda_trace_overhead_measurement
```

The output path must be new and outside the source checkout. The test uses five
warmup solves and ten alternating trace-off/on pairs. It reports the threshold
result without imposing a noisy wall-clock assertion on ordinary CI.

The frozen evaluator supports finite-horizon inner SAC and transition-based
update counts. It rejects `inner_outer_replay_fraction > 0` because these model
snapshots do not include real replay. Test replay mixing through a populated
training agent. See the [inner SAC options](../../RL/tdmpc2_core/README.md#finite-horizon-inner-sac-and-real-replay-mixing)
for the target convention and schedule restrictions.

### Existing research matrices

The active end-to-end configs live directly under `configs/ambi/`.
The frozen-checkpoint matrix is
`configs/research/ambi_inner_decoupling.json`, outside the active AMBI training
tree so it cannot be confused with the independently trained branch and horizon
comparisons.

# Frozen-checkpoint preset workflow

`ambi_inner_decoupling.json` is a compact matrix of one-axis-at-a-time overrides
based on `configs/algs/AntAMBITDMPC2.json`. The canonical AMBI reference is
fresh, action-local, fully cloned inner SAC.
The remaining operators, LoRA and persistence settings below deliberately test
auxiliary ablations or comparators; they are not alternative definitions of
AMBI. The matrix covers:

- none, SAC, TD3, and compute-matched MPPI inner operators;
- the reference five-head distributional-Q model and a scalar twin-Q ablation;
- actor, critic, and temperature adaptation controls;
- train-only actor, online-critic, and joint prior writeback at beta 0.01, 0.1,
  and 1.0, plus the no-writeback reference;
- temperature, imagined behavior, and returned-action exploration;
- explicit J/N/H/G collection and joint-update schedules;
- action, episode, and run lifecycles;
- replay sampling, bootstrap source, rollout horizon, clone/LoRA, LoRA rank,
  and outer-policy anchoring controls.

Within a variant's `alg_params`, `null` removes an inherited base key. The
operator comparison uses this to keep SAC's J/N/G controls out of MPPI and
no-improvement configurations.

List the matrix without importing the training stack:

```bash
python3 evaluate_ambi_checkpoint.py --list-presets
```

Materialize ordinary algorithm configs that can be referenced by the existing
`main.py --run ... --alg-dir ...` workflow:

```bash
python3 evaluate_ambi_checkpoint.py \
  --comparison inner_operator \
  --materialize-dir configs/algs/generated
```

Materialization also writes `AMBIResearchExperiment.json`, preserving the
matrix's environment parameters. Run it with both paths pointed at that output:

```bash
python3 main.py \
  --run configs/algs/generated/AMBIResearchExperiment.json \
  --alg-dir configs/algs/generated
```

Evaluate the default operator comparison from one frozen checkpoint:

```bash
python3 evaluate_ambi_checkpoint.py \
  --checkpoint /path/to/ambi-checkpoint.pt \
  --device cuda \
  --seeds 101 102 103 104 105 \
  --output logs/ambi_inner_operator_eval.json
```

Select individual presets with repeatable `--preset comparison/variant`, or an
entire axis with repeatable `--comparison comparison`. The evaluator creates a
fresh model for every preset, uses paired environment seeds, always returns the
policy mean to the real environment, never calls the outer update, and hashes
outer model/optimizer/temperature state before and after each run. Its output
contains per-episode real returns and all finite model-predicted inner metrics.
Output is written atomically, and an existing `--output` path is preserved
unless `--overwrite` is supplied explicitly.
When a comparison's reference preset is selected, it also reports seed-paired
return deltas for every selected variant.

Q representation is part of the checkpoint architecture. The reference
checkpoint uses five distributional heads. It can compare checkpoint-compatible
inner operators and controls, but it cannot be evaluated as a scalar twin model
(or vice versa). Train and supply a matching checkpoint for each side of the
Q-representation comparison, using one preset per invocation:

```bash
python3 evaluate_ambi_checkpoint.py --checkpoint distributional.pt \
  --preset q_representation/distributional_five --output distributional-eval.json
python3 evaluate_ambi_checkpoint.py --checkpoint scalar.pt \
  --preset q_representation/scalar_twin --output scalar-eval.json
```

The evaluator rejects a mixed-architecture selection before running either
side. It also rejects the train-only `execution_noise` and `prior_writeback`
axes. Deterministic evaluation returns the policy mean, which collapses the
execution-noise variants. Prior writeback is deliberately disabled outside
training, so its variants would likewise collapse under a frozen outer
checkpoint. Materialize and train those axes instead.
