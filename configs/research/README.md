# Frozen-checkpoint AMBI research

## Mean-prefix calibration with twenty seeds

[`ambi_prior_mean_prefix_h1_20seeds.json`](ambi_prior_mean_prefix_h1_20seeds.json)
keeps the prior-initialized H1/N128/J4/C32/A4 learner unchanged at 150k and
200k, adds mean-action prefixes followed by sampled frozen-prior continuations,
and expands paired episode coverage to seeds 101–120. See the
[measurement and Oscar execution protocol](MEAN_PREFIX_OSCAR.md) for seed
sharding, validation and complete-panel publication.

## Prior-initialized refinement at one checkpoint

[`ambi_prior_refinement_h1_200k.json`](ambi_prior_refinement_h1_200k.json) is the
fast iteration reference on the hash-pinned `mey3rxj8` **200k checkpoint only**.
Its default `initialization/inherited` copies the checkpoint's actor and online
critic at every decision. The actor keeps its learned mean and standard
deviation; the scratch-only `inner_actor_initial_std` override is cleared.
Optimizers and replay start fresh, and outer weights remain frozen.

Relative to `ambi_scratch_takeoff_h1_alpha_zero.json`, the only algorithm changes
are actor initialization, critic initialization, and clearing that standard-
deviation override. H1/N128/J4/B256, C32 then A4 each round, alpha zero, the
checkpoint's frozen saved Q scale, and all seeds, roots, repetitions and
continuation lengths are retained. One checkpoint uses one seventh of the
previous panel's total work; wall time does not scale by the same factor when
checkpoint tasks run concurrently.

Use the existing evaluators with an explicit single checkpoint and preset:

```bash
AMBI_REFINEMENT_MATRIX=configs/research/ambi_prior_refinement_h1_200k.json
# Set these to the verified checkpoint and fresh output paths on the runtime.
AMBI_REFINEMENT_CHECKPOINT=/absolute/path/to/verified-200000.pt
AMBI_REFINEMENT_RESULTS=/absolute/path/to/fresh-refinement-results
AMBI_REFINEMENT_PY=environments/dmcontrol/.venv/bin/python
AMBI_REFINEMENT_ATTEMPT='<explicit-new-attempt>'
"$AMBI_REFINEMENT_PY" evaluate_ambi_checkpoint.py \
  --matrix "$AMBI_REFINEMENT_MATRIX" --preset initialization/inherited \
  --checkpoint "$AMBI_REFINEMENT_CHECKPOINT" --device cuda \
  --bundle-dir "$AMBI_REFINEMENT_RESULTS/episodes" \
  --reference-bundle /absolute/path/to/verified-prior-episode-bundle
"$AMBI_REFINEMENT_PY" evaluate_ambi_calibration.py run \
  --matrix "$AMBI_REFINEMENT_MATRIX" --preset initialization/inherited \
  --checkpoint "$AMBI_REFINEMENT_CHECKPOINT" --device cuda \
  --bundle-dir "$AMBI_REFINEMENT_RESULTS/real" \
  --attempt-label "$AMBI_REFINEMENT_ATTEMPT" \
  --save-root-bank "$AMBI_REFINEMENT_RESULTS/root-bank.json" \
  --reference-cache "$AMBI_REFINEMENT_RESULTS/prior-reference"
```

Run compute through the scheduler on Oscar after the usual tested-commit
synchronization and CUDA smoke. The guarded `ambi_takeoff_campaign.py` launcher also accepts this matrix:
set `AMBI_TAKEOFF_MATRIX` for both GPU workers and diagnostic publishers, and
provide an inventory containing only its pinned 200k checkpoint. Array index 0
is ordinary episode evaluation and index 1 is real calibration. Smoke and
production each use `--array=0-1`, with fresh output directories and an explicit
new publication attempt. The reference configuration itself does not submit
or publish a run.

Assess paired return changes from the actual initialization at actor-update
counts 0/4/8/12/16, using the same roots and independent diagnostic noise.
Round-zero inherited actions must match the frozen prior under paired noise.
Ordinary episodes execute mean actions; existing shared-root calibration uses
sampled actions. A mean-action shared-root diagnostic remains follow-up work:
with alpha zero, sampled gains may arise from reduced variance without a
better mean action. Preserve this distinction when interpreting results.
The 125k and 300k checkpoints can later test transfer of a selected recipe;
they are not included in this iteration's checkpoint contract. Success on 200k
alone is exploratory evidence, not a claim across training stages.

### Parallel checkpoint coverage

[`ambi_prior_refinement_h1_parallel.json`](ambi_prior_refinement_h1_parallel.json)
retains the same prior-initialized recipe and selects six checkpoints: 125k,
150k, 200k, 300k, 500k and 2M. The 200k-only configuration remains the focused
iteration reference. This panel uses twelve independent GPU tasks, so it fits
in one wave when twelve GPUs and the required CPU/memory allowance are free.
The 100k checkpoint is omitted because the prior remains near the return floor.
Use the same matrix for workers and publishers; the inventory must list exactly
these six checkpoints in order. Smoke indices `2,8` both select 200k. Production
indices `0-5` evaluate episodes and `6-11` run real calibration. Set concurrency
from live resources, rather than assuming this allowance always exists.

## Scratch inner SAC and to-go calibration

[`ambi_scratch_takeoff_h1.json`](ambi_scratch_takeoff_h1.json) is the early-stage
pilot on seven hash-pinned `mey3rxj8` checkpoints: 100k, 125k, 150k, 200k, 300k,
500k, and 2M. Its `initialization/scratch` uses J4/N128/H1/B256, random actor and
critic, initial pre-tanh standard deviation 0.3, and 32 critic then 4 actor
updates after each complete collection round. Replay capacity stays at 2,048
to retain all data even in a later N512 comparison. Alpha and each checkpoint's
saved Q scale remain fixed. H1 isolates action-value fitting from multistep
target propagation. These are exploratory starting values, not a tuned recipe.
The model probe and real calibration protocol is documented in
[TOGO_CALIBRATION.md](TOGO_CALIBRATION.md#early-checkpoint-h1-pilot).

[`ambi_scratch_togo_reference.json`](ambi_scratch_togo_reference.json) pins the
reward/Q-scale `mey3rxj8` 2M backbone with J5/N512/H3/B512, fresh random inner
networks, fixed alpha and saved Q scale, and a frozen outer horizon bootstrap.
It enables model-return probes at initialization and every round. A separate
simulator evaluator compares saved actors and the prior at shared states, with
configurable coverage, continuing Monte Carlo tails, paired noise, and portable
HTML/W&B diagnostics. See [commands, return definitions, and overhead](TOGO_CALIBRATION.md).

## TDAMBI on native TD-MPC2 checkpoints

`tdambi_humanoid_inner_benchmark.json` selects evaluation-only native TD-MPC2
actor–critic adaptation. It supports single-task state checkpoints with their
saved target critic and adjacent `.metadata.json`; an AMBI SAC checkpoint is
not interchangeable. Encoder, dynamics, reward model and outer state stay
frozen. See the [learner contract](../../RL/tdmpc2_core/README.md#tdambi-native-inner-learning)
for the native losses and scale initialization.

The default `inner_budget/tdambi_3` uses six rounds, 512 imagined rollouts per
round, horizon three, batch size 512, replay capacity 12,288, and three paired
critic/actor updates per round. Select `inner_budget/tdambi_6` or
`inner_budget/tdambi_12` explicitly for larger budgets. Learning rates, fixed
entropy coefficient, value coefficient, discount, target interpolation, and
gradient clipping inherit the saved native settings. No sweep is launched.

New native training checkpoints include their learned Q scale. By default each
inner solve starts from that saved value, then updates only its local copy.
Older checkpoints without a scale use the existing first-collection calibration.
For a matched comparison at the default three-update budget, select
`q_scale/checkpoint` (requires saved `S`) or `q_scale/calibrate` (calibrates from
imagined replay). These presets retain the same rollout, update and episode
budgets. Results identify the actual scale source and initial/final values;
the saved frozen scale is included in the outer-state immutability check.

Choose a checkpoint and fresh output paths. This command evaluates five
episodes (environment seeds 101–105, controller seed 12345, at most 500 decisions
each), saving all inner-update traces locally:

```bash
AMBI_EVAL_PY=environments/dmcontrol/.venv/bin/python
TDAMBI_CHECKPOINT=/absolute/path/to/checkpoint.pt
TDAMBI_RESULTS=/absolute/path/to/fresh-results
"$AMBI_EVAL_PY" evaluate_ambi_checkpoint.py \
  --matrix configs/research/tdambi_humanoid_inner_benchmark.json \
  --preset inner_budget/tdambi_3 --checkpoint "$TDAMBI_CHECKPOINT" \
  --device cuda --bundle-dir "$TDAMBI_RESULTS/bundle" \
  --output "$TDAMBI_RESULTS/results.json"
"$AMBI_EVAL_PY" report_ambi_benchmark.py \
  --bundle "$TDAMBI_RESULTS/bundle" --output "$TDAMBI_RESULTS/report.html"
```

Add `--reference-bundle /absolute/path/to/native-paired-results.json` to reuse
the prior episodes saved by the TD-MPC2 paired evaluator. Ingestion verifies
checkpoint hash, source, environment, seeds, episode limit and central-action
rule, and preserves the reference's controller RNG provenance. It does not
rerun or republish the prior or MPPI results. A three-decision smoke cannot use
a 500-decision reference. TDAMBI v1 supports episode traces; the SAC bank probes
are rejected because their temperature and soft-return definitions do not
describe this learner.

### Explicit checkpoint-curve publication

Use the common New/Append workflow from a clean, tested checkout. Generate the
identity template before GPU jobs with the same evaluation arguments above,
replacing output options with `--eval-series-spec-dir /absolute/path/to/specs`.
Supply `--checkpoint-inventory` if source provenance needs the validated
checkpoint inventory. Then explicitly choose **one** action:

```bash
# New attempt: returns its selected run directory and W&B ID.
"$AMBI_EVAL_PY" eval_series.py create --root /absolute/path/to/curve-registry \
  --spec /absolute/path/to/specs/inner_budget__tdambi_3.json \
  --attempt-label first --owner oscar-rgao48

# Append: validate the identity against a specifically selected existing run.
"$AMBI_EVAL_PY" eval_series.py append /absolute/path/to/selected-run \
  --spec /absolute/path/to/specs/inner_budget__tdambi_3.json
```

Pass `--eval-run-dir /absolute/path/to/selected-run` to every checkpoint job for
that curve. Jobs atomically save and queue results; the existing CPU publisher
owns the W&B session (`eval_series.py publish ... --owner oscar-rgao48`). Its
`--watch --jobs JOB_ID ...` mode checks every 15 seconds. Results appear on the
native checkpoint curves in `ambi-inner-bench`. An upload failure can be retried
without GPU evaluation. Measured Q scale is a diagnostic; the initialization
rule and native learning settings are part of planner identity.

### Oscar smoke and CUDA trace measurement

After committing, pushing and checking out the tested commit through Git,
submit the supplied smoke script from that clean checkout. Pin
`EXPECTED_ACTION_MODES_SHA`, `TDAMBI_CHECKPOINT` and a fresh absolute
`TDAMBI_SMOKE_OUTPUT_ROOT` in the submission environment, then use:

```bash
sbatch --export=ALL --output=/absolute/path/to/logs/tdambi-%j.out \
  --error=/absolute/path/to/logs/tdambi-%j.err \
  slurm/run_tdambi_checkpoint_smoke_oscar.sbatch
```

The script runs focused correctness tests, three Humanoid decisions and an
offline HTML report. Set `TDAMBI_MEASURE_TRACE=1` before submission to include
the existing alternating trace-off/on CUDA benchmark. Its receipt separates
CUDA and wall control time, calibration time, warmup, CPU trace append and gzip
serialization; no diagnostic probes or uploads run in that benchmark. The
5% median raw-tracing overhead remains a measured hardware acceptance target,
not a guarantee established by CPU tests. No training or publication runs as
an implicit part of this smoke.

## Humanoid inner-SAC benchmark

`ambi_humanoid_inner_benchmark.json` evaluates a frozen checkpoint using its
adjacent `.metadata.json` as the source of algorithm, observation, and environment
settings. The matrix declares source run `rwgao_b-brown-university/ambi/u13m14st`;
this is provenance supplied by the experiment configuration, not proof that an
arbitrary checkpoint came from that W&B run. Pin the intended checkpoint file
and keep its sidecar. Every result records the actual checkpoint SHA256, saved
settings, code fingerprint (including uncommitted source), and runtime versions.

The frozen evaluator supports finite-horizon inner SAC, transition-based update
counts, and optional [step–update interleaving](../../RL/tdmpc2_core/README.md#stepupdate-inner-sac)
with `inner_update_timing="step"`. It rejects `inner_outer_replay_fraction > 0` because these model
snapshots do not include real replay. Test replay mixing through a populated
training agent. See the [inner SAC options](../../RL/tdmpc2_core/README.md#finite-horizon-inner-sac-and-real-replay-mixing)
for the target convention and schedule restrictions.

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
  --save-root-bank results/inner-bench/roots.json --wandb
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
  --bundle-dir results/inner-bench/confirmation --wandb
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

The legacy extra probes run only on the bank, initially and after every round. They use
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

### Existing research matrices

The active end-to-end configs live directly under `configs/ambi/`.
The frozen-checkpoint matrix is
`configs/research/ambi_inner_decoupling.json`, outside the active AMBI training
tree so it cannot be confused with the independently trained branch and horizon
comparisons.

### Critic-only LoRA-RL comparisons

The [LoRA-RL implementation contract](../../RL/tdmpc2_core/README.md#critic-only-lora-rl)
keeps the actor dense and applies low-rank updates only to selected critic
matrices. The initial reference is rank 96, direct scale one, and AdamW adapter
weight decay `2e-4`. Output heads, biases, and normalization parameters remain
trainable. All scientific inner state resets at each real decision.

The frozen-checkpoint matrix `ambi_inner_decoupling.json` provides these
explicit selectors without changing its default operator comparison:

- `critic_lora_placement/clone`: full actor and critic adaptation.
- `critic_lora_placement/input_hidden_r96`: LoRA on the critic input and hidden
  matrices; the default placement for AMBI's learned-latent input.
- `critic_lora_placement/hidden_r96`: LoRA on hidden matrices only, closer to
  the paper's placement, with a normally trainable input projection.
- `critic_lora_rank/rank_64`, `rank_96`, and `rank_128`: input-and-hidden
  placement with identical scale, decay, actor adaptation, and SAC budgets.

These replace the old `adaptation_mechanism` and `lora_capacity` axes. Old
selectors and active `"lora"` configurations do not silently select the new
method. Reproducing a historical run requires its historical source; start a
new evaluation attempt when adopting `"lora_rl"`.

Four matching algorithm/experiment pairs under `configs/dmcontrol/` provide
single-seed Humanoid Walk training screens:

- `ambi_humanoid_walk_base_v2_lora_rl_input_hidden_r64`
- `ambi_humanoid_walk_base_v2_lora_rl_input_hidden_r96`
- `ambi_humanoid_walk_base_v2_lora_rl_input_hidden_r128`
- `ambi_humanoid_walk_base_v2_lora_rl_hidden_r96`

Each retains the base-v2 recipe: seed 55, 14 million real decisions, J=8, N=32,
H=3, G=1, replay capacity 768, and the original evaluation/checkpoint settings.
Only critic adaptation and descriptive run identity change. These are full
training configurations, not smoke tests; no sweep runs by adding them. Rank 96
and decay `2e-4` borrow the paper's DMC SimbaV2 settings, while the rank and
placement alternatives test the transfer to AMBI. The trained prior and zero-B
initialization deliberately differ from the paper's main setup. Neither a
return improvement nor a speedup has been established for these configurations.

# Frozen-checkpoint preset workflow

`ambi_inner_decoupling.json` is a compact matrix of one-axis-at-a-time overrides
based on `configs/algs/AntAMBITDMPC2.json`. The canonical AMBI reference is
fresh, action-local, fully cloned inner SAC.
Source configurations and materialized presets inherit `rho=0.5` and the
canonical TD-MPC2 temporal reduction: transition losses divide by the training
horizon, and the outer actor divides by the number of latents (`H+1`). Retired
normalization-selection and reference-horizon fields are absent from maintained
presets. Historical checkpoint sidecars remain unchanged and parse through the
deprecated-input compatibility path; training under the new rule starts a new
lineage.
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
- replay sampling, bootstrap source, rollout horizon, critic-only LoRA-RL placement and rank,
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
