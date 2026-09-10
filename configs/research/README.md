# Frozen-checkpoint AMBI research

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

Select `update_timing/step_j5_c1_a1` for five rounds with an update after each
imagined rollout step. Each vector step adds 512 transitions, then performs
one critic→actor→target update before collecting the next depth with the
adapted policy. Horizon three gives 15 critic and 15 actor updates per real
decision. This uses the shared step scheduler (`inner_steps_per_update=512`)
and calibrates the native Q scale from the first collection before updating.
The default round-end preset remains unchanged.

Select `update_timing/step_j1_c1_a1` or `update_timing/step_j3_c1_a1` for the
smaller-round comparison. These perform 3/9 paired actor, critic and target
updates and collect 1,536/4,608 imagined transitions per decision, respectively.
Both retain N512, horizon three, batch512, the same native learning settings
and one paired update per vector step. Replay capacity is 32,768, matching the
J7/J10 rounds sweep; all populated transitions are retained. These presets
change only the number of rounds relative to J7/J10 and leave the default
selection unchanged.

Select `update_timing/step_j5_c2_a2` for two critic→actor→target updates after
each vector step, using `inner_steps_per_update=256`. Collection still adds
512 transitions at once. This gives 30 critic, actor, and target updates per
real decision, with the same 7,680 imagined model steps and inherited settings.

Select `update_timing/step_j7_c1_a1` or `update_timing/step_j10_c1_a1` to
increase fresh rollout rounds at one paired update per vector step. These use
21/30 actor, critic and target updates and 10,752/15,360 imagined transitions
per decision, respectively. Both allocate replay capacity 32,768 so the J10
solve retains every transition; the older 12,288 capacity would evict data.
This capacity change is explicit in each new planner identity. Increasing
unused capacity preserves the J5 behavior because sampling uses populated
replay entries. Other native learning settings remain inherited unchanged.

On Oscar, use `slurm/run_tdambi_checkpoint_eval_oscar.sbatch` with the same
input variables as the Hydra launcher and the desired `TDAMBI_PRESET`.
It uses one L40S, six CPUs and 32 GiB per checkpoint, with no array throttle;
choose parallelism from live account/QOS and memory headroom. The Oscar smoke
launcher also accepts `TDAMBI_PRESET`, defaulting to the original round-end
preset. Give different planner settings separate output roots and explicitly
created curve registries.

For the J5/C1 actor-entropy ablation, select `entropy/squashed_eta1e_5`,
`entropy/squashed_eta1e_4`, or `entropy/squashed_eta1e_3`. These use the joint
tanh-corrected entropy with fixed coefficients `1e-5`, `1e-4`, and `1e-3`;
`entropy/native_j5` is exactly the existing `update_timing/step_j5_c1_a1`
reference. The default preset is unchanged. All four retain J5/N512/H3/B512,
replay capacity 12,288, 15 paired updates, native learning rates, reward-only
critic targets and the same per-decision resets. No retraining is involved.
The coefficient grid spans two orders of magnitude because native scaled
entropy and joint squashed entropy have different scales; it does not claim
to match their gradient strengths. Compare paired returns and saturation as
well as entropy. The episode entropy diagnostics average samples over each
controller's changing visited/imagined states, not a fixed observation bank.

Each coefficient requires its own explicitly created curve. Existing native
J5 results can be used for comparison after a default-behavior compatibility
check; do not append the new implementation to an older scientific identity.

The Hydra launcher accepts `TDAMBI_PRESET` and array indices 2–20 for the
100k–1M checkpoint grid. It requests one GPU per checkpoint from the L40S/A6000 pool with no array
throttle or fixed node, so pending checkpoints can use either node as slots
free. Override resource constraints explicitly for other validated hardware.

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
