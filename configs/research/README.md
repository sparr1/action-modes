# Frozen-checkpoint AMBI research

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
  --bundle-dir results/inner-bench/screen --wandb
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
