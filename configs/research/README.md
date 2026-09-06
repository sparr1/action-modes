# Frozen-checkpoint AMBI research

## AMBI-XQC inner J6 checkpoint campaign

`ambixqc_humanoid_inner_j6_benchmark.json` evaluates native inner XQC with
J6/N512/H3/G3, batch 512, replay capacity 9,216, and sampling with replacement.
It borrows these rollout and update-slot budgets from
`AMBITDMPC2-humanoid-walk-base-v2-d512-4-j6-seed55`; it retains the XQC learning
rule. The budget source is the canonical configuration
`configs/dmcontrol/algs/ambi_humanoid_walk_base_v2_d512_4_j6.json` at commit
`698bd074551bc48128cb34f78adaf8caaab1f188`, file SHA-256
`96f16f50207e20b118b5c30f913c89c853e71369a02464777c73062ff6e34ae5`.
This identifies the budget reference, separately from the evaluated checkpoint.

Each real decision collects 3,072 imagined branches and 9,216 model
transitions. Eighteen XQC update slots produce **18 critic, six actor, and six
temperature optimizer steps** with the checkpoint's policy delay of three.
Actor objectives and BatchNorm training forwards still run in every slot.
Each decision copies the persistent actor, online critic, and temperature into
a fresh inner learner with empty optimizer state and replay. The inner target
starts from the copied online critic. Tensor allocations may be reused. Both real controllers
execute `tanh(mu)`; imagined behavior and inner adaptation remain stochastic.

The matrix inherits the checkpoint's architecture, XQC optimizer semantics,
learning rates, and reward normalization. The seed-55 prior bank uses inner
actor/critic learning rates `5e-5` and `frozen_real_scale`; temperature uses the
inner actor learning rate. Its real reward scale and return accumulator stay
frozen. These settings differ from the named SAC configuration's learning
rates, critic representation, and actor/temperature update rule. Evaluation
checks complete outer state, including BatchNorm, targets, optimizers,
temperature, normalization, counters, and the outer learner RNG.

The default selects only `controller/xqc`, with `controller/prior` as its
reference, environment seeds 101–105, controller seed 12345, and at most 500
decisions per episode. Reuse a completed prior reference from the paired MPPI
campaign when its checkpoint hash and full episode protocol match. The
reference's MPPI rows, if present, are ignored when computing paired gains.
The campaign runner filters them from its XQC-versus-prior reports; the generic
report command below displays all runs from its selected bundles.
The older default XQC matrix uses controller seed 55, so its references do not
match this protocol.

```bash
XQC_EVAL_PY=environments/dmcontrol/.venv/bin/python
XQC_J6_MATRIX=configs/research/ambixqc_humanoid_inner_j6_benchmark.json
XQC_CHECKPOINT=/absolute/path/to/checkpoint.pt
XQC_PRIOR_BUNDLE=/absolute/path/to/matching/prior-bundle

"$XQC_EVAL_PY" evaluate_ambi_checkpoint.py \
  --matrix "$XQC_J6_MATRIX" --checkpoint "$XQC_CHECKPOINT" --device cuda \
  --reference-bundle "$XQC_PRIOR_BUNDLE" \
  --bundle-dir results/xqc-j6/inner

"$XQC_EVAL_PY" report_ambi_benchmark.py \
  --bundle "$XQC_PRIOR_BUNDLE" --bundle results/xqc-j6/inner \
  --output results/xqc-j6/comparison.html
```

If a matching prior reference is unavailable, select `--preset controller/prior`
with this same matrix and save it in a fresh bundle first.
Repeat with separate output paths for each selected checkpoint; the established
bank comparison uses all 30 checkpoints at 50,000-decision intervals through
1.5 million decisions. Commands run on the current host; use scheduler compute
nodes for full evaluations. Add `--wandb` for explicit publication to
`ambi-inner-bench`. Reports contain raw episode returns and paired gains,
decision timing, and aggregate normalized XQC value metrics with actual
optimizer counts. They do not provide per-update optimizer traces.

The versioned Oscar launcher is `slurm/run_ambixqc_inner_eval_oscar.sbatch`.
It uses one L40S, six CPUs, 32 GB, and a two-hour limit per checkpoint task,
with the existing locked runtime. Run from a clean checkout at the pushed
evaluation commit. The campaign manifest records checkpoint and prior-reference
hashes; its first and last rows also carry `smoke_reference_bundle` and
`smoke_reference_manifest_sha256` for matching short smoke episodes. Create the
durable `slurm` output directory before submission.

```bash
export EXPECTED_ACTION_MODES_SHA=$(git rev-parse HEAD)
export CHECKPOINT_MANIFEST=/absolute/scratch/campaign/checkpoint-manifest.json
export RESULT_ROOT=/absolute/scratch/campaign/smoke
export AMBIXQC_EVAL_MODE=smoke
sbatch --array=0,29%2 --output=/absolute/scratch/campaign/slurm/smoke-%A_%a.out \
  --error=/absolute/scratch/campaign/slurm/smoke-%A_%a.err \
  slurm/run_ambixqc_inner_eval_oscar.sbatch

# After both endpoint smoke tasks pass, evaluate the full 30-checkpoint bank.
export RESULT_ROOT=/absolute/scratch/campaign/production
export AMBIXQC_EVAL_MODE=production
sbatch --array=0-29%12 --output=/absolute/scratch/campaign/slurm/eval-%A_%a.out \
  --error=/absolute/scratch/campaign/slurm/eval-%A_%a.err \
  slurm/run_ambixqc_inner_eval_oscar.sbatch

"$XQC_EVAL_PY" summarize_ambixqc_inner_eval.py \
  --manifest "$CHECKPOINT_MANIFEST" --results-root /absolute/scratch/campaign \
  --expected-source-sha "$EXPECTED_ACTION_MODES_SHA" \
  --output /absolute/scratch/campaign/summary.json \
  --html /absolute/scratch/campaign/comparison.html --wandb
```

The summary checks completed checkpoint bundles and matching prior references
before publishing paired return curves. For local downloads whose reference
paths have moved, use `--reference-root` pointing to the copied prior campaign
root; its `production/step_N/bundle` manifests must retain their original hashes.

## AMBI-XQC prior versus MPPI

`ambixqc_humanoid_mppi_benchmark.json` compares the persistent policy with an
evaluation-only MPPI controller on the same AMBI-XQC checkpoint. It defaults to
both `controller/prior` and `controller/mppi`, seeds 101–105, controller seed
12345, and at most 500 decisions per episode. It does not run inner XQC updates.
Training and checkpoint formats are unchanged: the evaluator loads the saved
XQC model as a frozen prior and attaches a separate planner.

```bash
XQC_EVAL_PY=environments/dmcontrol/.venv/bin/python
XQC_MPPI_MATRIX=configs/research/ambixqc_humanoid_mppi_benchmark.json
XQC_CHECKPOINT=/absolute/path/to/checkpoint.pt

"$XQC_EVAL_PY" evaluate_ambi_checkpoint.py \
  --matrix "$XQC_MPPI_MATRIX" --checkpoint "$XQC_CHECKPOINT" --device cuda \
  --bundle-dir results/xqc-mppi/paired

"$XQC_EVAL_PY" report_ambi_benchmark.py \
  --bundle results/xqc-mppi/paired --output results/xqc-mppi/paired.html
```

To reuse prior outcomes, first select only `--preset controller/prior` and save
its bundle. Then select only `--preset controller/mppi` with
`--reference-bundle /path/to/prior-bundle` and a new `--bundle-dir`. The prior
reference must match this checkpoint hash and the episode protocol, including
controller seed 12345; a reference generated with the older XQC matrix's
controller seed 55 is not a match. Add `--wandb` to publish the selected run(s)
to `ambi-inner-bench`. These commands do not submit scheduler jobs.

The planner uses horizon 3, 512 candidates, 64 elites, 24 policy trajectories,
standard deviation bounds 0.05–2, and temperature 0.5. Its configured six
iterations become eight for action dimensions of at least 20, including
Humanoid Walk's 21 actions. Search uses native stochastic samples and executes
a native weighted elite action with no additional Gaussian execution noise.
This differs from the prior's deterministic `tanh(mu)` action; each run records
its own action rule. The shifted plan mean persists within an episode and is
cleared after warmup and before every scored episode. Private planner RNG is
seeded independently of saved training RNG and preset execution order.

Scoring adds raw TOLD predicted rewards and a terminal value from the online
twin XQC critics, averaged across heads and multiplied by the checkpoint's
frozen real reward scale. This gives consistent raw reward units while retaining
the learned soft-Q tail; it does not turn XQC's critic into a separately trained
reward-only critic. No critic, actor, temperature, BatchNorm statistic, optimizer,
real reward normalizer, or outer RNG is updated. Decision records report actual
model work and zero optimizer steps, alongside returns and control time.

Planner settings live in a variant's `evaluation_controller` object, separate
from training `alg_params`. Only the eight documented planner settings are
accepted. MPPI presets cannot be materialized into training configurations.
Use the existing XQC matrix below for an explicitly requested inner-XQC
comparison.

The Oscar campaign launcher, `slurm/run_ambixqc_mppi_eval_oscar.sbatch`, evaluates
the seed-55 prior bank at all 30 checkpoints from 50,000 through 1,500,000
decisions. Its JSON manifest contains the source run and one ordered row per
checkpoint with `step`, absolute `path`, `sha256`, and `metadata_sha256`.
`run_ambixqc_mppi_evaluation.py` validates these hashes and training metadata
before allocating an environment. Each array task writes a fresh `step_N`
directory containing paired results, decision traces, validation, and HTML.

Use a clean checkout of the pushed evaluation commit and the existing locked
Oscar runtime. Create the campaign's `slurm` directory before submission:

```bash
export EXPECTED_ACTION_MODES_SHA=$(git rev-parse HEAD)
export CHECKPOINT_MANIFEST=/absolute/scratch/campaign/checkpoint-manifest.json
export RESULT_ROOT=/absolute/scratch/campaign/smoke
export AMBIXQC_EVAL_MODE=smoke
sbatch --array=0,29%2 --output=/absolute/scratch/campaign/slurm/smoke-%A_%a.out \
  --error=/absolute/scratch/campaign/slurm/smoke-%A_%a.err \
  slurm/run_ambixqc_mppi_eval_oscar.sbatch

# After both GPU smoke tasks pass, evaluate five full episodes per controller.
export RESULT_ROOT=/absolute/scratch/campaign/production
export AMBIXQC_EVAL_MODE=production
sbatch --array=0-29%12 --output=/absolute/scratch/campaign/slurm/eval-%A_%a.out \
  --error=/absolute/scratch/campaign/slurm/eval-%A_%a.err \
  slurm/run_ambixqc_mppi_eval_oscar.sbatch

"$XQC_EVAL_PY" summarize_ambixqc_mppi_eval.py \
  --manifest "$CHECKPOINT_MANIFEST" --results-root /absolute/scratch/campaign \
  --expected-source-sha "$EXPECTED_ACTION_MODES_SHA" \
  --output /absolute/scratch/campaign/summary.json \
  --html /absolute/scratch/campaign/comparison.html --wandb
```

The smoke uses two seeds and three decisions with the full MPPI search budget,
runs CUDA regression checks, and disables W&B. Production publishes checkpoint
runs to `ambi-inner-bench`; the optional final summary publishes one campaign
run with native curves indexed by training checkpoint. Summary validation
requires all 30 checkpoints by default. `--allow-partial` can summarize missing
checkpoint directories but still rejects failed or incomplete results that are
present. Completed episode bundles are preserved if later evaluation fails.

## AMBI-XQC episode comparison

This workflow measures deployment-time improvement from fresh inner XQC on an
AMBI-XQC world model and persistent policy trained without inner improvement.
The checkpoint metadata determines the environment, wrappers, observation and
action contracts, architecture, and outer learning settings. Keep every model
file with its adjacent `.metadata.json`. Standalone XQC checkpoints do not
contain the required TOLD world model.

Run commands from the repository root using the existing locked DMControl
interpreter. Set `XQC_EVAL_PY` to its absolute path if reusing an environment in
another worktree. Production training and full evaluations belong on scheduler
compute nodes; the commands below do not submit jobs themselves.

```bash
XQC_EVAL_PY=environments/dmcontrol/.venv/bin/python
XQC_MATRIX=configs/research/ambixqc_humanoid_inner_benchmark.json
XQC_RESULTS=/absolute/path/to/durable/xqc-checkpoints

"$XQC_EVAL_PY" main.py \
  --run configs/dmcontrol/experiments/ambixqc_humanoid_walk_outer_prior_no_inner_checkpoint_bank_1p5m.json \
  --alg-dir configs/dmcontrol/algs --log-dir "$XQC_RESULTS"
```

The training profile uses Humanoid Walk state, seed 55, 1.5 million decisions,
and retains all 60 checkpoint/metadata pairs every 25,000 decisions. Online
evaluation is disabled. After the unchanged random warmup, real collection
samples the persistent actor. TOLD and outer XQC keep learning, including the
chronological real reward normalizer; no inner learner or imagination runs.
Choose durable output storage through the normal training harness. This is a
single-seed exploratory configuration, and checkpoints are not exact trainer
resumes.

Choose one checkpoint and fresh result directories. Evaluate the prior once:

```bash
XQC_CHECKPOINT=/absolute/path/to/checkpoint.pt

"$XQC_EVAL_PY" evaluate_ambi_checkpoint.py \
  --matrix "$XQC_MATRIX" --checkpoint "$XQC_CHECKPOINT" \
  --preset controller/prior --device cuda \
  --bundle-dir results/xqc-eval/prior
```

Then evaluate inner XQC against those saved prior episodes:

```bash
"$XQC_EVAL_PY" evaluate_ambi_checkpoint.py \
  --matrix "$XQC_MATRIX" --checkpoint "$XQC_CHECKPOINT" \
  --preset controller/xqc --device cuda \
  --reference-bundle results/xqc-eval/prior \
  --bundle-dir results/xqc-eval/inner

"$XQC_EVAL_PY" report_ambi_benchmark.py \
  --bundle results/xqc-eval/prior \
  --bundle results/xqc-eval/inner \
  --output results/xqc-eval/comparison.html
```

Repeat this sequence with separate output paths for each checkpoint. A reference
must match the checkpoint hash and episode protocol, including the requested
seeds. Select both presets in one invocation to evaluate them together. Omitting
`--preset` selects only the prior. The default evaluation is seeds 101–105, at
most 500 decisions per episode, and controller seed 55. No source W&B run is
invented by the matrix. Append `--wandb` for explicit publication to
`ambi-inner-bench`, or `--wandb --wandb-mode offline` for offline SDK output.

Both real controllers execute `tanh(mu)`. Inner imagined behavior, minibatches,
and learner updates remain stochastic. The evaluation solver is reseeded after
checkpoint loading and before each episode, independently of checkpoint RNG and
preset execution order. Every inner action starts from the persistent priors.
Outer weights, BatchNorm buffers, targets, temperature, optimizers, counters,
normalizer state, and outer learner RNG are checked for changes.

The inner preset inherits the saved inner settings. For the standard checkpoint
bank, these are J2/N32/H3/G4, batch 64, replay capacity 192, learning rates
`5e-5`, and `frozen_real_scale`. To compare another budget, add a variant to a
separate matrix using the existing `inner_*` override mechanism; keep replay
capacity large enough for `J*N*H`. Changing outer architecture or learning
settings is rejected. The checkpoint's real reward scale and return accumulator
stay frozen throughout evaluation. With `action_local_imagined`, each solve
seeds its disposable return context from that saved accumulator and adapts only
its inner normalization statistics.

Each bundle records real returns, seed-paired return gains, per-decision timing
and aggregate inner diagnostics, checkpoint identity, saved/evaluated settings,
and a source/runtime fingerprint. Full-episode rewards are raw environment
rewards; XQC critic values and targets use normalized reward units. Optimizer
loss metrics summarize update slots, including slots where delayed actor or
temperature optimizer steps were skipped. The default eight critic slots
produce three accepted actor and temperature steps, not eight. Reports use the
actual optimizer counters and offer decision-level plots, without implying
within-solve learning traces. Shared-observation banks and probes are rejected.

Completed episode shards survive a later failure; partial episodes and nonfinite
measurements remain explicit. Bundle directories must be new, and report files
are preserved unless `--overwrite` is requested. Report generation reads saved
data only and never invokes a model, simulator, or W&B session.

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
side. It also rejects `execution_noise`, because deterministic evaluation must
return the policy mean and would collapse those training-only variants.
