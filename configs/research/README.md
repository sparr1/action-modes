# Frozen-checkpoint AMBI research

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
