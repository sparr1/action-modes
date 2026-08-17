# Frozen-checkpoint AMBI research

The active end-to-end configs live directly under `configs/ambi/`.
The frozen-checkpoint matrix is
`configs/research/ambi_inner_decoupling.json`, outside the active AMBI training
tree so it cannot be confused with the independently trained branch and horizon
comparisons.

# Frozen-checkpoint preset workflow

`ambi_inner_decoupling.json` is a compact matrix of one-axis-at-a-time overrides
based on `configs/algs/AntAMBITDMPC2.json`. It covers:

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

# AMBI inner-loop latency benchmark

`ambi_latency_benchmark.json` defines a frozen-checkpoint action-selection
benchmark around the current Humanoid Walk G4 configuration. It directly uses
`configs/dmcontrol/algs/ambi_humanoid_walk_updates_g4.json`; it does not inherit
the older Ant defaults in `ambi_inner_decoupling.json`. Each cell changes only
the canonical round, rollout, and update controls J/N/G, plus the required
action-local replay capacity `J*N*H`. The horizon remains H=3 and the inner
batch size remains B=64.

The 15 unique cells are the union of four sweeps. The shared center
`J=2,N=32,G=4` is materialized only once:

| Family | Cells | Work held fixed |
| --- | --- | --- |
| G | `J=2,N=32,G={0,2,4,8,16}` | 192 imagined transitions |
| N | `J=2,N={8,16,32,64,128},G=4` | 8 update slots |
| Natural J | `J={1,2,4,8},N=32,G=4` | N and G |
| Matched-work J | `(J,N,G)={(1,64,8),(2,32,4),(4,16,2),(8,8,1)}` | 192 imagined transitions and 8 update slots |

For every action, the expected counters are:

- rollout paths: `J*N`;
- imagined transitions and replay capacity: `J*N*H`;
- critic/actor update slots: `J*G` each;
- replay draws: `J*G*B`;
- temperature steps: zero, because the base inherits the outer temperature;
- policy evaluations: `J*N*H + 2*J*G*B + 1`;
- Q evaluations: `3*J*G*B`.

The matrix records these expectations per cell. The runner rejects counter
mismatches and compiled-region fallbacks instead of silently publishing
incomparable timings.

## Measurement contract

`benchmark_ambi_latency.py` loads the same frozen outer checkpoint for every
cell and calls only the public AMBI prediction path. It builds a fixed bank of
64 observations with seeded environment resets, but never calls `env.step`,
`learn`, or an outer update. W&B and diagnostics are disabled. Each process
records one cold call, discards 49 warmup calls, then retains 200 synchronized
CUDA wall-time and AMBI phase measurements. The observation bank cycles across
the 250 total calls.

The synchronized wall timer surrounds the public `predict` call, so it also
includes observation conversion and action unscaling. Production training's
logged inner-action timer excludes those small wrapper operations. Use the
benchmark for controlled cross-cell scaling and report its absolute wall
latency as a separate metric rather than equating it with the W&B series.

Run one process-isolated cell locally with the locked DMControl runtime:

```bash
MUJOCO_GL=egl environments/dmcontrol/.venv/bin/python \
  benchmark_ambi_latency.py \
  --config configs/research/ambi_latency_benchmark.json \
  --checkpoint /absolute/path/to/checkpoint-compatible-humanoid-walk.pt \
  --cell 2,32,4 \
  --output /absolute/path/to/center_j2_n32_g4.json
```

Omitting `--cell` runs all configured cells with fresh model instances in one
process. That is useful for development, but it is not the maintained
process-isolated cluster protocol.

## Oscar array

Do not change or reuse the production checkout that owns active or requeueable
training jobs. Submit the latency array from a separate clean scratch clone or
worktree at the intended commit. The launcher defaults to that checkout's
locked interpreter, while `AMBI_BENCHMARK_PYTHON` can point at an already-built
locked environment elsewhere:

```bash
export AMBI_LATENCY_CHECKPOINT=/absolute/path/to/checkpoint-compatible-humanoid-walk.pt
export AMBI_LATENCY_OUTPUT_ROOT=/oscar/scratch/rgao48/ambi-latency/2026-08-17
export AMBI_BENCHMARK_PYTHON=/oscar/home/rgao48/action-modes/environments/dmcontrol/.venv/bin/python
export AMBI_LATENCY_EXPECTED_COMMIT=$(git rev-parse HEAD)
sbatch slurm/run_ambi_latency_oscar.sbatch
```

Use an operator-approved, new directory under Oscar scratch for
`AMBI_LATENCY_OUTPUT_ROOT`; do not route these outputs to the grace-expired data
allocation. Both the checkpoint and output root must be absolute paths. The
launcher refuses to overwrite an existing block directory.

Oscar scratch is purgeable, so copy the completed JSON artifacts off the
cluster promptly, including each block's `COMPLETE` marker.

The array has three independent blocks (`0-2%2`). Every block benchmarks all 15
cells in a deterministic block-specific random order, yielding three
process-level replicates per cell. Each cell gets a fresh runner process and
private TorchInductor, Triton, and CUDA caches. Cell JSON plus stdout/stderr are
written beneath
`$AMBI_LATENCY_OUTPUT_ROOT/job_<array-job>/block_<array-index>/`; the repository
only receives the small Slurm launcher logs. The job requests one L40S, six
CPUs, 32 GiB RAM, and three hours on Oscar's `gpu` partition. It does not train,
checkpoint, resume, or initialize W&B.

After all three blocks finish, copy the job directory to a durable local
artifact location. In a local plotting environment that provides Matplotlib,
validate, aggregate, and plot it outside the source checkout:

```bash
python3 plot_ambi_latency.py \
  /absolute/local/path/job_<array-job> \
  --output-dir /absolute/local/artifacts/latency_plots
```

The plotter treats each process-isolated cell JSON as one observation within
its `block_N` replicate. The 200 action timings within a file estimate that
process's latency distribution; they are not 200 independent experimental
replicates. Directory inputs must contain all three `COMPLETE` markers and the
exact 45-file matrix before synthesis is allowed.
