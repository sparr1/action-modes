We would like to develop a relatively complete and mature system of transferrable action-abstractions from the ground up where as much is learned as possible. 

To do so, we introduce the action mode framework.

An action mode is a triple of 
1. mode support: a support of states on which the mode is defined
2. mode actions: a bounded, n-dimensional action space
3. projection: a function taking the mode actions to a base action space

To solve a problem with a collection of M completely defined action modes, you follow a four step process: 

1. take the state s you are in, and restrict your attention to the k supported modes for which s is in their mode support.
2. select a mode (let's call it mode i) from the k supported options.
3. select a latent action-within-mode z_i from the ith mode actions.
4. project z_i to a base action a, using the ith projection function. 
5. take action a in the base space, and repeat! 

Given that the mode supports and projection functions are completely packaged together with the modes, the only part left is to solve 2-3, which can essentially be solved via masking (using the supports) your favorite PAMDP solver. 

So now that we know how to use modes, that's great! But how did we get these projection functions and support sets to begin with? I'm glad you asked! 

There may be many ways to get all kinds of different modes, but not all of the modes you end up with are going to be particularly useful. Right now, in order to learn modes, our current thinking is that it suffices to learn parameterized skills which solve average reward tasks on a restriction of the state space and a low-dimensional compression of the action space. These skills will become our projection functions. There will be some special auxillary losses which help these skills become useful.

The mode actions then become interpretable as saying "if I execute this mode, and this particular mode action, I will eventually settle in the high-dimensional base space, into a pattern of behavior which is high reward under this task. For instance, walking forwards, or rotating, etc. 

## What AMBI is

Anytime Model-Based Policy Improvement (AMBI) learns a task-oriented latent
dynamics (TOLD) world model together with persistent actor and critic control
priors. At every real environment decision, AMBI creates a fresh local actor,
critic, target critic, optimizer state, and imagined replay from those priors;
generates root-local experience with the frozen TOLD model; performs local
off-policy actor-critic updates; and executes an action from the adapted actor.
The local learner is then discarded, while the world model and control priors
continue learning from real replay.

The reference implementation uses full cloned actor and critic priors with
inner SAC. LoRA adaptation, TD3, no inner improvement, persistent inner state,
and MPPI are auxiliary ablations or comparison operators. In particular, MPPI
is the TD-MPC-style planning comparator; it is not AMBI's core action-selection
algorithm.

`AMBIXQC/AMBIXQC` is the XQC-backed form of the same AMBI pattern. It keeps
TOLD and its recurrent multi-step BPTT training, replaces the persistent SAC
control priors with the released XQC actor and twin categorical critics, and
runs a fresh full-copy inner XQC learner at every non-warmup decision. Its
first implementation is state-observation only and deliberately has no MPPI,
TD3, LoRA, or persistent-inner switches. The reduced Humanoid Walk integration
check is
`configs/dmcontrol/experiments/ambixqc_humanoid_walk_state_smoke.json` and must
be run with `--alg-dir configs/dmcontrol/algs`; it is not a benchmark.

The resolved AMBI-XQC defaults use the released four-layer XQC actor
(`256` units per layer), twin four-layer critics (`512` units per layer), 101
atoms on `[-5, 5]`, `alpha=0.01`, policy delay 3, and `tau=0.005`. The fresh
inner solve defaults to `J=2`, `N=32`, `H=3`, and `G=4`, with a 64-row batch
and fixed `5e-5` actor/critic learning rates. Temperature always uses the actor
learning rate. These are initial method defaults, not a frozen benchmark
protocol; every experiment must record its resolved settings.

AMBI-XQC uses the same sparse W&B key as inner SAC for final-policy drift:
`train/inner_final_outer_policy_kl` reports the closed-form root-state
`KL(final adapted actor || outer prior)` and is observational rather than part
of the XQC actor loss.

AMBI-XQC v1 supports portable model checkpoints for evaluation and weight
transfer. It is not allowlisted for exact trainer resume because that stronger
contract also requires replay and environment state. Eager execution remains
the canonical one-million-decision screen. CUDA runs may enable four
fixed-shape compiled XQC loss regions—the persistent and action-local actor and
critic regions—with `compile=true`; `compile_strict=true` makes any graph or
runtime fallback fatal. The recurrent TOLD computation, optimizer mutations,
and action-local lifecycle orchestration remain eager. A compile request is
inactive on CPU rather than silently changing the device contract.

### AMBI-XQC compiled execution and paired timing

The canonical eager Humanoid Walk configuration remains
`configs/dmcontrol/algs/ambixqc_humanoid_walk_state.json`. Its strictly compiled
single-seed sibling is
`configs/dmcontrol/experiments/ambixqc_humanoid_walk_state_compiled.json`, run
with the same algorithm directory:

```bash
environments/dmcontrol/.venv/bin/python main.py \
  --run configs/dmcontrol/experiments/ambixqc_humanoid_walk_state_compiled.json \
  --alg-dir configs/dmcontrol/algs
```

The sibling preserves the eager run's model, XQC heads, inner schedule, seed,
one-million-decision budget, evaluation cadence, and checkpoint policy. Only
strict compilation and the W&B identity differ. It is still a single-seed
exploratory run, not a learning-quality comparison.

For compute-only profiling, use the environment-free exact-shape benchmark:

```bash
environments/dmcontrol/.venv/bin/python \
  tests/benchmarks/ambixqc_compute_throughput.py \
  --device cuda --warmup 10 --measured 50 \
  --compile --compile-strict --require-compiled
```

Every iteration uses the production Humanoid observation/action dimensions,
model-size-5 latent width, full released XQC networks, canonical
`J=2/N=32/H=3/G=4` inner solve, and one synthetic recurrent outer update with
`H=3` and `B=256`. It excludes DMControl, replay sampling, evaluation, logging,
and checkpoint I/O. The JSON result records the first cycle with synchronized
host wall-clock so compiler startup is fully included, then uses CUDA events
for warmed and measured per-cycle p50 and p95. It also records throughput, peak
CUDA allocation and reservation, all four compile statuses and fallbacks,
exact counters, projection residual, outer-prior and workspace lifecycle
checks, global RNG isolation, versions, source SHA, and configuration hash.
`--output` uses exclusive creation and refuses to replace an existing file.

The end-to-end timing manifest is
`configs/dmcontrol/experiments/ambixqc_humanoid_walk_state_timing_pair.json`.
Its eager and strict-compiled arms differ only in their compile flags. Each arm
runs 1,502 decisions with the inclusive 500-step seed boundary, producing 501
random actions, 1,001 planned inner actions, and 1,002 outer updates. Both keep
the production networks, inner schedule, and outer batch. Evaluation, W&B, and
trajectory logging are disabled; one final `latest` checkpoint at step 1,502 is
retained solely to validate counters, finiteness, projection, and loading.
Each canary records both its cold whole-process wall time and synchronized
training-compute timings. The warmed compute total excludes the first planned
inner action and the first outer update independently, because those two calls
exercise the inner and outer compilation regions. The remaining 1,000 planned
actions and 1,001 updates form the end-to-end performance canary.

Oscar is the default venue for the paired GPU workflow. First commit and push
the locally tested changes, then update a clean Oscar checkout through Git to
that exact commit. From the Oscar checkout root, submit into the designated
SHA-scoped scratch directory outside the checkout:

```bash
ACTION_SHA="$(git rev-parse HEAD)"
RESULTS_ROOT="/oscar/scratch/rgao48/ambi/benchmarks/ambixqc-compile/$ACTION_SHA"
mkdir -p "$RESULTS_ROOT"
slurm/submit_ambixqc_compile_pair_oscar.sh \
  --expected-action-modes-sha "$ACTION_SHA" \
  --results-root "$RESULTS_ROOT" \
  --dry-run
```

Remove `--dry-run` only after reviewing the command. The results root is fixed
to `/oscar/scratch/rgao48/ambi/benchmarks/ambixqc-compile/<full-SHA>`; the job
creates exactly one `<Slurm-job-id>` child, so every artifact lands under
`/oscar/scratch/rgao48/ambi/benchmarks/ambixqc-compile/<full-SHA>/<job-id>`.
The submitter launches one two-hour L40S job. Before timing, that job runs the
mandatory AMBI-XQC and shared compile-region CUDA correctness suites with a
fresh cache. It then runs five independent compute processes per mode and
alternates eager/compiled order across repetitions inside the job. Every
process receives a new node-local XDG, TorchInductor, Triton, and CUDA cache, so
compiled timing never benefits from a prior process. Each process retains 10
cold/warmup cycles and 50 measured cycles. The job finally runs one eager and
one compiled 1,502-step canary, also with separate fresh caches. Online W&B
stays disabled. The job validates both final checkpoints, checks that the
checkout remained clean, and writes a terminal `PASS` file. Neither script
updates the checkout or resolves dependencies on the cluster.

The job artifact directory contains five `compute-<mode>-rep<N>.json` files per
mode, `compute-aggregate.json`, per-arm cold-process timing, warmed-compute and
checkpoint-validation JSON, logs, GPU/runtime metadata, and `comparison.json`.
The mandatory gate requires all learned state and metrics to be finite, exact
counters and lifecycle/RNG checks, no compile fallback, all four compiled
regions in every compiled repetition, and projection residual at most `1e-6`.
It fails the job on a correctness violation.

Performance is classified separately. The compiled median p50 must be at most
`0.90` of eager, compiled median p95 at most `0.95`, the maximum p50 coefficient
of variation at most `0.05`, no paired compiled repetition above `1.10` of its
eager repetition, and median peak allocation at most `1.10` of eager. The
warmed Humanoid compute canary must be at least five percent faster. A
performance miss is recorded in `comparison.json` and `PERFORMANCE_MISS` but
does not fail an otherwise-correct job. The aggregate also estimates
compilation break-even as the median cold-cycle cost delta divided by steady
p50 savings per action. The synchronized warmed canary timers bracket complete
planned-action and outer-update calls, including eager TOLD recurrence,
backward passes, optimizer work, and XQC orchestration, while excluding
environment stepping, logging, and checkpoint I/O. The separately reported
cold process time includes compilation, environment setup and stepping, replay,
logging setup, and the final checkpoint, but it is not used for the five-percent
classification. Reported speedup ratios greater than one favor compilation.

## Training checkpoints

Within-run checkpoint retention is configured independently from end-of-trial
model saving:

```json
{
  "checkpoint_every": 100000,
  "save_strat": ["best", "latest"],
  "checkpoint_best_window": 100,
  "save_trials": "none"
}
```

`save_strat` accepts `all`, `best`, and `latest`, either as one string or as a
list. `all` keeps the existing step-numbered checkpoints. `latest` continually
replaces one alias and is also updated after a clean training completion.
`best` replaces one alias when the rolling mean completed-episode return
improves. A partial initial window is allowed. `last` is accepted as an alias
for `latest`; `none` must be used alone. If `save_strat` is omitted, a positive
`checkpoint_every` retains the backwards-compatible `all` behavior. Set
`checkpoint_every` to `null` to disable within-run checkpointing.

`save_trials` remains a separate policy for final models across trials.
Checkpoint files are model snapshots and do not universally contain replay or
environment state for full training resume.

The compact AMBI branch-count and imagination-horizon suite lives under
`configs/ambi/`. Each of its five algorithm files has one matching runnable
manifest under `configs/ambi/experiments/`; the anchor entry point is
`configs/ambi/experiments/ambi_anchor.json`, used with
`--alg-dir configs/ambi/algs`.

## Single-task DMControl

DMControl uses an isolated, locked `uv` environment because its MuJoCo and
Gymnasium versions intentionally differ from the repository's legacy robotics
stack. The root `requirements.txt` is unchanged. From this repository root:

```bash
uv sync --project environments/dmcontrol --locked
environments/dmcontrol/.venv/bin/python main.py \
  --run configs/dmcontrol/experiments/walker_walk_state.json \
  --alg-dir configs/dmcontrol/algs
```

The default observation is the same state representation used by TD-MPC2's
single-task benchmarks: DMControl observation components are flattened in their
native insertion order to one `float32` vector. The adapter also supports
TD-MPC2-compatible pixels as an explicit alternative, not as an addition to
state. Set `env_params.obs` to `"rgb"` to select a three-frame `(9, 64, 64)`
`uint8` stack. If `alg_params.obs` is also supplied, it must match the
environment setting. The provided RGB manifest is a functional smoke test;
its small replay and training budgets are not benchmark settings.

Both modes repeat each action for two raw control steps, sum the two rewards,
and expose a 500-decision time limit as `truncated=True` and
`terminated=False`. Pixel observations use camera 2 for quadruped and camera 0
for other tasks. A one-million-row pixel replay needs about 36.9 GB just for
raw observations, so production RGB runs should choose `buffer_size`
deliberately and use summary logging.

See [the DMControl runtime guide](environments/dmcontrol/README.md) for exact
versions, rendering backends, example manifests, and lock maintenance. Set
`MUJOCO_GL=egl` before Python for headless NVIDIA rendering or
`MUJOCO_GL=osmesa` for CPU software rendering on a suitably provisioned Linux
host. Video rendering is supported through
`render_checkpoint.py --video-dir`; the adapter does not implement
Gymnasium's interactive `human` render mode.

For RGB checkpoints, `deterministic=True` selects the deterministic policy
action but intentionally does not disable TD-MPC2's random-shift image
augmentation; that augmentation is active during upstream acting and
evaluation as well.

### TD-MPC2 state benchmark

The first official-style comparator suite covers the six standard DMC tasks
`cartpole-swingup`, `cheetah-run`, `cup-catch`, `finger-spin`,
`reacher-easy`, and `walker-walk`. Each task uses seeds 1--3, four million
agent decisions, action repeat 2, the 5M TD-MPC2 model, and the upstream
single-task hyperparameters, including `rho=0.5`. Compilation remains disabled
because this port does not support the upstream compiled path exactly.

On Hydra, create the locked environment once and submit the 18 task/seed cells
from this repository root:

```bash
uv sync --project environments/dmcontrol --locked
sbatch slurm/run_tdmpc2_dmcontrol_state.sbatch
```

The Slurm array runs at most four jobs concurrently and uses one
scheduler-selected GPU per job. Production benchmark manifests disable model
checkpoints and trajectories. W&B syncs online from node-local temporary
storage, so its cache does not consume Hydra home-directory capacity. Runs use
the `rwgao_b-brown-university/ambi` project and task/seed-specific names. The
benchmark also runs the upstream-style ten online evaluation episodes at step 0
and every 100,000 agent decisions, writing only `step,reward,seed` beneath
`results/dmcontrol/tdmpc2_state/<task>/seed_<seed>/`. Each per-seed CSV is only a
few kilobytes. Slurm stdout and stderr remain under `slurm/`.

Humanoid Walk uses the shipped TD-MPC2 protocol of 14 million agent decisions
for each of seeds 1--3. Submit its three-cell A6000 array separately:

```bash
sbatch slurm/run_tdmpc2_humanoid_walk_state.sbatch
```

The launcher is for TD-MPC2 comparator runs only. Add AMBI benchmark cells only
after explicitly freezing and profiling an inner-loop compute schedule; the
sparse historical Walker AMBI template inherits a substantially larger default
inner workload and is not a confirmatory benchmark configuration.

### XQC Humanoid Walk comparison on Hydra

The XQC comparison runs four independent jobs concurrently: official JAX seed
0, official JAX seed 1, Action Modes PyTorch seed 0, and Action Modes PyTorch
seed 1. Each job owns one GPU and runs one actual seed; neither launcher
vectorizes nor serializes seeds. From a clean, pushed Hydra checkout at the
intended commit, submit the matrix with:

```bash
ACTION_SHA="$(git rev-parse HEAD)"
COMPARISON_ID="xqc-hwalk-${ACTION_SHA:0:7}-$(date -u +%Y%m%dT%H%M%SZ)"
slurm/submit_xqc_humanoid_walk_hydra.sh \
  --expected-action-modes-sha "$ACTION_SHA" \
  --official-dir /cs/home/rgao48/projects/xqc \
  --results-root /cs/home/rgao48/xqc-humanoid-walk-parity \
  --comparison-id "$COMPARISON_ID"
```

The helper refuses dirty or mismatched checkouts, verifies both dependency-lock
hashes, and refuses to reuse an artifact directory. At submission time it uses
`gpu2301` when four one-GPU slots are available, otherwise `gpu2201`; if neither
node can hold all four but their combined free capacity can, it splits the four
jobs across them. The availability check is a scheduler snapshot, so the jobs
can still queue if cluster state changes immediately afterward. Add `--dry-run`
to print the four `sbatch` commands without creating artifacts or submitting.

All runs share the comparison ID, while W&B groups are method-specific:
`<comparison-id>-official-jax` and `<comparison-id>-action-pytorch`. This makes
each curve aggregate exactly seeds 0 and 1. The common comparison metrics are
`comparison/raw_frame`, `comparison/train_return`, and
`comparison/eval_return`; canonical charts use raw frame as their step axis.
The PyTorch jobs require the compiled learner regions, automatically select
fused Adam on CUDA, and keep hot-path debug checks disabled. Each run also
records its implementation, seed, task, source SHA, and comparison ID in W&B
configuration. Durable logs, evaluation CSVs, and W&B state live under
`<results-root>/<comparison-id>/`, with implementation, seed, and Slurm job ID
in every job directory name.

## Rendering a checkpoint

Use the dedicated renderer instead of `init.py`:

```bash
python render_checkpoint.py /path/to/checkpoint --display
python render_checkpoint.py /path/to/checkpoint --video-dir videos
python render_checkpoint.py /path/to/checkpoint --results-json evaluation.json
```

The renderer runs one complete deterministic episode by default. Use
`--episodes`, `--seed`, `--device`, `--max-steps`, or `--stochastic` to change
the rollout. Video mode writes one MP4 per episode and refuses to replace an
existing output unless `--overwrite` is supplied.

`--results-json` runs without a rendering backend and atomically writes strict
JSON. For the AMBI comparison protocol, evaluate the two aliases separately:

```bash
python render_checkpoint.py /path/to/model_best \
  --results-json evaluation/best.json --episodes 5 --seed 101
python render_checkpoint.py /path/to/model_latest \
  --results-json evaluation/latest.json --episodes 5 --seed 101
```

These commands use deterministic policy means and environment seeds 101–105.
The JSON includes the checkpoint identity, per-episode returns and lengths,
summary statistics, and resolved runtime metadata.

New checkpoints include an adjacent `.metadata.json` file containing the exact
environment, wrapper, and algorithm settings. Existing checkpoints in their
original `logs/<run>/models` directory are supported through the run's
`settings.json` and per-trial `alg_settings.json`. For copied legacy files, pass
`--trial-settings` and `--experiment-settings` explicitly.
