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
the canonical one-million-decision screen. CUDA runs may enable five
fixed-shape regions with `compile=true`: the persistent and action-local XQC
actor and critic losses plus the dense, fixed-horizon non-episodic inner
rollout. `compile_strict=true` makes any graph or runtime fallback fatal. The
outer recurrent TOLD computation, optimizer mutations, action-local lifecycle
orchestration, and episodic rollout path remain eager. A compile request is
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

### AMBI-XQC heavy inner-loop v1 suite

The `ambixqc_humanoid_walk_heavy_inner_v1` suite is an XQC-native heavy-compute
screen derived from the strict compiled production configuration above. Every
cell preserves the XQC/TOLD model, optimizer,
target, policy-delay, and recurrent-training settings; uses seed 55, strict
compilation, horizon `H=3`, and a 14-million decision budget; and explicitly
sets `inner_reward_normalization="action_local_imagined"`. Relative to the
strict compiled production configuration, the base changes only the decision
budget, evaluation cadence, declared inner schedule, inner reward-normalization
mode, W&B identity, and manifest logging/checkpoint policy; the other cells
change only the inner schedule and cell identity relative to that base. The
source matrix is native AMBI commit
`698bd074551bc48128cb34f78adaf8caaab1f188`. Matching algorithm and experiment
JSON files live under `configs/dmcontrol/algs` and
`configs/dmcontrol/experiments`. The table abbreviates the common
`ambixqc_humanoid_walk_heavy_inner_v1` stem.

| Cell suffix | J | N | H | G | Batch | Replay capacity | Critic slots | Accepted actor / temperature optimizer steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base (no suffix) | 8 | 32 | 3 | 1 | 64 | 768 | 8 | 3 |
| `d256_g1_j6` | 6 | 256 | 3 | 1 | 256 | 4,608 | 6 | 2 |
| `d256_g1` | 8 | 256 | 3 | 1 | 256 | 6,144 | 8 | 3 |
| `d256_g3` | 8 | 256 | 3 | 3 | 256 | 6,144 | 24 | 8 |
| `d512_g1_j6` | 6 | 512 | 3 | 1 | 512 | 9,216 | 6 | 2 |
| `d512_g1` | 8 | 512 | 3 | 1 | 512 | 12,288 | 8 | 3 |
| `d512_g3_j6` | 6 | 512 | 3 | 3 | 512 | 9,216 | 18 | 6 |
| `d512_g3` | 8 | 512 | 3 | 3 | 512 | 12,288 | 24 | 8 |
| `d512_b256_g6` | 8 | 512 | 3 | 6 | 256 | 12,288 | 48 | 16 |

The source-to-XQC projection is explicit:

| Native AMBI cell | AMBI-XQC cell suffix |
| --- | --- |
| base-v2 | base (no suffix) |
| D256-1-J6 | `d256_g1_j6` |
| D256-1 | `d256_g1` |
| D256-2 | `d256_g3` |
| D512-1-J6 | `d512_g1_j6` |
| D512-1 | `d512_g1` |
| D512-4-J6 | `d512_g3_j6` |
| D512-2, D512-3, or D512-4 | `d512_g3` |
| D512-5 | `d512_b256_g6` |

Here `G` is the existing shared XQC optimizer-slot count per round; this suite
does not add a component-specific scheduler or alter XQC update-slot ordering.
Each slot updates the critic, while policy delay 3 accepts actor and automatic
temperature optimizer steps at zero-based slots `0, 3, 6, ...`, so the accepted
count is `((J*G - 1) // 3) + 1`. The actor objective and gradient are still
evaluated on every slot, and actor BatchNorm buffers still update on every
slot. Online critic BatchNorm buffers likewise update on critic training
batches. Target-critic BatchNorm retains XQC's `batch_no_update` behavior, and
rollout/execution forwards consume rather than modify the learned running
statistics. Replay capacity is exactly `J*N*H`.

The heavy suite's reward normalization is action-local. At every planned
action, the imagined-return moments start empty, while every independent branch
seeds its discounted-return accumulator from the current outer real-return
accumulator. Branch accumulators never flow into neighboring branches and reset
to that same real-root seed at the start of each round. All realized imagined
transitions update the local moments once during collection; the resulting
scale is then used by that round's XQC optimizer slots. Replay rewards remain
raw. The local statistics are discarded after action selection and never write
back to the chronological outer real-reward normalizer. With no imagined sample
yet, the outer real scale is the action's fallback initial scale. Other
AMBI-XQC configurations retain the backward-compatible
`inner_reward_normalization="frozen_real_scale"` default unless they opt in
explicitly. W&B records the initial, final, and delta reward scales, the local
moment counts, and the number of imagined transitions incorporated.

The projection preserves `J`, `N`, `H`, batch size, replay capacity, and critic
slot dose. It does not preserve native SAC actor/alpha scheduling, phased
ordering, minibatch-draw counts, or optimizer semantics. In particular, when
the native AMBI `D512-2`, `D512-3`, and `D512-4` cells are projected by treating
their three critic updates per round as shared XQC `G=3`, all three resolve to
`d512_g3`; this is not a claim of component-schedule equivalence.

These are single-seed, non-confirmatory exploratory configurations. Every
manifest keeps the source suite's timestamped summary logging while disabling
evaluation and model checkpoints, and AMBI-XQC still has no exact
trainer-resume path.

The dedicated Hydra launcher intentionally exposes only the two requested
source-projection cells: `d512_g3_j6` (native D512-4-J6) and `d512_g3` (native
D512-2, also the collapsed D512-3/D512-4 projection). It submits one atomic
two-task array, with the array index serving as the fixed allowlist; there is no
caller-selectable manifest. Each task requests one GPU, eight CPUs, 64 GiB of
host memory, and at most 30 days. The submitter pins both tasks to `gpu2301`,
requires two immediately available matching GPU/CPU/memory slots, and has no
fallback node. It requires eight GiB of durable free space for the pair; each
array task rechecks that at least four GiB remains and requires eight GiB in
its node-local scratch filesystem. Automatic requeue is disabled because the
runs have no exact-resume state.

First commit and push the tested changes, then update a clean Hydra checkout
through Git to that exact commit. Create a new durable result root outside the
checkout and point `--python` at an existing locked DMControl environment. The
environment may belong to a different clean checkout only when its `uv.lock`
is byte-identical:

```bash
ACTION_SHA="$(git rev-parse HEAD)"
RUN_ID="ambixqc-heavy-inner-${ACTION_SHA:0:7}-$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_ROOT="/cs/home/rgao48/ambixqc-heavy-inner-v1"
DMCONTROL_PYTHON="/absolute/path/to/action-modes/environments/dmcontrol/.venv/bin/python"
mkdir -p "$RESULTS_ROOT"
slurm/submit_ambixqc_humanoid_walk_heavy_inner_pair_hydra.sh \
  --expected-action-modes-sha "$ACTION_SHA" \
  --results-root "$RESULTS_ROOT" \
  --run-id "$RUN_ID" \
  --python "$DMCONTROL_PYTHON" \
  --dry-run
```

Remove `--dry-run` only after reviewing the single `sbatch` command and the
reported `gpu2301` capacity. The availability check is a scheduler snapshot;
the array can still queue if capacity changes after the check. Each task gets a
unique cell/job artifact directory and W&B run ID. Torch, TorchInductor,
Triton, CUDA, NumPy/Numba, plotting, temporary, and W&B state are isolated on
node-local storage. Durable storage receives the timestamped summary output,
provenance/runtime probes, job log, and a final `PASS` marker. There is no
evaluation or model-checkpoint acceptance check because both manifests
deliberately disable those outputs, and a failed job cannot resume exactly.
Before Python starts, inherited compiler-debug and CUDA synchronization/cache
overrides are cleared. The runtime probe requires exactly one visible NVIDIA
L40 with at least 44 GiB of memory. Both cells require strict compilation, so
any compile fallback is fatal. Their optimizer backend remains `auto`: fused
Adam is intended on CUDA, but the runtime selection is authoritative and the
launcher does not claim that fused execution occurred merely from the config.
No job is submitted by adding these launch files.

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
CUDA allocation and reservation, all five compile statuses and their aggregate
fallback, inner-rollout p50 and p95, the exact 64 full `H=3` rollouts and
192-transition replay fill, exact update/replay counters, projection residual,
outer-prior and workspace lifecycle checks, global RNG isolation, versions,
source SHA, and configuration hash.
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
RESULTS_ROOT="/oscar/scratch/rgao48/ambi/benchmarks/ambixqc-dense-rollout/$ACTION_SHA"
mkdir -p "$RESULTS_ROOT"
slurm/submit_ambixqc_compile_pair_oscar.sh \
  --expected-action-modes-sha "$ACTION_SHA" \
  --results-root "$RESULTS_ROOT" \
  --dry-run
```

Remove `--dry-run` only after reviewing the command. The results root is fixed
to `/oscar/scratch/rgao48/ambi/benchmarks/ambixqc-dense-rollout/<full-SHA>`; the job
creates exactly one `<Slurm-job-id>` child, so every artifact lands under
`/oscar/scratch/rgao48/ambi/benchmarks/ambixqc-dense-rollout/<full-SHA>/<job-id>`.
The submitter launches one two-hour L40S job. Before timing, that job runs the
mandatory AMBI-XQC and shared compile-region CUDA correctness suites with a
fresh cache, including the fifth dense-rollout region. It creates a clean
node-local Git checkout of the fixed four-region baseline
`b0c39193e8b9c922d091063b30d92b00dfd9f28f`; no network update or source overlay
is used. The job then runs five independent compute processes per mode for
candidate eager, strict-compiled baseline, and strict-compiled candidate code,
alternating the outer order across repetitions. Every process receives a new
node-local XDG, TorchInductor, Triton, and CUDA cache, so compiled timing never
benefits from a prior process. Each process retains 10 cold/warmup cycles and
50 measured cycles. The job finally runs one 1,502-step canary for each of the
three modes, also with separate fresh caches. Online W&B stays disabled. The
job validates all three final checkpoints, checks that both checkouts remained
clean, and writes a terminal marker. `PASS` unambiguously means every mandatory
correctness and performance gate passed and the candidate is retention
eligible. A nonfatal performance miss writes `CORRECTNESS_PASS` together with
`PERFORMANCE_MISS`, but never `PASS`. Neither script updates a persistent
checkout or resolves dependencies on the cluster.

The candidate SHA scopes the result root and the fixed baseline SHA is recorded
in the artifacts. The job artifact directory contains five
`compute-<mode>-rep<N>.json` files for each of `eager`, `baseline`, and
`candidate`, plus `compute-aggregate.json`, per-arm cold-process timing,
warmed-compute and checkpoint-validation JSON, logs, GPU/runtime metadata, and
`comparison.json`. The mandatory gate requires all learned state and metrics
to be finite, the exact production rollout/replay/counter contract—including
replay cursor/fullness/sample ID and policy/Q evaluation counts—and
lifecycle/RNG checks, no compile fallback across the five candidate regions,
all five regions compiled in every candidate repetition, and projection
residual at most `1e-6`. It fails the job on a correctness violation.

Performance is classified separately. The compiled candidate median p50 must
be at most `0.90` of candidate eager, compiled candidate median p95 at most
`0.95`, the maximum p50 coefficient of variation across candidate eager and
compiled arms at most `0.05`, no paired candidate repetition above `1.10` of
its eager repetition, and median peak allocation at most `1.10` of eager. The
warmed Humanoid candidate compute canary must be at least five percent faster
than eager. The incremental gates
also require candidate p50 and warmed-canary compute to each be at most `0.97`
of the compiled four-region baseline, while candidate p95 may not exceed the
baseline and candidate peak allocation may not exceed `1.10` of baseline. A
performance miss is recorded in `comparison.json` and `PERFORMANCE_MISS` but
does not fail an otherwise-correct job; that job is explicitly not eligible for
retention. The aggregate also estimates
compilation break-even as the median cold-cycle cost delta divided by steady
p50 savings per action. The synchronized warmed canary timers bracket complete
planned-action and outer-update calls, including eager TOLD recurrence,
backward passes, optimizer work, and XQC orchestration, while excluding
environment stepping, logging, and checkpoint I/O. The separately reported
cold process time includes compilation, environment setup and stepping, replay,
logging setup, and the final checkpoint, but it is not used for either warmed
performance classification. Reported speedup ratios
greater than one favor compilation.

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
