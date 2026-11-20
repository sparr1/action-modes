# Four-task Meta-World setup

These manifests train independent state-based policies for **Assembly,
Hand-Insert, Push, and Stick-Pull**. Coffee-Push is omitted. This is a selected
single-task subset, not MT10, MT50, or a meta-learning protocol.

The prepared campaign uses **AMBI 5M, post-tanh entropy with sampled training
actions**. The selected paper
budget is **1,000,000 agent decisions = 2,000,000 repeated raw training steps**
per task, with **one seed (seed 55)**: four tasks × one seed = **4 prepared runs**.
Evaluation and the reset no-op are additional simulator steps outside that
training budget.

The adapter uses goal-observable, 39-dimensional v2 state observations and four
continuous actions. Episodes last 100 decisions, each action repeats twice,
and rewards are summed. At decision 100 the environment returns
`truncated=true, terminated=false`, and the training harness resets it.
Success does not end an episode early. The learner's `episode_length=100`
explicitly matches the adapter's fixed horizon.

The discount is 0.95. `seed_steps=1000` collects the initial 1,000 random-action
decisions, and `pretrain_steps=1000` sets the initial update count. This warmup
is included in the **1,000,000 total training decisions**. Horizon 3, batch
size 256, and UTD 1 retain the shared training settings.

The optional [TD-MPC2 configuration](algs/tdmpc2_state.json) is retained as a
hyperparameter reference only. None of the prepared manifests selects it;
TD-MPC2 comparison results will come from the upstream repository.

AMBI uses post-tanh entropy in both actor objectives, reward-only
critics, and fixed entropy coefficient **0.0004** for the four action dimensions.
Its inner SAC recipe uses **H3 / J8 / N128**, 16 critic and four actor updates
per round, fresh cloned actor/critic priors at every decision, sampled training
actions, mean actions for online evaluation, and per-update percentile Q scaling.
`inner_rounds=8` counts AMBI's inner SAC rounds at each decision; TD-MPC2's
MPPI `iterations` is a separate setting.
The entropy coefficient preserves the source recipe's `action_dim × 1e-4`
convention: four dimensions give 0.0004 rather than Humanoid's 0.0021.
The discount is 0.95. Compiler
and inner CUDA-graph settings are enabled; critic geometry diagnostics remain
enabled every 1,000 updates. See the complete [AMBI recipe](algs/ambi_state.json)
for the remaining scientific and execution settings.
The source recipe is `configs/dmcontrol/algs/ambi_inner_entropy_posttanh_outer_adapt.json`
at commit `5f6089fe70441b968ed814d1b1e73ddbd3f08a71`.
Auxiliary return critics and distillation are disabled.

Evaluation runs at step zero and every 25,000 decisions, using ten episodes.
`success_rate` is mean **final-timestep** success, not whether success occurred
at any earlier point. Evaluation returns and successes go into a separate CSV
for each task/algorithm/seed. The ten-episode cadence and one-seed selection are
setup choices, not claims that this four-task subset is an official benchmark.

Each manifest keeps a numbered, portable model checkpoint every 25,000 agent
decisions, aligned with evaluation. A completed run has 40 numbered
checkpoints through decision 1,000,000. The matching metadata sidecar
records the task and model configuration. The `latest` and `best` aliases
are also retained; `best` uses rolling training return.

For resumable training on Oscar, use
[`slurm/run_metaworld_oscar.sbatch`](../../slurm/run_metaworld_oscar.sbatch).
Array indices 0–3 correspond to Assembly, Hand-Insert, Push, and Stick-Pull.
Submit from a clean checkout with `AMBI_PROJECT_DIR`, `AMBI_EXPECTED_COMMIT`,
and `AMBI_METAWORLD_ROOT` exported. Place the campaign root and Slurm
`--output` / `--error` files on shared storage outside the checkout. Supply
the GPU, CPU, memory, and walltime requests for the measured workload.
`AMBI_METAWORLD_PYTHON` can point to a separate install of the pinned runtime's
launcher, and `AMBI_METAWORLD_NATIVE_ENV` can point to the site's native
library environment script.

The Oscar launcher materializes a complete production recipe in each segment's
console directory before launching. W&B reports to
`rwgao_b-brown-university/ambi`, group `metaworld-tdmpc2-4task-s55`,
with a separate run identity per task. Event-indexed metrics are collected
every 1,000 decisions and published as generations commit. Training uses PyTorch
float32 matmul precision `high`, one interop thread, and one intraop thread
per allocated CPU. A continuation
uses the checkpointed W&B run ID. An explicit rollback branches with a fresh
ID. The base algorithm and preparation commands keep W&B disabled for smoke
checks and timing.

All allocations for a task use the same `<campaign>/<task>/ambi_state/seed_55/lineage`.
The trainer commits full training state (including replay) every 25,000
decisions, on its 30-minute timer, and when draining at an episode boundary.
It keeps the latest recovery generations separately from the historical
evaluation checkpoints. Each segment has its own `evaluation.csv` and
checkpoint directory under `lineage/segments/`. The launcher verifies the
handoff before requeuing that array element and verifies the durable `DONE`
record on completion. Resubmitting a single element under the same root
opens its latest committed generation. Concurrent writers are rejected.

## Runtime and compatibility

The [isolated runtime](../../environments/metaworld/README.md) uses the
[Meta-World revision pinned by TD-MPC2](https://github.com/Farama-Foundation/Metaworld/tree/04be337a12305e393c0caf0cbf5ec7755c7c8feb)
with MuJoCo 2.1.0, `mujoco-py==2.1.2.14`, and `gym==0.21.0`.
The Gymnasium adapter keeps TD-MPC2's uncounted zero-action step after each
reset and returns the observation from before that step. The manifests record
this protocol explicitly so each result can be matched to the upstream curves.

All prepared configurations have W&B disabled. No jobs are submitted by the
preparation tool. Production W&B publication, resource sizing, a short GPU
validation, and scheduling on Oscar remain part of a later authorized launch.

## Inspect the prepared runs without launching

From the repository root, print the four prepared commands and their unique output paths:

```bash
python3 slurm/metaworld_campaign.py plan \
  --results-dir /absolute/path/to/new/metaworld-results
```

This command only reads JSON and prints a plan. It does not create environments,
result directories, W&B runs, or jobs. The printed `main.py` commands are for a
later scheduled compute allocation. They select exactly one algorithm/seed cell
with `--num-runs 1` and set `TDMPC2_EVAL_CSV` separately for every cell.
Keep the working directory at the repository root. Reusing an existing
evaluation CSV is rejected by the learner.

## Prepare a variant or change the seed/budget

The prepared AMBI algorithm file can be reused directly. For a new variant,
create an explicit algorithm JSON with `env="MetaWorld-v0"` and
`alg_params.obs="state"`. Review its discount, warmup, inner compute budget,
and entropy targets for the four-dimensional action space; copying a
Humanoid-specific entropy target unchanged would be a different experiment.
Set online evaluation, disable paired-controller comparison, remove any fixed
`eval_csv_path`, and leave W&B disabled. State/single-Q AMBI must retain critic
geometry diagnostics; the learner automatically enables the standard telemetry
when its setting is absent or null.

Then preview an AMBI campaign without writing or launching:

```bash
python3 slurm/metaworld_campaign.py prepare \
  --algorithm-config configs/metaworld/algs/ambi_state.json \
  --output-dir /absolute/path/to/new/campaign-configs \
  --results-dir /absolute/path/to/new/metaworld-results \
  --decisions 1000000 --trials 1 --base-seed 55 --dry-run
```

Remove only `--dry-run` to write the new complete JSON files. Preparation still
never launches anything and refuses to replace an existing output directory.
It changes only the top-level decision budget, base seed, and episode-count
limit; algorithm parameters are preserved without a shallow partial merge.
Algorithm-level checkpoint settings retain their normal precedence over the
manifest defaults. The original source configuration is never edited.
