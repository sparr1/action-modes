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

Set `AMBI_METAWORLD_SEED=56` or `57` to launch another independent seed;
the default remains 55. For Assembly alone, submit `--array=0` separately
for each seed with its measured resource requests. This changes the top-level
training seed and publication labels only. Each invocation remains one
algorithm/trial cell at trial index zero; do not use `--trial-index` for
resumable seed selection. The equivalent config command is
`python3 slurm/metaworld_campaign.py production-config --task assembly --seed 56 --output /absolute/path/to/ambi_state.json`.

The Oscar launcher materializes a complete production recipe in each segment's
console directory before launching. W&B reports to
`rwgao_b-brown-university/ambi`, group `metaworld-tdmpc2-4task-s<seed>`,
with a separate run identity per task. Event-indexed metrics are collected
every 1,000 decisions and published as generations commit. The launcher selects
W&B's core backend (`WANDB__REQUIRE_CORE=true`) with the pinned SDK, matching
the working seed-55 deployment. Training uses PyTorch
float32 matmul precision `high`, one interop thread, and one intraop thread
per allocated CPU. A continuation
uses the checkpointed W&B run ID. An explicit rollback branches with a fresh
ID. The base algorithm and preparation commands keep W&B disabled for smoke
checks and timing.

All allocations for a task and seed use the same
`<campaign>/<task>/ambi_state/seed_<seed>/lineage`.
The trainer commits full training state (including replay) every 25,000
decisions, on its 30-minute timer, and when draining at an episode boundary.
It keeps the latest recovery generations separately from the historical
evaluation checkpoints. Each segment has its own `evaluation.csv` and
checkpoint directory under `lineage/segments/`. The launcher verifies the
handoff before requeuing that array element and verifies the durable `DONE`
record on completion. Resubmitting a single element under the same root
opens its latest committed generation. Concurrent writers are rejected.
Unexpected nonzero exits remain fail-closed rather than automatically requeued;
after resolving the cause, resubmit the same task and seed under the same root
to restore its durable `LATEST` generation. A different seed uses a separate
lineage and W&B identity, without modifying the seed-55 run.

## Assembly with Q5 actor mean and target minimum

The explicit [Q5 Assembly recipe](algs/ambi_state_mrsq_q5.json) changes the
baseline's random-pair reductions to use all five critics: `q_pair_size=5`,
inner and outer actors use `mean_all`, and inner/outer targets and terminal
Q use `min_all`. All other scientific settings remain the Assembly baseline,
including sampled training actions, entropy coefficient 0.0004, discount
0.95, the H3/J8 planner, 1M decisions, critic geometry diagnostics, and ten
online evaluation episodes every 25k decisions.

Submit only Assembly (`--array=0`) with `AMBI_METAWORLD_VARIANT=mrsq_q5`
and an explicit seed (55 for the initial comparison), using the existing Oscar
launcher and a new campaign root. The launcher selects
[`assembly_mrsq_q5.json`](experiments/assembly_mrsq_q5.json) and saves the full
recovery lineage under `<campaign>/assembly/ambi_state_mrsq_q5/seed_<seed>/lineage`.
This is a fresh training variant, not a mutation or continuation of the original
Assembly policy. It uses the same full-state recovery and verified walltime
handoff behavior described above. The default `baseline` variant retains the
existing configuration, output layout, and W&B identities.

Preview the exact production recipe without creating a run:

```bash
python3 slurm/metaworld_campaign.py production-config \
  --task assembly --seed 55 --variant mrsq_q5 \
  --output /absolute/new/path/ambi_state_mrsq_q5.json
```

The Q5 variant is currently prepared only for Assembly. The launcher rejects
other task indices or unknown variant names before creating a lineage.

## Assembly with Q10 actor mean and target minimum

The [Q10 Assembly recipe](algs/ambi_state_mrsq_q10.json) increases both the
ensemble and the full-ensemble reduction size to ten. It uses the same
Assembly training settings and ten-episode online evaluation at 25k decisions.
Submit Assembly (`--array=0`) with `AMBI_METAWORLD_VARIANT=mrsq_q10` and a new
shared-scratch campaign root. Its full-state lineage is
`<campaign>/assembly/ambi_state_mrsq_q10/seed_<seed>/lineage`.

Before production, run the scheduled L40S check in
`slurm/smoke_metaworld_mrsq_q5.py` with `--variant mrsq_q10`.
Use its compiled training and native evaluation timings to size the allocation.

Push and Stick Pull also support `mrsq_q10`. For the Hydra pair, submit
`slurm/run_metaworld_q10_hydra.sbatch` with its two task indices, a pinned
checkout, the locked native MetaWorld interpreter, and shared project
storage. Each allocation validates its actual L40 before training in place.
The shared launcher saves full state every 15 minutes at an episode boundary
or every 25k decisions, and verifies the handoff before a walltime requeue.

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
