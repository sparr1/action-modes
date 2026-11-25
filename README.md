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
inner SAC. Random inner initialization, LoRA adaptation, TD3, no inner
improvement, persistent inner state, and MPPI are auxiliary ablations or
comparison operators. In particular, MPPI is the TD-MPC-style planning
comparator; it is not AMBI's core action-selection
algorithm.

The [TD-AMBI training preset](RL/tdmpc2_core/README.md#td-ambi-training-preset)
uses TD-MPC2's actor and critic loss formulas in both the outer and inner
learners, with fixed entropy coefficients and Q percentile scaling. It keeps
AMBI's inner transition minibatches and action-local adaptation. Its matching
Humanoid Walk algorithm and experiment files are named `TD-AMBI.json`.

The [TD-AMBI objective study](configs/dmcontrol/TD-AMBI-STUDY.md) extends that
base into four prior-only checkpoint runs and six full AMBI runs. It compares
reward-only and entropy-augmented critics, Q scaling and learned temperatures,
and fixed versus adaptive inner Q scales. Full runs use one paired actor/critic
update per batched imagined step with six rounds and 512 branches.

The [SAC prior parameterization study](configs/dmcontrol/SAC_PRIOR_DIAGNOSTICS.md)
compares clipping and smooth log-std mappings with corrected action-entropy
targets −21 and −10.5. Its four configurations run two seeds each and add
outer-policy saturation, entropy, temperature and fixed-observation diagnostics
to the prior-only checkpoint banks.

The optional [critic-only LoRA-RL adapter](RL/tdmpc2_core/README.md#critic-only-lora-rl)
uses `inner_critic_adaptation="lora_rl"` with a densely trained actor. The default
rank is 96, direct adapter scale is one, and AdamW adapter weight decay is
`2e-4`. Its explicitly named Humanoid base-v2 presets compare input-and-hidden
versus hidden-only placement and ranks 64, 96, and 128; they retain the base
training and evaluation budgets. They replace the retired joint actor/critic
LoRA presets without reusing their configuration or run names.

### Optional independent return critic

AMBI also supports an [independent return critic and optional return actor](RL/tdmpc2_core/README.md#independent-auxiliary-return-critic).
Set `aux_return_mode="sac"` to evaluate reward return under the SAC prior, or
`"return_actor"` to train a separate stochastic actor against that reward-only
critic. Actor and critic sources for inner initialization and the horizon tail
are independently selectable. The existing split return/entropy heads remain
available as a separate, unchanged variant; the two variants cannot be enabled
together. The auxiliary mode defaults to `"off"`.

### Critic ensemble execution

AMBI and TDAMBI default to `alg_params.critic_ensemble_backend="vmap"`.
Existing main-experiment configurations inherit this setting automatically.
It vectorizes dense critic heads in the primary and auxiliary critics, their
targets, cloned inner critics, and eager PAVE geometry. Parameters and optimizer
slots retain their existing ownership and checkpoint names. Set the option to
`"loop"` to select the historical head loop. LoRA and custom heads retain their
compatible loop execution; the standalone TD-MPC2 comparator is unchanged.

Portable checkpoints load across either backend. Vmap preserves independent
dropout across heads but changes RNG ordering and floating-point arithmetic,
so exact continuation requires the same backend. See the
[execution and compatibility details](RL/tdmpc2_core/README.md#critic-ensemble-execution).

### Optional Bernoulli critic transfer

Set `alg_params.inner_critic_bernoulli_p=0.5` to retain each scalar critic
parameter from the previous successful inner SAC solve with probability 50%.
Other entries reset to the **current** outer critic prior. This includes biases
and normalization parameters, while buffers always reset to the prior. The
default `0.0` preserves fresh initialization; `1.0` carries all critic weights.

Transfer stays within an episode. The actor, target, Adam moments, imagined
replay, and temperature are freshly initialized at every decision; the target
starts from the selected online critic. Both loop and vmap ensembles work.
This supports main training, prediction, and isolated paired evaluation.
See [the transfer contract](RL/tdmpc2_core/README.md#bernoulli-critic-transfer)
for supported settings, logging, and exact resume behavior.

The full-training manifests `ambi_transfer_training.json` (J6) and
`ambi_transfer_training_j8.json` (J8) pair H1/H2/H3 with fresh or Bernoulli 50%
critics. Both retain all checkpoints every 25k decisions through 500k decisions.
The Hydra and Oscar transfer-training launchers default to J6. Set
`AMBI_TRANSFER_MANIFEST=configs/dmcontrol/experiments/ambi_transfer_training_j8.json`
for both the GPU smoke job and its dependent training array to select J8, with
a separate `AMBI_TRANSFER_OUTPUT` directory. The gate binds the selected manifest
and its configurations to the pinned commit; a J6 gate cannot authorize J8.

The single-cell `configs/dmcontrol/experiments/ambi_soft_training.json` control
uses fresh H3/J8 inner SAC with the same C16/A4/N128/B256 update schedule, primary
outer training recipe, seed 55 and complete 25k checkpoint cadence. It disables
auxiliary critic allocation and training, initializes the inner critic from the
primary SAC critic, uses that critic for the horizon bootstrap, and selects
entropy-augmented inner targets. The finite-horizon boundary keeps the existing
sampled outer action and online primary-Q `mean_pair` bootstrap, without an
additional terminal entropy bonus. Removing auxiliary training also removes its
loss contribution to the shared representation. There is no Bernoulli transfer.

For a short CPU-allocation comparison, run
`python slurm/benchmark_ambi_training_cpu.py --output /absolute/scratch/directory`
inside one Slurm allocation with a GPU, eight physical CPU cores
(`--cpus-per-task=8 --hint=nomultithread`) and 32 GB RAM. It measures the J8 H3
Bernoulli recipe in separate 1/2/4/8-core subprocesses using the same GPU and
seed. Each case excludes 20 active warmup decisions, then measures 128 complete
training decisions including the environment, replay and outer updates. JSON
results include throughput, controller latency, CPU occupancy and per-decision
timings; W&B and checkpoints are disabled. This early-replay screen excludes
long-run logging, periodic diagnostics and checkpoint costs. It does not change
the training recipes or establish a production walltime guarantee.

The `ambi_posttanh_outer_adapt_humanoid_stand.json` algorithm and experiment
manifest reproduce the Humanoid Walk run `fc7c433ab4734a5aa7eb1717e2165969`
on Humanoid Stand. The learner remains pinned to source `5f6089fe`: seed 55,
one million decisions, sampled execution, H3/J8/K1, N128/C16/A4/B256, fixed
post-tanh entropy 0.0021, and action-local adaptive Q scaling. All checkpoints
are retained every 25k decisions and critic geometry is observed every 1k
updates. `slurm/posttanh_humanoid_stand_oscar.py prepare` verifies source parity
and existing CPU-scaling evidence; `slurm/run_posttanh_humanoid_stand_oscar.sbatch`
requires a clean pinned checkout and runs fresh GPU startup, timing, geometry,
and checkpoint checks with W&B disabled before starting online training.

The three `ambi_posttanh_outer_adapt_dog_{run,walk,trot}.json` algorithm and
experiment pairs transfer that same recipe to Dog's 223 state observations and
38 actions. The fixed **joint** entropy coefficient remains 0.0021. Like the
official TD-MPC2 wrapper at `8bbc14eb`, each action is held for two DMControl
steps, their rewards are summed, and an episode contains 500 policy decisions.
The one-million-decision budget therefore covers two million control steps.
`slurm/run_posttanh_dog_oscar.sbatch gate`, with `AMBI_DOG_TASK` selecting the
task, checks 38-action CUDA parity and benchmarks the actual task on one and
two physical CPU cores with W&B disabled. On success it submits that task's
training job with the smallest competitive CPU allocation and measured
runtime plus 15% and 30 minutes for startup. It checks combined live QOS limits,
verifies the source and checkpoint capacity, installs W&B progress panels,
and releases the training job only after those checks succeed. The three tasks
can run concurrently; failures are shown in the workspace and Slurm logs.

The `ambi_posttanh_outer_adapt_dog_{walk,trot,run}_seeds56_57.json` manifests
add independent seeds 56 and 57 to each sampled baseline. Only the seed and
publication labels change. `slurm/run_posttanh_dog_seed_replicates_oscar.sbatch`
takes the task and seed index (`0` for 56, `1` for 57), with a separate campaign
directory for each task. The launcher verifies unchanged training code, exact
reference recipe hashes, task-specific CUDA parity, CPU-scaling evidence,
checkpoint checks and the L40S runtime before enabling W&B training. Use the
measured one-CPU allocation and derive walltime from completed reference runs.

The `ambi_posttanh_outer_adapt_dog_stand_seeds55_56_57.json` manifest adds
Dog Stand with fresh seeds 55, 56 and 57 and the same sampled learner. Run
`slurm/posttanh_dog_stand.py prepare` from a clean pinned checkout, then submit
`slurm/run_posttanh_dog_stand_oscar.sbatch gate` with two physical CPUs and an
L40S. The gate checks the actual Dog Stand environment and benchmarks one versus
two CPUs with W&B disabled, including geometry, checkpoint and graph/native
continuation checks. Production takes `train INDEX`, with indices 0–2 mapping
to seeds 55–57. All three share the verified gate because their resolved
learner settings differ only by seed and publication labels. Size production
from `gate.json`; its walltime uses the larger of the fresh measured estimate
and completed Dog reference runtimes, plus 15% and 30 minutes for startup.

The `ambi_dog_stand_mrsq_ensemble.json` manifest compares five and ten critics
on fresh seed-55 Dog Stand runs. It transfers MRS.Q's aggregation rules:
`min_all` for both TD targets and the inner terminal bootstrap, `mean_all` for
both actor objectives, and no critic-pair sampling. All remaining sampled
post-tanh Dog Stand settings are preserved. This is an AMBI aggregation study,
not a reproduction of the full MRS.Q method; its released actor uses the
ensemble mean ([official implementation](https://github.com/facebookresearch/MRSQ/blob/b12abaf309833f28619dfd87f4251074abed3ae3/MRSQ/MRSQ.py#L446-L469)).
`slurm/dog_stand_mrsq.py prepare` checks unchanged learner source and prior CUDA
evidence. The Oscar launcher takes `gate INDEX` or `train INDEX` (0 for five,
1 for ten critics). Each arm requires its own fresh CUDA aggregation parity,
one/two-CPU timing, geometry, checkpoint and graph/native continuation checks.
Production allocations use measured throughput, observed full-run overhead,
15% margin and 30 minutes for startup. Gates keep W&B disabled.

`ambi_dog_walk_mrsq_q5_seed55.json` applies that five-critic recipe to Dog Walk,
changing only the task and publication labels. It retains the one-million-decision
budget, 25k checkpoint bank and critic geometry logging. Use
`slurm/dog_walk_mrsq.py prepare`, followed by
`slurm/run_dog_walk_mrsq_oscar.sbatch gate 0` and `train 0` from the pinned clean
checkout. The task-specific gate checks real Dog Walk action-repeat/reward parity,
full-ensemble CUDA parity, one/two-CPU throughput, checkpointing and geometry
before production is enabled.

`ambi_dog_walk_mrsq_q10_seed55.json` changes only that Dog Walk run's critic
count to ten, with all-head aggregation and the same seed and training budget.
Set `AMBI_MRSQ_NUM_Q=10` for both `prepare` and the gate/training submissions;
the default remains five. The Q10 manifest, campaign, gate receipts, checkpoints
and W&B identity are separate from Q5. Q10 requires fresh ten-head CUDA parity
and Dog Walk one/two-CPU timing; size its allocation from
`gates/q10/validated.json`, including the measured memory, observed full-run
overhead, 15% timing margin and 30-minute startup allowance.

The matching seed-55 Q5 suite adds Dog Trot, Dog Run, Humanoid Stand,
Humanoid Walk and Humanoid Run through `ambi_<task>_mrsq_q5_seed55.json`
manifests. Only the environment task and publication labels differ from Dog
Walk Q5. Each task uses its own manifest, output directory and W&B identity.
Set `AMBI_MRSQ_TASK=<task>` for `slurm/mrsq_q5_suite.py prepare`, then use
`slurm/run_mrsq_q5_suite_oscar.sbatch gate <task>` and `train <task>` from the
pinned checkout. Every task passes environment action-repeat/reward parity,
Q5 family CUDA parity, one/two-core throughput, checkpoint and geometry checks
before production. Keep benchmark W&B disabled; derive production allocations
from each task's validated receipt and schedule against live QOS limits.
Training jobs may be queued held while these gates wait for GPUs. An `afterok`
CPU job running `slurm/release_mrsq_q5_suite.py` verifies each gate's pinned
source/configuration, updates every Slurm CPU field plus memory and walltime,
verifies the W&B workspace, and releases only that task's training job. The
controller records its own commit separately from the unchanged training
checkout; failed gates leave their production jobs held.

### Optional random inner initialization

Set `inner_actor_initialization="random"` and/or
`inner_critic_initialization="random"` to initialize those inner networks from
scratch for every active inner solve. Both independently default to `"prior"`.
Initialization is separate from adaptation: `inner_actor_adaptation="clone"`
and `inner_critic_adaptation="clone"` select full dense training even when
initialization is random. Random critic initialization also works with
`inner_critic_adaptation="lora_rl"`: selected random critic matrices freeze,
while adapters, critic biases, normalization, and value heads train; the actor
continues dense training.

This option supports single-policy, action-local inner SAC and the native
TDAMBI frozen-checkpoint evaluator. Native TDAMBI retains its TD-MPC2 objectives
and explicit Q-scale initialization policy. Setting `inner_rounds=0` retains
the existing outer-policy bypass. The learned world model and outer control
priors retain their existing training behavior. See
[inner-network initialization](RL/tdmpc2_core/README.md#inner-network-initialization)
for examples, initialization rules, compatibility limits, and checkpoint use.

### Optional adapted-prior writeback

AMBI normally discards each action-local learner. An opt-in training ablation
can instead interpolate its final cloned actor and/or online critic back into
the corresponding persistent prior:

```text
prior <- (1 - coefficient) * prior + coefficient * adapted_to_go
```

Set `inner_actor_writeback_coef` and `inner_critic_writeback_coef` independently
to finite values in `[0, 1]`; both default to `0.0`, while `1.0` fully replaces
that prior after every planned training action. A nonzero coefficient is
restricted to the canonical J/N/H/G schedule with action-local, full-clone
inner SAC, optimizer updates enabled for the corresponding component, and
matching inner/outer actor log-standard-deviation semantics. The writeback
occurs only after the returned action and diagnostics have been computed, and
it persists across environment episode resets because it changes the outer
prior itself.

Evaluation and direct prediction still run inner improvement without changing
the outer priors. Writeback preserves the outer parameter objects and their Adam
state. Critic writeback targets only the online outer critic; its target critic
continues its normal EMA updates. The inner temperature and target critic are
never written back. When either coefficient is active, the lightweight
`inner_actor_writeback_coef`, `inner_critic_writeback_coef`,
`inner_actor_writeback_applied`, and `inner_critic_writeback_applied` metrics
report the configured strengths and whether each writeback occurred.

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

For state-observation TD-MPC2/AMBI, opt in with top-level
`"checkpoint_replay": true` beside `checkpoint_every` (algorithm settings take
precedence over the experiment manifest). Each saved model then retains a
`<model>.replay/` companion containing every resident replay row, episode IDs,
writer/sampler state, and any collected current-episode rows not yet inserted
into replay. CPU replay placement is fixed before collection. Capture streams
owned 32,768-row shards while training is paused, bounding additional memory;
model serialization remains asynchronous. The companion manifest binds the
model step and SHA-256 and checksums every replay payload. The model metadata
sidecar is published only after both are complete. Load and verify a pair with
`utils.checkpoint_replay.load_checkpoint_replay(model_path, buffer)`; pending
episode rows are returned separately to preserve the original replay sampling
population. This option preserves retained replay for later experiments; exact
training continuation still uses the native resume lineage, which additionally
stores environment, learner, optimizer, carried inner state, and RNG state.

Exact-resume W&B verification accepts API-added null padding only for metric
keys present elsewhere in the saved event history. Event indices, environment
steps and logged values still match exactly; missing logged fields, unknown
fields and extra non-null values remain errors.

Resuming a pre-fix lineage from a telemetry repair commit requires an explicit
reviewed-source proof. Set `AMBI_RESUME_TELEMETRY_BASE_COMMIT` to the lineage's
original full commit and `AMBI_RESUME_TELEMETRY_PATCH_SHA256` to the SHA-256 of
`git diff --binary --full-index --no-ext-diff --no-textconv BASE HEAD --`
after reviewing that patch. Full object IDs keep the digest independent of
repository size and Git abbreviation settings. Both checkouts must
be clean, the execution commit must descend from the base, and the change must
stay within the enumerated W&B repair, compatibility, tests and documentation
files. All training configuration, dependencies, model code and gate artifact
hashes must still match. Use `AMBI_RESUME_MODE=required` and the original lineage.
The original lineage and checkpoint files are preserved; the new `SEGMENT.json`
and campaign launch receipt record the actual execution commit, original
source, patch digest and compatibility rule. Without this explicit proof, the
strict source-identity requirement is unchanged.

Native TD-MPC2 periodic and final checkpoints also save the learned Q percentile
scale `S`, frozen at the same step as the networks. Frozen TDAMBI evaluation
uses that saved scale to initialize each independent inner solve. AMBI already
preserves its actor Q scale when scaling is enabled. Older native snapshots
without `S` retain the documented local-calibration fallback; see the
[evaluation scale options](RL/tdmpc2_core/README.md#tdambi-native-inner-learning).

### Oscar durable checkpoint storage

The general Oscar resume launchers for `rgao48` store resumable state under
`/oscar/home/rgao48/ambi-durable`. The quota preflight must select the user home
row that owns that path:

```bash
AMBI_DURABLE_ROOT=/oscar/home/rgao48/ambi-durable
AMBI_LINEAGE_DIR=/oscar/home/rgao48/ambi-durable/lineages/<unique-lineage>
AMBI_DURABLE_QUOTA_LABEL=rgao48
AMBI_DURABLE_QUOTA_PATH=/oscar/home
```

Pass all four values through `sbatch --export`. Use a new lineage for a fresh
run and the same lineage only for its exact resume. `data+rbalestr` is a
separate allocation and is not an approved AMBI durable target; both Oscar
launchers reject that label and any quota selection other than exactly
`rgao48` plus `/oscar/home`. On Oscar they also require the exact durable root
shown above, so the checked quota row and the checkpoint filesystem cannot
diverge.

The Q10 launcher stores active-run state on the shared Oscar scratch
filesystem, under the campaign's `<task>/lineage` directory. Its quota check
uses the `rgao48` row for `/oscar/scratch` and accounts for staged generations
and the separate `<task>/models` checkpoint bank. The same lineage is used
after a requeue or restart on another node. Oscar may purge individual scratch
files after 30 days without access, so archive checkpoints that need longer
retention. Job-local storage is used only for temporary work and caches.

Humanoid Stand keeps its existing task recipe with ten critics and the
full-ensemble Q10 reductions. The resumable Dog Walk job starts a fresh
seed-55 attempt. The earlier interrupted run saved model checkpoints without
enough training state to restore its replay and random streams.

For the matched Dog Run, Humanoid Run, Humanoid Stand, and Humanoid Walk
seed-56 jobs, pass `--seed 56` and the task to
`slurm/queue_mrsq_q10_suite.py queue`. The dispatcher reuses the validated
seed-55 resource measurements after checking the recipe and source, then checks
the assigned GPU at startup. Each seed-56 job uses its own `<task>-s56/lineage`
and tests save-and-resume on its first trained checkpoint before continuing.

Hydra accepts Dog Trot, Dog Walk, and Humanoid Stand seed 57 through
`slurm/run_mrsq_q10_dogs_hydra.sbatch <task> 57`. Request one L40S or L40 with
the corresponding Slurm GPU type. The startup check permits at most 0.38
seconds per decision on L40S or 0.46 on L40. Each run has a separate
`<task>-s57/lineage` and `<task>-s57/models` under the Hydra campaign.

The compact AMBI branch-count and imagination-horizon suite lives under
`configs/ambi/`. Each algorithm file has one matching runnable
manifest under `configs/ambi/experiments/`; the anchor entry point is
`configs/ambi/experiments/ambi_anchor.json`, used with
`--alg-dir configs/ambi/algs`.

AMBI and the local TD-MPC2 baseline now use TD-MPC2's temporal weighting with
`rho=0.5` in defaults and maintained presets: model losses average `rho**t`
weighted terms over the `H` transitions, while the outer actor averages over
all `H+1` latents. There is no reference-horizon rescaling. Historical temporal
configuration fields still parse with a deprecation warning for checkpoint
evaluation; weight transfer into this training rule starts a new lineage.
See [the exact reductions and compatibility contract](RL/tdmpc2_core/README.md#temporal-training-weights).

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

The Humanoid Walk MC-value base also has three paired analytic behavior-KL
entry points under `configs/dmcontrol/experiments`: `ambi_anchor_kl_smooth`,
`ambi_anchor_kl_quantile`, and `ambi_anchor_kl_dual`. They retain the same
five-head `min_all`, reward-only critic configuration, three seeds, one-million
decision budget, and value-evaluation/checkpoint cadence.

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

That Humanoid configuration also runs five fixed-seed paired auxiliary
episodes at step zero and each 100,000-decision evaluation point. Under the shared AMBI
metric names, `outer` is TD-MPC2's deterministic network policy and
`fresh_inner` is eval-mode MPC. This adds five MPC episodes plus five cheap
network-policy episodes beyond the ordinary ten MPC episodes, while leaving
training compute and the training environment unchanged.

The launcher is for TD-MPC2 comparator runs only. Add AMBI benchmark cells only
after explicitly freezing and profiling an inner-loop compute schedule; the
sparse historical Walker AMBI template inherits a substantially larger default
inner workload and is not a confirmatory benchmark configuration.

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
