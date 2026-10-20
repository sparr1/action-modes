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
replay, and temperature are freshly initialized at every inner solve; the target
starts from the selected online critic. Both loop and vmap ensembles work.
This supports main training, prediction, and isolated paired evaluation.
See [the transfer contract](RL/tdmpc2_core/README.md#bernoulli-critic-transfer)
for supported settings, logging, and exact resume behavior.

### Execute several real actions per inner solve

Set `alg_params.inner_solve_interval=K` to run inner SAC once per K real
decisions. The default is 1. This is independent of `inner_rollout_horizon=H`;
for example, adding `"inner_solve_interval": 3` to a full H3/J8 recipe solves
at episode decisions 0, 3, 6, and so on, with eight rounds at each solve.

Each intervening action evaluates the adapted actor at the new real
observation using a frozen snapshot of the encoder from that solve. The actor
and encoder stay fixed for the block, while the outer learner and real replay
continue their ordinary updates. Training retains the configured sampled or
mean action execution; deterministic evaluation executes policy means.
Bernoulli critic transfer, when enabled, occurs between solves. Episode
boundaries discard the held policy, including a partial final block.

This supports the current state-observation fresh/Bernoulli SAC recipes with
primary soft or auxiliary-return critics. Existing experiment configurations
remain unchanged. See [held-policy execution](RL/tdmpc2_core/README.md#held-policy-execution-in-full-training)
for compatibility, metrics, checkpointing, and the opt-in CUDA smoke gate.

The one-cell `configs/dmcontrol/experiments/ambi_held_training.json` runs fresh
aux-return H3/J8/K3 with sampled actions, seed 55, 500k decisions, and all 25k
model snapshots. Its Oscar wrapper is
`slurm/run_ambi_held_training_oscar.sbatch`. Set `AMBI_PROJECT_DIR`, the locked
`AMBI_PYTHON`, full `AMBI_EXPECTED_COMMIT`, and an external `AMBI_HELD_OUTPUT`.
Submit `AMBI_HELD_MODE=smoke` with three physical CPU cores and explicit GPU,
memory and walltime resources. This runs the exact K3 CUDA case followed by
1/2/3-core timing of complete training decisions with CPU replay, matching
production exact resume. The gate writes `gate/receipt.json` bound to the source,
configuration, runtime and GPU model. The CPU comparison alone can be run inside
an allocated GPU job as:

```bash
python slurm/benchmark_ambi_training_cpu.py \
  --manifest configs/dmcontrol/experiments/ambi_held_training.json \
  --cell-index 0 --cpu-counts 1,2,3 --warmup 12 --decisions 120 \
  --output /absolute/scratch/held-cpu-comparison
```

After the gate passes, submit the same wrapper with `AMBI_HELD_MODE=train`, a
measured CPU count, the same GPU model, an explicit `AMBI_DRAIN_AFTER_SECONDS`
below the requested walltime, and a unique `AMBI_LINEAGE_DIR` under
`/oscar/home/rgao48/ambi-durable`. A production-only `--signal=USR1@900` provides
an additional drain trigger; omit it for the short smoke. The canonical Oscar launcher owns quota checks,
episode-boundary exact resume, W&B continuity, and verified requeue. Numbered
model snapshots remain under each durable segment's `model_snapshots`; pruning
old resume generations does not remove these snapshots. The wrapper specifies
no default CPU, GPU, memory, or walltime request.

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

The one-cell `ambi_return_only_mppi_training.json` and
`ambi_return_only_sac_training.json` manifests instead train the primary
five-head distributional critic with reward-only Bellman targets and disable
the auxiliary critic. The actor retains automatic entropy regularization,
initialized at 1.0 with target entropy −10.5. The `sac` source selectors refer to
this primary learner; its critic now predicts reward return. MPPI keeps H3,
8 iterations, 512 candidates, 64 elites and 24 policy trajectories, replanning
every decision (K1). Fresh inner SAC keeps H3/J8/C16/A4/N128/B256 with reward-only
inner targets and K3 held-policy execution. Both keep sampled training actions,
seed 55, 500k decisions, all 25k checkpoints and durable exact resume, and publish
to AMBI group `ambi-return-only-training-20261019`. Their finite-horizon terminal
score uses online primary-Q `mean_pair` without an added entropy bonus.

The matching one-cell `ambi_return_only_no_inner_training.json` control samples
the current outer actor directly after the same random warmup. It performs no
planning, imagined transitions, inner updates, transfer or held-policy execution.
The outer H3 replay unroll, world model, reward-only five-head primary critic,
automatic actor entropy target −10.5, seed 55, 500k decisions and all 25k
checkpoints remain the same. It publishes as
`No inner | reward-only Q | auto entropy | seed55` in the same AMBI group, so the
comparison isolates the collection policy while preserving outer learning.

The separate one-cell manifests
`ambi_return_only_no_inner_1m_entropy10p5.json` and
`ambi_return_only_no_inner_1m_entropy21.json` start fresh seed-55 runs for
1M decisions (2M raw environment steps), with automatic actor entropy targets
−10.5 and −21 respectively. They preserve the no-inner reward-only control's
remaining settings, including its 1M-row replay capacity, and retain all forty
25k model checkpoints. Their names include the entropy target and 1M-decision
budget in the same AMBI group. The earlier 500k recipe remains available;
these are new training lineages, not continuations of that run.

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

Native TD-MPC2 periodic and final checkpoints also save the learned Q percentile
scale `S`, frozen at the same step as the networks. Frozen TDAMBI evaluation
uses that saved scale to initialize each independent inner solve. AMBI already
preserves its actor Q scale when scaling is enabled. Older native snapshots
without `S` retain the documented local-calibration fallback; see the
[evaluation scale options](RL/tdmpc2_core/README.md#tdambi-native-inner-learning).

### Oscar durable checkpoint storage

AMBI jobs submitted by `rgao48` on Oscar use
`/oscar/home/rgao48/ambi-durable` by default. The quota preflight must select the
filesystem row that owns the chosen root:

```bash
AMBI_DURABLE_ROOT=/oscar/home/rgao48/ambi-durable
AMBI_LINEAGE_DIR=/oscar/home/rgao48/ambi-durable/lineages/<unique-lineage>
AMBI_DURABLE_QUOTA_LABEL=rgao48
AMBI_DURABLE_QUOTA_PATH=/oscar/home
```

Pass all four values through `sbatch --export`. Use a new lineage for a fresh
run and the same lineage only for its exact resume. `data+rbalestr` is a
separate allocation and is not an approved AMBI storage target.

For new reward-only training lineages, scratch is an explicit alternative:

```bash
AMBI_DURABLE_ROOT=/oscar/scratch/rgao48/ambi-training-lineages
AMBI_LINEAGE_DIR=/oscar/scratch/rgao48/ambi-training-lineages/<unique-lineage>
AMBI_DURABLE_QUOTA_LABEL=rgao48
AMBI_DURABLE_QUOTA_PATH=/oscar/scratch
```

The reward-only wrapper and canonical resume launcher accept only these two
root/quota pairs. Both retain containment, quota, write/fsync and exact-resume
checks. A lineage keeps its resume generations, segment logs and every requested
25k portable checkpoint together; scratch survives job boundaries and requeues,
but is not a backup. Oscar [purges scratch files not accessed for 30 days](https://docs.ccv.brown.edu/oscar/managing-files/filesystem),
so archive results needed for long-term retention. Job-local temporary storage
is not supported for lineages. Existing home lineages remain in place.

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
