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
