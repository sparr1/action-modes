# Consolidated AMBI experiments

This directory is the source of truth for the new Ant-v4 comparison suite.
Algorithm files are flat under `algs/`; experiment manifests are grouped by
research question under `experiments/`.

## Baseline entry point

The clearly labeled default is:

```bash
python main.py \
  --run configs/ambi/experiments/canonical/baseline_comparison.json \
  --alg-dir configs/ambi/algs
```

It starts three independent one-million-step runs from seed 55:

- native distributional SAC with the deliberately retained twin-Q critic;
- standard TD-MPC2 with train horizon 3 and planning horizon 3;
- full-copy AMBI TD-MPC2 with train/plan/inner horizons 3/3/3.

All three keep `best` and `latest` aliases at 100,000-step boundaries. `best`
means the best rolling mean of completed stochastic training episodes; it is
not a side-effect-free deterministic evaluation score. Checkpoints are model
snapshots and are not guaranteed to contain replay, environment, and all RNG
state needed for exact training resumption.

Each checked-in research manifest uses exactly one trial at seed 55 and is
explicitly labeled `single_seed_exploratory_screening` in the manifest itself.
These runs support engineering and exploratory comparisons only: do not attach
confidence intervals, significance claims, or confirmatory conclusions. Every
configuration is trained from scratch and may generate its own trajectory and
replay distribution.

## Trial groups

- `horizon/`: the 2-by-2 train-horizon 3/6 and inner-horizon 3/6 study.
- `fixed_budget/`: inner horizons 1, 2, 3, 4, and 6 while holding imagined
  transitions at 192 per round by changing the rollout count.
- `breadth_depth/`: four controls that vary inner horizon and rollout count
  independently, with replay capacity set to the generated transition count.
- `update_dose/`: 64 versus 192 update slots per round.
- `batch_size/`: inner batch size 64 versus 128.
- `round_schedule/`: 1, 2, 4, or 8 collect/update rounds while holding total
  imagined transitions and total update slots fixed at 768 per action.
- `smoke/`: short CPU checks for five-head construction, train-6/plan-3 routing,
  and the executable inner-horizon extrapolation warning, plus a GPU throughput
  pilot that runs several actions with the full J4/N64/H3/G192 inner schedule.

Every full experiment has an individual manifest, and each group also has an
`all_*_trials.json` convenience manifest. `experiments/all_unique_trials.json` runs each
behaviorally unique full condition once; it deliberately excludes duplicate
anchor aliases listed in `research/trial_matrix.json`.

Example:

```bash
python main.py \
  --run configs/ambi/experiments/horizon/horizon_train6_inner6.json \
  --alg-dir configs/ambi/algs
```

## Environment semantics

The local comparator uses Gymnasium 0.29.1 `Ant-v4` with:

```text
exclude_current_positions_from_observation = true
max_episode_steps = 1000
terminate_when_unhealthy = false
```

This produces a 27-dimensional observation and an 8-dimensional action. Ant's
internal MuJoCo frame skip is five (0.05 simulated seconds per agent action).
There is no additional action-repeat wrapper. Because unhealthy termination is
disabled, the configured environment ends through the 1,000-step `TimeLimit`
truncation rather than a true terminal transition. The learners bootstrap
through that truncation, so training is a continuing discounted objective over
finite reset segments. AMBI's `episodic=false` likewise keeps imagined
rollouts at their configured fixed length.

Upstream TD-MPC2 does not use Gym Ant in its published 104-task benchmark. At
the repository revision pinned for this comparison, its default task is
`dog-run` with state observations, `episodic=false`, 10 million training agent
steps, and ten evaluation episodes every 50,000 steps. DeepMind Control tasks
use action repeat two and a 1,000-physics-step time limit, giving 500 agent
decisions per reset segment. TD-MPC2 suppresses timeout termination for those
tasks and therefore bootstraps through the timeout. The default learner uses
five Q heads, 101 symlog bins over [-10, 10], and random pairs of two. The files
here apply those learner choices to the repository's local Ant comparator.

## Controlled compute

All AMBI research files explicitly resolve the three horizons and use
`reference_weighted_mean` with reference horizon 3 and rho 0.7. The default
inner schedule is J=4, N=64, H=3, G=192, batch 64, and replay capacity 768.
Replay uses replacement and remains action-local and cumulative across rounds.

The suite metadata records transitions per round/action, optimizer slots,
replay-row draws per action, and capacity for every condition. It is validated
against the ready-to-run files by the config tests.

## Safe execution options

The full AMBI anchor performs 768 sequential joint SAC update slots per real
action, so use the routing smoke first and then
`smoke/smoke_ambi_anchor_throughput.json` before allocating long jobs.
Independent conditions may be scheduled in parallel. `compile_strict=false`
allows AMBI to fall back safely when compilation is unsupported; standard
TD-MPC2 remains uncompiled in this repository, so wall-clock comparisons should
retain and report that distinction. These are execution choices only; none
changes the research configurations, shares checkpoints, or forces identical
replay.

The older frozen-checkpoint, one-axis preset matrix is retained under
`legacy/`. It is useful for diagnostics but must not replace the end-to-end
comparisons above.

## Post-run outcomes

Use training-return area under the learning curve versus real environment
steps, final deterministic real-environment return, and return versus wall-clock
time as the primary outcomes. Do not select configurations by the same world
model's predicted return.

Evaluate both aliases headlessly and deterministically over environment seeds
101 through 105:

```bash
python render_checkpoint.py /path/to/model_best \
  --results-json evaluation/best.json --episodes 5 --seed 101
python render_checkpoint.py /path/to/model_latest \
  --results-json evaluation/latest.json --episodes 5 --seed 101
```

The renderer increments the initial seed once per episode and uses policy means
unless `--stochastic` is explicitly requested.
