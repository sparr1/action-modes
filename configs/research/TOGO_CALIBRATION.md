# To-go return diagnostics

## Early-checkpoint H1 pilot

`ambi_scratch_takeoff_h1.json` selects source `mey3rxj8` at 100k, 125k, 150k,
200k, 300k, 500k, and 2M decisions, with explicit checkpoint hashes. Use
`--preset initialization/scratch` with either evaluator below and replace the
matrix path. The primary development checkpoint is 200k; the 100–150k group
samples takeoff, and 500k/2M provide later controls. All use the same inner
recipe. The old 2M reference below is preserved as a separate configuration.

The pilot collects 128 one-step model transitions with the actor held fixed,
then performs 32 critic updates followed by 4 actor updates. Four rounds give
512 model transitions, 128 critic updates, and 16 actor updates per real
decision. Batch size is 256 with replacement; the first batch therefore reuses
some of the 128 collected rows. Replay capacity is fixed at 2,048 so later
N64/N128/N512 experiments can retain all four rounds without changing capacity.
Increasing collection later requires matching total queries and updates when
testing only the collection frequency. This initial recipe is an exploratory
pilot, with no established optimality claim.

Each solve starts a random dense actor and critic. The actor's initial pre-tanh
standard deviation is exactly 0.3 at every state; its mean remains random.
Both learning rates are `3e-4`, alpha is fixed at `1e-4`, and the actor uses
the checkpoint's own saved Q percentile scale frozen for the entire solve.
There is no common numeric Q scale across checkpoints. The checkpoint's
five-head, 101-bin distributional architecture and dropout 0.01 are retained.
Inner actor Q uses `mean_pair`, inner target Q uses `min_pair`, and the frozen
outer terminal bootstrap uses online `mean_pair` Q at a sampled prior action.
With H1 every transition directly targets its predicted reward plus this
frozen outer bootstrap; no multistep local-target propagation is involved.

Model probes run at initialization and rounds 1–4: actor counts 0/4/8/12/16 and
critic counts 0/32/64/96/128. The 32-rollout probes add 192 model transitions,
384 policy sample rows, and 192 Q evaluation rows per decision, without extra
optimizer updates. This is 37.5% additional model-transition work, which is
not a wall-time estimate. Dedicated real calibration retains five source
episodes, five roots per episode, three independent solves per root, four
sampled branches per captured actor, and 1,000 prior continuation decisions.
With five snapshots and H1, the full panel per checkpoint uses 1,501,500 actor
branch decisions, 100,100 reusable prior decisions, and 2,500 source decisions.

Ordinary episodes execute actor means and can use `controller/prior` for a
matched mean-policy reference. Real calibration executes sampled actors and
prior continuations as specified below. These are separate protocols.
Frozen evaluations measure improvement with a model available at that stage;
an eventual online experiment is required to establish earlier outer-learning
takeoff when adapted actions change the training data.

## Original H3 reference

`ambi_scratch_togo_reference.json` pins the final 2,000,000-decision checkpoint
from `rwgao_b-brown-university/ambi/mey3rxj8` by SHA256. Its reference is dense
inner SAC with random actor and critic initialization, J5/N512/H3/B512,
replay capacity 7,680, and 15 paired actor/critic updates per real decision.
The independent target critic starts from the new online critic. Interior
Bellman targets use the inner target; the horizon boundary uses the frozen
outer actor and online Q. Alpha stays at `1e-4`, both learning rates are `3e-4`,
and each solve freezes the checkpoint's Q scale (17.54598045349121).

`initialization/inherited` changes only the actor/critic initialization.
`update_dose/x1`, `x2`, and `x4` retain collection, batch size, and scratch
initialization, with 15, 30, and 60 paired updates respectively. The world
model and outer actor/Q remain frozen. These are evaluation configurations;
they do not launch outer training or production jobs.

## Model probes during ordinary episode evaluation

```bash
AMBI_EVAL_PY=environments/dmcontrol/.venv/bin/python
AMBI_CHECKPOINT=/absolute/path/to/verified-2000000.pt
AMBI_RESULTS=/absolute/path/to/fresh-results
"$AMBI_EVAL_PY" evaluate_ambi_checkpoint.py \
  --matrix configs/research/ambi_scratch_togo_reference.json \
  --preset initialization/scratch --checkpoint "$AMBI_CHECKPOINT" \
  --device cuda --bundle-dir "$AMBI_RESULTS/episodes"
"$AMBI_EVAL_PY" report_ambi_benchmark.py \
  --bundle "$AMBI_RESULTS/episodes" --output "$AMBI_RESULTS/episodes.html"
```

`evaluation.togo_return_rollouts=32` enables probes before learning and after
all five rounds. Outside this new reference matrix, the default remains zero.
The reported score is

```text
G_model = sum(t=0..H-1, gamma^t r_model(z_t,a_t))
          + gamma^H Q_outer_online(z_H, a_H),  a_H ~ pi_outer(z_H).
```

The first H actions use the actor at the measured boundary. The terminal
actor uses the outer policy's own log-standard-deviation semantics. Q reduction
matches `mppi_terminal_q_reduction`. True termination masks subsequent reward
and bootstrap. Reported returns have no entropy bonus or Q normalization.
Diagnostic Q calls use the eager frozen ensemble through the ordinary Q
decoder and reduction, without changing the learner's compilation cache.

Each event retains current, initial, and prior reward/bootstrap/total means,
paired gains, actual actor/critic counters, and diagnostic work/time. Noise and
critic-pair selection are private and fixed per root; probes do not change
learner RNG streams, actions, optimizer state, outer state, or module modes.
The initial snapshot is the actual new actor; `inner_rounds=0` is an outer-policy
bypass and is not used to represent it. Existing `inner_return_mean` remains the
undiscounted reward collected during learning, with its existing meaning.

The bundle keeps complete trace events, `togo_probe_rows`, and per-round
summaries, including per-episode summaries. The existing checkpoint-axis curves
remain supported. To create a separate round-axis diagnostic series:

```bash
"$AMBI_EVAL_PY" evaluate_ambi_calibration.py export-model \
  --bundle "$AMBI_RESULTS/episodes" --selector initialization/scratch \
  --attempt-label scratch-model-first --output "$AMBI_RESULTS/model-series"
```

This export labels controller-specific episode states separately from shared
prior roots. Observation-bank model probes can also be exported when source
episode provenance is present; they are distinct from simulator-state banks.

## Separate continuing-simulator calibration

The opt-in `--prefix-action-rule mean` (or matrix
`real_calibration.prefix_action_rule="mean"`) executes `tanh(mu)` for the H
prefix decisions of both the saved actor and its prior reference. All tail
actions remain sampled, with unchanged tail noise and exact endpoint-Q
handoff. Default `sampled` preserves the existing protocol. Mean-prefix
references have a separate cache identity; sampled 32-rollout training probes
are unchanged. The [twenty-seed experiment](MEAN_PREFIX_OSCAR.md) uses this
diagnostic without changing inner optimization.

```bash
"$AMBI_EVAL_PY" evaluate_ambi_calibration.py run \
  --matrix configs/research/ambi_scratch_togo_reference.json \
  --checkpoint "$AMBI_CHECKPOINT" --preset initialization/scratch \
  --device cuda --attempt-label scratch-real-first \
  --save-root-bank "$AMBI_RESULTS/simulator-roots.json" \
  --reference-cache "$AMBI_RESULTS/prior-continuations" \
  --bundle-dir "$AMBI_RESULTS/real-series"
```

The default panel uses five prior-mean source episodes, seeds 101–105, and
states immediately before decisions 0/100/200/300/400. Each state has three
independent inner solves, each with four sampled real rollouts for its initial
actor and every completed round. A captured actor is held fixed for H real
decisions, encoding the new observation each time, then hands control to the
frozen prior for 1,000 decisions. There is no further optimization on a branch.

The simulator bank contains immutable, checksummed MuJoCo integration state,
task and space RNGs, Gym/raw-DMControl bookkeeping, exact observations, and
runtime/model identity. Restoration is supported initially for **Humanoid Walk
state observations**, using the reviewed standard Gym wrappers. Banks must
match the runtime that captured them. Both Gym's and raw DMControl's clocks
are disabled only in dedicated continuation environments. Source episodes and
ordinary evaluations retain their normal time limits.

To compare another setting, supply `--root-bank` instead of `--save-root-bank`,
reuse `--reference-cache`, and select a fresh bundle and explicit attempt label.
Prior references are cached per root and verified against the checkpoint,
state, noise, horizon, discount, Q reduction, effective outer policy distribution,
and source semantics. Repetitions and rounds reuse the same prior samples.
Incompatible or corrupted references fail; they are not silently reused.

Coverage can use exactly one of `--decisions 0 100 200`, `--every-n 100`, or
`--every-decision`. Other overrides are `--rounds 0 5`, `--solver-repetitions 3`,
`--rollout-repetitions 4`, `--tail-steps 1000`, `--model-probe-rollouts 32`,
`--seeds 101 102 103 104 105`, `--max-steps 500`, and
`--bootstrap-resamples 2000`. Equivalent defaults live in `real_calibration`
and `evaluation` in the matrix. Continuations must be long enough to reach the
original cutoff for every selected root; shorter partial sums are not reported
as full cutoff returns. A small smoke can shorten both `--max-steps` and
`--tail-steps`, and select corresponding early roots/rounds.

Each paired measurement contains:

- Model H-step rewards plus frozen outer online Q.
- Real H-step rewards plus Q at the real endpoint.
- Real H-step rewards plus the measured discounted prior continuation.
- Endpoint bootstrap error, total prediction error, and gains over the paired
  prior in model, real-bootstrapped, and real-Monte-Carlo returns.
- Discounted and undiscounted reward through the original episode cutoff.

Model and real prefixes use identical policy noise. At the real endpoint, Q
is queried with the exact sampled action executed as the first prior-tail
step. Monte Carlo tails never receive a critic bootstrap at their cutoff.
For per-decision reward bound `Rmax`, the omitted tail at the endpoint is
`Rmax * gamma^T / (1-gamma)`. With gamma .99, Rmax 2, and T 1,000 this is
0.00863425 (approximately 0.00864); at the original root it is multiplied by
`gamma^H`. True terminal branches have a zero remaining bound. Unexpected
truncation makes the calibration incomplete and prevents publication.

## Reports and publication

A completed bundle contains a manifest, compressed complete paired rows,
`report.html`, source matrix/sidecar, simulator bank, prior references, and
per-solve model-probe shards. The sibling `.work` directory retains completed
shards and a failure receipt if evaluation is interrupted. Fresh invocations
require new output directories; original results are never overwritten.

The standalone HTML includes all measurements, a root selector, round/update
axes, means, variability, confidence intervals, protocol and timing, and a JSON
download. It needs no server or network. Rebuild another copy using:

```bash
"$AMBI_EVAL_PY" evaluate_ambi_calibration.py report \
  --bundle "$AMBI_RESULTS/real-series" --output "$AMBI_RESULTS/comparison.html"
```

Publish only after choosing the explicit attempt and reviewing the local
bundle. The default publication mode is offline. Explicit online publication:

```bash
"$AMBI_EVAL_PY" evaluate_ambi_calibration.py publish \
  --bundle "$AMBI_RESULTS/real-series" --mode online \
  --entity rwgao_b-brown-university --project ambi-inner-bench
```

One diagnostic run is one checkpoint, inner setting, scope, protocol, and
explicit attempt. Native W&B metrics use `diagnostic/actor_updates`,
`diagnostic/critic_updates`, and `diagnostic/round`; full data and HTML are in a
`return-diagnostics` artifact. Existing checkpoint-axis runs and workspace
layouts are not rewritten. Publication receipts prevent duplicate successful
uploads; an uncertain interrupted upload requires inspection before retrying.

Aggregation averages rollouts within solver, solvers within root, roots within
source episode, and episodes equally. Variability at each level and a 95%
episode-cluster percentile bootstrap interval (2,000 resamples) are retained.
One source episode has no interval. Missing rows/metrics remain explicit and
cannot be published as a complete panel. Five episodes from one backbone remain
an exploratory diagnostic, not independent evidence from 25 environments.

## Overhead and validation

For J5/H3 and 32 probes, six boundaries plus one cached prior probe add 672
model transitions (8.75% of the 7,680 learning transitions), 896 sampled policy
rows, and 224 ensemble-Q rows, with no optimizer updates. Small eager batches
and mode traversal mean wall-time overhead need not equal this percentage.

The full real panel executes 1,805,400 inner-branch simulator decisions and
100,300 cached prior decisions, plus 2,500 source-episode decisions. Prior
inference batches active continuations. Actor copies are processed one solve
at a time. This approximately 1.9-million-decision diagnostic runs separately
from ordinary episodes.

Timing records separate optimization, model probes, simulator stepping,
frozen policy/Q inference, snapshot serialization, root collection, warmup,
bundle/report generation, and publication. A completion receipt includes
final bundle persistence; the publication receipt includes SDK shutdown.
`--benchmark-repetitions 7` adds warmed, alternating probes-off/probes-on
measurements at the first root and checks their actions agree. The real panel
has its own separately measured runtime and work count. Zero repetitions
adds no timing solves.

Focused validation:

```bash
MUJOCO_GL=disable TORCHINDUCTOR_COMPILE_THREADS=1 "$AMBI_EVAL_PY" -m pytest -q \
  tests/test_ambi_togo_trace.py tests/test_ambi_calibration_cli.py \
  tests/test_ambi_real_calibration.py tests/test_ambi_diagnostic_series.py \
  tests/test_ambi_benchmark_evaluation.py tests/test_report_ambi_benchmark.py
AMBI_RUN_REAL_DMCONTROL_TESTS=1 MUJOCO_GL=disable \
  TORCHINDUCTOR_COMPILE_THREADS=1 "$AMBI_EVAL_PY" -m pytest -q \
  tests/test_ambi_real_calibration.py tests/test_ambi_calibration_cli.py
```

`MUJOCO_GL=disable` is useful for state-only tests on macOS hosts where GLFW
initialization aborts. It cannot validate RGB rendering. Use `MUJOCO_GL=egl`
on compatible NVIDIA execution hosts. CPU tests and tiny-model timings do not
establish reference-checkpoint wall time or CUDA readiness; run the same checks
and timing panel on the intended GPU before production submission.
