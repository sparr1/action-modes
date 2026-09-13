# SAC prior parameterization study

This Humanoid Walk state-observation study compares two log-standard-deviation
parameterizations and two entropy targets. It trains the stochastic outer prior
without inner optimization or MPPI. The four configurations each run seeds 55
and 56: eight backbones in one W&B group.

| Algorithm configuration | Log-std mapping | Bounds | Joint entropy target |
| --- | --- | --- | --- |
| [`ambi_prior_sac_clip_target21`](algs/ambi_prior_sac_clip_target21.json) | Direct clipping | [−10, 2] | −21 nats |
| [`ambi_prior_sac_clip_target10p5`](algs/ambi_prior_sac_clip_target10p5.json) | Direct clipping | [−10, 2] | −10.5 nats |
| [`ambi_prior_sac_smooth_target21`](algs/ambi_prior_sac_smooth_target21.json) | Smooth tanh mapping | [−10, 2] | −21 nats |
| [`ambi_prior_sac_smooth_target10p5`](algs/ambi_prior_sac_smooth_target10p5.json) | Smooth tanh mapping | [−10, 2] | −10.5 nats |

The [two-trial manifest](experiments/ambi_prior_sac_parameterization_study.json)
contains the complete matrix. Its base seed is 55; the training harness adds the
trial index, producing seeds 55 and 56. Each run lasts 2 million environment
decisions and retains every checkpoint at 25,000-decision intervals: 80 per run,
640 across the study. This is an exploratory two-seed comparison.

## Shared training contract

All cells use corrected post-tanh action entropy, automatic temperature
initialized at 1 with learning rate `3e-4`, and no actor Q normalization. The
actor uses the mean of a sampled pair of critics; the target uses the minimum
of a sampled pair. All five critics have distributional outputs and an
entropy-augmented Bellman target.

The shared backbone is model size 5 with 512-dimensional latent and MLP layers,
training horizon 3, batch size 256, UTD 1, temporal weight `rho=0.5`, discount
0.99, and target update coefficient 0.01. Explicit `seed_steps=2500` preserves
the existing inclusive action-selection boundary: decisions indexed 0 through
2500 use random actions, so there are 2,501 random decisions. The 2,500-update
pretraining burst begins after decision 2501. World-model, critic and
actor learning rates, optimizer epsilons, dropout, loss coefficients and the
remaining settings are explicit in each algorithm file. Environment behavior
remains the existing adapter contract: action repeat 2 with summed rewards,
500-decision episode cutoff, and timeout as truncation.

Native random network initialization is preserved. There is no fixed initial
standard deviation or specially initialized std-output head. For the same seed,
the four cells begin with matching raw network weights. Their initial policy
distributions differ across mappings: this is a comparison of native
parameterization recipes, including their initial exploration. At a hypothetical
raw std-head output of zero, clipping gives sigma 1 and smooth mapping gives
sigma `exp(-4)`. These are analytical reference values, not measurements of an
initialized network.

Algorithm files contain the complete `alg_params` mapping. The manifest avoids
partial nested overrides: the harness replaces `overrides_alg.alg_params`
wholesale rather than deep-merging it.

## Diagnostic settings and measurement sources

Diagnostics are opt-in and currently require state observations and prior-only
collection (`inner_operator=none`, `mpc=false`). W&B diagnostics also require
event-indexed transport. The new study enables these settings:

| Configuration key | Value | Meaning |
| --- | ---: | --- |
| `outer_policy_diagnostics` | `true` | Enable outer-policy telemetry |
| `outer_policy_diagnostics_early_every` | 100 | Completed-update cadence during early learning |
| `outer_policy_diagnostics_early_until` | 10,000 | Inclusive early-cadence boundary |
| `outer_policy_diagnostics_every` | 1,000 | Completed-update cadence afterward |
| `outer_policy_diagnostics_states` | 32 | Fixed observation-bank size |
| `outer_policy_diagnostics_samples` | 32 | Policy samples per bank observation |
| `outer_policy_diagnostics_seed` | 12,345 | Private policy-noise seed |
| `wandb_event_indexed` | `true` | Retain multiple events at the same environment step |

Initial and pretraining-boundary observations distinguish the untouched actor,
the actor before its first learner update, and the actor after the pretraining
burst. Phase labels are `initialization`, `pretrain_before`, `pretraining`,
`pretrain_after`, `training`, and `final`. Subsequent measurements use actual
completed-update counts, including
updates within that burst. Each event records its phase and environment step;
the actor snapshot timing is explicit. Learner packets describe the policy
sample **before** its actor optimizer step and include before/after update
counts and before/after alpha. Fixed-bank probes describe the policy at the
reported completed-update count. Do not infer snapshot timing solely from a
graph's horizontal coordinate.

Measurement sources distinguish the initial observation and the completed
reference bank, as well as learner and executed-action telemetry:

- **Learner samples** (`learner`): summarize the samples already used by the actor update.
  Report depth zero separately from pooled actor-training depths. This avoids
  inserting a second learner-policy sample into the optimization RNG stream.
- **Fixed observations** (`reference_bank`): collect up to 32 evenly spaced observations from the
  seeded random warmup, retain their values, and re-encode them with the current
  encoder at each probe. The observations are fixed; their latent encodings are
  allowed to change as the backbone learns. Reuse 32 private Gaussian noise
  samples per observation, giving 1,024 policy sample rows for a full bank.
  The `initial_observation` probe uses one observation with 32 samples.
- **Executed actions** (`executed_uniform` / `executed_prior`): accumulate actual normalized control coordinates during
  environment collection. Random warmup actions and actions from the learned
  prior have separate sources and denominators.

Fixed-bank inference preserves module training modes and Python, NumPy and
PyTorch RNG state, including exception cleanup. It uses evaluation mode and
adds no optimizer update, environment transition, model rollout or critic
evaluation. Learner summaries operate on existing tensors. Their extra cost
is measurement, small policy/encoder probes, transfer, serialization and
publication; it must be measured rather than inferred from sample counts.
On CUDA, host timing around device-to-host transfers can include outstanding
learner kernels. Component timings therefore describe host accounting, not an
exact separation of GPU execution time; compare total elapsed time on the same
GPU runtime before estimating production overhead.

## Reading the metrics

Action saturation is reported both near the bounds (`abs(action) >= 0.99`) and
at exactly −1 or +1. Fractions are over action coordinates, not the fraction of
whole decisions having any saturated coordinate. Mean-action statistics refer
to `tanh(mu)`, the usual deterministic control action; they are not the
expectation of the squashed stochastic action. Separate sampled-action metrics
capture the effect of policy noise.

Pre-tanh mean magnitudes, log-std and sigma statistics describe the Gaussian
before action squashing. Sigma is `exp(log_std)`, not the standard deviation of
the bounded action. Log-std is a natural logarithm. Histograms retain 24
equal-width bins over the configured interval, counts and edges. Exact-bound
occupancy and occupancy within 0.1 log units of either bound distinguish hard
clipping from approaches toward a smooth bound.

Entropy is the joint post-tanh Monte Carlo estimate `-log pi(a|s)` in nats,
summed across all 21 action dimensions. It may be negative. With training
horizon 3, actor updates include four latent depths, 0 through 3. Their relative
weights are `1, 0.5, 0.25, 0.125`:

- `entropy_rho_mean` divides the weighted entropy sum by 1.875. This is the
  state mixture used by automatic temperature tuning.
- `entropy_unweighted_mean` averages the four depths equally. The established
  `train/actor_entropy` also uses this unweighted average.
- `entropy_actor_loss_weighted` divides the weighted sum by 4, matching the
  actor objective's temporal normalization. This is not an entropy target.

`entropy_shortfall` is target minus measured entropy. A positive value means
entropy is below the requested target, which pushes the automatic temperature
up. The temperature loss alone is insufficient to diagnose this: it can be
zero at initial `log(alpha)=0` while its gradient is nonzero. Learner diagnostics
therefore expose the residual, alpha before/after, entropy contribution,
critic values and actor gradient norm.

Q values are decoded, unnormalized **soft values**: their targets include
entropy. They are not reward-only Monte Carlo returns. Multiplying entropy by
alpha gives its contribution in the same value units. Removing actor Q
normalization does not remove the distributional critic representation or
entropy from the Bellman target.

## W&B and local artifacts

Each trained backbone has its own W&B run in project `ambi`, under group
`ambi-prior-sac-parameterization-20260913`. Configuration files deliberately
omit `wandb_run_name`: the existing fallback produces
`AMBITDMPC2-<configuration-basename>-seed55` or `...-seed56`. Eight distinct
names and resolved seed metadata are tested.

Ordinary training curves retain the `env_step` axis. Diagnostics use
`outer_diag/updates_completed`, with source and phase recorded explicitly.
W&B's internal history step is a monotonic event index so the 2,500-update
pretraining burst can produce multiple retained records at the same
environment step. The configuration does not silently replace an existing
checkpoint-axis evaluation series.

Each run's diagnostic directory contains:

- `events.jsonl`: append-only event records, including scalar summaries,
  histogram counts/edges, source, phase, environment step, and learner counts.
- `reference.json`: warmup observation indices, stored observations and the
  policy-noise bank used to pair repeated probes.
- `manifest.json`: schema, completion status, resolved configuration, reference
  SHA-256, row count, units and timing fields.

The complete directory is published as an `outer-policy-diagnostics` W&B
artifact. The local bundle remains available when W&B is disabled. Failed or
interrupted runs must be interpreted using their completion status rather than
treated as a complete panel. Collection, serialization and publication timings
are separate; publication timing measures enqueue calls, not confirmed remote
upload completion. Histogram arrays and denominators are retained in the
local/artifact JSON; W&B history contains scalar diagnostic curves.

The adjacent `<bundle-name>_publication.json` receipt records artifact enqueue
time, the final component timing counters, status, and the manifest hash without
changing the captured bundle. A `queued` receipt means the SDK accepted the
artifact, not that remote upload was verified. Ordinary runs also report
`outer_diag/time/artifact_enqueue_seconds`. Resumable runs attach only after the
final durable checkpoint and metric-journal publication; attachment prints its
timing and writes the receipt without adding an uncommitted history event.

Checkpoint sidecars retain the full input diagnostic settings under
`trial_run_params.alg_params`, with the resolved active diagnostic settings also
in `trial_run_params.resolved_runtime.outer_policy_diagnostics`. W&B and the
diagnostic manifest additionally retain the full resolved configuration.
Portable model weights are not a substitute
for the diagnostic bundle or an exact mid-training resume.

## Local CPU timing measurement

On 2026-09-13, a bounded synthetic-update microbenchmark ran on macOS 26 ARM64
with the locked PyTorch 2.3.1 environment, one CPU thread and compilation off.
It used 67 observation coordinates, 21 action coordinates, H3/B256, five
101-bin critics and dropout 0.01. Encoder, MLP and latent dimensions were each
reduced to 64. The actor used clipping [−10, 2], target −21, automatic alpha and
soft critic targets. This fixture is smaller than the production backbone.

After 20 warmup updates, each condition started from identical model and Adam
state and used the same synthetic batch and RNG seed for 300 updates. Five
repetitions rotated condition order. Diagnostics ran at updates 100, 200 and
300. Timings include local JSONL event serialization, but exclude construction,
initialization probes, environment interaction, replay sampling and W&B.

| Condition | Seconds per 300 updates, mean ± sample SD | Mean paired change from disabled |
| --- | ---: | ---: |
| Disabled | 15.2458 ± 0.0594 | — |
| Learner summaries only | 15.2583 ± 0.0661 | +0.083% |
| Learner summaries + fixed bank, 32 states × 32 samples | 15.3203 ± 0.0649 | +0.489% |

The paired changes varied substantially relative to these small means:
learner-only differences ranged from −0.549% to +1.009%, and learner-plus-bank
differences ranged from −0.015% to +1.044%. Sample SDs of the paired changes
were 0.570 and 0.401 percentage points respectively; these are variability
statistics, not confidence intervals. Negative repetitions reflect timing
noise, not a diagnostic speedup.

Direct recorder timings averaged 0.00331 seconds of collection plus 0.000974
seconds of serialization for the three learner events, and 0.01337 seconds of
collection plus 0.001540 seconds of serialization for the three learner events
and three fixed-bank events. These component timers do not capture every
indirect runtime effect and should not replace the wall-time comparison.
Learner summaries add no policy, encoder, dynamics or Q forward. The full-bank
condition adds three encoder calls (96 state rows) and three policy calls
(3,072 sampled rows), with no extra dynamics/Q call or optimizer update.
Final model parameters were checked for exact equality across every condition
and repetition.

Reproduce the bounded measurement from the repository root:

```bash
environments/dmcontrol/.venv/bin/python -m tests.benchmark_ambi_outer_policy_diagnostics \
  --updates 300 --repeats 5 --warmup 20 \
  --output /tmp/ambi-outer-policy-cpu-timing.json
```

The output retains every repetition, the fixture and component timings. This
measurement does not establish production overhead or GPU readiness: CUDA was
unavailable, the networks were reduced, and neither GPU compilation nor live
publication was exercised. In CUDA runs, transfer synchronization may include
previously queued optimizer work, so component attribution is approximate;
measure marginal wall time with a compatible diagnostics-off/on benchmark.

## Validation and execution boundary

The focused configuration checks run without MuJoCo or live W&B:

```bash
environments/dmcontrol/.venv/bin/python -m pytest -q \
  tests/test_ambi_prior_sac_study_configs.py \
  tests/test_td_ambi_study_configs.py
```

They cover the four-cell Cartesian product, shared settings, both seeds,
checkpoint counts, W&B identities, and native initialization. Diagnostic tests
separately validate reductions, timing labels, RNG/mode preservation, local
serialization and publication behavior.

The broader 21-file regression batch reported **437 passed, 8 skipped** on
2026-09-13: seven skips require CUDA (including strict compiled updates and
publication RNG isolation); one requires the opt-in real DMControl test runtime.
It included the new configuration, learner, recorder, harness, resume and W&B
tests plus existing entropy, checkpoint, configuration and adapter tests.
Its assertions passed, but macOS subprocess abort traces occurred around
MuJoCo import/subprocess creation and the multiprocessing resource tracker.
The runtime limitation below therefore still applies. The separate legacy
`tests/test_wandb_logging.py` could not be collected in this locked environment
because it imports `stable_baselines3`, which is not installed; no dependencies
were changed to force that cross-environment test to run.

The subsequently extended `tests/test_ambi_outer_diagnostics_resume.py` passed
all three cases, including artifact-publication failure after the final durable
checkpoint followed by recovery at the same target. Recovery performs no extra
reset, environment step, replay draw or optimizer update and preserves the
saved diagnostic rows, observation/noise bank, learner state and RNG. Together
these focused batches exercised 438 distinct passing tests. No live W&B service
or cluster job was used for validation.

On 2026-09-13, this local macOS test batch used the locked
`environments/dmcontrol/.venv/bin/python` interpreter:

```bash
environments/dmcontrol/.venv/bin/python -m pytest -q \
  tests/test_outer_policy_diagnostic_recorder.py \
  tests/test_ambi_outer_diagnostics_integration.py \
  tests/test_wandb_event_indexed.py \
  tests/test_wandb_utils.py \
  tests/test_wandb_resume.py
```

Pytest reported **105 passed, 1 CUDA skip**, and its main process exited with
code 0. However, a multiprocessing resource-tracker child emitted
`Fatal Python error: Aborted` while TorchRL allocated replay storage in
`test_recording_preserves_complete_harness_and_exposes_pretraining`. The stack
included `multiprocessing.util.spawnv_passfds` and
`resource_tracker.ensure_running`; a later warning stated that the tracker
process died unexpectedly and was relaunched, with possible resource leaks.
This is an observed subprocess-runtime failure despite the passing assertions,
not a clean validation of that macOS execution path. A subsequent batch of the
four recorder/W&B files above, excluding the harness integration file, completed
with **92 passed, 1 CUDA skip** and no child abort; it does not resolve the
replay subprocess limitation.

A compatible CUDA smoke and measured wall-time comparison are separate from
these CPU checks and must precede production GPU claims. Creating the configs,
documentation and tests does not submit training or publish a live run. Cluster
smoke, training submission and live publication remain separate authorized
execution steps.

## Hydra launch and CUDA gate

`slurm/run_ambi_prior_sac_study_hydra.sbatch` maps array tasks 0–7 to
`algorithm_index = task // 2`, `trial_index = task % 2`. Each cell requests one
L40S, eight CPUs and 32 GiB RAM. There is no array throttle; check current
account limits and capacity before submission. Reserve at least 48 GiB for the
640 checkpoints plus diagnostics. Compilation and W&B staging use job-private
node-local storage. Checkpoints and scheduler logs belong outside the checkout.

The launcher requires `AMBI_EXPECTED_COMMIT` (the exact full Git SHA),
`AMBI_STUDY_OUTPUT_ROOT` (absolute output directory), and a clean source tree.
`AMBI_DMCONTROL_PYTHON` can select an existing interpreter synchronized with the
identical DMControl lock. Requeue and duplicate output directories are rejected.
Run `slurm/validate_ambi_prior_sac_hydra.sbatch` on an allocated GPU before the
production array; it uses fake/offline W&B and includes the production-shape
real-Humanoid gate plus compiled diagnostic invariance tests.

The real-environment gate keeps the approved model size, H3/B256, five critics,
and production warmup settings, but calls only two direct fixture updates per
mapping. Its 32-observation test bank is collected in 32 real decisions rather
than executing production pretraining. It reports compilation fallback flags
explicitly. This bounded validation is distinct from a full training run.
