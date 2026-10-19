# Diagnosing inherited SAC critics before inner learning

`probe_ambi_gradients.py` tests whether the inherited critic or a differentiable
model rollout supplies useful local improvement directions. It complements the
completed full-episode H3/J8 evaluations. It does not rerun them, execute an
inner solve, train a model, or implement MAGE. All measurements are conditional
on a frozen checkpoint and its primary SAC continuation policy.

## Objective and comparisons

For a captured real state, encode `z = enc(s)` and fix the first action `a`.
Compare the decoded reward critic `Q(z,a)` with

```text
B_H(s,a) = sum(t=0..H-1) gamma^t r_model(z_t,a_t)
           + gamma^H Q(z_H,a_H)
a_0 = a;  a_t ~ frozen_primary_prior(z_t), t >= 1.
```

The probe differentiates Q, B_1, and B_H with respect to the first action.
Actor and model weights are frozen, but their input derivatives remain
connected. B_1 is the one-step differentiated Bellman target relevant to a
MAGE-style diagnostic. B_H tests the longer model objective; it is not MPPI's
optimization over an open-loop action sequence.
Both use the frozen online reward critic, not an EMA target; this is a
consistency diagnostic for the inherited value surface, not a reproduction of
the inner SAC target implementation.

For the current full-training auxiliary runs, Q is the **auxiliary reward
critic**, even though the primary SAC critic is soft. A reward-only primary
critic is also supported. Soft-only, split-value, separate return-actor,
episodic, and pixel checkpoints are rejected. No entropy term or Q scaling is
inserted into any return. The continuation is sampled, matching the critic's
policy-evaluation semantics, even though the root anchor is an actor mean.

Default `mean_all` removes critic-pair sampling noise and equals the expectation
of `mean_pair`. This is a deliberate diagnostic reduction, not an assertion
that the learning update uses all heads. `--q-reduction mean_pair` samples one
private pair per root and holds it fixed for every objective and perturbation;
the pair indices are recorded. Minimum reductions are also explicit options.

Each source root stores its exact simulator snapshot and the source prior's
pre-tanh mean `u`, where `a = tanh(u)`. To respect action bounds, interventions
use `tanh(u +/- epsilon*d)`, with d a unit vector in pre-tanh coordinates. The
critic/model directions are their action gradients multiplied by `(1-a^2)`,
then normalized. Reports retain gradients and norms in both coordinate systems,
cosines, actual action displacements, and skipped zero/saturated directions.
The finite differences therefore refer to u, not directly to a.

For every intervention, restore the identical simulator state. Execute the
fixed first action, then the checkpoint's frozen prior. Measure:

| Measurement | What it isolates |
| --- | --- |
| `critic_bias` | Root reward-Q minus measured continuing return |
| `model_return_bias` | Model H-step return with Q tail minus measured return |
| `model_error_with_common_endpoint_critic` | Model H-step return minus real H-step return using the same endpoint critic |
| `real_bootstrapped_return_bias` | Critic error at the real H-step endpoint |
| `*_gain` | Positive-direction intervention minus the unperturbed baseline |
| `*_central_difference` | Paired positive-minus-negative intervention, divided by `2*epsilon` |

Gradient estimation uses separate noise from intervention evaluation. All
interventions share evaluation noise, indexed by absolute decision and rollout.
At the real H-step endpoint the Q action is exactly the first Monte Carlo tail
action. There is no hidden terminal Q at the end of the Monte Carlo rollout.
The saved model predictions use the same policy noise as the real branches.

The simulator continues beyond the ordinary 500-decision timeout. With the
Humanoid decision-reward bound 2, the omitted return after H+T decisions is
bounded by `2*gamma^(H+T)/(1-gamma)`. Rows and contrasts record the corresponding
return, gain, and finite-difference tail bounds. An unexpected truncation fails
the run. A short smoke tail validates execution only; it cannot establish Q
calibration. The production default is T=1000; increase it when the bound is
material relative to the observed gain. Monte Carlo uncertainty is separate
from this deterministic truncation bound.

## Common roots across backbones

A saved bank pins its source checkpoint, environment, root observations,
integration-state checksums, and anchor actions. Loading it for another
backbone intentionally allows a different checkpoint hash. It requires the
same environment/wrappers, validates the bank and snapshot checksums, and keeps
the first actions identical. Each target still uses **its own** continuation
prior and critic; this is not a pure critic-swap experiment with a common
continuation policy.

Use both source banks to avoid diagnosing only the successful policy's state
distribution. The first pilot is four runs at 400k:

| Root/anchor source | Evaluated backbone |
| --- | --- |
| MPPI auxiliary run | MPPI auxiliary run |
| MPPI auxiliary run | SAC auxiliary run |
| SAC auxiliary run | SAC auxiliary run |
| SAC auxiliary run | MPPI auxiliary run |

The existing 400k full-episode result remains the behavioral reference. Repeat
the diagnostic at 150k to examine the onset of the training divergence after
the first pilot establishes which signal is informative. This protocol does
not claim that the 400k checkpoint diagnoses what caused the earlier failure.

Default pilot coverage is two source episodes (101,102), roots at decisions
0 and 100, four sampled continuations, 32 gradient samples, H=3, two epsilon
values (0.03,0.1), and one random control direction. With all gradients nonzero,
there are 17 candidate actions per root and at most **272,816 branch simulator
decisions per run**, plus 202 source-collection decisions when creating a bank.
This is a compute count, not a walltime estimate. Benchmark the actual device
before choosing scheduler resources. Use checkpoint/bank combinations as
independent jobs; preserve each bundle's complete root coverage.

```bash
AMBI_PROBE_PY=environments/dmcontrol/.venv/bin/python
AMBI_MPPI_CKPT=/absolute/path/to/mppi-aux-400k
AMBI_SAC_CKPT=/absolute/path/to/sac-aux-400k
AMBI_PROBE_RESULTS=/absolute/path/to/fresh-gradient-pilot

# Validate metadata and inspect the work count without loading weights or MuJoCo.
"$AMBI_PROBE_PY" probe_ambi_gradients.py --checkpoint "$AMBI_MPPI_CKPT" \
  --output "$AMBI_PROBE_RESULTS/mppi-on-mppi" --plan-only

"$AMBI_PROBE_PY" probe_ambi_gradients.py --checkpoint "$AMBI_MPPI_CKPT" \
  --output "$AMBI_PROBE_RESULTS/mppi-on-mppi" --device cuda
"$AMBI_PROBE_PY" probe_ambi_gradients.py --checkpoint "$AMBI_SAC_CKPT" \
  --root-bank "$AMBI_PROBE_RESULTS/mppi-on-mppi/root-bank.json" \
  --output "$AMBI_PROBE_RESULTS/sac-on-mppi" --device cuda

"$AMBI_PROBE_PY" probe_ambi_gradients.py --checkpoint "$AMBI_SAC_CKPT" \
  --output "$AMBI_PROBE_RESULTS/sac-on-sac" --device cuda
"$AMBI_PROBE_PY" probe_ambi_gradients.py --checkpoint "$AMBI_MPPI_CKPT" \
  --root-bank "$AMBI_PROBE_RESULTS/sac-on-sac/root-bank.json" \
  --output "$AMBI_PROBE_RESULTS/mppi-on-sac" --device cuda
```

Set `MUJOCO_GL` appropriately before Python. For a local state-only macOS
smoke, `MUJOCO_GL=disable` avoids creating a rendering context. Existing output
directories are never overwritten. `--seeds` and `--decisions` customize newly
collected banks; a reused bank defines its own coverage. Existing observation-
only evaluation banks lack the required physical snapshots/anchor actions.

Each bundle contains `plan.json`, checkpoint metadata, `root-bank.json`, and
one complete record per root under `roots/`. `result.json` appears only after
all roots succeed and the model, target networks, optimizers, temperature, and
auxiliary learner pass the existing frozen-state digest check. Source files
and checkpoints are hashed. Failures retain completed root records and a
`failure.json`; retry with a fresh directory. No W&B publication is implicit.

Summaries average sampled rollouts within each root, roots within each source
episode, then episodes equally. The reported standard error is across source
episodes. With only two episodes this is an exploratory screen; it is not a
training-seed confidence interval. Raw rollout pairs are retained for analysis.

## Interpreting the pilot and follow-up experiments

| Observation | Follow-up it motivates |
| --- | --- |
| Model-gradient steps improve real returns; critic-gradient steps do not | Test gradient supervision/MAGE on a frozen backbone, then verify inner optimization |
| Both model and critic directions fail in reality | Inspect model/endpoint errors, coverage, and continuation quality before adding gradient supervision |
| Q values are biased but directional steps work | Do not infer a gradient problem from value error alone |
| Improvement exists on MPPI roots but not SAC roots | Test state-distribution/coverage effects and an MPPI-to-SAC collection switch |
| Both inherited directions work, but adapted SAC still fails | Inspect post-update inner critics, entropy, data coverage, and actor update stability |

These are diagnostic implications, not causal conclusions. B_1 and B_H still
depend on the learned terminal critic, so their errors cannot automatically be
attributed solely to dynamics. Small action gradients also do not reproduce
SAC's entropy term or its network-parameter update across many replay states.

The **primacy/reset test** remains a separate training intervention: preserve
the same real replay, compare continued learning with critic+target+optimizer
reset, then actor+critic reset, while initially holding the representation
fixed. Weight-only checkpoints cannot reconstruct historical replay or exact
optimizer continuation. The **actor/planner mismatch test** likewise needs
inner-policy snapshots/behavior metadata and an explicit training intervention
for KL regularization or planner distillation. This probe does not claim to
implement either intervention or to measure gradients after the J8 solve.

Focused validation:

```bash
AMBI_RUN_REAL_DMCONTROL_TESTS=1 MUJOCO_GL=disable \
  environments/dmcontrol/.venv/bin/python -m pytest -q \
  tests/test_ambi_gradient_probe.py tests/test_ambi_real_calibration.py
```

## Oscar campaign launcher

`slurm/ambi_gradient_probe_campaign.py prepare` consumes an external inventory
with exactly the two 400k checkpoints above, their source run IDs, absolute
checkpoint/metadata paths, and SHA256 hashes. Pass `--inventory`, `--root`, and
`--expected-sha`. Preparation verifies the hashes and requires a clean checkout
at that commit; campaign outputs must be outside the checkout.

`slurm/run_ambi_gradient_probe_oscar.sbatch` takes `CAMPAIGN_ROOT`,
`EXPECTED_ACTION_MODES_SHA`, `PYTHON_BIN`, `STAGE`, and `CELL_INDEX` (or a Slurm
array index). Supply scheduler resources explicitly after inspecting live
capacity. The stages are:

- `banks`, indices 0/1: MPPI/SAC source banks, with the full pilot root coverage.
  H1/T1/R1 probes validate restoration cheaply. These returns are smoke results
  and are excluded from production summaries.
- `benchmark`, indices 0/1/2: CPU with one thread, CPU with two threads, and
  CUDA with one thread. Each uses the actual MPPI checkpoint at one root,
  H3/T100/R4 and the production direction/sample settings. Allocate at least
  two CPUs for the CPU2 cell and a GPU for CUDA.
- `production`, indices 0/1/2/3: the four combinations in the table above.
  Set `DEVICE` and `TORCH_THREADS` from the benchmark. Every cell reads a
  validated completed bank, so all four can run independently after bank
  creation. Choose concurrency and walltime from live limits and measurements.

Every stage writes an immutable receipt with source/input verification, complete
root and result checksums, elapsed time, child CPU time, and peak child memory.
The runner requires `srun` inside a Slurm allocation. None of these commands
submits jobs or publishes W&B results by itself.
