# TD-AMBI actor and critic objective study

This study contains four prior-only training configurations and six full AMBI
training configurations derived from `algs/TD-AMBI.json`. Every configuration
uses Humanoid Walk state observations, seed 55, and 2 million decisions. These
are single-seed exploratory comparisons. They use the trainable
`AMBITDMPC2/AMBITDMPC2` implementation; they are not frozen-checkpoint native
TDAMBI evaluations.

## Study matrix

All filenames below have the prefix `td_ambi_` and suffix `.json`, with an
algorithm file in `algs/` and a matching individual manifest in `experiments/`.

| Family | Prior-only configuration | Full AMBI configuration(s) | Critic target | Outer temperature | Inner temperature | Actor Q normalization |
| --- | --- | --- | --- | --- | --- | --- |
| Reward / Q scale | `prior_reward_qscale` | `full_reward_qscale_frozen`, `full_reward_qscale_adaptive` | Reward only | Fixed `0.0001` | Inherit outer; fixed during each solve | Percentile scale |
| Entropy / Q scale | `prior_entropy_qscale` | `full_entropy_qscale_frozen`, `full_entropy_qscale_adaptive` | Entropy augmented | Fixed `0.0001` | Inherit outer; fixed during each solve | Percentile scale |
| Entropy / auto temperature | `prior_entropy_autotemp` | `full_entropy_autotemp` | Entropy augmented | Learned; initially `0.0001` | Learned; initialized from outer each solve | Off |
| Reward / auto temperature | `prior_reward_autotemp` | `full_reward_autotemp` | Reward only | Learned; initially `0.0001` | Inherit outer; fixed during each solve | Off |

`frozen` and `adaptive` refer only to the inner Q normalization scale. In both
cases, each solve starts from the current learned outer scale. Frozen uses
`inner_actor_loss_scale_update="per_action"` and holds that value throughout
the solve. Adaptive uses `"per_update"` and updates a private inner scale before
each actor loss with percentile EMA rate `0.01`. Neither mode writes its inner
scale back to the outer learner. The outer scale continues adapting during
training in both variants.

## Entropy and value units

All active actors use `tdmpc2_scaled` entropy, the TD-MPC2 tanh log-standard-
deviation mapping, and bounds `[-10, 2]`. Write this entropy statistic as
`H_scaled`, the entropy coefficient as `alpha`, and the positive Q percentile
scale as `S`. The actor minimizes the negative of:

```text
Q / S + alpha * H_scaled     when Q normalization is enabled
Q     + alpha * H_scaled     when Q normalization is disabled
```

The scale divides only the actor's Q term. The critics continue predicting
returns in raw reward units. The explicitly chosen entropy-augmented critic
convention therefore adds `alpha * S * H_scaled` to the next-state bootstrap
when Q normalization is enabled, and `alpha * H_scaled` when it is disabled:

```text
reward-only target:         r + gamma * continuation * Q_target
entropy target, Q scaled:   r + gamma * continuation * (Q_target + alpha * S * H_scaled)
entropy target, unscaled:   r + gamma * continuation * (Q_target + alpha * H_scaled)
```

Here `continuation` accounts for true termination, and the entropy statistic
and target Q use the sampled next action. The critic update uses the currently
available scale; an adaptive inner actor then updates its scale for its own
loss and subsequent update slots.

The user-selected target is explicitly **`-441` wherever temperature is
learned**, in the units of `tdmpc2_scaled` entropy. Humanoid has 21 action
dimensions, so this equals `-21²`. It is a chosen tuning value; it is not a
claim that every old squashed-entropy target has an exact constant conversion.
Both learned outer temperatures start at `0.0001` through
`ent_coef="auto_0.0001"`. Only `full_entropy_autotemp` learns an inner
temperature, initializing it from the current outer value and targeting
`-441`. Its inner temperature gradients are unclipped
(`inner_temperature_grad_clip_norm=null`), matching the outer temperature
update. `full_reward_autotemp` keeps the inherited inner temperature fixed.

## Full AMBI update protocol

All six full configurations use `J=6`, `N=512`, `H=3`, inner minibatch size
512, and replay capacity 9,216. After each batched imagined depth step,
512 new transitions trigger one paired critic-then-actor update. Target
interpolation follows that pair. The entropy/auto-temperature variant also
updates its temperature after the actor. Each solve therefore generates
9,216 transitions and performs **18 critic updates and 18 actor updates**;
the entropy/auto-temperature variant additionally performs 18 temperature
updates. Replay retains the complete solve and samples accumulated transitions
with replacement.

The actor, critic, target critic, and optimizer state are private to each real
decision. Actor and critic weights are cloned from the outer learner; the
inner target starts from the outer target critic. Adam moments start fresh.
The world model is frozen during inner optimization. Inner losses average
transition minibatches, as requested, and do not apply sequence-level `rho`
weighting. Outer sequence losses retain horizon 3 and `rho=0.5`.

The matched actor and critic settings include learning rates `0.0003`, actor
Adam epsilon `1e-5`, critic Adam epsilon `1e-8`, critic loss coefficient `0.1`,
actor/critic gradient clipping at 20, target interpolation rate `0.01`, and
target updates every step. The distributional critic has five heads, 101
symlog bins spanning `[-10, 10]`, mean-of-two actor Q reduction, and
minimum-of-two target Q reduction. The model-size-5 backbone, outer batch 256,
UTD 1, and one-million-transition outer replay are inherited from TD-AMBI.
Additional exploration mechanisms, writeback, policy regularizers, controller
evaluation, and value diagnostics are disabled.

## Prior collection and checkpoints

All four prior configurations set `inner_operator="none"` and `mpc=false`.
After the standard random warmup they collect actions from the stochastic
outer policy. Inactive inner-only settings are neutralized so they pass the
same validation as other no-inner configurations; no inner updates run.

All ten configurations retain every model checkpoint at 25,000-decision
intervals (`checkpoint_every=25000`, `save_strat="all"`), producing 80
scheduled checkpoints per completed 2-million-decision run. Checkpoints
include the learned outer Q scale when Q normalization is enabled. Keep each
checkpoint with its adjacent metadata sidecar. The cadence and retention are
pinned in both the algorithm files and manifests because algorithm-level
checkpoint fields take precedence in the training harness. Separate final
trial saves are disabled; the 2-million-decision checkpoint already belongs
to the periodic schedule.

The intended cluster split is **four prior-only runs on Oscar** and **six
full AMBI runs on Hydra**. A completed campaign therefore retains 320
scheduled checkpoints on Oscar and 480 on Hydra, for 800 checkpoints total,
plus their metadata sidecars. This split is a workload plan; it does not
represent a launch or completion status.

## Manifests

`experiments/td_ambi_prior_study.json` contains the four prior configurations;
`experiments/td_ambi_full_study.json` contains the six full configurations.
Each individual manifest runs one configuration. All manifests use timestamped
output directories and preserve the complete algorithm parameter mapping;
they do not supply a partial `overrides_alg.alg_params` replacement.

The standard training entry point accepts either an individual manifest or a
suite manifest:

```bash
environments/dmcontrol/.venv/bin/python main.py \
  --run configs/dmcontrol/experiments/td_ambi_prior_study.json \
  --alg-dir configs/dmcontrol/algs

environments/dmcontrol/.venv/bin/python main.py \
  --run configs/dmcontrol/experiments/td_ambi_full_study.json \
  --alg-dir configs/dmcontrol/algs
```

These commands start the complete training workloads. Run authorized cluster
training through the scheduler using the tested repository commit and the
locked DMControl environment. Configuration validation lives in
`tests/test_td_ambi_study_configs.py`.

## Correctness checks and comparison scope

The reference is official TD-MPC2 commit
`8bbc14ebabdb32ea7ada5c801dc525d0dc73bafe`. The reward-only, Q-scaled baseline
matches its actor/critic equations and update order. The other families apply
the changes in the study matrix; inner training retains transition minibatches
and fresh per-decision optimizers, as described above.

The tests check numerical behavior as well as configuration fields:

| Contract | Independent check |
| --- | --- |
| Complete outer update | `tests/test_td_ambi_upstream_review.py` transcribes the upstream policy, dense two-hot CE, Q decoder, percentile calculation and temporal reductions. It compares gradients, Adam state, target interpolation and RNG consumption across consecutive updates, including active dropout and clipping. |
| Inner objective | `tests/test_td_ambi_inner_objectives.py` and `tests/test_ambi_scaled_critic_entropy.py` compare losses, gradients and optimizer updates against explicit formulas, including saturation, entropy target units and the scale used before each actor EMA. |
| Numerical boundaries and temperature | `tests/test_td_ambi_numerical_review.py` checks analytic entropy at 21 actions, distributional support boundaries, and 18 temperature updates against scalar Adam equations. |
| Actual experiment execution | `tests/test_td_ambi_study_execution.py` resolves the suite manifests through `main.py`, executes all six full schedules for two decisions, checks prior-only routing, and verifies immutable asynchronous checkpoints. |
| Training continuation | `tests/test_td_ambi_checkpoint_review.py` saves and reloads trained learners, then compares their next update and complete resulting state under identical RNG. |

Runtime tests narrow hidden layers for CPU while retaining 21 actions and the
full inner population, batch size, horizon and rounds. They do not establish
CUDA/Inductor parity or long-run training performance. Sparse CE and
`log1p`/`expm1` preserve the equations with small floating-point differences
from upstream, so this is not a bitwise-equivalence claim. The literal
TD-MPC2 entropy ratio also retains upstream's sensitivity when its log-density
denominator approaches zero; `tests/test_tdmpc2_scaled_entropy.py` explicitly
checks that behavior rather than silently cancelling the ratio.
