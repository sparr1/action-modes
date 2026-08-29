# TD-MPC2 core

This package vendors the algorithmic core from the official TD-MPC2 implementation
at upstream commit `8bbc14ebabdb32ea7ada5c801dc525d0dc73bafe`.

Copied/adapted here:

- `agent.py`
- `common/buffer.py`
- `common/init.py`
- `common/layers.py`
- `common/math.py`
- `common/scale.py`
- `common/world_model.py`

Intentionally not copied:

- Hydra training entrypoint
- official trainer loop
- official logger / W&B setup

AMBI should construct the environment and own the top-level training harness. A `RL/TDMPC2.py` wrapper should import this package and implement AMBI-compatible `learn`, `predict`, `save`, and `load` methods.

## Compatibility scope

The port preserves the single-task TD-MPC2 planning and training equations for
both state and RGB observations. The project-level `DMControl-v0` adapter
recreates the upstream single-task environment order: normalized actions,
two-step action repeat with reward summation, an optional three-frame 64x64 RGB
stack, and a 500-decision timeout. State is the default; RGB explicitly replaces
state when requested. The compatibility layer deliberately differs in these
ways:

- critics use ordinary `ModuleList` ensembles instead of TensorDict-vectorized
  modules;
- CPU and explicit device selection are supported;
- arbitrary finite Box action bounds are normalized to `[-1, 1]` by the wrapper;
- `torch.compile` is not supported by the compatibility ensemble and is rejected;
- the environment boundary follows Gymnasium's NumPy/five-value API rather than
  upstream's Torch/four-value wrapper API;
- RGB is restricted to the upstream-compatible `(9, 64, 64)` `uint8` layout and
  requires `latent_dim == 16 * num_channels` (the model-size-5 default is 512);
- current official vectorized-critic checkpoints are converted on load.

Replay keeps complete overlapping RGB frame stacks as `uint8`, matching
upstream behavior. Random-shift augmentation remains active during acting and
evaluation. AMBI isolates the acting-time crop in its private RNG stream, but
does not alter the learned latent inner-loop algorithm. Checkpoints record an
observation signature so state/RGB mismatches fail before any live model state
is changed; legacy and official checkpoints without that metadata remain
loadable through strict architecture preflight.

`common/soft_world_model.py` and `ambi_agent.py` are AMBI extensions, not
upstream TD-MPC2. They retain the encoder, latent dynamics, reward model,
SimNorm representation, and multi-step consistency/reward training, but replace
TD-MPC2's policy prior with a squashed-Gaussian, entropy-regularized SAC actor.
Core AMBI learns persistent actor and critic control priors, fully clones them
into a fresh root-local learner at every real decision, trains that learner on
imagined TOLD transitions, and acts with the adapted actor. The reference model
uses TD-MPC2's model-size-driven distributional ensemble (five Q heads at model
size 5), inner SAC, and a learned action-local entropy temperature initialized
from the current outer temperature. Scalar twin critics, LoRA, TD3, no inner
improvement, and persistent inner scopes are explicit ablations. The MPPI inner
operator is a compute-matched TD-MPC-style comparator, not AMBI's planner.

### Optional adapted-prior writeback

The reference behavior keeps the outer control priors immutable during action
selection and discards the adapted action-local learner. The writeback ablation
allows the final action-local actor and online critic to update their matching
outer priors after a planned training action. For either component with outer
parameters `w`, final to-go parameters `w_togo`, and configured coefficient
`c`, the update is

```text
w <- (1 - c) * w + c * w_togo.
```

`inner_actor_writeback_coef` and `inner_critic_writeback_coef` are independent,
strict finite numeric values in `[0, 1]`. Both default to `0.0`, preserving the
fresh-prior AMBI path exactly. Setting either value to `1.0` is a hard
replacement; intermediate values apply a Polyak interpolation. Each nonzero
coefficient requires the canonical J/N/H/G inner schedule, inner SAC, `clone`
adaptation, action-local parameters, and at least one optimizer update for its
corresponding component. Any active writeback configuration also requires
identical inner and outer log-standard-deviation mappings and bounds, keeping
the full cloned learner in the same policy family as its persistent actor
prior.

Writeback is explicitly authorized only by the training collection path. Seed
actions, direct `predict` and rendering calls, online evaluation, and frozen
checkpoint evaluation cannot write back. Evaluation still performs its normal
root-local optimizer steps; `eval_mode` controls the returned action, not inner
learning. This separate gate prevents step-zero or periodic evaluation from
silently changing the subsequent training prior.

Within an authorized action, AMBI first finishes local updates, produces the
environment action, and computes diagnostics against the unchanged outer
prior. It then applies writeback immediately before tearing down the
action-local workspace. Consequently, diagnostics such as
`inner_final_outer_policy_kl` retain their pre-writeback meaning. Because the
destination is the outer prior, the result survives episode resets and is
included in later model checkpoints; optimizer, replay, temperature, and other
inner lifetimes remain action-local.

Both resolved coefficients are recorded in portable runtime metadata and in
the exact-resume scientific identity. For lineage purposes, omitted defaults,
integer zeros, and explicit `0.0` values canonicalize to the same disabled
setting.

The interpolation is in-place. It preserves the outer `Parameter` identities
and the existing outer Adam moments, optimizer counters, entropy coefficient,
and learner counters. Actor writeback targets only the outer policy. Critic
writeback targets only the online outer critic; it does not copy the local
target critic. The outer target critic remains untouched by writeback and
continues to follow the online critic through the configured ordinary EMA.
Neither the local entropy temperature nor any target-network state is written
back.

When at least one coefficient is nonzero, action metrics include
`inner_actor_writeback_coef` and `inner_critic_writeback_coef`, together with
the binary `inner_actor_writeback_applied` and
`inner_critic_writeback_applied`. These are scalar bookkeeping metrics only;
they add no imagined rollouts, policy or critic forwards, or optimizer steps.

AMBI's actor log-standard-deviation transform is configurable independently of
its bounds. `log_std_mapping="direct_clamp"` preserves the SAC-style default by
clamping the actor head directly to `log_std_min`/`log_std_max` (default
`[-20, 2]`). `log_std_mapping="tdmpc2_tanh"` uses TD-MPC2's smooth transform,
`low + 0.5 * (high - low) * (tanh(raw) + 1)`. Exact upstream mapping and bounds
are selected with `log_std_mapping="tdmpc2_tanh"`, `log_std_min=-10`, and
`log_std_max=2`. The mapping never changes the bounds implicitly. The inner
actor inherits these settings when `inner_log_std_mapping`,
`inner_log_std_min`, and `inner_log_std_max` are `null`; each can be overridden
independently for an ablation.

Policy optimization and critic evaluation can be ablated independently.
`outer_critic_target` and `inner_sac_critic_target` each accept
`"entropy_augmented"` (the default, which bootstraps with
`Q - alpha * log_pi`) or `"reward_only"` (which bootstraps with `Q`). The
reward-only setting changes only the corresponding Bellman target; it does not
otherwise change actor sampling, the `alpha * log_pi - Q` actor loss, or the
configured temperature behavior. To mirror TD-MPC2's reward-return critic in
both places, set both fields to `"reward_only"`. Ensemble selection remains a
separate control: `min_all` and `mean_all` use every Q head, while the `*_pair`
modes use `q_pair_size` sampled heads.
Outer and inner actor updates also report the configured-reduction Q alongside
same-forward `actor_q_mean_all`, `actor_q_min_all`, and
`actor_q_mean_all_minus_min_all` diagnostics. These metrics are observational;
the configured actor reduction remains the only Q signal used by optimization.

Action-time SAC diagnostics also report `inner_final_outer_policy_kl` (published
to W&B as `train/inner_final_outer_policy_kl`), the
closed-form `KL(final adapted actor || current outer policy prior)` at the
encoded real-decision root after all root-local updates. It uses the actors'
diagonal pre-tanh Gaussian parameters and the same numerically stabilized
closed-form helper as the KL regularizer. It is evaluated under `no_grad` and
never contributes to the actor loss. It follows `inner_diagnostics_every` and
is absent on unsampled actions; maintained AMBI presets sample it every 1,000
real environment steps.
This final observational metric is distinct from `inner_outer_policy_kl`, the
intermediate actor-update statistic associated with a positive
`inner_outer_policy_kl_coef`; when that regularizer is disabled, its update-time
metric is omitted rather than reported as a false zero.

### Optional replay behavior-policy regularizer

`outer_behavior_policy_kl_schedule` optionally enables a replayed
behavior-policy regularizer. `outer_behavior_policy_objective="reverse_kl"`
(the default) preserves the analytic reverse KL from the current outer actor to
the action-generating replay policy. Both are represented by diagonal pre-tanh
Gaussians, so their shared invertible tanh leaves this KL unchanged.

`outer_behavior_policy_objective="action_space_cross_entropy"` instead uses
the exact normalized squashed-action cross entropy. Its Gaussian expectation
is analytic, while the exact tanh log-Jacobian is evaluated stably on the same
reparameterized pre-tanh sample already used by the SAC actor and critic. This
partially analytic single-sample estimator has the exact CE objective and an
unbiased gradient. Cross entropy can be negative and is coordinate dependent;
ordinary SAC entropy remains active and separate.

For either objective, the replayed component is an empirical Jensen upper-bound
surrogate for the unavailable historical policy mixture. The loss uses only the
`H` actor states with corresponding actions, divides by action dimension, and
renormalizes over valid rows; seed and random actions are invalid rather than
zero-valued targets. Reverse KL includes its current-policy log-density term and
is intentionally distinct from the released TD-M(PC)² sampled `-log mu(a)`
regularizer.

The default `"none"` preserves the legacy replay and checkpoint contracts.
The active choices are `"smooth"` (a readiness-paused smoothstep ramp to
`outer_behavior_policy_kl_coef`), `"quantile_gate"` (coefficient active only
while the just-updated P95-P5 Q-range EMA is strictly above its threshold), and
`"dual"` (a separately optimized log coefficient targeting
`outer_behavior_policy_kl_target`). If actor-loss scaling is enabled, the
entire raw SAC-plus-regularizer objective is divided by the shared Q-range
scale. The CE objective supports `"smooth"` and `"quantile_gate"` but rejects
`"dual"`, because CE has no invariant zero-valued constraint target. Active
modes require stochastic inner SAC execution. Replay and agent states move to
versions 2 and 5/6 respectively only while this feature is active; exact
checkpoint metadata rejects cross-objective continuation.

Learned outer and inner SAC entropy coefficients have a numerical floor of
`1e-8`. Fixed entropy coefficients retain their configured values. The floor
prevents an underflowed outer coefficient from becoming an invalid zero when a
fresh inner SAC solve inherits it.

### Value-equivalence live monitor

AMBI includes an optional, observational value-equivalence monitor for inner
SAC. It has **fresh-prior semantics**: each sampled outer replay update evaluates
the Bellman targets seen from the current outer actor, critic, and temperature
initialization, before any root-local inner improvement. It therefore measures
whether the current TOLD predictions preserve the TD targets that a newly
initialized inner learner would receive; it is not a measurement of the critic
after root-local adaptation. For persistent inner-scope ablations, the monitor
still evaluates this fresh outer prior rather than the already-adapted modules
retained by the inner workspace.

The monitor is sparse. It runs every configured number of **outer learner
updates**, independently of action-time `inner_diagnostics_every`, and its
metrics are absent on unsampled updates rather than reported as zeros. Paired
Monte Carlo samples reuse their random choices across the real-transition and
model-transition sides of each comparison. The monitor does not contribute to
any loss or optimizer update. The regular training accumulator publishes
sampled values under `train/ve_prior_*`, including the normal metric count,
minimum, and maximum summaries. Headline outputs cover target MAE, RMSE, bias,
normalized RMSE, absolute-error p95, reference-target RMS, reward RMSE,
bootstrap RMSE, and reward/bootstrap cancellation. `ve_prior_*_depth_1`
measures the replay-supported transition from the encoded root; larger depth
suffixes use recurrent model states under the recorded action sequence and
therefore include accumulated rollout error. On episodic tasks, rows after
either matched branch has already terminated are excluded, and a fully
unsupported later depth is omitted.

Enable it with:

```json
{
  "value_equivalence_diagnostics": true,
  "value_equivalence_every_updates": 1000,
  "value_equivalence_mc_samples": 4
}
```

`value_equivalence_every_updates` and `value_equivalence_mc_samples` must be
strictly positive integers. The monitor currently requires
`inner_operator="sac"`; it remains disabled by default.

For an actor-only inner-SAC ablation, set
`inner_actor_adaptation="clone"`, `inner_critic_adaptation="frozen"`, and
`inner_temperature_mode="inherit_outer"`. The canonical schedule then resolves
the inner critic and temperature update counts to zero while retaining actor
updates. This combination is also available as
`adapted_components/actor_only` in
`configs/research/ambi_inner_decoupling.json`.

Inner-critic dropout is independently switchable with
`inner_critic_dropout_enabled`. Its default is `true`, preserving TD-MPC2's
configured critic dropout during trainable inner critic and actor-Q updates.
Set it to `false` for dropout-free, deterministic per-head forwards while
leaving outer critic training unchanged. Ensemble-pair selection and SAC action
sampling remain stochastic. The switch disables all dropout in the adapted
critic, including a nonzero `inner_critic_lora_dropout`; target critics remain
in eval mode in either setting.

AMBI can optionally use TD-MPC2's running P95-P5 actor-value scale through
`sac_actor_loss_scale_mode="tdmpc2_percentile_range"` (the default is
`"none"`). Unlike TD-MPC2's fixed-entropy policy prior, AMBI divides the entire
soft SAC actor objective by the detached scale, so Q and the learned entropy
coefficient remain in consistent units. The outer learner updates the scale
from real-replay actor values; each root-local SAC solve freezes one snapshot
for the whole action. Rewards, Bellman targets, temperature losses, TD3, and
MPPI are not normalized by this option.

Model saves are suitable for evaluation and weight transfer. They do not include
replay, environment state, or all trainer counters, so they are not exact
mid-run resume checkpoints.

### Figure 1 value-calibration protocols

The public reference for the value-calibration evaluator is TD-MPC² commit
[`d1c2632c36effd2f7b661bfe5f822a3db8054d40`](https://github.com/DarthUtopian/tdmpc_square_public/commit/d1c2632c36effd2f7b661bfe5f822a3db8054d40),
specifically
[`tdmpc_square/tdmpc_square/trainer/online_trainer.py::eval_value`](https://github.com/DarthUtopian/tdmpc_square_public/blob/d1c2632c36effd2f7b661bfe5f822a3db8054d40/tdmpc_square/tdmpc_square/trainer/online_trainer.py#L70-L99).
For each value measurement, that code first averages 100 complete discounted
rollouts under the deterministic mean of the nominal policy. It then draws an
independent second batch of 100 environment resets, evaluates the current
online critic at each initial state and deterministic mean action, selects two
distinct Q heads at random from the five-head ensemble, decodes them, and
averages the pair. Finally it reports the separate scalar means as `mc_value`
and `q_value`. The released trainer calls ordinary policy evaluation first,
runs the value measurement at step zero and at the first episode boundary
after each requested cadence, and reuses the training environment and global
NumPy/Torch random streams. Consequently, the upstream diagnostic itself
changes later environment and learner randomness.

AMBI exposes two explicitly separate protocols rather than treating those
side effects as part of the scientific estimator:

- `paper_deterministic` preserves the reference estimator's deterministic
  mean-policy rollouts, independent 100-reset MC and Q batches, and random
  two-of-five current-online-head mean. Its head pairs are sampled through a
  private, namespaced seed instead of the learner's global NumPy stream.
- `stochastic_bellman` evaluates the stochastic policy represented by AMBI's
  reward-only Bellman critic. It pairs Q and Monte Carlo at the same seeded
  initial state, evaluates Q at a sampled first action, executes that exact
  action in the rollout, and then samples subsequent policy actions. It reports
  the mean and minimum over all online Q heads instead of a random pair.

As in the public evaluator, a Monte Carlo rollout stops on either environment
termination or time-limit truncation and does not append a critic bootstrap at
the boundary. Thus `stochastic_bellman` is Bellman-matched in its sampled action
and subsequent policy, while its reported return remains the finite real-episode
return used by the Figure 1 protocol.

Both protocols use a dedicated evaluation environment, fixed namespaced seeds,
and private random streams, so they do not alter the training environment,
replay, or learner RNG state. These are deliberate controlled divergences from
the released trainer. `paper_deterministic` is the compatibility-facing curve;
`stochastic_bellman` is the Bellman-matched calibration curve, and their values
must remain separately labeled and aggregated.

Enable the observational probe through the existing online-evaluation cadence:

```json
{
  "eval_freq": 50000,
  "eval_value": true,
  "eval_value_samples": 100,
  "eval_value_seed": 12345,
  "eval_value_protocols": [
    "paper_deterministic",
    "stochastic_bellman"
  ]
}
```

It is state-observation-only and requires a reward-only outer critic. The
paper-facing W&B aliases are `eval/mc_value`, `eval/q_value`, and
`eval/q_minus_mc`. Bellman-matched outputs are prefixed with
`eval/stochastic_`; they include Monte Carlo value, all-head mean and min,
paired mean-head bias/RMSE, and critic-head spread. All values are merged into
the ordinary evaluation event, while `time/value_eval_seconds` records the
additional wall time. The sample dispersions are within one evaluation event;
they are not uncertainty estimates across training seeds.

The maintained Humanoid Walk entry point is
`configs/dmcontrol/experiments/ambi_humanoid_walk_base_min_all_reward_only_value_calibration.json`.
It derives from the five-head `min_all`, reward-only base, runs one million
agent decisions for trials 55--57, evaluates both protocols at step zero and
every 50,000 decisions with 100 samples and evaluation seed 12345, and retains
`all`, `best`, and `latest` checkpoints every 50,000 decisions. It intentionally
omits `wandb_run_name`, allowing the normal AMBI run name to incorporate each
resolved trial seed.

#### Fifty-percent outer-policy trajectory ablation

The paper also describes a middle-row intervention in which 50% of collection
trajectories use the nominal policy. The public TD-MPC² repository does not
include a configuration, branch, or patch implementing that collector, so its
exact randomization and switching code cannot be audited. AMBI therefore makes
the paper-stated trajectory semantics explicit: at each eligible fully
post-seed episode start/reset, `outer_policy_episode_probability` controls one
Bernoulli draw that selects the collector for the whole episode. The first
episode that crosses the seed-collection boundary stays AMBI for its partial
post-seed remainder; it is not randomized mid-episode. A value of `0.5` means
that half of eligible episodes in expectation are collected by the unadapted
outer policy; the other episodes use ordinary AMBI action-time improvement.
The choice never switches within an episode and is not a per-decision 50/50
action mixture.

The Bernoulli choice is a stateless, namespaced hash of the training seed and
episode-start environment step. Selected episodes use a separate episode-local
Torch generator to sample the current stochastic outer actor, bypassing
`agent.act` and the inner SAC engine. The selection and action streams do not
advance the learner's global Python, NumPy, or Torch RNGs. Both trajectory types
enter the ordinary replay and receive the unchanged outer UTD=1 updates.

This intervention changes training collection only. Both value-calibration
protocols still use their dedicated seeded evaluation environment and retain
the estimator definitions above. Episode events expose
`rollout/outer_policy_episode` and
`rollout/outer_policy_episode_eligible`; training windows expose the outer and
inner behavior action counts, the outer-action fraction, and separate action
timing. The public patch is unavailable, so AMBI's episode-level implementation
is the documented operationalization of the paper's condition, not a claim of
byte-for-byte reproduction.

The paired campaign entry point is
`configs/dmcontrol/experiments/ambi_humanoid_walk_value_calibration_outer_policy_ablation.json`.
It runs the unchanged zero-probability baseline and the `0.5` intervention for
the same seeds 55--57, one-million-decision budget, evaluation seed 12345, and
50,000-decision evaluation/checkpoint grid. The intervention retains all
baseline W&B tags and adds `outer-policy-trajectory-ablation` and the
paper-facing condition label `50pct-outer-policy-trajectories`; neither
condition hard-codes a run name.

After the three runs finish, render the exact-grid, equal-seed aggregate with:

```bash
python plot_value_calibration.py \
  --run ENTITY/PROJECT/RUN_55 \
  --run ENTITY/PROJECT/RUN_56 \
  --run ENTITY/PROJECT/RUN_57 \
  --output-prefix value_calibration
```

The same command accepts repeated `--history-csv` arguments for exported W&B
histories. It writes a two-panel PNG and PDF plus an aggregate CSV, performs no
interpolation or smoothing, and labels the band as mean plus or minus one
across-seed population standard deviation.

For the paired campaign, use the companion plotter to validate and render all
six histories as one condition-by-protocol 2x2 figure:

```bash
python plot_value_calibration_ablation.py \
  --baseline-run ENTITY/PROJECT/BASELINE_RUN_55 \
  --baseline-run ENTITY/PROJECT/BASELINE_RUN_56 \
  --baseline-run ENTITY/PROJECT/BASELINE_RUN_57 \
  --fifty-run ENTITY/PROJECT/FIFTY_RUN_55 \
  --fifty-run ENTITY/PROJECT/FIFTY_RUN_56 \
  --fifty-run ENTITY/PROJECT/FIFTY_RUN_57 \
  --output-prefix value_calibration_outer_policy_ablation
```

For exported histories, replace the W&B inputs with repeated seed-qualified
`--baseline-history-csv SEED=PATH` and
`--fifty-history-csv SEED=PATH` arguments. `--intervention-run` is an alias for
`--fifty-run`, and `--intervention-history-csv` is an alias for
`--fifty-history-csv`. The companion rejects missing or mismatched seeds and
anything other than the shared exact 21-point grid. It writes one 2x2 PNG, PDF,
and combined aggregate CSV. The paper-facing quantities are `eval/mc_value`,
`eval/q_value`, and `eval/q_minus_mc`; the separately prefixed
`eval/stochastic_*` outputs provide the Bellman-matched secondary analysis.

### Figure 2 paired-controller protocol

The online paired-controller probe measures whether action-local inner
optimization improves control in the real environment. For each evaluation
episode it resets independent auxiliary environments with the same explicit
seed, then compares deterministic outer-only control against deterministic
execution after a fresh inner SAC solve at every visited state. The primary
quantity is the episode-paired fresh-inner-minus-outer return; the fixed-target
root-Q action gain collected along the fresh-inner trajectory is a
model-predicted diagnostic, not a substitute for that real return difference.
Positive predicted gain with a negative real return delta is evidence of
critic/model optimism or compounded distribution shift.

Enable the observational probe with:

```json
{
  "eval_inner_comparison": true,
  "eval_inner_comparison_episodes": 5,
  "eval_inner_comparison_seed": 12345,
  "inner_diagnostic_rollouts": 0
}
```

Zero diagnostic rollouts retains the fixed-target root-Q comparison without
adding imagined diagnostic trajectories. The probe shares the frozen outer
model read-only, disables inner writeback, and restores private inner and
global RNG state; it does not touch the training environment or replay. Its
numeric outputs merge into the ordinary evaluation event. Figure-facing
uncertainty must be aggregated across the three independent training seeds;
within-event episode dispersion and correlated per-state Q gains are only
diagnostics.

The paired real-control curves are
`eval/paired_outer_episode_reward` and
`eval/paired_fresh_inner_episode_reward`; their episode-paired difference is
`eval/paired_fresh_inner_minus_outer`. Population standard deviations and the
strict inner-win fraction are logged beside them. The complete fixed-critic
root distribution uses the
`eval/paired_fresh_inner_fixed_target_q_action_gain` stem (count, mean,
population standard deviation, extrema, linear 5/25/50/75/95 percentiles, and
positive fraction). Optimization-only model steps per action and control time
with diagnostic time removed are logged separately from the observational
diagnostic cost and total paired-evaluation wall time.

The three `ambi_anchor_kl_{smooth,quantile,dual}` Humanoid Walk manifests run
this five-episode protocol at the existing 50,000-decision cadence alongside
Figure 1 value calibration. Their paired-controller settings are observational;
after removing those settings and their W&B labels, the only learning-axis
differences from the base are the declared replay behavior-policy-KL controls.
