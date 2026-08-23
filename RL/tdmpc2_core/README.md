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
