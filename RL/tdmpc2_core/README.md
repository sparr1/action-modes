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

Fresh critics always match the initialization in the official checkout at
`8bbc14ebabdb32ea7ada5c801dc525d0dc73bafe`. Its TensorDict critic parameters
retain the samples drawn by each `nn.Linear` constructor, so the ordinary
`ModuleList` critics are excluded from the global truncated-normal initializer:

- hidden linear weights and every linear bias, including the output bias,
  retain their uniform samples in `[-1/sqrt(fan_in), 1/sqrt(fan_in)]`;
- only the final linear layer's weights are zeroed;
- LayerNorm weights remain one and biases remain zero.

This contract applies to AMBI, TDAMBI, and the local TD-MPC2 baseline across
ensemble sizes and scalar/distributional variants, with no configuration switch.
Other model components retain their existing initialization and order. Target
critics and prior-initialized inner critics copy their source weights; random
inner critics use the same fresh-critic rules. Checkpoint loading
restores saved weights without reinitializing them. Checkpoint schemas and Q
APIs are unchanged. Fresh runs differ from the earlier ModuleList initialization,
which applied truncated-normal critic weights and zero biases; initial Q values
are now determined by the sampled output biases and need not be zero.

The maintained port deliberately replaces upstream TD-MPC2's epsilon-floored
tanh log-Jacobian with the exact stable pre-tanh identity. This changes policy
log-probabilities, entropy diagnostics, and their gradients, but not sampled
actions, RNG consumption, or checkpoint structure. Continuing training across
this boundary is therefore a new experimental lineage, even from a loadable
older checkpoint. Actor diagnostics report mean and maximum absolute pre-tanh
values, the fraction beyond the former floor's `7.600902` crossover, and the
fraction of actions rounded exactly to either tanh bound.
AMBI's optional `tdmpc2_scaled` actor entropy statistic separately reproduces
the upstream floored Jacobian and ratio; it leaves these stable action
log-probabilities and diagnostics intact.

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
from the current outer temperature. Scalar twin critics, random inner
initialization, LoRA, TD3, no inner improvement, and persistent inner scopes are
explicit ablations. The MPPI inner operator is a compute-matched TD-MPC-style
comparator, not AMBI's planner.

Outer Adam defaults match TD-MPC2: `actor_lr=critic_lr=3e-4`,
`actor_adam_eps=1e-5`, and `adam_eps=1e-8` for world-model and critic updates,
with betas `(0.9, 0.999)`, zero weight decay, and AMSGrad disabled. Explicit
`actor_adam_eps` values override the default; `null` inherits `adam_eps`.
Inner optimizer settings are configured independently.

### Temporal training weights

AMBI and the local TD-MPC2 baseline use the upstream temporal reductions with
`rho=0.5` by default and in maintained presets. For training horizon `H`, each
per-time loss has already averaged its replay batch:

```text
consistency / reward / critic = sum(rho**t * loss[t], t=0..H-1) / H
outer actor                 = sum(rho**t * loss[t], t=0..H)   / (H+1)
```

The actor includes the terminal latent. Critic training also averages all
ensemble heads. AMBI's optional value-equivalence loss uses the transition
reduction above. `rho` remains configurable; there is no reference horizon or
rescaling to hold total weight constant as `H` changes. For example, at
`rho=0.5`, transition weights sum to `1` at `H=1`, `7/12` at `H=3`, and
`21/64` at `H=6`. Reductions preserve each caller's existing upstream-compatible
floating-point operation order.

Runtime metadata records `temporal_loss_normalization="divide_horizon"` and
`rho`. Historical `temporal_loss_normalization` inputs (`"divide_horizon"` or
`"reference_weighted_mean"`) and positive `temporal_loss_reference_horizon`
inputs are accepted with a deprecation warning, then discarded. They cannot
select the retired algorithm, and the resolved configuration contains no
reference horizon. This keeps historical checkpoint sidecars loadable for
evaluation without changing saved weights or their original metadata; the
temporal change requires no checkpoint tensor-schema migration. Exact resumes
across the change are rejected by the existing source/configuration identity
checks. Intentional weight transfer starts a new lineage.

AMBI's temperature and behavior-regularizer reductions retain their own
normalization and use the configured `rho`. Inner SAC minibatch losses have no
timestep weighting. Termination BCE keeps its unweighted upstream reduction.
The optional Q-only percentile scaling described below is unchanged.

### Inner-network initialization

`inner_actor_initialization` and `inner_critic_initialization` independently
accept `"prior"` (the default) or `"random"`. A prior-initialized component
starts from its latest learned control prior. A random component receives fresh
samples for every active inner solve, including when the engine reuses its parameter
storage. The networks retain the configured architecture, device, and dtype.
They do not reset the world model or the outer priors.

Initialization controls starting weights; adaptation controls trainable
parameters. In particular, the existing `"clone"` adaptation name means full
dense training and also applies to a network initialized from scratch. To train
both AMBI inner networks densely from scratch, add these algorithm parameters:

```json
{
  "inner_operator": "sac",
  "inner_actor_initialization": "random",
  "inner_critic_initialization": "random",
  "inner_critic_target_initialization": "online",
  "inner_actor_adaptation": "clone",
  "inner_critic_adaptation": "clone"
}
```

Either initialization can instead be `"prior"` to isolate the actor or critic
ablation. For a dense random actor and a random critic with LoRA updates, use:

```json
{
  "inner_operator": "sac",
  "inner_actor_initialization": "random",
  "inner_critic_initialization": "random",
  "inner_critic_target_initialization": "online",
  "inner_actor_adaptation": "clone",
  "inner_critic_adaptation": "lora_rl"
}
```

Random actors follow AMBI's model-construction initializer: truncated-normal
linear weights with standard deviation `0.02`, zero linear biases, and reset
LayerNorm weights/biases of one/zero. Random critics follow the fresh-critic
rules above: constructor-uniform linear weights and biases, zero final-layer
weights, and reset LayerNorm. The critic's output biases remain random.
Every active solve resets its targets, optimizer moments, and imagined replay.
Seeded runs reproduce the initialization sequence; subsequent decisions draw
new samples rather than restoring one fixed random network. Setting
`inner_rounds=0` retains the existing outer-policy bypass and does not instantiate
inner networks.

Optional `inner_actor_initial_std` sets a random actor's initial **pre-tanh**
Gaussian standard deviation exactly (for example `0.3`). The default `null`
preserves the initializer above. The requested log standard deviation must lie
strictly inside the resolved inner bounds. Initialization retains the random
mean network, zeroes only the std output rows, and fills their biases using the
inverse of `tdmpc2_tanh` or `direct_clamp`. Those rows remain trainable. Every
action reset reapplies the scale without additional random draws or outer
changes. The option requires `inner_actor_initialization="random"`, enters
experiment and resume identity, and is preserved in immutable actor snapshots.
With the default tanh mapping and bounds `[-10, 2]`, the unmodified near-zero
std head instead starts near `exp(-4)`, approximately `0.018`.

A random online critic initializes its independent local target from that same
new critic. `inner_critic_target_initialization="outer_target"` is rejected
when the critic initialization is random. LoRA targets use the merged effective
weights and preserve the existing target-update rule. The initialization choice
does not disable explicitly configured outer bootstrap sources, policy anchors,
inherited entropy temperature, or Q scaling. These remain separate controls;
random initialization alone does not remove all uses of learned priors.

Random initialization is supported for single-policy inner SAC and the
[native TDAMBI evaluator](#tdambi-native-inner-learning), with all seven actor,
critic, temperature, replay, and corresponding optimizer scopes set to
`"action"`. Dense adaptation uses `"clone"`; AMBI SAC also supports `"frozen"`
components under the existing zero-update rules. Native TDAMBI retains paired
updates with a `"clone"` actor and a `"clone"` or `"lora_rl"` critic. Critic
`"lora_rl"` adaptation always requires a `"clone"` actor. Persistent scopes,
explorer populations, other inner operators, nonzero prior writeback
coefficients, a positive `value_equivalence_loss_coef`, and
`value_equivalence_diagnostics=true` are rejected for this ablation. The
value-equivalence features retain their fresh-prior interpretation.

Omitting the initialization fields or setting both to `"prior"` preserves
existing behavior and historical scientific identities. Random choices are
included in configuration and experiment identities. AMBI exact inner-state
training checkpoints for random initialization use version 4 to record and
validate the initialization protocol before loading; an exact resume cannot
change that protocol. Portable outer checkpoints remain loadable for a new
scratch-initialized evaluation or study. Native TDAMBI remains an evaluation
wrapper that loads frozen native model checkpoints and their saved scale.

### Critic-only LoRA-RL

Set `inner_critic_adaptation="lora_rl"` and retain
`inner_actor_adaptation="clone"` for the paper-inspired inner SAC variant.
The native TDAMBI evaluator accepts the same critic adaptation while retaining
its native TD-MPC2 actor and critic objectives.
Dense actor/critic adaptation remains the reference default. The replacement
is based on [LoRA-RL](https://arxiv.org/abs/2604.18978) and its released
[implementation at commit `0395bb27b24016f6c680ca12c6d5e367f77db054`](https://github.com/paulzyzy/LoRA_RL/tree/0395bb27b24016f6c680ca12c6d5e367f77db054).
It borrows critic update regularization while retaining AMBI's architecture,
actor loss, critic representation, target reduction, and budgets. Initialization
uses learned priors by default and supports the independent random options above.

```json
{
  "inner_actor_adaptation": "clone",
  "inner_critic_adaptation": "lora_rl",
  "inner_critic_lora_layers": "input_hidden",
  "inner_critic_lora_rank": 96,
  "inner_critic_lora_scale": 1.0,
  "inner_critic_lora_weight_decay": 0.0002
}
```

Each selected inner-critic matrix uses `W = W_base + scale * B @ A`.
`W_base` is the copied prior weight or a fresh random weight, frozen during the
solve; both adapter factors train. The output heads, biases, and LayerNorm
parameters remain normally trainable in separate
inner copies. There is no adapter dropout. The ordinary configured critic
dropout still follows `inner_critic_dropout_enabled`. Actor updates remain
dense, and gradients through the adapted critic reach actor actions without
updating any outer model parameter.

`inner_critic_lora_layers="input_hidden"` selects the input and hidden-to-hidden
matrices in each critic. `"hidden"` selects only hidden-to-hidden matrices and
leaves the input projection normally trainable. AMBI's input is a learned latent
plus action: at model size 5, its input matrix is similar in size to its
512-by-512 hidden matrix. The default therefore constrains both; hidden-only is
the closer placement analogue to the paper's residual-block-only adapters.
Neither placement changes the network architecture or adapts the value head
with LoRA.

The DMC [SimbaV2 launcher](https://github.com/paulzyzy/LoRA_RL/blob/0395bb27b24016f6c680ca12c6d5e367f77db054/SAC_SimbaV2/scripts/run_lora_example.sh)
uses rank 96, alpha 96, and adapter weight decay `2e-4`; the DMC
[BRC launcher](https://github.com/paulzyzy/LoRA_RL/blob/0395bb27b24016f6c680ca12c6d5e367f77db054/SAC_BRC/scripts/run_lora_example.sh)
uses rank 128, alpha 128, and `6e-4`. Our direct scale is alpha/rank, so scale
one follows both settings without changing scale when rank changes. The default
decay borrows the SimbaV2 setting; it is not a claim that AMBI matches SimbaV2's
effective capacity or hyperspherical normalization. Adapters use AdamW at the
existing inner critic learning rate and epsilon; other trainable critic
parameters retain AMBI's ordinary Adam behavior through an AdamW group with
zero weight decay. BRC instead preserves its own baseline AdamW decay of
`1e-4` on those parameters, as shown in its
[optimizer groups](https://github.com/paulzyzy/LoRA_RL/blob/0395bb27b24016f6c680ca12c6d5e367f77db054/SAC_BRC/jaxrl/agent/brc_learner.py).
This difference deliberately preserves each baseline's dense optimization.
Frozen matrices never enter an optimizer.

Target updates follow the released
[BRC effective-weight rule](https://github.com/paulzyzy/LoRA_RL/blob/0395bb27b24016f6c680ca12c6d5e367f77db054/SAC_BRC/jaxrl/lora.py):
Polyak-average the merged online weights into an independent dense target,
including the normally trainable parameters. Averaging A and B separately
does not implement this rule. The target follows the existing inner target
cadence and tau. Frozen outer bootstrap choices retain their existing meaning.

All actor, critic, replay, temperature, and optimizer scopes are action-local
for this variant. Every active solve restores the latest learned prior or draws
fresh random weights according to each initialization setting, initializes
fresh A from a normal distribution with standard deviation `1/sqrt(rank)` and
zero B, and resets target, optimizer, and replay state in reused storage.
Zero B makes the initial adapted critic exactly reproduce its selected dense
base. Random initialization retains zero B and the existing trainable biases,
LayerNorm, and value heads; it does not freeze the entire critic or the actor.
In particular, keeping the zero-initialized critic value-head weight trainable
allows learning to reach the hidden layers. This remains an intentional
departure from the paper's random nonzero A/B initialization and specialized
weight-normalization projection, which AMBI does not implement.
The public SimbaV2 snapshot contains its launcher/configuration but lacks the
`simbaV2` implementation imported by its agent factory; detailed target and
trainable-parameter behavior is checked against BRC, not an unavailable SimbaV2
module. We make no latency or return-improvement claim without AMBI experiments.

The retired `inner_actor_adaptation="lora"`,
`inner_critic_adaptation="lora"`, and joint `inner_adaptation="lora"` requests
fail with migration guidance; they are not aliases for the replacement.
Historical results retain their old identities. Inactive legacy LoRA fields in
dense saved configurations can be ignored with a deprecation warning. To use
the new method, explicitly select `lora_rl`, remove old actor/adapter-dropout
options, and choose a new experimental lineage. The old Humanoid rank-8/rank-16
and AntLegAdapt joint-LoRA preset names are retired. See the
[new research selectors and Humanoid configurations](../../configs/research/README.md#critic-only-lora-rl-comparisons).

New exact resumes record and validate the method, placement, rank, scale, and
decay before loading live state, even though the action-local workspace is empty
at the checkpoint boundary. Historical action-local payloads did not record the
adapter method; their saved configuration is required to identify and reject an
obsolete active LoRA run. Portable outer-model weights remain usable for a new
study with an explicitly selected adaptation method.

### TD-AMBI training preset

[`configs/dmcontrol/algs/TD-AMBI.json`](../../configs/dmcontrol/algs/TD-AMBI.json)
and its [experiment manifest](../../configs/dmcontrol/experiments/TD-AMBI.json)
select TD-MPC2's actor and critic loss formulas for both learners on the AMBI
training backbone. The outer sequence losses keep upstream `rho=0.5` and
division by the horizon. The inner learner retains AMBI's individual imagined
transition minibatches and their ordinary means, as an explicit design choice;
it does not adopt upstream sequence batching or its terminal-depth actor term.

Both actors minimize `-mean_pair(Q) / S - 1e-4 * H_scaled`. Both critics use
`reward + discount * (1 - terminated) * min_pair(target_Q)` as their Bellman
target, with distributional two-hot cross entropy averaged across five heads
and coefficient `0.1`. The actor statistic is the literal upstream scaled
entropy helper, both policies use the smooth tanh log-standard-deviation map
over `[-10, 2]`, and neither learner creates a temperature optimizer. Actor
and critic learning rates are `3e-4`, their Adam epsilons are `1e-5` and `1e-8`,
gradient clipping is `20`, and target interpolation is `0.01` each update.

The preset explicitly sets `inner_critic_loss_coef=0.1`,
`inner_actor_adam_eps=1e-5`, `inner_actor_loss_scale_update="per_update"`, and
`inner_critic_target_initialization="outer_target"`. At each real action the
inner target starts from the outer target critic, and the inner scale starts
from the current outer scale. Inner scale EMA updates then use the same actor
samples and Q values as the loss, without extra sampling or outer-state mutation.
The default settings remain coefficient `1`, actor epsilon inherited from
`inner_adam_eps`, scale frozen for the action, and target copied from online Q.
Nondefault coefficient/scale/target options currently require ordinary inner
SAC without an explorer population; outer-target initialization additionally
requires an action-local cloned critic.

The Humanoid Walk preset retains base-v2's seed 55, 14-million-decision budget,
J=8/N=32/H=3/G=1 collection schedule, minibatch/replay sizes, action-local
learners, and disabled evaluation and model saves. It is a single-seed
exploratory recipe. It uses `AMBITDMPC2/AMBITDMPC2` for training; the separate
`TDAMBI/TDAMBI` wrapper below remains for native frozen-checkpoint evaluation.

```bash
environments/dmcontrol/.venv/bin/python main.py \
  --run configs/dmcontrol/experiments/TD-AMBI.json \
  --alg-dir configs/dmcontrol/algs
```

This is a full training command; submit it through the scheduled-compute
workflow when running an experiment.

### TDAMBI: native inner learning

`RL/TDAMBI.py` adapts native TD-MPC2 checkpoints for frozen evaluation using the
shared `InnerImprovementEngine`. By default, each active solve restores a local
actor, online critic, and the checkpoint's **saved target critic**, clears
optimizer moments and replay, and resets the local Q scale. Allocated storage
can be reused. The actor and online critic independently support `"prior"` or
`"random"` initialization; the actor trains densely and the critic supports
`"clone"` or `"lora_rl"` adaptation. All outer parameters, including encoder,
dynamics, reward, and saved control priors, remain immutable. Only single-task
state observations are supported; the wrapper rejects training and SAC
shared-observation probes.

For example, add these planner settings to a native TDAMBI evaluation to train
a fresh random actor and a critic with random frozen LoRA base matrices:

```json
{
  "inner_actor_initialization": "random",
  "inner_critic_initialization": "random",
  "inner_critic_adaptation": "lora_rl"
}
```

The wrapper selects the target initialization default from the critic settings:
`"outer_target"` for a prior-initialized `"clone"` critic, and `"online"` for a
random critic or any `"lora_rl"` critic. Historical `"online"` settings with a
prior-initialized dense critic normalize to `"outer_target"`, preserving that
controller's saved-target behavior and identity. Random or LoRA critics reject
`"outer_target"`; their independent local targets start from the new online
critic's effective weights. This applies on both initial allocation and reused
storage. The source checkpoint must still contain its complete saved native
target critic. Actor initialization does not change which target source is used.

These settings change initialization and critic trainability while preserving
native TD-MPC2 learning below. `TDAMBI/TDAMBI` remains the frozen native evaluator;
the separately named `TD-AMBI.json` presets use `AMBITDMPC2/AMBITDMPC2` for
training with AMBI's configurable SAC implementation.

Imagined stochastic-policy transitions are detached and sampled as ordinary
minibatches, without TD-MPC2's outer timestep batching or rho weighting. A
paired update performs critic, actor, then target interpolation. Critic learning
uses native distributional cross entropy averaged across heads, multiplied by
the native value coefficient. Its target is predicted reward plus discounted
local target Q (minimum of two random heads) at a sampled local-policy action.
Rollout cutoffs bootstrap and do not introduce synthetic terminal flags. There
is no entropy bonus in that target.

Actor learning maximizes average-pair online Q divided by the running scale,
plus the saved fixed coefficient times native `scaled_entropy`. This preserves
the native policy standard-deviation mapping and this port's stable tanh
calculations. Online critic dropout remains active during training; the target
critic remains in evaluation mode. Dense updates retain native Adam defaults,
actor epsilon `1e-5`, critic epsilon `1e-8`, clipping, and target interpolation.
With LoRA, critic updates use the existing AdamW adapter groups: selected base
weights freeze, adapters receive configured weight decay, and critic biases,
normalization, and value heads remain trainable with zero weight decay. Its
target averages merged effective weights. No local temperature optimizer is
created. The real action is the adapted `tanh(mean)`.

Native portable checkpoints now include `scale = {value, percentiles}` beside
the model, including periodic asynchronous checkpoints. The value is captured
at the same training step as the networks. Loaders validate its shape, dtype,
finiteness, unit floor and fixed `[5, 95]` percentiles before mutating live state.
Native exact-training checkpoints already included this state and keep their
existing format. Loading an older model-only snapshot into the native learner
resets its unavailable scale to one.

TDAMBI's `tdambi_scale_initialization` selects one of three evaluation policies:

- `checkpoint_or_calibrate` (default): copy the saved training scale into each
  new inner solve when present; otherwise use the historical calibration below.
- `checkpoint`: require saved scale state and fail clearly if it is missing.
- `calibrate`: use historical local calibration even when a saved scale exists,
  for a controlled initialization comparison.

The saved reference scale stays frozen. Each solve owns its own copy and applies
the native EMA on subsequent actor minibatches; the next real decision starts
from the saved training value again. With checkpoint initialization, no
calibration samples or calibration model calls are performed. Older snapshots
cannot reconstruct a training scale that was never saved.

Historical calibration samples a reproducible minibatch from the first imagined
collection before the first optimizer update, evaluates frozen-prior actions
with frozen online average-pair Q, and initializes `max(P95 - P5, 1)`. It has an
isolated RNG stream and separate evaluation counts and timing. Random network
initialization and LoRA preserve this explicit scale policy: saved-scale modes
still use the checkpoint scale, and calibration still evaluates the frozen
control priors rather than the random inner networks. Subsequent actor updates
always update the local scale from the adapted online critic. Evaluation
records the initialization policy and actual source; decision metrics include
`inner_tdambi_scale_from_checkpoint`, `inner_tdambi_q_scale_initial`, and the
existing final scale. Initialization policy is part of planner identity, while
the learned numeric scale remains checkpoint-specific data.

Detached raw traces include native entropy, its weighted contribution, raw and
scaled Q, scale before/after, losses, gradients, target statistics and work
counts. Losses are pre-optimizer measurements. No per-update synchronization,
extra diagnostic forward, or publication is introduced by tracing. See the
[frozen TDAMBI workflow](../../configs/research/README.md#tdambi-on-native-td-mpc2-checkpoints)
for presets, matched native prior references, HTML reports and Oscar smoke use.

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
`Q + alpha * H_selected` without Q scaling, or `Q + alpha * S * H_selected`
with Q scaling) or `"reward_only"` (which bootstraps with `Q`). The
reward-only setting changes only the corresponding Bellman target; it does not
otherwise change actor sampling, the selected actor entropy objective, or the
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
`outer_behavior_policy_kl_target`). If actor-loss scaling is enabled, only
the SAC Q term is divided by the shared Q-range scale. Entropy and the
behavior-policy regularizer retain their coefficients. The CE objective
supports `"smooth"` and `"quantile_gate"` but rejects
`"dual"`, because CE has no invariant zero-valued constraint target. Active
modes require stochastic inner SAC execution. Replay and agent states move to
versions 2 and 5/6 respectively only while this feature is active; exact
checkpoint metadata rejects cross-objective continuation.

Behavior-regularized outer actor updates accumulate their L2 gradient norm in
float64. Large finite KL/CE gradients therefore retain the ordinary clipping
direction instead of overflowing a float32 norm and being zeroed. The actor,
world model, critics, and their optimizer moments remain float32. The dual
schedule keeps its single log coefficient and Adam moments in float64 so that
squaring a large finite constraint violation cannot overflow its second
moment; the coefficient is cast to the actor loss dtype when used there. This
changes numerical precision without capping the KL, changing the objective, or
adding a gradient-clipping threshold for the dual update. Runs with the
behavior regularizer disabled retain their existing clipping implementation.

Portable loads promote finite legacy float32 dual coefficients and moments.
Exact resume remains strict about dtype and requires a float64 dual state;
continuing a legacy dual checkpoint with this precision change is a new
training lineage. Checkpoints with already non-finite dual moments are rejected
before any live state is changed, since precision promotion cannot reconstruct
the lost values. Their model weights can still be transferred to a fresh agent.

Learned outer and inner SAC entropy coefficients have a numerical floor of
`1e-8`. Fixed entropy coefficients retain their configured values. The floor
prevents an underflowed outer coefficient from becoming an invalid zero when a
fresh inner SAC solve inherits it.

### Continuing-task and value-equivalence contract

Maintained AMBI research follows TD-MPC2's continuing-task formulation with
`episodic=false`. Time-limit truncations end data-collection episodes but do
not mask Bellman bootstraps. In this mode TOLD has no termination head or
termination loss, imagined inner replay always continues for its configured
horizon, and value-equivalence training is deliberately **value-only**. It
matches the discounted successor-value component while TOLD's existing reward
cross-entropy supervises rewards separately. It does not predict, mask, or
differentiate through termination. Supporting termination in this objective
would be a separate research decision rather than an implicit extension.

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

### Value-equivalence training loss

AMBI can also train TOLD to preserve the value component of the fresh-inner SAC
Bellman operator. At recurrent depth `i`, it minimizes the raw squared residual
`(gamma * mean_k[V_k(z_model[i+1]) - V_k(z_real[i+1])]) ** 2`, reduced with
TOLD's existing temporal weights. The replay action advances the recurrent
model, while each successor branch samples its own state-conditioned fresh-inner
policy action. Paired branches share Gaussian base noise and critic-head choices,
and the configured Monte Carlo values are averaged before the residual is
squared. Actor and critic parameters remain fixed, but their input gradients
train the source encoder and latent dynamics. The reward head receives only its
existing supervised TOLD reward loss.

Enable the loss explicitly with, for example:

```json
{
  "value_equivalence_loss_coef": 1.0,
  "value_equivalence_loss_mc_samples": 4
}
```

The coefficient must be a finite non-negative number and the sample count a
strictly positive integer. A positive coefficient requires
`inner_operator="sac"` and `episodic=false`. The default coefficient is zero,
so existing training and experiment configurations remain unchanged. Loss
sampling is independent of `value_equivalence_mc_samples`, which controls only
the observational live monitor.

### Finite-horizon inner SAC and real-replay mixing

The opt-in [to-go calibration workflow](../../configs/research/TOGO_CALIBRATION.md)
scores the current actor at initialization and completed rounds using H model
steps plus the frozen outer online-Q tail. `InnerActionTrace` can also capture
immutable actors for real H-step prefixes followed by continuing prior rollouts.
Diagnostic randomness, timing, and work counts are separate from optimization;
ordinary training and evaluation keep these diagnostics disabled by default.

`inner_finite_horizon=true` makes `inner_rollout_horizon=H` a boundary for
the inner solve. The final imagined transition, from depth `H-1` to `H`,
uses the frozen outer policy and value prior for its Bellman continuation:

```text
depth < H-1: y = r + gamma * (1 - terminated) * V_inner(next_z)
depth = H-1: y = r + gamma * (1 - terminated) * Q_outer(next_z, a_prior)
                                             a_prior ~ pi_outer(next_z)
```

Interior rows retain the configured `inner_bootstrap_source`,
`inner_q_target_reduction`, and `inner_sac_critic_target`. The boundary follows
AMBI's TD-MPC2 MPPI terminal score: it samples the stochastic outer policy with
its outer log-standard-deviation settings and evaluates the outer **online**
critic with `mppi_terminal_q_reduction` (default `"mean_pair"`). These priors
remain fixed throughout the inner solve. No additional entropy term is added at
the boundary. The outer critic retains whichever return semantics it was
trained with; set both `outer_critic_target` and `inner_sac_critic_target` to
`"reward_only"` for the TD-MPC2 reward-return interpretation. An
entropy-augmented outer critic still contributes its learned soft Q tail.

The horizon boundary is stored separately from true environment termination.
A true termination suppresses every continuation, including a termination on
the last imagined transition. A collection horizon or real time-limit
truncation does not by itself suppress bootstrap. The default
`inner_finite_horizon=false` retains the existing interpretation of `H` as a
collection cutoff with the same inner continuation on every replay row.

The actor and critic deliberately remain functions of latent state and action,
without a remaining-horizon input. States revisited at different rollout
depths can therefore have different desired targets that one stationary critic
must approximate. This is an intentional experiment rather than an exact
finite-horizon dynamic-programming solution. The boundary follows the
blueprint-value handoff in
[RL Search, Section 3](https://arxiv.org/pdf/2109.15316); that paper also
describes mixing global replay into its Q-value fine-tuning updates.

`inner_outer_replay_fraction=p` mixes real transitions into critic minibatches,
with a default of `0.0` and allowed values in `[0, 1]`. The requested number of
real rows is `floor(p * inner_batch_size + 0.5)`; the remaining rows come from
imagined replay. Real transitions are sampled uniformly with replacement from
valid consecutive observations in the same stored episode. Real observations
and successors are encoded with
the current frozen encoder at sample time, and their recorded actions and
rewards are retained. Actor and temperature updates still receive a full
imagined minibatch. Real transitions have no imaginary rollout depth and use
the ordinary interior Bellman continuation, even when finite-horizon solving
is enabled. If no outer replay is attached or no real transitions are
available, critic sampling falls back to imagined replay and warns once. The
frozen-checkpoint evaluator rejects a positive fraction because model snapshots
do not contain replay; evaluate this ablation with a populated training agent.
When mixing is enabled, `inner_outer_replay_samples` reports the total real rows
used, `inner_outer_replay_fraction` reports the realized fraction, and
`inner_outer_replay_available` reports replay availability across critic
updates. These sampling controls apply only to inner SAC.

This is a valid off-policy critic update: the behavior policy need not match
the current inner policy, and the actor does not need to update on every
critic-training state. It can improve coverage and reduce reliance on model
errors, but it also changes the critic's fitting distribution. Real successor
latents and imagined successor latents can disagree because the model is
imperfect, and old real states may provide poor action coverage for a rapidly
adapted inner policy. Shared critic parameters can still change actor gradients
at imagined states, so excluding real states from the actor loss does not
isolate the actor from this change. At `p=1`, the critic learns entirely from
real data while the actor continues to improve on imagined states; finite
horizon boundary rows then supply no direct critic loss.

### Inner SAC updates per imagined transition

`inner_steps_per_update` provides an alternative to a fixed number of updates
after each rollout round. It defaults to `null` and otherwise accepts a finite
positive number of imagined transitions per joint critic/actor update. For
example, add these settings to a canonical SAC configuration and omit the
existing `inner_updates_per_round` setting:

```json
{
  "inner_rounds": 3,
  "inner_rollouts_per_round": 2,
  "inner_rollout_horizon": 2,
  "inner_steps_per_update": 5
}
```

After each collection round, the cumulative requested update count is
`floor(imagined_transitions_so_far / inner_steps_per_update)`. The learner runs
the updates newly due at that round boundary, so this example performs
`0, 1, 1` joint updates across its three rounds. Every slot updates the critic
and actor once; automatic temperature tuning follows the actor. A value below
one permits more than one update per imagined transition. Actual transitions
count, including early rollout termination; replay draws and real replay rows
do not earn additional updates. Fractional remainder carries between rounds
and resets at the next real action. By default, `inner_update_timing="round"`
groups these updates after complete collection rounds. The optional step timing
below applies the same transition interval between imagined timesteps.

Counts use the configured decimal interval without accumulating floating-point
remainders: for example, 49 transitions with an interval of `0.07` earn exactly
700 updates. An explicit `null` for a competing per-round count lets an
override remove that count before selecting the interval schedule.

This option requires canonical inner SAC with both actor and critic updates
enabled. It rejects explicit non-null shared or component per-round update
counts, legacy per-iteration/per-action budget controls, model-step budgets,
and explicit explorer component update counts. Symmetric explorer schedules
can inherit the primary update cadence. Use the component schedule below for
asymmetric actor/critic update counts. Nominal runtime totals are calculated as
`floor(inner_rounds * inner_rollouts_per_round * H / inner_steps_per_update)`;
realized totals can be smaller when rollouts terminate early.

The three options can be combined, for example:

```json
{
  "inner_operator": "sac",
  "inner_rounds": 3,
  "inner_rollouts_per_round": 32,
  "inner_rollout_horizon": 3,
  "inner_updates_per_round": null,
  "inner_steps_per_update": 96,
  "inner_finite_horizon": true,
  "inner_outer_replay_fraction": 0.25,
  "compile": true
}
```

With no early terminations this requests one joint update after each round.
Set any inherited component gradient counts to `null` as well. All options
are recorded in scientific run identity and active checkpoint target metadata;
exact resume rejects changed settings. Ordinary weight loading still permits
inner-solver ablations. Legacy disabled replay layouts remain unchanged.

Horizon flags share packed replay storage; the compiled target uses fixed-size
masks and never constructs variable-sized boundary batches. Real replay lookup
metadata is cached until replay changes, and current/successor observations are
encoded together. The ordinary cloned critic and actor kernels support strict
graph capture (`compile_strict=true`), including the horizon tail. Custom/LoRA
detached critics retain the existing general stateless path and may require
non-strict compilation on the locked PyTorch version. Compile failures remain
visible through the existing fallback metrics. Benchmark compilation warm-up
separately from steady-state actions.

### Step–update inner SAC

Set `inner_update_timing="step"` together with `inner_steps_per_update` to
interleave collection and learning inside each rollout round. Each step advances
all still-live imagined trajectories by one timestep, immediately appends their
transitions to replay, and runs the joint SAC updates newly earned by those
transitions. The next timestep continues from those successor latents and samples
actions with the updated actor. Each new round restarts the trajectories at the
real decision's root latent, retaining the inner learner and cumulative replay
until the usual end of that decision.

For example, this collects 512 parallel trajectories and performs one joint
critic/actor update after each timestep, with automatic temperature tuning
following the actor:

```json
{
  "inner_operator": "sac",
  "inner_rounds": 6,
  "inner_rollouts_per_round": 512,
  "inner_rollout_horizon": 3,
  "inner_updates_per_round": null,
  "inner_update_timing": "step",
  "inner_steps_per_update": 512,
  "inner_batch_size": 512
}
```

Here a **parallel timestep** produces up to 512 **transitions**. An interval of
512 gives one update per full parallel step; an interval of 1 would give 512
updates per full step. In the absence of early termination, the example uses
9,216 transitions and 18 joint updates per real decision. Setting its timing to
`"round"` preserves those totals and groups three updates after each full round,
providing a comparison that changes timing alone. Equal sample counts do not
imply equal sampled transitions: interleaving lets earlier learning change later
imagined actions and states.

Updates sample minibatches from all currently retained inner replay, including
earlier rounds. There is no additional implicit warmup. With-replacement sampling
can train when the first collected batch is smaller than `inner_batch_size`;
without-replacement sampling requires enough replay rows at the first earned
update. The configuration checks this for full-length populations; early
termination can still leave too few rows and produce the existing sampling error.
Only actual generated transitions earn updates, including those ending in true
termination. Fractional credit crosses timestep and round boundaries and resets
per real decision, independently of replay capacity or lifetime. Horizon flags
and optional real-replay critic mixing retain their existing target semantics.

Step timing currently supports single-policy canonical SAC
(`inner_explorer_mode="none"`) and requires the explicit transition interval.
The interval's existing restrictions on shared/component gradient counts and
frozen actor/critic adaptation still apply. The default `"round"` path retains
the existing schedules and explorer populations. This is an additional schedule
ablation; the manuscript's grouped-rollout algorithm remains the default.

Non-episodic collection compiles one model timestep at a time, while the existing
actor and critic kernels remain separately compiled. Episodic collection keeps
the existing eager compaction of terminated branches. Traces record a collection
event per timestep, with `collection_rollout_step`, followed by its update events;
optional fixed-noise probes remain at initialization and completed round
boundaries. The resolved configuration and active checkpoint identity record the
timing, so exact training resume rejects a timing change.

### Separate root-local critic and actor update counts

Canonical SAC and TD3 schedules can give the root-local critic and actor
different fixed update counts after each rollout round. Specify both component
fields and omit the shared `inner_updates_per_round` field. For example, this
large-rollout SAC schedule collects 512 length-three trajectories in each of
eight rounds, then performs three critic updates followed by one actor update:

```json
{
  "inner_rounds": 8,
  "inner_rollouts_per_round": 512,
  "inner_rollout_horizon": 3,
  "inner_critic_updates_per_round": 3,
  "inner_actor_updates_per_round": 1,
  "inner_batch_size": 512,
  "inner_replay_capacity": 12288
}
```

Each critic or actor update draws its own minibatch through the existing
cumulative action-local replay sampler. Automatic SAC temperature optimization
follows the actor and runs once per actor update; TD3 and fixed-temperature SAC
add no temperature step. The resolved runtime metadata records both per-round
counts, their derived per-action critic, actor, and temperature totals, and the
summed critic-plus-actor replay rows drawn. A component count may be zero.

This component schedule is mutually exclusive with the shared canonical
`inner_updates_per_round` control and with the deprecated per-action total
budget controls. Existing configurations that use a shared integer or
`"auto"` retain their joint update-slot behavior, batch sharing, and random
number ordering. These controls do not change replay capacity, retention, or
sampling semantics.

For an actor-only inner-SAC ablation, set
`inner_actor_adaptation="clone"`, `inner_critic_adaptation="frozen"`, and
`inner_temperature_mode="inherit_outer"`. The canonical schedule then resolves
the inner critic and temperature update counts to zero while retaining actor
updates. This combination is also available as
`adapted_components/actor_only` in
`configs/research/ambi_inner_decoupling.json`.

### Explorer policy populations

The action-local SAC solver can split its fixed rollout population between the
prior-initialized learner `P` and an exploratory population `R`. Set
`inner_explorer_mode` to `frozen_random`, `shared_mixture`,
`separate_critics`, or `adaptive_param_noise`; the default `none` preserves the
one-policy implementation and its random-number ordering.
`inner_prior_rollout_weight` is always the weight of `P`. It must divide the
per-round rollout count exactly, so the split adds no model transitions to the
configured `J*N*H` optimization budget. Actor identity is selected once per
imagined trajectory and retained for its full horizon. Both sources append to
one uniformly sampled root-local replay.

`adaptive_param_noise` is behavior-only: it creates `K` independent Gaussian
parameter perturbations around the current clean inner actor before every
round. `inner_param_noise_actor_count` is required, and the exploratory count
must divide evenly by `K`; every perturbed actor receives the same number of
rollouts. For example, `N=512`, `inner_prior_rollout_weight=0.5`, and `K=4`
produce 256 clean rollouts and 64 rollouts from each of four perturbed actors.
Each perturbation is fixed for the full imagined horizon. Hidden linear
weights and biases plus the final mean rows are perturbed, while LayerNorm and
the final log-standard-deviation rows remain clean. Rollouts sample around the
perturbed mean using the clean actor's state-dependent log standard deviation,
so replicas from one perturbed actor are distinct without allowing parameter
noise to change SAC entropy.

The raw parameter scale is reset at every real decision and recalibrated before
each round to `inner_param_noise_target_action_rms`, measured as per-coordinate
post-tanh RMS between deterministic clean and perturbed means on the same
latents. The calibrated scale is warm-started only between rounds of that one
decision and discarded after the clean actor executes. Perturbed actors are
never optimized, used for Bellman continuation, or eligible for real-action
execution.

The first calibration uses only the root latent. Later rounds use the root plus
up to `inner_param_noise_calibration_batch_size-1` uniformly sampled latents
from completed rounds in the action-local replay. A probe sequence samples
`inner_param_noise_calibration_directions` Gaussian directions once and reuses
them while adjusting sigma. Each miss applies
`sigma *= clip(sqrt(target / measured), 0.5, 2.0)` and clamps the result to
`[inner_param_noise_sigma_min, inner_param_noise_sigma_max]`. Calibration stops
within 10% of the target, on reaching a bound, or after
`inner_param_noise_calibration_max_probes` probes. Defaults are target RMS
`0.10`, sigma `0.001` in `[0.000001, 0.1]`, 8 directions, 32 calibration
latents, and 8 probes.

This mode requires `inner_behavior_action="policy_sample"`, positive
`inner_behavior_std_scale`, primary-only execution, nonempty exact primary and
explorer splits, and zero explorer actor/critic/temperature updates. Diagnostics
report the exact actor/replica allocation, target and realized action RMS,
sigma path, calibration probes/evaluations/time, target and bound hits, action
saturation, and source-conditioned rollout/replay counts. W&B interval summaries
pool RMS and saturation by their realized row/probe counts, including episodic
rollouts whose populations terminate at different depths.

The materialized random-actor modes deliberately have different Bellman
meanings:

- `frozen_random` never updates `R`; its transitions train the ordinary local
  `P` critic, whose continuation policy remains `P`.
- `shared_mixture` trains both actors against one critic for the stepwise action
  mixture `mu = w*pi_P + (1-w)*pi_R`. Its SAC entropy is the exact marginal
  `-log mu(a|z)`, evaluated by applying both Gaussian densities to the same
  pre-tanh action. The categorical continuation can be stratified across a
  minibatch or integrated with a weighted evaluation of both actors.
- `separate_critics` gives each actor its own online critic, target critic,
  temperature, and optimizer. Both critics use every replay transition, but
  `Q_P` bootstraps through `P` and `Q_R` through `R`; replay sharing is the only
  cross-policy information channel.

All explorer scientific state is freshly reset under AMBI's private
initialization stream at every real decision; module and Adam tensor allocations
may be pooled, but parameters, moments, steps, and temperatures never carry over.
Active modes therefore require
the canonical action-local clone configuration and reject persistent scopes,
LoRA, prior writeback, and an outer-policy anchor loss. The explorer actor uses
an independent optimizer when it learns. `separate_critics` additionally owns
an independent critic optimizer and, under automatic entropy tuning, an
independent temperature optimizer. `shared_mixture` instead has one shared
critic and one shared temperature; the exact marginal actor loss performs one
joint backward pass so its cross-density gradients intentionally reach both
actors.

The returned environment action defaults to `P`. The
`inner_execution_policy_source` ablation can instead use `R`, sample a component
with the same `w`, select between deterministic component means with the frozen
outer target critic, or compare stochastic `H`-step soft handoffs. The handoff
score averages `inner_execution_handoff_samples` trajectories per actor, uses
one frozen outer entropy coefficient for both prefixes, and adds exactly one
outer-prior target value at the final latent. This execution selector is
independent of `inner_finite_horizon`. With that option disabled, the ordinary
imagined horizon remains a collection cutoff, and its final replay transition
receives the same one-step bootstrap as every other row.

Whenever a two-policy random-explorer mode (`frozen_random`, `shared_mixture`,
or `separate_critics`) is active, execution also records an RNG-free
counterfactual from the fixed outer target critic, regardless of which selector
actually supplies the environment action. It evaluates the deterministic `P`
and `R` means with `min_all`, assigns exact ties to `P`, and logs both values,
the signed `Q_R-Q_P` margin, the counterfactual source and mean action, agreement
with the actual execution source, and distance to the executed action. The
`outer_q_gate` selector reuses this evaluation rather than running a duplicate
pair of policy and critic forwards. No corresponding metrics or computation are
introduced without a concrete `R` actor.

Inner-critic dropout is independently switchable with
`inner_critic_dropout_enabled`. Its default is `true`, preserving TD-MPC2's
configured critic dropout during trainable inner critic and actor-Q updates.
Set it to `false` for dropout-free, deterministic per-head forwards while
leaving outer critic training unchanged. Ensemble-pair selection and SAC action
sampling remain stochastic. The switch disables the configured dropout in the
adapted critic; LoRA-RL introduces no separate adapter dropout. Target critics
remain in eval mode in either setting.

AMBI can optionally use TD-MPC2's running P95-P5 actor-value scale through
`sac_actor_loss_scale_mode="tdmpc2_percentile_range"` (the default is
`"none"`). It applies the scale to Q only, matching TD-MPC2's placement:

```text
S <- (1 - tau_s) * S + tau_s * max(P95(Q[0]) - P5(Q[0]), 1)
actor objective = -alpha * H_selected - Q / stop_gradient(S)
```

The scale starts at one. Each outer actor update estimates linearly
interpolated percentiles across the replay batch at the first latent timestep,
using the same decoded Q values and configured ensemble reduction as the actor.
The updated scale is used immediately at every rollout depth.
`sac_actor_loss_scale_tau` controls `tau_s` and defaults to `0.01`, matching
upstream's default `tau`; set it to the comparator's `tau` when that value is
overridden. The configured AMBI actor-Q reduction remains independent of
scaling (upstream's actor uses `mean_pair`).

By default, inner SAC applies the same Q-only division, with one snapshot of
the outer scale frozen for the whole real action. Setting
`inner_actor_loss_scale_update="per_update"` instead updates that local snapshot
from each actor minibatch's selected Q values before using it, with the same
percentiles, floor and `sac_actor_loss_scale_tau`. This requires ordinary inner
SAC without an explorer population. The local updates never modify the outer
scale, and each new real action starts from the current outer scale again.
Entropy and outer/inner policy regularizers in the actor objective are
unscaled, as are rewards and reward-only Bellman targets.
TD3 and MPPI do not use this option. The diagnostics
`actor_effective_ent_coef` and `inner_effective_alpha` report the unchanged
entropy coefficient appearing in the optimized objective.

Q scaling requires fixed temperatures: set a positive numeric `ent_coef` and
use `inner_temperature_mode="fixed"` or `"inherit_outer"` for active inner SAC.
Outer `auto`/`auto_<initial>` and inner `auto` remain rejected. Ordinary SAC
without explorers also supports entropy-augmented critics with Q scaling:

```text
critic target = r + gamma * (1 - terminated) * (Q_target + alpha * stop_gradient(S) * H_selected)
```

The critic stores raw returns, so its entropy coefficient is `alpha * S`.
Outer targets use the current outer scale before that update's actor EMA.
Inner targets use the current private action-local scale before the paired
actor update; after the actor advances it, the next critic update uses the
new local value. A `per_action` scale remains frozen throughout the solve.
Value-equivalence probes compare fresh inner solves using the inherited outer
scale. Coefficient-unit provenance is added to checkpoint target metadata only
for this combination; existing reward-only and unscaled identities are unchanged.
An adaptive scale changes the effective regularized-return objective over time,
and the critic and actor within a pair use their respective pre/post EMA snapshots.

Entropy-augmented Q scaling is rejected for explorer populations. Unused inner
SAC settings do not restrict non-SAC operators. Percentile tracking solely for
the behavior regularizer's `quantile_gate` schedule remains compatible with
adaptive temperatures and entropy-augmented critics when Q scaling is off.

The historical `ambi_humanoid_walk_base_percentile_normalized` algorithm preset
and experiment manifest retain their recorded scientific settings. They combine
Q scaling with adaptive temperatures and entropy-augmented critics, so current
configuration validation rejects them. To start a new experiment, explicitly
select compatible settings and a new run identity; the historical preset is
not a supported current training recipe.

Earlier versions divided the entire SAC-plus-regularizer loss by the scale.
The checkpoint specification now records `application="q_only"`; structured
checkpoints carrying the old application are rejected before state mutation.
For intentional weight transfer, load the raw model state into a fresh agent,
which resets the running scale. The existing mode name and default `"none"`
are retained; enabling normalization is still an explicit configuration choice.

### Selectable actor and critic entropy

`outer_actor_entropy_mode` and `inner_actor_entropy_mode` independently select
`"squashed"` (the default) or `"tdmpc2_scaled"` for actor and temperature
optimization and the corresponding entropy-augmented critic. The default uses
the existing stable squashed-action entropy sample, `H_selected = -log_pi`.
With Q scaling off, each learner's entropy-augmented Bellman target is:

```text
y = reward + discount * (1 - terminated) * (Q_target + alpha * H_selected_next)
```

The outer critic uses the outer mode and temperature. Inner SAC uses the inner
mode; separate critics use their own policies and temperatures. Value-equivalence
losses and diagnostics use the inner target statistic, preserving gradients
through predicted latents for the loss. Critic targets themselves stop gradients.
Reward-only targets have no entropy bonus. With Q scaling enabled, the supported
fixed-temperature entropy-augmented case uses the `alpha * S` coefficient
described above. Action-entropy metrics, the finite-horizon prior handoff, and
the native TDAMBI operator retain their existing definitions.

In scaled mode the critic estimates returns augmented by the selected surrogate
statistic. The manuscript's standard maximum-entropy SAC equations describe
the default `squashed` mode; scaled entropy is an explicit experimental variant.

Scaled mode uses the same Gaussian sample and action, with no additional
policy forward or random draw. For joint pre-tanh Gaussian log-probability
`ell_G`, normalized action `a`, and action dimension `d`, it computes the
literal [upstream implementation](https://github.com/nicklashansen/tdmpc2/blob/8bbc14ebabdb32ea7ada5c801dc525d0dc73bafe/tdmpc2/common/world_model.py):

```text
ell_TD = ell_G - sum(log(relu(1 - a**2) + 1e-6))
H_scaled = -ell_TD * ((d * ell_G) / (ell_TD + 1e-8))
actor objective = -Q / S - alpha * H_selected
temperature objective = log_alpha * stop_gradient(H_selected - H_target)
```

Gradients flow through the entire scaled expression. The implementation does
not cancel the ratio or detach its scale. The ordinary `log_prob` and
`actor_entropy` diagnostics remain the actual stable squashed-action quantities.
When scaled mode is active, `actor_scaled_entropy` reports the selected
statistic and `actor_entropy_bonus` reports `alpha * H_selected`; inner and
explorer metrics use their existing prefixes. Outer actor losses retain the
`rho**t / (H+1)` temporal reduction, outer temperature updates retain normalized
temporal weights, and inner updates use minibatch means. Policy regularizers
remain independent and unscaled.

Outer `ent_coef` and the existing inner temperature settings keep their current
meanings, initialization, and numerical floors. Automatic tuning in scaled
mode requires an explicit numeric `target_entropy` or `inner_target_entropy`
for that learner: `"auto"` and `"inherit_outer"` are rejected as scaled tuning
targets. A squashed inner learner may inherit its target only from a squashed
outer learner; when outer entropy is scaled, set `inner_target_entropy="auto"`
for the usual inner action-entropy target, or supply a numeric inner target.
Unused targets do not impose these restrictions on fixed coefficients.
Coefficient inheritance copies the coefficient without converting its units.
`inner_temperature_grad_clip_norm=null` disables inner temperature-gradient
clipping while retaining its norm diagnostic, matching the outer temperature
optimizer. The default remains `20.0`; finite positive thresholds retain their
existing behavior. Explicit scaled-entropy targets are in the selected statistic's
units; for example, the TD-AMBI objective study uses `-441` for both learned
outer and learned inner temperatures.

Scaled inner entropy supports ordinary SAC, frozen-random and adaptive
parameter-noise Gaussian explorers, and both actors in `separate_critics`.
It is rejected for `shared_mixture` and non-SAC operators. Selecting scaled
outer entropy does not change the inner selection.

For the fixed-coefficient TD-MPC2 outer-actor comparison, use this configuration
alongside matching network dimensions and initialization:

```json
{
  "outer_actor_entropy_mode": "tdmpc2_scaled",
  "ent_coef": 0.0001,
  "outer_q_actor_reduction": "mean_pair",
  "num_q": 5,
  "q_pair_size": 2,
  "log_std_mapping": "tdmpc2_tanh",
  "log_std_min": -10,
  "log_std_max": 2,
  "sac_actor_loss_scale_mode": "tdmpc2_percentile_range",
  "sac_actor_loss_scale_tau": 0.01,
  "rho": 0.5,
  "actor_lr": 0.0003,
  "actor_adam_eps": 0.00001,
  "grad_clip_norm": 20,
  "outer_behavior_policy_kl_schedule": "none",
  "outer_critic_target": "reward_only",
  "inner_sac_critic_target": "reward_only",
  "inner_temperature_mode": "inherit_outer"
}
```

The inner actor retains its default squashed entropy statistic and inherits the
fixed outer coefficient. Both critics use reward-only targets as required by
Q scaling.
The outer actor architecture and objective match TD-MPC2 given matching latent
inputs, critic values, policy noise, sampled critic heads, normalization state,
and optimizer state. Other AMBI training and inner-loop choices remain independent;
this recipe does not make the full algorithms equivalent. Automatic tuning of scaled entropy is an
AMBI extension; the fixed-coefficient recipe is the TD-MPC2 comparison. Existing
experiment presets are unchanged.

Checkpoint `critic_target_spec.entropy_semantics` and resolved run metadata
`critic_entropy` record each critic's bonus as `none`, `squashed_action_entropy`,
`squashed_mixture_entropy`, or `tdmpc2_scaled_entropy`. Historical checkpoints
missing these fields used squashed entropy for entropy-augmented targets, even
when their actors selected scaled entropy. Exact resumes reject a changed critic
entropy statistic before mutating state; portable target metadata remains
provenance for intentional weight transfer. Checkpoint entropy specifications
also record actor modes and temperature-target semantics.
Scientific run identity treats missing modes and explicit
`"squashed"` defaults equivalently. Incompatible training resumes fail during
preflight. Loading a structured outer checkpoint permits inner-objective
changes for evaluation, while an exact training resume checks both objectives.
Historical missing mode metadata means `"squashed"`. Explicit raw-model weight
transfer remains available for starting a fresh learner with a new objective.
Evaluation matrices still apply their explicit overrides: a matrix that sets
`inner_target_entropy="inherit_outer"` must use `"auto"` or a numeric target
for squashed inner SAC on scaled-outer weights. For a prior-only variant that
sets `inner_operator="none"`, use `inner_actor_entropy_mode="squashed"` if the
saved run selected scaled inner entropy. Existing matrices are not rewritten.

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

The online paired-controller probe compares an unoptimized network controller
against an online optimized controller in the real environment. For each
evaluation episode it resets independent auxiliary environments with the same
explicit seed. In AMBI, `outer` is deterministic outer-policy control and
`fresh_inner` performs a fresh inner SAC solve at every visited state. In the
TD-MPC2 baseline, the same names mean deterministic network-policy control and
eval-mode MPC/MPPI, respectively. Keeping the metric schema identical makes the
two methods directly comparable without relabeling plots.

The primary quantity is the episode-paired fresh-inner-minus-outer return. The
fixed-target root-Q action gain collected along the optimized-controller
trajectory is a model-predicted diagnostic, not a substitute for that real
return difference. Positive predicted gain with a negative real return delta
is evidence of critic/model optimism or compounded distribution shift.

Enable the observational probe with:

```json
{
  "eval_inner_comparison": true,
  "eval_inner_comparison_episodes": 5,
  "eval_inner_comparison_seed": 12345
}
```

AMBI configurations must additionally pin `inner_diagnostic_rollouts=0`; this
retains the fixed-target root-Q comparison without adding imagined diagnostic
trajectories. The TD-MPC2 probe requires state observations, `mpc=true`, and at
least one policy-prior trajectory. Both probes preserve learner and global RNG
state, use auxiliary environments rather than the training environment, and
merge their numeric outputs into the ordinary evaluation event. Figure-facing
uncertainty must be aggregated across independent training seeds; within-event
episode dispersion and correlated per-state Q gains are only diagnostics.

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

For TD-MPC2, the root diagnostic is the all-head mean target-critic difference
`Q_target(z, a_mpc) - Q_target(z, a_network)`. It adds one batched critic call
per MPC-controlled root and no diagnostic model rollout. Reported planner model
steps count policy-prior transitions plus every candidate transition across
the effective MPPI iterations. The maintained 14M Humanoid Walk state baseline
enables five fixed-seed pairs at step zero and every 100,000 decisions. Because its existing
evaluation already runs ten MPC episodes, the probe adds approximately 50%
more evaluation-time planner work plus five comparatively cheap network-policy
episodes; training compute is unchanged.

The three `ambi_anchor_kl_{smooth,quantile,dual}` Humanoid Walk manifests run
this five-episode protocol at the existing 50,000-decision cadence alongside
Figure 1 value calibration. Their paired-controller settings are observational;
after removing those settings and their W&B labels, the only learning-axis
differences from the base are the declared replay behavior-policy-KL controls.
