# Reward-only critic prior comparison

This study prepares six reward-only critic backbones at seed 55: four matched
counterparts to the [SAC prior parameterization study](SAC_PRIOR_DIAGNOSTICS.md)
and two fixed-temperature comparisons. The four automatic-temperature cells
change the outer critic's Bellman target while retaining the actor recipe,
collection, initialization, model and training budget. The files prepare a
comparison; they do not submit jobs.

| Algorithm configuration | Log-std mapping | Joint post-tanh entropy target |
| --- | --- | --- |
| [`ambi_prior_reward_clip_target21`](algs/ambi_prior_reward_clip_target21.json) | Direct clipping, [−10, 2] | −21 nats |
| [`ambi_prior_reward_clip_target10p5`](algs/ambi_prior_reward_clip_target10p5.json) | Direct clipping, [−10, 2] | −10.5 nats |
| [`ambi_prior_reward_smooth_target21`](algs/ambi_prior_reward_smooth_target21.json) | Smooth tanh, [−10, 2] | −21 nats |
| [`ambi_prior_reward_smooth_target10p5`](algs/ambi_prior_reward_smooth_target10p5.json) | Smooth tanh, [−10, 2] | −10.5 nats |
| [`ambi_prior_reward_clip_fixed0p0001`](algs/ambi_prior_reward_clip_fixed0p0001.json) | Direct clipping, [−10, 2] | Inactive; fixed α = 0.0001 |
| [`ambi_prior_reward_smooth_fixed0p0001`](algs/ambi_prior_reward_smooth_fixed0p0001.json) | Smooth tanh, [−10, 2] | Inactive; fixed α = 0.0001 |

The [manifest](experiments/ambi_prior_reward_critic_study.json) uses one trial,
seed 55, for six new backbones. Each starts from native random
initialization and trains for 2 million decisions with 80 scheduled checkpoints
at 25,000-decision intervals: 12 million decisions and 480 checkpoints total.
This single-seed screen does not estimate variability across training seeds.
Existing seed-55 SAC runs provide the four automatic-temperature controls;
do not resume their learned weights or temperatures into these runs. Each new
automatic-temperature cell should be compared with its matching mapping,
target and seed. The original two-seed SAC study remains unchanged.

## Objective and controlled differences

For the four automatic-temperature cells, the sole active algorithm change
from their SAC controls is `outer_critic_target="reward_only"`.
For a sampled next action `a' ~ pi(. | z')` and a sampled critic pair `P`,
the target changes from

```text
y_soft   = r + gamma * (1 - terminated) * (min_P Q_target(z', a') - alpha * log pi(a' | z'))
y_reward = r + gamma * (1 - terminated) *  min_P Q_target(z', a')
```

The actor still minimizes the temporally weighted expectation of
`alpha * log pi(a | z) - mean_P Q(z, a)`, using corrected post-tanh density.
In those four cells, automatic alpha still starts at 1, learns at `3e-4`, and uses the same
normalized rho-weighted entropy residual. Alpha's subsequent trajectory may
differ because the changed critic changes the actor. Actor Q normalization
remains disabled. The reward-only critic evaluates rewards induced by a
stochastic entropy-regularized policy: removing the explicit future-entropy
term does not disable exploration.

The two fixed-temperature cells instead use `ent_coef=0.0001` from the first
actor update, with no temperature optimizer. The retained `target_entropy=-21`
and `ent_coef_lr` fields are inactive: there is no reason to duplicate each
fixed-alpha run for both target entropies. Their tags mark the target inactive.
Comparisons against automatic alpha test the entire temperature recipe,
including automatic alpha's initial value of 1 versus fixed 0.0001. They do not
isolate the effect of adaptation independently of early temperature magnitude.

The fixed coefficient `0.0001` is a low-entropy-incentive experiment requested
for this study. It matches TD-MPC2's numerical coefficient, but it does not
match TD-MPC2's regularization strength: this study uses raw Q without actor
Q normalization and corrected joint post-tanh entropy. A coefficient learned
with entropy-augmented critics does not establish a suitable fixed value for
reward-only critics. Fixed alpha also does not fix action variance or entropy;
inspect early learning, action variability, entropy and saturation. Both actor
mappings use the same coefficient. A second fixed value would add two
backbones, not four.

The log-std bounds remain `[-10, 2]`, allowing pre-tanh standard deviations
from `exp(-10)` to `exp(2)` (approximately 0.000045 to 7.39). Direct clipping
maps raw output zero to log-std zero; smooth tanh maps it to -4. These are
native parameterizations with different initial policy distributions, even
when raw weight initialization is paired. Changing the bounds would also
change the smooth mapping's offset and sensitivity, so bounds are held fixed.

All remaining active settings match the SAC counterparts, including H3,
batch 256, UTD 1, rho 0.5, discount 0.99, replay capacity 1 million, model size
5, five distributional critics, and the explicit 2,500-step seed/pretraining
settings with their existing inclusive warmup boundary. Ordinary evaluation,
MPPI and inner optimization remain disabled. Dormant inner settings are copied
unchanged, including `inner_sac_critic_target="entropy_augmented"`; they execute
no updates and do not define a future inner-adaptation or bootstrap protocol.
That handoff requires its own explicit configuration.

This uses entropy in the actor and excludes it from the critic target, as in
TD-MPC2. All six cells retain no Q normalization; four use automatic temperature
and two use fixed temperature. Official TD-MPC2 uses a fixed entropy coefficient
of `1e-4`, percentile-based Q normalization and its scaled entropy calculation.
Corrected post-tanh entropy is also a deliberate change from that calculation.
The official implementation's mean-pair actor and min-pair target reductions
match both arms of this comparison. The four automatic-temperature comparisons
isolate critic entropy; the fixed-temperature additions compare actor recipes
within reward-only critics. Neither is an exact TD-MPC2 reproduction or the
full soft SAC objective.

Implementation provenance: the upstream TD-MPC2 checkout was inspected at
commit `8bbc14ebabdb32ea7ada5c801dc525d0dc73bafe`, including
`tdmpc2/tdmpc2.py` actor/target updates and
`tdmpc2/common/world_model.py` policy entropy. The TD-MPC2 paper, Eq. 3,
also specifies a reward-only TD target. AMBI already implements both target
modes in `RL/tdmpc2_core/ambi_agent.py`; no learner-equation change is needed.

## Diagnostics and interpretation

All [existing diagnostic sources, cadence and definitions](SAC_PRIOR_DIAGNOSTICS.md)
remain enabled: initialization, the pretraining boundaries, every 100 updates
through 10,000, every 1,000 thereafter, and the final update; 32 fixed warmup
observations and 32 private noise samples per observation.

Reward-only diagnostic manifests identify decoded Q as a **predicted
reward-only return, not a measured return**. Soft critic manifests retain
their existing soft-Q label. The resolved configuration and checkpoint metadata
record the critic target mode. Do not interpret a cross-arm change in Q level
as a performance gain: the targets have different meanings. Compare actual
episode rewards, action saturation, entropy/temperature and learning speed at
matched environment steps; use separate frozen-policy evaluation for a
checkpoint performance claim. Soft Q and reward-only Q are not interchangeable
bootstrap targets in later calibration.

New runs use W&B project `ambi`, group `ambi-prior-reward-critic-20260914`,
and automatic names containing the configuration and resolved seed. Their tags
identify reward-only critics. Event-indexed transport, environment-step and
optimizer-update axes, local bundles and artifact publication are unchanged.
The manifest avoids partial nested `alg_params` overrides.

## Validation

Run with the locked DMControl interpreter from the repository root:

```bash
environments/dmcontrol/.venv/bin/python -m pytest -q \
  tests/test_ambi_prior_reward_critic_study_configs.py \
  tests/test_ambi_prior_sac_study_configs.py \
  tests/test_ambi_outer_entropy_modes.py \
  tests/test_ambi_tanh_logprob_integration.py \
  tests/test_ambi_actor_loss_scaling.py \
  tests/test_outer_policy_diagnostic_recorder.py
```

The matched configuration tests compare full raw and resolved settings for the
automatic-temperature cells against the original seed-55 arms, permitting only
the outer target and W&B identities to change. Fixed-temperature cells are
compared against their matching reward-only automatic-temperature settings.
Tests also check the six cells, checkpoint budget, inherited inner settings
and publication metadata. Existing target tests verify entropy-free
targets, termination masking and ignoring policy density in reward-only mode.
Recorder tests cover reward-only, soft and legacy manifest labels through
artifact serialization. These local checks do not establish CUDA readiness
for a new launch; allocation-specific validation remains a launch step.

For this launch, the user requested production submission before smoke tests.
Validate the six configurations and GPU training path after submission, then
record the results in the launch receipt. Historical local checks from earlier
configuration revisions do not establish acceptance of this exact launch.
