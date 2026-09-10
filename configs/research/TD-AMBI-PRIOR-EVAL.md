# Frozen TD-AMBI prior bank evaluation

The four `td_ambi_prior_bank_*.json` matrices evaluate every saved checkpoint
from 25,000 through 2,000,000 training decisions, at 25,000-decision intervals.
Each bank contains 80 checkpoints from one Humanoid Walk state-observation
training run with seed 55. These are exploratory comparisons on four learned
backbones, not independent training replications.

| Matrix suffix | Source W&B run | Critic target | Actor Q term | Adaptive inner scalar |
| --- | --- | --- | --- | --- |
| `reward_qscale` | `rwgao_b-brown-university/ambi/mey3rxj8` | Reward only | `Q / S` | `S` |
| `entropy_qscale` | `rwgao_b-brown-university/ambi/zdnoyhbt` | Entropy augmented | `Q / S` | `S` |
| `entropy_autotemp` | `rwgao_b-brown-university/ambi/hw0nlj29` | Entropy augmented | `Q` | `alpha` |
| `reward_autotemp` | `rwgao_b-brown-university/ambi/jirflxz1` | Reward only | `Q` | `alpha` |

Every matrix derives its algorithm and environment from the checkpoint metadata.
It overrides inner controls only, preserving the saved outer objective, actor
and critic architectures, and scalar state. Keep checkpoint metadata sidecars
adjacent to their weights. Default selection evaluates the frozen prior and
MPPI first; the SAC selectors are explicit.

| Selector | Evaluation |
| --- | --- |
| `controller/prior` | Frozen policy mean, no imagined work |
| `controller/mppi` | Native TD-MPC2 MPPI on the frozen world model and online critic |
| `inner/fixed` | J5 C1/A1 inner SAC, checkpoint `alpha` and `S` held fixed throughout each solve |
| `inner/adaptive` | Same SAC schedule, adapting only the scalar in the table above |

All controllers use the same five environment seeds 101–105, controller seed
55 and 500-decision episodes. The evaluator returns policy means for prior/SAC.
MPPI uses native stochastic weighted-elite selection without additional
execution noise, with H3, 512 samples, 64 elites, 24 policy trajectories,
standard-deviation bounds 0.05–2, and score temperature 0.5. Six configured
iterations become eight for Humanoid's 21 action dimensions. Its mean warm
starts from the previous decision within an episode. The score is raw predicted
reward plus the frozen online critic's mean-pair terminal value at a sampled
prior action; it adds no separate entropy correction and applies no Q scale.
The reward/Q-scale backbone is the TD-MPC2-equation comparator. The entropy
backbones retain their learned soft-Q tails.

SAC uses J5, N512, H3, minibatch size 512, and replay capacity 7,680. Each vector
step generates 512 transitions and triggers one paired critic-then-actor update,
followed by target interpolation. Each decision therefore has 15 critic updates,
15 actor updates and 15 target updates. Adaptive-alpha variants additionally
perform 15 temperature updates. Replay samples accumulated transitions with
replacement and retains the full solve. Inner losses average transition
minibatches; they do not introduce sequence-level `rho` weighting.

Each real decision clones the saved actor and online critic, initializes its
private target critic from the saved outer target, and starts fresh Adam moments
and replay. The actor/critic learning rates are 0.0003, actor epsilon is `1e-5`,
critic epsilon is `1e-8`, critic loss coefficient is 0.1, gradient clipping is
20, and target interpolation rate is 0.01. Both actors use `tdmpc2_scaled`
entropy with TD-MPC2 tanh log-standard-deviation mapping and bounds `[-10, 2]`.
Actor Q uses mean-of-two heads; Bellman targets use minimum-of-two heads.
The saved prior-only configurations neutralized inactive inner options; these
SAC selectors explicitly restore the intended active options.

The fixed mode copies the saved `S` into private inner state and holds it fixed
(`inner_actor_loss_scale_update="per_action"`). The adaptive Q-scale mode starts
from that same saved value and updates only its private scale with percentile
EMA 0.01 before each actor loss (`"per_update"`). Alpha remains fixed in both
Q-scale banks. For the entropy/Q-scale critic, the entropy bootstrap contribution
is `alpha * S * H_scaled`, using the current scale before the subsequent actor
EMA update. The reward-only critic adds no entropy contribution.

Both auto-temperature banks keep Q scaling disabled. Fixed mode inherits the
saved alpha; adaptive mode initializes a private learned alpha from the saved
value on every decision, with learning rate 0.0003, unclipped temperature
gradients and target entropy -441 in `tdmpc2_scaled` units. This adaptation also
applies to the reward-only bank: its actor adapts alpha while its critic remains
reward-only. No local scalar, optimizer, actor, critic or target is written back.

Workers should save a completed `controller/prior` bundle per checkpoint and
pass it through `--reference-bundle` to each later controller. This pairs all
reported return improvements with the same checkpoint's prior episodes without
repeating them. The matrix's internal `inner` reference is `fixed`; the external
prior reference supplies prior-relative episode deltas to both SAC variants.
Use explicit new evaluation-series attempts for these four banks and each
resolved controller, with one CPU publisher per curve and checkpoint/planner
parallel GPU workers. Preserve completed output paths and verified reference
bundles when resuming.

Focused validation is in `tests/test_td_ambi_prior_bank_presets.py`,
`tests/test_td_ambi_frozen_scalars.py`, and the frozen MPPI controller tests.
It checks checkpoint-objective preservation, portable saved-scalar restoration,
full J5 paired update counts, action-local reset behavior and immutable outer
state. Broader objective derivations and training differences are documented in
[`TD-AMBI-STUDY.md`](../dmcontrol/TD-AMBI-STUDY.md).
