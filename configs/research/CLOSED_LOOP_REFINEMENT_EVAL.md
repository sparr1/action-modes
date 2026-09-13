# Full-episode closed-loop prior-refinement evaluation

**Short name:** closed-loop refinement eval.

**Reference recipe identifier:** `closed-loop-refinement-v1`.

The name identifies an evaluation procedure. The backbone, checkpoint selection,
entropy objective/coefficient, and requested compute comparisons are separate
experiment choices. The v1 recipe below records the established starting point;
it is not a claim that these hyperparameters are optimal.

## What is evaluated

Evaluate the real controller throughout an ordinary episode. At **every real
decision**, encode the new observation, create fresh copies of the frozen prior
actor and online critic, initialize an inner target critic from that online
critic, and create fresh optimizers and replay. Run the requested inner solve,
then execute the adapted actor's deterministic `tanh(mu)` action. Repeat this
process at the next real observation. The previous solve's actor, critic,
optimizers, and replay do not carry over to the next real decision.

The outer encoder, dynamics, reward model, actor, critics, and saved scale stay
frozen. No outer learning or actor/critic writeback occurs. Learning and imagined
collection inside each solve remain stochastic even though execution uses the
mean action.

The primary outcome is **undiscounted raw environment reward over the ordinary
full episode**, without an entropy bonus or a bootstrap after the final real
decision. For the Humanoid Walk state reference, this is up to 500 decisions;
each decision repeats its action for two raw control steps and sums their
rewards. Preserve both Gym and raw DMControl time limits and genuine termination.

## Reference recipe v1

| Setting | Reference value |
|---|---|
| Environment | Humanoid Walk, state observations |
| Inner actor / critic initialization | Inherited prior / inherited online critic |
| Replay and optimizer lifetime | One real decision; retained across its inner rounds |
| Collection per round | N128 imagined rollouts, H1 |
| Update order per round | Collect, then C32 critic updates, then A4 actor updates |
| Minibatch | B256 |
| Replay | Capacity 2,048, sampling with replacement |
| Actor / critic learning rates | 0.0003 / 0.0003 |
| Critic targets | Finite-horizon, reward-only; frozen outer terminal bootstrap |
| Q reductions | Actor and outer terminal mean-pair; inner target min-pair |
| Q scale | The selected checkpoint's saved divisor, fixed during the solve |
| Temperature | Fixed; record the explicitly selected entropy mode and coefficient |
| Policy parameterization | Inherit mean and std; reference uses TD-MPC2 tanh log-std mapping, bounds −10 to 2 |
| Execution | Final actor mean action after every fresh solve |
| Environment seeds | 101–120 |
| Controller seeds | 55, 56, 57, paired across conditions |
| Episode limit | 500 real decisions |

The established compute panel is **J1/J2/J4**. J counts complete rounds:

| Budget | Imagined transitions per real decision | Critic updates | Actor updates |
|---|---:|---:|---:|
| J1 | 128 | 32 | 4 |
| J2 | 256 | 64 | 8 |
| J4 | 512 | 128 | 16 |

Select the requested J values explicitly. Increasing J changes data collection,
critic fitting, and actor updates together; it does not isolate actor compute.
Keep the resolved optimizer, dropout, target-network, compilation, and policy
settings with every result. The reference has critic dropout enabled, no KL or
action-distance penalty, and no real-replay mixing.

For a new backbone, use its actual checkpoint, metadata, source identity,
discount, and saved Q scale. Verify compatibility with this recipe. In
particular, a soft-Q critic or a different policy parameterization requires an
explicitly documented variant; do not silently relabel its semantics as the
reward-only v1 reference. Do not copy the old checkpoint's numeric scale or
assume every checkpoint has a saved scale.

## Baselines, pairing, and uncertainty

Include a frozen-prior mean-action episode reference. The established
prior-initialized refinement baseline uses **inner alpha = 0**. Inheriting actor
and critic weights does not imply inheriting a nonzero entropy coefficient.
An entropy comparison must specify its mode and coefficient separately; the
protocol name does not automatically select alpha 0.0021 or scale alpha by the
action dimension.

Reuse compatible existing prior and alpha-zero results after verifying the
checkpoint hash, scientific implementation, runtime/protocol, seeds, complete
coverage, and immutable bundle hashes. Do not automatically rerun alpha zero.
References from a different checkpoint are not reusable performance baselines.

Form paired differences at matched environment and controller seeds. Average
the three controller repetitions within each environment seed, then weight the
20 environment seeds equally. Report means, variability, and paired 95%
percentile bootstrap intervals using **2,000 environment-seed cluster resamples**
(reference bootstrap seed 20260912). There are 20 top-level clusters, not 60
independent training seeds. Intervals are exploratory and unadjusted for
multiple comparisons. Pairing seeds does not keep later states or imagined
minibatches identical after different actions are executed.

## Model diagnostics accompanying the episode evaluation

Retain **32 sampled model-return probes** at the actual actor initialization and
after every completed round, using diagnostic randomness independent of the
learner. For v1/H1, the score is predicted reward plus checkpoint-discounted
frozen outer online Q at a sampled outer-policy terminal action. It uses raw
discounted reward units, with no entropy bonus and no Q normalization.

Retain reward/bootstrap components, paired gains over initialization and prior,
actual cumulative update counts, per-decision summaries, and inner traces.
For J4 the actor-update probe axis is 0/4/8/12/16 and the critic axis is
0/32/64/96/128. Record the real initialization snapshot; `inner_rounds=0`
executes the prior and is not a substitute for a learned-controller snapshot.

These sampled-policy diagnostics average the controller's own visited states;
execution uses mean actions. They are not predictions of the undiscounted
500-decision episode return. Label sampled actor-training saturation separately
from saturation of executed mean actions; the historical episode traces do not
record executed action vectors.

Preserve complete paired bundles and standalone HTML, with full-episode returns,
paired gains, J/optimizer-work axes, and model-probe round/update axes in the
selected W&B comparison run. Keep existing checkpoint-axis curves intact.
Validate complete coverage before publishing final comparisons. Report probe
and control time separately and identify historical reference timing.

## Distinct evaluation procedures

This protocol refines the controller again at every real decision. A shared-root
model-only probe, or one adapted action/H-step prefix followed by a frozen-prior
Monte Carlo continuation, answers a different question. Name those procedures
explicitly; they do not replace full-episode closed-loop refinement evaluation.
The latter branching calibration is documented in
[TOGO_CALIBRATION.md](TOGO_CALIBRATION.md).

## Reproducible reference implementation

The established September 2026 comparison uses source commit
`d25e29dc0d84a2e978a33b27dd08154f487153de`, retaining the earlier alpha-zero
round-scaling results. At that commit, consult
`configs/research/ENTROPY_EPISODES.md`, the six
`configs/research/entropy_{native,squashed}_j{1,2,4}_h1_200k.json` matrices,
`slurm/ambi_entropy_episode_campaign.py`, and
`slurm/ambi_entropy_episode_report.py`. Those experiment-branch files may not
be present in a different checkout; verify the intended implementation and
resolved configuration before execution.

That reference used `rwgao_b-brown-university/ambi/mey3rxj8` at 200,000 training
decisions, gamma 0.99, and saved Q divisor 11.182598114013672. These identify
the historical reference, not a mandatory backbone or numeric scale for future
uses of this named evaluation.

Example request:

> Use closed-loop refinement eval, reference recipe v1, on [backbone/checkpoints].
> Compare [J values and explicitly selected entropy settings]. Keep the paired
> full-episode protocol and model probes, and reuse verified prior/alpha-zero
> results where compatible.
