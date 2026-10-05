# Sampled diagnostics for Bernoulli transfer

This is the measurement contract for the [575k Bernoulli transfer campaign](BERNOULLI_TRANSFER_575K.md).

Unlike the original lean discovery campaign, this campaign requires sampled
measurements at zero-based decisions **0,1,25,100,250,499**. At each sampled
root the evaluator compares the frozen prior, preceding donor when present,
actual post-transfer initialization and final adapted learner. It captures only
the initial/final actor and critic state dictionaries; it does not run extra SAC
forks, record each optimizer update or save large persistent network snapshots.

The shared probe bank contains 32 states: the actual root and 31 independently
sampled one-step successors under the frozen prior. Eight candidate actions per
state include prior/donor means and paired samples. Eight frozen-model Monte
Carlo continuations estimate reward-only finite-H returns with the same frozen
prior continuation and horizon tail. **Every stage uses exactly the same action
bank and labels**, identified by a target hash. These labels are independent of
the evaluated critic, but remain model references rather than real-return truth.

The measurements are:

- Critic fixed-target RMSE, bias, centered/action-relative error, empirical
  action-selection regret and root action-ranking statistics. Winner-based
  regret uses the finite sampled bank; it is not a population optimum estimate.
- Critic action-gradient magnitude and paired model-return change for positive
  versus negative pre-tanh perturbations of size 0.05 at each stage's mean action.
  Continuation remains the frozen prior, so changes in the critic cannot change
  the evaluator. Conditional Monte Carlo uncertainty and zero-gradient fraction
  accompany the mean. States and Monte Carlo replicates are not extra episodes.
- Actor Gaussian KL, mean displacement and standard deviation, first-action
  model-return gain with fixed prior continuation, plus root return using that
  stage's current actor as continuation. The latter evaluates a different policy,
  while the underlying world model, reward and tail remain fixed.
- Penultimate feature effective rank, sample-rank ceiling and their ratio for
  the actor and each critic head. Relative mean-absolute activity and variability
  use 10% of the layer average; they do not classify a Mish neuron as dead.
  The 32-state centered rank ceiling is at most 31, not the network width.

At decisions **25 and 250**, prior and actual-initialization clones additionally
fit identical fixed labels for four Adam steps with paired minibatches and
isolated dropout RNG. Even/odd state indices define disjoint train/held-out
sets. Critic fitting uses decoded per-head Q regression; actor fitting imitates
the empirically best mean action in the candidate bank. Step-zero error, final
error, parameter movement and gradients are retained. This small stationary
learning test is a diagnostic of fitting progress, not conclusive evidence of
plasticity loss; the actor test does not assess covariance fitting.

Probe RNG is independent of policy, minibatch and Bernoulli-mask streams.
Synchronized snapshot/probe wall time is recorded as `diagnostic_seconds` and
excluded from `control_seconds`; ordinary donor construction/export remains
controller work, including the lightweight mask-count and parameter-distance
statistics. Small trace-bookkeeping overhead can remain on the six sampled
latencies. Uninstrumented decisions provide the dominant timing evidence.
The immutable manifest records all diagnostic settings and source hashes.

A labeled smoke forces sampling at decisions 0 and 1 and one stationary probe
at decision 1. It first runs an uninstrumented paired episode, then repeats it
with diagnostics. Exact actions, rewards, training metrics, final donor state
and private learner RNG must match; otherwise the smoke fails. Each completed
smoke episode records `diagnostic_isolation_verified: true`. Coverage validation
also requires finite numeric measurements from every requested stage/family,
not merely enabled configuration flags. Production requires all smoke receipts.

The existing campaign evaluator and Oscar launcher accept this matrix via
`--campaign` / `CAMPAIGN_MATRIX`. Publication uses the separate
`ambi-inner-bench` project and includes distinct mechanism colors, return/time
comparisons, sampled initialization-versus-final metrics, stationary fit gains,
and diagnostics overhead. No historical results are silently substituted into
new episode output.
