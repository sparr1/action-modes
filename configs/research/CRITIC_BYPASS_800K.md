# Full-episode critic bypass at 800k

This evaluation tests whether selecting an existing SAC replay action improves
real closed-loop control relative to executing the adapted actor mean. It uses
the pinned auxiliary-return checkpoint and the fresh H1/J6 C16/A4/N128/B256
return/return recipe from the matched-state action audit.

The three intervention arms execute the actor mean, the highest expected
inner-Q candidate, or the highest directly scored H1 model-return candidate.
Every solve collects all 768 replay transitions. The candidate bank contains
the adapted actor mean, frozen prior mean, and all 768 actual replay actions.
The means come first so an exact tie does not arbitrarily choose a replay row.
The learned-Q reduction is expected mean-pair (all-head mean in evaluation
mode). Model scoring uses predicted reward plus discounted frozen auxiliary
return Q, with the original frozen terminal actor and bounds, no entropy bonus,
and the same expected terminal reduction as the current target. Thirty-two
shared terminal-policy noise draws score all candidates. Held-out diagnostics
use 32 independent draws and never affect execution or learner randomness.

Only the final selector changes between the SAC arms. Actor, critic, target,
replay, optimizers, and temperature initialize fresh at every real decision;
the full outer learner stays frozen. Named solve seeds depend on episode and
decision, not arm. Root-local diagnostics compare selectors on an identical
bank; full-episode trajectories naturally diverge after different actions.
Actor-mean versus greedy selection additionally changes how an entropy-trained
policy is executed. The learned-Q versus direct-model contrast is the cleaner
test of critic approximation in action selection.

Run twenty paired environment seeds 101–120 with controller seed 55, up to 500
real decisions per episode, honoring true termination and time-limit truncation.
Report undiscounted real episode return, differences from the prior and actor
baseline, seed win rates, sample variance, and environment-seed uncertainty.
MPPI H3 uses the action-audit's return-Q settings, prior proposal shifting, and
per-decision named RNG. Historical MPPI episodes with a persistent planner RNG
are not identical reference realizations. Verified compatible prior episodes
may be reused with explicit provenance; historical hardware timings must not
be pooled with fresh A5000 timings.

All new controllers run eagerly on A5000 GPUs. Synchronized controller timings
include the solve, necessary bank handling, action scoring, and conversion to
the environment action. Sparse shadow scoring and held-out validation are
recorded separately and excluded from controller latency. Report mean and p95
latencies, selection overhead, total evaluation time, and timing coverage.
Do not compare this eager experiment's latency directly with compiled historical
curves or claim a deployable speedup without a matching runtime comparison.

The CLI prepares an immutable campaign identity and uses one independently
owned arm/seed task per output directory. Workers reject different commits,
checkpoint hashes, metadata, configurations, or existing output directories.
Every completed episode verifies the frozen outer-state digest. A separate CPU
publisher validates the complete task grid and sets up a distinct visible W&B
workspace in `ambi-inner-bench`. Partial coverage remains explicitly pending;
final arm summaries require all twenty seeds. Raw banks are not uploaded.

Smoke evaluation uses all five arms, the full J6 update dose and 32-draw scoring,
but one seed and four real decisions. Smoke and production identities and
output directories are distinct. Smoke results are never benchmark episodes.
