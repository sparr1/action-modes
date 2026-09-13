# Entropy during adaptation of frozen priors

This diagnostic study varies the inner actor's entropy objective while holding
its data and critic fixed. It uses existing frozen checkpoint banks; it does not
train a new backbone or append to the ordinary checkpoint evaluation curves.

The reference cases are reward/Q-scale `mey3rxj8` at 200k, reward/automatic-alpha
`jirflxz1` at 500k, and the original squashed-entropy SAC bank `u13m14st` at
500k. These are different trained systems. Interpret changes within a
checkpoint; comparisons between banks do not isolate the causal effect of
training entropy, Q scaling, or policy parameterization.

## Controlled protocol

For each saved prior-trajectory root, collect N128/H1 data and perform C32
critic updates with minibatch B256. Retain the resulting fixed critic and
dataset for the actor-only comparison. Save the actual inherited actor at
A0, then actor-update counts A1, A4 and A16. Extra actor updates must not
collect additional data or refit the critic. A0 is the captured initialization;
`inner_rounds=0`, which dispatches the outer policy, is not its substitute.

Arms are `off`, `prior_recipe`, and `squashed_matched`. An optional
`gaussian_control` is included only when preflight declares it for that
checkpoint. Preflight records exact entropy formulas, coefficients, saved
Q scale, matching rule and units. The prior's recipe can use TD-MPC2's scaled
statistic, which is different from the corrected action-distribution entropy.
When the old SAC checkpoint makes `prior_recipe` and `squashed_matched`
identical, compute may be shared only after exact identity validation. Retain
both requested arm labels and record the alias in every reused measurement.

The reference coverage is source episode seeds 101–120; decisions 0, 100, 200,
300 and 400; three independent solver repetitions per root; four simulator
rollout repetitions per captured actor; and both `mean` and `sampled`
prefix rules. Use 32 model-probe rollouts and a 1,000-decision sampled frozen
prior continuation. Workers record the declared coverage and immutable source,
checkpoint, root, solver, rollout and policy-noise identities.

The evaluator records predicted model-prefix rewards plus outer Q, real-prefix
rewards plus endpoint Q, real-prefix rewards plus measured prior continuation,
paired gains over the prior, and the relevant entropy, gradient and saturation
diagnostics. Prefix rules remain separate. No critic value is added at the
Monte Carlo cutoff. Old SAC's endpoint critic retains its soft-value semantics:
omitting an explicit entropy term from the diagnostic does not convert that
critic into a reward-only value. Keep actual reward returns and bootstrapped
scores separately named and compare their paired changes.

## Reporting contract

`utils/ambi_entropy_reporting.py` accepts measurement dictionaries containing:

```text
source_run, checkpoint_step, checkpoint_sha256, prefix_action_rule,
episode_seed, root_id, solver_repetition, rollout_repetition,
arm, actor_updates, metrics={metric_name: finite_number}
```

Additional fields retain provenance, alias identities, work counters and raw
decompositions. Pass the preflight's identity-only `expected_rows` separately;
never infer expected coverage from the results available so far. The evaluator
may pass `required_metrics` to enforce its complete measurement contract.
Missing, duplicate, unexpected or invalid rows are rejected. Final science is
published only after every declared row is present and pairing is valid.
Nonfinal reports display coverage and missing identities without partial
scientific curves. Arm-specific metrics remain in their own curves; paired
differences are computed for metrics measured on both sides.

Aggregation averages rollouts within solver, solvers within root, roots within
source episode, and source episodes equally. Every metric retains its
source-episode means, episode standard deviation and pointwise 95% percentile
interval from 2,000 episode-cluster bootstrap resamples. The same episode draws
are reused across matched points. Resampling occurs after forming paired
differences, preserving covariance. A single episode cannot estimate an
episode-cluster confidence interval and is reported without one. These
pointwise intervals are exploratory and do not adjust for selected checkpoints
or multiple comparisons. There is no pooling across checkpoint banks or
between mean and sampled prefixes.

Reports include each arm's measurements, paired change from its actual A0
initialization, arm differences versus `off`, and the primary
`squashed_matched − prior_recipe` contrast. Original raw measurements remain
in `paired-rows.jsonl` and `report.json`. The standalone `report.html` embeds
the complete report as compressed JSON and renders inline SVG charts for the
selected metric and comparison, without network dependencies. All recorded
metrics remain available through the metric selector; full raw JSON can be
downloaded without reparsing it when the report first opens.

## Publication

The CPU publisher creates one explicit new comparison run in `ambi-inner-bench`
using `start_entropy_wandb`. Its saved configuration identifies the campaign,
source checkpoints and complete protocol. Progress is logged immediately.
`publish_entropy_wandb` logs custom actor-update axes for each checkpoint,
prefix rule, arm and contrast; uploads JSON, paired rows and HTML together in
an artifact; and attaches the HTML to the run. It never appends these data to
ordinary checkpoint-axis evaluation curves. The caller owns completion and
failure status. Reconnecting to the same publisher requires an explicit
`resume="must"`; new runs use `resume="never"`.

W&B is imported only for explicit online publication. Disabled mode is usable
in local tests without credentials or network access. CPU reporting checks do
not establish CUDA learner, simulator-restoration or Oscar launch readiness;
those require the evaluator's separate validation and GPU smoke.
