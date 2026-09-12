# Mean-prefix measurement experiment

`ambi_prior_mean_prefix_h1_20seeds.json` pins the `mey3rxj8` reward/Q-scale
backbone at 150k and 200k. The prior-refinement learner is unchanged: prior
actor and online critic, inherited standard deviation, H1/N128/J4/B256,
C32 followed by A4 each round, fixed alpha zero, frozen saved Q scale, and
fresh action-local optimizers/replay. Outer learning remains off.

The diagnostic changes only the H-step prefix action rule to `tanh(mu)`.
Both the adapting actor and its prior reference use that rule, then follow
sampled frozen-prior actions for 1,000 decisions. Model prefixes also execute
means; endpoint Q uses the exact sampled first real continuation action.
Discounting, raw reward units and Monte Carlo cutoff remain unchanged.
Generate noise in the original order, then zero only prefix noise, preserving
tail pairing. Mean-prefix references have their own identity. Separate
32-rollout training probes remain sampled, with unchanged learner randomness.

Ordinary prior/refined episodes and calibration source episodes use seeds
101–120. Calibration captures decisions 0/100/200/300/400, three solves/root,
four continuations/saved actor, and all five actors at updates 0/4/8/12/16.
Each checkpoint has 100 roots, 300 solves and 6,000 paired actor rows:
6,006,000 actor-branch simulator decisions, 400,400 prior-reference decisions
and 10,000 source decisions. Ordinary evaluations add 20 refined and 20 prior
episodes, each 500 decisions. Roots/references use the new verified source
identity; historical results remain unchanged. This is a measurement change,
without critic-budget or actor-budget ablations.

## Parallel execution and publication

The matrix partitions seeds into [101–107], [108–114], [115–120]. Controller
RNG resets independently per episode/root using existing namespaced seeds.
Twelve workers can run concurrently when live GPU/CPU/memory limits permit.

| Task indices | Checkpoint | Work |
| --- | --- | --- |
| 0/1/2 | 150k | Ordinary episode shards |
| 3/4/5 | 200k | Ordinary episode shards |
| 6/7/8 | 150k | Mean-prefix calibration shards |
| 9/10/11 | 200k | Mean-prefix calibration shards |

Workers own fresh `shards/step_N/MODE/shard_I` directories. Episode workers
evaluate both `initialization/prior` and `initialization/inherited`, retaining
inline paired references, and seal every bundle file with SHA256. GPU workers
never publish or stage partial results.

`merge_ambi_seed_shards.py` requires exact seed coverage, complete artifacts,
matching science/configuration/runtime and unchanged outer state. It retains
all raw measurements and original artifacts, recomputes full-panel summaries
and episode-cluster intervals, and rebuilds the complete root-bank identity.
Missing, overlapping, corrupted or mismatched shards fail. Worker-time sums
are labeled separately from campaign wall time. Original seed-shard banks and
reference caches remain the reusable units; the merged bank has a new identity
and does not relabel its archived references as a compatible full-panel cache.

Prepare the checkpoint inventory and a new full-20-seed evaluation run through
the metadata-only evaluator spec and `eval_series.py create`. Do not append
this expanded protocol to the old five-seed attempt. Required variables:
`EXPECTED_ACTION_MODES_SHA`, `AMBI_MEAN_PREFIX_MANIFEST`,
`AMBI_MEAN_PREFIX_ROOT`, `AMBI_MEAN_PREFIX_ATTEMPT`, and `EVAL_RUN_MAP` for
production merge. `AMBI_MEAN_PREFIX_MATRIX` selects the matrix;
`AMBI_DMC_PYTHON` selects the existing locked interpreter.

Use `slurm/run_ambi_mean_prefix_oscar.sbatch`, then CPU checkpoint merges via
`slurm/run_ambi_mean_prefix_merge_oscar.sbatch`. The merge generates ordinary
HTML, exports sampled model probes and stages the complete inherited-policy
checkpoint record. Afterwards, `eval_series.py publish` publishes the curve;
the existing `slurm/run_ambi_takeoff_publisher_oscar.sbatch`, with this matrix
and root, publishes the four merged model/real diagnostic bundles. Each real
diagnostic retains the mean-prefix/sampled-tail protocol and explicit attempt.
No seed shard is published as a complete checkpoint result.

## Acceptance

First set `AMBI_MEAN_PREFIX_SMOKE=1` and run indices 3,4,9,10: the full 200k
checkpoint, seeds 101/108, two ordinary decisions and one calibration root/solve
per seed, all five actors and four full continuations. Every worker requires
CUDA and runs the analytic, simulator and RNG-preservation checks. CPU smoke
merge uses checkpoint index 1 and both seed shards. Verify the complete merged
artifacts and exact initialization/prior equality before production.

Focused tests are `tests/test_ambi_calibration_cli.py`,
`tests/test_ambi_real_calibration.py`, `tests/test_ambi_seed_shards.py` and
`tests/test_ambi_mean_prefix_campaign.py`. Enable real local simulator tests
with `AMBI_RUN_REAL_DMCONTROL_TESTS=1`; local CPU checks do not establish CUDA
readiness. Record host failures and CUDA skips, then run the Oscar GPU smoke.
