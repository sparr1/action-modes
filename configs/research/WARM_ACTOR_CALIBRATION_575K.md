# Matched-state calibration of transferred actors at 575k

This diagnostic reruns the corrected actor-transfer controller, saves selected
real simulator states and actor stages, and branches evaluation from those
identical states. It does not train a new backbone and does not implement a new
stopping criterion. It concerns the original every-decision replanning study,
not the later hold-H controller.

## Frozen scope

Source controllers are actor-only warm transfer at H3/J6, H3/J8, H3/J10, and
H1/J8, each with environment seeds 101–105 and the frozen 575k auxiliary-critic
checkpoint. Each decision uses its selected J, including the first. Only the
actor weights carry between real decisions; critic, optimizer, replay, and
temperature use the original fresh-solve initialization. Other source-controller
settings retain the corrected actor-transfer recipe.

Each source episode captures decisions 25, 75, 150, 250, 350, and 450, giving
120 roots across 20 source episodes. At each root, save warm actors after
rounds 0, 1, 2, 4, 6, 8, and 10 where the stage does not exceed source J; retain
cold solves at the same root and the frozen prior as references. Round zero
is the inherited actor before this solve's updates. A round-2 candidate from a
J10 source episode retains J10's history; it is not a J2 experiment.

Cold candidates answer what a fresh solve would produce at a warm-visited
state. They are not complete cold-controller episodes. Snapshot capture and
diagnostic branches must leave the source controller's subsequent actions and
RNG stream unchanged; saved environment state must restore observations and
transitions reproducibly.

## Prefix and terminal calibration

Evaluate both sampled-action and mean-action candidate prefixes for H decisions,
then a sampled frozen-prior continuation of 1,000 decisions, with 32 paired
replicates per root/candidate/mode. Share policy innovations between model and
real branches; the same replicate identifier pairs candidates and references.
Use the reward-only auxiliary critic, raw rewards, and gamma 0.99.

For each branch, report:

| Symbol | First H decisions | Continuation |
|---|---|---|
| A | Learned model and captured actor | Frozen terminal Q |
| B | Real simulator and captured actor | Same frozen Q at the real endpoint |
| C | Real simulator and captured actor | Real sampled-prior return |

The modeled-prefix error is A − B, the terminal-value error is B − C, and the
total error is A − C. Prefix error includes dynamics, reward, and subsequent
policy-action effects. Evaluate terminal Q at the exact prior action executed
at the first tail decision. Mean-pair, min-pair, and min-all estimates can be
calibrated from the same frozen head outputs without modifying the solver.

Continuing calibration branches pass the source episode's time limit; distinguish
that from ordinary 500-decision episode returns. For rewards bounded by two per
decision, the unmeasured discounted return after a 1,000-step tail is at most
`2 * gamma**(H+1000) / (1-gamma)`, less than 0.00864. This is a truncation bound,
not uncertainty from the 32 stochastic continuations.

## Actual controller continuation

At roots 75 and 350 only (40 roots), branch inherited, round-4, final warm actors,
and the prior reference. Execute the candidate mean action once and then resume
every-decision actor-only transfer with the original source J. Each candidate
branch carries its own actor. Replanning output is the remaining 500-decision
episode's **undiscounted** reward; it must never be pooled numerically with the
discounted prefix/terminal calibration values. This local intervention measures
whether stopping the current solve sooner improves its continuation. Testing a
permanent early-stopping controller still requires a separate full-episode study.

## Output and analysis contract

The campaign manifest pins `overview_run_id`, `wandb_entity`, `wandb_project`,
checkpoint/source identities, `cells`, `expected_prefix_shards=120`, and
`expected_replan_shards=40`. Completed roots appear in
`prefix/<source_cell>/seed-<seed>/decision-<decision>.json` or the corresponding
`replan/` directory. Default publisher globs exclude capture snapshots and
temporary files.

Every completed shard has `schema_version=1`, `status="complete"`, source-cell
identity (`source_cell`, `H`, `J`, `seed`, `decision`) and `records`. Records
contain `branch_kind` (`prefix` or `replan`), `actor_family` (`warm`, `cold`, or
`prior`), `round`, `action_mode` (`sample` or `mean`), `replicate`, and
`real_mc_return`. Prefix rows also contain `predicted_model_return` and
`real_bootstrapped_return`. Optional root diagnostics include `critic_head_sd`
and `critic_preference_gap`. Extra per-head estimates, provenance, and execution
details can remain in the raw shard. Atomic complete files are the publication
boundary; incomplete files are not counted as measurements.

Gain comparisons subtract a reference at the same source cell, seed, decision,
branch kind, action mode, and replicate before aggregation. Analysis averages
replicates within root, roots within episode seed, then seeds equally. Intervals
resample episode seeds, not branches: they are exploratory 95% cluster bootstrap
intervals with only five seeds, not confirmatory significance tests. Partial
publication reports its exact observed root/seed counts and never replaces
missing values with zeros.

The compact report uses four figure panels: real/predicted gains versus round,
modeled-prefix/terminal errors, mean-versus-sampled action effects, and replanning
returns. Exact values, optional critic diagnostics, seed counts and paired gains
over both inherited actor and prior remain in the measurements table. No figure
is generated for every decision. Raw rows and shard hashes remain available for
audit; a frozen critic's disagreement is not treated as ground truth.

`report/summary.json` also includes `period_measurements` for early decisions
below 100, middle decisions 100–299, and late decisions from 300 onward. Each
period uses the same paired, seed-balanced aggregation and exploratory cluster
intervals. The primary charts retain all captured roots, so period-specific
curves do not crowd the overview. Prefix calibration has two selected roots
per period; replanning has only the selected early and late roots.

## Execution and publication

`slurm/run_warm_actor_calibration_oscar.sbatch` validates the pinned clean Git
checkout before dispatch. The caller chooses current CPU/GPU resources and
array concurrency; the launcher has no arbitrary low cap. Set `SOURCE_DIR`,
`PYTHON_BIN`, `CAMPAIGN_ROOT`, `EXPECTED_ACTION_MODES_SHA`, and `EVAL_MODE`.
`EVAL_MODE=worker` takes `KIND=capture|prefix|replan` and an array index;
`EVAL_MODE=prepare` forwards arguments to the campaign preparer. Compute workers
disable W&B. One CPU publisher owns the overview run.

```bash
python slurm/warm_actor_calibration_publish.py report --root /path/to/campaign
python slurm/warm_actor_calibration_publish.py publish --root /path/to/campaign
python slurm/warm_actor_calibration_publish.py watch --root /path/to/campaign
```

`report` writes local HTML and exact JSON without external mutations; PNG/PDF
figures are added when matplotlib is available. Native W&B charts require
neither matplotlib nor Pillow and work in the unchanged locked runtime.
`publish` performs one W&B update; `watch` updates as complete roots arrive. It
initially publishes explicit pending progress and four native chart panels,
with means in the charts and uncertainty intervals in the measurements table.
It installs its own
stable results-section IDs through `utils/wandb_results_layout.py`, preserves
existing unrelated workspace sections/filters, and reads back the saved layout.
Layout errors are visible in run summary and receipt files. The exact delivered
run URL still needs an authenticated browser check; successful API storage
alone is not proof that the user sees the result panels.
