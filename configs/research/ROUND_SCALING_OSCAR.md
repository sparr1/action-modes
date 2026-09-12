# Full-episode inner-round scaling

The three `round_scaling_j{1,2,4}_h1_200k.json` matrices compare one, two and
four rounds of prior-initialized inner SAC on the frozen reward/Q-scale
`mey3rxj8` 200k checkpoint. The only learner setting varied is `inner_rounds`.
Each round retains H1/N128, batch256,32critic updates then4actor updates,
fixed alpha0, inherited actor/online critic and standard deviation, saved
Q scale, and fresh action-local replay/optimizers. Outer weights stay frozen.

Every real decision receives a new solve and executes its mean action.
Twenty environment seeds101–120 each run for500decisions. Controller seeds
55,56,57 provide three independent solver repetitions; the same controller
seed is paired across J settings. The first round shares its RNG sequence
at an identical state; complete trajectories can diverge after intervention.
This experiment measures full controllers, not one action followed by a
sampled prior tail. Separate32-rollout sampled model probes remain enabled
at initialization and every completed round.

| Rounds | Imagined transitions/decision | Critic updates | Actor updates |
| --- | ---: | ---: | ---: |
| 1 | 128 | 32 | 4 |
| 2 | 256 | 64 | 8 |
| 4 | 512 | 128 | 16 |

These counts exclude model probes and repeated critic/actor batch evaluation.
Preserve actual counters and control wall time rather than equating model
transitions to total compute. The evaluator reports probe time separately
and subtracts it from control time before storing `control_seconds`.

## Scheduling and ownership

An immutable campaign manifest pins the checkpoint, each cell's matrix,
controller seed, complete seed list, source bundle, and publication identity.
The completed J4/controller55 panel can be reused only after source identity,
checkpoint, full seed coverage, file seals and frozen-state checks pass.
Reuse retains its existing native W&B run and model-series identity; it does
not relabel the historical result as a freshly evaluated run.

The other eight cells use weighted seed shards: J1 one20-seed shard, J2 two
10-seed shards, and J4 four5-seed shards. Seventeen GPU tasks prioritize J4,
then J2, then J1. When available, twelve L40S workers run concurrently using
6CPUs/32GiB each. Check live account and QOS headroom before submission.
Workers run the existing evaluator, keep inline prior references and save
sealed disjoint outputs. No partial seed shard is published as a completed
checkpoint. CPU merges validate all shards, export model summaries, and stage
complete native checkpoint records. Full per-decision traces remain in
bundles; the comparison report avoids embedding them into huge HTML files.

## Reporting

Each fixed J/controller-seed cell retains a native checkpoint-curve identity.
A separate shared comparison run appears immediately and reports progress.
Only the complete9-cell panel supplies final compute-axis scientific curves.
The comparison artifact stores all raw paired episode returns. Average the
three controller repetitions within each environment seed, then weight the
20environment seeds equally. Use2,000 paired environment-seed bootstrap
resamples for exploratory95% intervals, including differences from prior
and J1. Do not treat60episodes as60independent environment clusters.

Selected checkpoints and one outer training seed make this an exploratory
study. CUDA reductions and long simulator trajectories need not reproduce
bit for bit across runs. Record exact source identities, GPU type and
allocation for timing comparisons; never overwrite older results.

## Validation

Run config, launcher, merger and reporting tests locally. Before production,
run the full200k checkpoint on Oscar with short paired episodes covering
J1,J2,J4 and at least two J4 shards; merge the smoke shards, verify true
round/update counts, unchanged outer state and complete per-round model
probes. A CPU smoke alone does not establish CUDA readiness. Production
submission is separate from defining the matrices.
