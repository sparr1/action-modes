# Post-tanh outer-adapt K1 with lazy prior distillation

This single-seed exploratory run changes the outer actor objective of the
post-tanh `outer_adapt` K1 recipe from normalized Q maximization to normalized
reverse KL against cached fresh inner-SAC teachers, plus the same entropy term.
The control is W&B run `fc7c433ab4734a5aa7eb1717e2165969` in project `ambi`.
Use `algs/ambi_posttanh_lazy_distillation.json` with the dedicated
`slurm/ambi_distillation_posttanh_oscar.py` gate and train entry points.

Preserved settings are Humanoid Walk state observations, seed 55, H3/J8/K1,
N128/C16/A4/B256, five distributional Q heads and pair size 2, mean-pair actor Q
and min-pair targets, reward-only critics, finite-horizon prior bootstrapping,
and fixed joint squashed-entropy coefficient 0.0021 at both levels. Each action
starts from fresh prior-initialized actor/critic/optimizer state. Its private Q
scale starts from the current outer scale and adapts per actor update. The KL
scale is separate. Critic geometry remains enabled every 1,000 outer updates.

The intended initial budget is 750,000 decisions, with all portable checkpoints
retained every 25,000 decisions. This shorter budget is deliberate: the original
million-decision K1 run already takes roughly 50 hours at its observed throughput.
A fresh end-to-end gate, including environment interaction, collection, outer
updates, refreshes, checkpoint cost and startup, must fit the supplied absolute
48-hour deadline before production is launched. No existing run is resumed or
modified. Compare returns at matching decision counts.

Lazy refresh initially uses 20 sequences every 200 outer updates. The interval is
larger than the default 10 to control runtime. All collection actions still
save their adapted teacher; refresh updates older replay labels. The complete
inner-SAC solve budget is preserved for every refreshed root. The learning
tradeoff from less frequent refresh is unmeasured and must be evaluated from
this run. Hardware batch width, concurrency and optional first-layer fusion
are selected from paired GPU timings; the resolved JSON is the source of truth.

`inner_cuda_graph=true` uses a production, checkpointed private RNG for fresh
K1 collection solves. Deterministic teacher extraction is included before
execution sampling; extraction itself consumes no random draws. Sparse native
diagnostic actions remain supported. Teacher-only reanalysis bypasses this
collection graph and uses its own isolated solver/RNG. Unsupported persistent
state, held-policy K>1, selector/explorer, and writeback combinations are
rejected by the collection graph's static validator. The disabled default
preserves the existing native path and exact-resume identities.

The gate runs with W&B disabled and retains raw timing/validation outputs in
scratch. Production uses a separate W&B run and an idempotently configured
comparison workspace with return, entropy, teacher coverage/age, KL, refresh
cost, and critic geometry panels. A gate receipt binds source commit, config,
locked environment, hardware, resource count and deadline; it cannot authorize
a different training recipe.
