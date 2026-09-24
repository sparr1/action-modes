# H3 round-budget comparison at 650k

The target −10.5/shared SAC backbone `aux6428346x0` has little closed-loop
improvement at 650,000 training decisions with H3/J10. This experiment measures
whether a different number of inner rounds improves actual full-episode return.
It is a separate evaluation at each J, because its executed actions produce a
different state trajectory. Per-round model probes remain accompanying
diagnostics rather than substitutes for these episode evaluations.

The matrix `ambi_closed_loop_h3_j_sweep_650k.json` selects J1, J2, J4, J6, J8,
J10, J12 and J14. H3, N128, C16/A4 per round and B256 are fixed. Both the inner
critic and terminal bootstrap use the auxiliary return critic. Targets are
one-step, reward-only; the actor retains its adaptive entropy coefficient.
Every real decision starts a fresh prior-initialized actor, critic, target,
optimizers and replay, executes `tanh(mu)` after the final round, and discards
the inner state. The world model and outer parameters remain frozen.

Replay capacity preserves the historical H3 recipe: 3,072 through J8, 3,840 at
J10, 4,608 at J12 and 5,376 at J14. These capacities retain all imagined
transitions for their respective round budgets. The selected checkpoint's
temperature is inherited at every solve: initial alpha approximately
0.0059470199, target entropy −10.5. The checkpoint uses no actor Q normalization;
preparation verifies this and reads its saved temperature directly.

Each new budget runs complete 500-decision episodes with environment seeds
101–105 and controller seed 55. DMControl repeats each action twice and sums
their rewards. All settings retain 32 model-return probes initially and after
each round, full learning traces, and frozen-state validation.

The hash-pinned prior and completed J10 bundles are reused. J10 retains its
original shared checkpoint-curve performance run and independent training run;
its specific 650k publication record is verified even though that curve's
publication counts include other checkpoints. Existing soft- and return-critic
MPPI results appear as fixed historical reference lines with their different
compute budgets labeled. No prior, J10 or MPPI episodes are rerun.

`slurm/ambi_closed_loop_j_sweep.py prepare` validates all references and the
checkpoint, builds actual evaluator specifications, and allocates seven new
independent performance/training identities. `worker` reuses the checkpoint
sweep's full result, work, runtime, pairing, probe and receipt validators. The
shared worker records each cell's actual H/J and checkpoint grid. The J14
three-decision smoke retains its full inner budget; production indices are
0,1,2,3,4,6,7 and run only after that smoke succeeds.

Use `slurm/run_ambi_closed_loop_j_sweep_oscar.sbatch` with explicit CPU prepare,
GPU worker or CPU watcher resources and `EXPECTED_ACTION_MODES_SHA`. GPU workers
do not publish. The bounded CPU watcher publishes complete per-budget returns
and training traces plus an overview with a numeric J axis, matched-prior
improvements, paired 95% bootstrap intervals, and all eight setting statuses.
Intervals use 2,000 paired environment-seed resamples with seed 20260912;
five episodes from one training seed remain exploratory evidence.
