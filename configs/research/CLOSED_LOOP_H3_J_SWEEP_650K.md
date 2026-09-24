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

## H1 and H2 extension

`ambi_closed_loop_h1_j_sweep_650k.json` and
`ambi_closed_loop_h2_j_sweep_650k.json` extend the same checkpoint and J grid to
H1 and H2. Every one of their eight settings is new: the completed H3/J10
episodes cannot be reused at another horizon. Each campaign therefore adds
40 complete episodes, for 80 additional episodes across H1 and H2. The original
H3 campaign and its publication identities remain independent.

All remaining algorithm, seed, diagnostic and execution settings match their
historical 575k horizon-specific recipes. Replay capacities intentionally stay
at 3,072 through J8, 3,840 at J10, 4,608 at J12 and 5,376 at J14 for **all three
horizons**, preserving those recipes rather than shrinking the buffer at H1/H2.
The shorter horizons collect fewer imagined transitions per round but still
perform C16/A4 updates.

Preparation accepts `--horizon 1`, `--horizon 2` or `--horizon 3` and selects the
corresponding matrix when `--matrix` is omitted. The launcher forwards
`EVAL_HORIZON`; omitting it preserves the original H3 behavior. H1/H2 production
indices are 0–7, gated on their own full-budget J14 three-decision smoke at
index 7. Each campaign receives separate performance, training and overview
identities, while its existing prior and fixed-budget MPPI references are
reused unchanged.

## H1 soft/soft comparison

`ambi_closed_loop_h1_soft_soft_j_sweep_650k.json` evaluates the complete
J1/J2/J4/J6/J8/J10/J12/J14 grid with the coherent soft-critic strategy used in
the historical 575k comparison. This changes exactly four settings from the
H1 return/return matrix: `inner_critic_source="sac"`,
`inner_horizon_critic_source="sac"`,
`inner_sac_critic_target="entropy_augmented"`, and
`inner_terminal_entropy="outer"`. The online SAC soft critic initializes the
inner critic; inner targets include entropy, and the frozen soft terminal
bootstrap includes the outer-policy entropy correction. The SAC actor retains
its inherited adaptive alpha and mean-action execution.

The selected 650k checkpoint, H1, N128/C16/A4/B256, replay capacities, five
complete paired episodes per setting, and 32 model probes remain matched to
the return/return sweep. All eight soft settings are new, totaling 40 episodes;
return/return results are not substitutes. Existing prior and MPPI references
remain reusable, and no historical refinement run is relabeled or republished.

Pass `--horizon 1 --critic soft` to preparation, or set `EVAL_HORIZON=1` and
`EVAL_CRITIC=soft` for the launcher. Omitting `--critic` retains `return_only`
and all previous default matrix, group and reuse behavior. Preparation compares
each requested soft recipe against its original 575k configuration and checks
the actual evaluator specification before allocating eight new performance and
training identities. Its J14 smoke gates all eight production settings.
