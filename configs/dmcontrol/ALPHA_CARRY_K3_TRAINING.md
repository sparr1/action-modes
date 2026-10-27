# K=3 run-carried alpha training

This single-seed Humanoid Walk run is the K=3 variant of
[source run be85c916fcd94d5a813d7857d0ef5efb](https://forge.coreweave.com/wandb/rwgao_b-brown-university/ambi/runs/be85c916fcd94d5a813d7857d0ef5efb),
originally launched from commit `939a5a6`.

The only algorithm setting changed from
`ambi_alpha_carry_h3_j8_run_detached_ent10p5.json` is
`inner_solve_interval: 3`. At episode decision indices 0, 3, 6, ... the agent
performs the ordinary H3/J8 solve. At each intervening observation it samples
from the retained refined actor using the solve-time encoder snapshot. The
inner alpha and its Adam state persist for the entire run and advance only on
solves. Real environment collection, outer training and replay updates still
occur at every decision. Episode boundaries restart the solve cadence.

The recipe preserves seed 55, one million decisions, 2,500 warmup decisions,
2,500 pretraining updates, detached auxiliary-return representation, SAC actor,
target entropy -10.5, C16/A4/N128/B256, and strict CUDA compilation. Every
25,000-decision model checkpoint retains its complete replay snapshot. The
separate native lineage writes rolling exact-resume generations every 15
minutes. This is a fresh training run, with no source checkpoint loaded. Root-policy
diagnostics retain the 1,000-decision cadence, deferred to the next actual
solve when the scheduled decision uses the held policy; their axis records
the decision actually sampled.

## Validation and resource selection

Use the locked DMControl interpreter and a clean checkout at the tested commit.
`slurm/run_ambi_alpha_carry_k3_oscar.sbatch` accepts
`AMBI_ALPHA_CARRY_K3_MODE=smoke`, `benchmark`, or `train`. Common required
variables are `AMBI_PROJECT_DIR`, `AMBI_PYTHON`, `AMBI_EXPECTED_COMMIT`, and
`AMBI_ALPHA_CARRY_K3_OUTPUT`; output must live under
`/oscar/scratch/rgao48/ambi/alpha-carry-k3/`.

Run smoke and benchmark in short L40S allocations, each with two physical CPU
cores and 16 GiB host memory. They can run independently in parallel. The smoke
keeps production model and update shapes but reduces warmup, pretraining and
replay capacity, disables W&B, and uses short post-warmup episodes. It checks
strict compilation, K3 cadence, zero adaptation work on held decisions,
unchanged held alpha/Adam state, run-alpha carry across episode boundaries,
interleaved outer updates, and checkpoint/model/replay roundtrips without RNG
changes. It writes `gate/receipt.json`.

The benchmark measures actual K3 training with one and two physical CPU cores,
24 active warmup decisions and 96 measured decisions per case. Each measurement
uses 32 complete solve/hold cycles and includes every outer update. It rejects
compilation inside the timing window and keeps the production CPU replay
placement. Subprocesses reuse the allocation-local compiler cache. It writes
`cpu-benchmark/results.json` and recommends the smallest core count within 5%
of the fastest measurement. Short early-replay timings exclude W&B, scheduled
diagnostics and checkpoint I/O; account for those and startup with a modest
explicit margin when selecting walltime.

## Production launch and continuation

Submit one L40S training job, overriding `--cpus-per-task` with the benchmark's
recommendation and `--time` with the measured estimate. Set
`AMBI_TRAIN_WALLTIME_SECONDS` to exactly that Slurm duration. The launcher verifies
this against the allocation. Its short default walltime is for validation,
not production. The default drain is 30 minutes before allocation expiry;
`AMBI_DRAIN_AFTER_SECONDS` can request an earlier planned drain. Production
validates that both receipts bind the same source commit, manifest, algorithm
config, lockfile, interpreter and GPU model.

The new native lineage is `<output>/lineage`; each scheduler segment gets its
own `<output>/launches/<job-id>.0` and console directory. Online W&B uses the
locked SDK's bundled core (`WANDB__REQUIRE_CORE=true`), as in the source run.
The native lineage owns the W&B run identity. The launcher does not set a
separate `WANDB_RUN_ID`. Keep the source configuration and commit fixed for
continuations and set `AMBI_RESUME_MODE=required` with the same campaign root.
Exit 75 means a verified clean handoff, not completion; exit 0 requires a valid
`DONE` marker. The launcher never automatically requeues. Install and verify
visible run/progress panels before sharing the live run link.
