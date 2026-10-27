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

The additional J10 configuration,
`ambi_alpha_carry_h3_j10_k3_run_detached_ent10p5.json`, changes only
`inner_rounds` from 8 to 10 relative to the K3/J8 recipe, plus the required
dependent replay sizing and publication labels. Each J10 solve performs 160
critic updates and 40 actor and temperature updates. Its imagined replay
capacity is 3,840 rows (`10 * 128 * 3`), retaining every transition generated
within the solve, as J8 does with 3,072 rows. The existing configuration contract
rejects a smaller action-local bank; no replay eviction or runtime semantics
are changed. H3, K3, all per-round settings, seed, training budget, checkpointing
and diagnostics remain unchanged. Its W&B group is
`ambi-alpha-carry-k3-j10-20261027`.

## Validation and resource selection

Use the locked DMControl interpreter and a clean checkout at the tested commit.
`slurm/run_ambi_alpha_carry_k3_oscar.sbatch` accepts
`AMBI_ALPHA_CARRY_K3_MODE=smoke`, `benchmark`, or `train`. Common required
variables are `AMBI_PROJECT_DIR`, `AMBI_PYTHON`, `AMBI_EXPECTED_COMMIT`, and
`AMBI_ALPHA_CARRY_K3_OUTPUT`; output must live under
`/oscar/scratch/rgao48/ambi/alpha-carry-k3/`.

Select J10 by exporting
`AMBI_ALPHA_CARRY_K3_MANIFEST=configs/dmcontrol/experiments/ambi_alpha_carry_k3_j10_training.json`
for its smoke, benchmark and training jobs. The default remains
`configs/dmcontrol/experiments/ambi_alpha_carry_k3_training.json` (J8). These are
the only accepted manifests. Give J10 a separate campaign output root and run
its own full CUDA smoke and CPU benchmark; J8 validation receipts cannot launch
J10. The selected manifest and configuration hashes bind each receipt, and the
benchmark explicitly passes the selection to each worker subprocess.

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

`slurm/submit_ambi_alpha_carry_k3.py` can run in a short CPU allocation with an
`afterok` dependency on both validation jobs. Supply `--source-dir` for the
clean, validated training checkout, `--expected-commit` for its full commit,
`--campaign-dir` for the output root, and `--python` for the locked interpreter.
For J10, also pass the selected manifest with `--manifest` or the environment
variable above. The submitter pins it in both its validation subprocess and
the submitted job, includes it in the durable submission identity, and names
the production job `ambi-alpha-k3-j10` (J8 retains `ambi-alpha-k3`).
The submitter selects the benchmark's recommended CPU count, one L40S and
16 GiB. It rounds walltime up to the next hour after adding 15% to the measured
million-decision duration, twice the GPU-gate runtime for startup, and ten
minutes for fixed overhead. An optional `--production-overhead-factor` (finite,
at least 1, default 1.0) multiplies the measured duration before the 15% margin.
The J10 launch uses 1.13 based on a matched K3/J8 production window: decisions
89,000 through 96,000 took 1,932.188 seconds, or 0.276027 seconds per decision,
versus the short benchmark's 0.245472, a ratio of approximately 1.1245. This
accounts for production logging, diagnostics and
checkpoint overhead while retaining the separate safety margin. The factor is
recorded in both the receipt's resources and submission identity, so a retry
cannot silently change it. Its durable `submission.json` records an intent
before calling Slurm and the returned job ID afterward. A successful rerun is
idempotent; an uncertain outcome requires reconciliation before retrying.
The submitter may live in a newer checkout while training remains pinned to
the exact source covered by the validation receipts.

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
visible run/progress panels before sharing the live run link. The companion
`slurm/ambi_alpha_carry_k3_workspace.py` compares the K1/J8 source, K3/J8 and
K3/J10 runs. It accepts omitted `--run-id` and `--j10-run-id` before startup,
showing pending status and selecting the source ID plus the two dedicated K3
campaign groups. Rerunning after registration binds the three exact run IDs.
