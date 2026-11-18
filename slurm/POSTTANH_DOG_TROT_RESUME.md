# Resumable Post-tanh sampled Dog Trot training

`run_posttanh_dog_trot_resume_oscar.sbatch` trains seed 56 for one million
decisions using the original Dog Trot recipe. The interrupted model-only run
cannot be resumed exactly, so this launcher creates a fresh attempt. Native
resume uses CPU replay; the new trajectory need not match the old GPU replay
trajectory. All other scientific settings, critic geometry, and all 40 model
snapshots at 25k intervals are preserved.

Full trainer, optimizer, replay, environment and RNG state is committed every
15 minutes at the next episode boundary. The atomic lineage store retains the
latest generation and its predecessor under Oscar home storage and verifies
checksums when restoring. A sudden failure loses only work since the last
committed generation. W&B events are published after checkpoint commit, so
online charts update in batches.

The first real training segment drains after ten minutes, verifies trained
state (including replay), and resumes the same lineage and W&B run in the same
allocation. Near walltime, a verified handoff permits automatic Slurm requeue.
Unexpected errors stop the job to avoid a repeating failure loop. After fixing
the cause, submit a **new job** with `AMBI_RESUME_MODE=required`, the same source
commit, manifest, lineage, campaign and snapshot root. Do not use `new` or
create a new W&B ID for that continuation. Confirm the old job is no longer
active before resubmitting; the lineage also enforces an exclusive writer lock.

The launcher requires `AMBI_EXPECTED_COMMIT`, `AMBI_PYTHON`,
`AMBI_WORKSPACE_PYTHON`, `AMBI_RESUME_CAMPAIGN`, `AMBI_LINEAGE_DIR`, and
`AMBI_TRAIN_WALLTIME_SECONDS`. Use the locked DMControl interpreter and the
existing workspace-publisher interpreter. Submit from the clean pinned checkout;
pass the selected CPU count and measured walltime explicitly to `sbatch`.
For production, also pass `--signal=USR1@1800`; omit this warning for the short
benchmark allocation. The trainer's internal deadline independently starts a
walltime handoff 30 minutes before allocation expiry.

First run `benchmark` with two physical CPUs, one L40S and W&B disabled. Its
`benchmark/plan.json` chooses the smallest CPU count within 5% of the fastest
measurement. The walltime estimate uses measured throughput and completed
matching runs, with a 20% margin plus half an hour for startup. Size host memory
for the measured peak, a full one-million-row CPU replay and checkpoint staging.

Every production allocation additionally measures its own GPU/CPU throughput
before starting or restoring training. It rejects more than 0.225 seconds per
decision, compared with about 0.18 on completed reference runs and about 0.31 on
the timed-out allocation. `gpu2708` is excluded because both slow originals ran
there; this is a scheduling precaution, not proof of a hardware fault.
Benchmarking disables W&B both in the environment and resolved learner config.

Keep campaign submission and verification receipts in scratch, outside source.
Record observed progress and throughput after restoration, and verify the W&B
workspace panels before delivering a live link.
