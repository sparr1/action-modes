# Humanoid Run and Stand seed replication

The `ambi_posttanh_outer_adapt_humanoid_{run,stand}_seeds56_57.json`
manifests add independent seeds 56 and 57 to each task's seed-55 experiment.
Only the seed and W&B labels change. Each run trains for one million decisions
with sampled execution, H3/J8/K1, fixed post-tanh entropy coefficient 0.0021,
action-local adaptive Q scaling, and reward-only critics. All 40 checkpoints
are retained at 25k-decision intervals, and critic geometry is observed every
1,000 outer updates.

`slurm/posttanh_humanoid_seed_replicates.py` checks each resolved recipe against
its versioned seed-55 reference. On an allocated GPU it also verifies identical
training code, configuration hashes, environment lock, runtime, GPU parity,
and task-specific checkpoint/graph-continuation validation before training.
The original CPU scaling benchmark selected one core per L40S. Reusing that
evidence requires the same runtime and GPU; it does not create benchmark runs
in W&B.

`slurm/run_posttanh_humanoid_seed_replicates_oscar.sbatch` takes `run` or `stand`
and index `0` or `1`. Its campaign manifest supplies unique run IDs and Slurm
job identities, and its output directories reject accidental reuse. The
publisher installs and verifies idempotent progress and diagnostic panels
before training and again on completion; publication failures stop the job.
