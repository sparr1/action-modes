# gSDE8 sampled: Dog Trot and Dog Run

These two manifests transfer the completed Humanoid Walk gSDE8 recipe
(`590be7c27a9b40eaa9a68218cdb74d0a`, learner source `edec58c`) to Dog Trot
and Dog Run. Each uses seed 55, one million agent decisions, and all 40
checkpoints at 25k intervals. The task and W&B identity are the only recipe
changes. Critic geometry remains enabled every 1,000 outer updates.

`slurm/gsde_dog_oscar.py prepare` verifies unchanged learner sources, recipe,
runtime lock, and the original seven passing CUDA parity cases before reusing
their evidence. Set `AMBI_GSDE_VARIANT` to `dog_trot` or `dog_run` and use the
matching campaign directory under
`/oscar/scratch/rgao48/ambi/gsde8-dog-trot-run-20261116`.

Run `slurm/run_gsde_dog_oscar.sbatch gate` on one L40S with two CPU cores
before training. Each task validates its production shapes, finite training,
CUDA graph reuse, native execution, sparse geometry diagnostics and checkpoint
roundtrip. Gate benchmarks disable W&B in both environment and configuration.
Select the smallest CPU count within 5% of the measured fastest throughput.
Use the larger of the measured full-run projection and completed reference
runtime, with a 15% margin plus 30 minutes, for the training walltime.

Submit `slurm/run_gsde_dog_oscar.sbatch train` with the measured resources.
The submission receipt must bind the job ID and verified W&B workspace to the
same source commit. Training checks GPU/runtime identity and gate evidence,
and its publisher installs and verifies progress, learning, and geometry
panels before starting. Preserve each task's outputs in its own directory.
