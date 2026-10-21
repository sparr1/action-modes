# Reward-only training replay placement

`ambi_reward_training.py` and `run_ambi_reward_training_oscar.sbatch` retain
exact continuation by default. This fixes replay on CPU, saves replay/trainer
generations, and permits scheduler requeue after a verified handoff.

The optional `--continuation off` path uses ordinary `main.py` training.
Replay placement is automatic: state replay uses CUDA when the buffer's memory
check finds sufficient free VRAM, otherwise CPU. Its GPU gate requires actual
CUDA replay. The production startup log reports the actual placement; the gate
does not guarantee identical free memory at production startup.

To select this mode with the Oscar wrapper, set
`AMBI_RETURN_CONTINUATION=off` for **both** the smoke and training submissions,
and pass `sbatch --no-requeue` for both. The command-line override supersedes
the wrapper's default `#SBATCH --requeue`. Omit `--signal=USR1@900` in this mode:
the drain signal handler belongs to exact continuation. Set the existing
`AMBI_RETURN_MODE=smoke` for the gate and `AMBI_RETURN_MODE=train` for training,
along with the normal pinned checkout, interpreter, manifest, output, and
measured resource selections. For example, after setting those campaign
variables:

```bash
export AMBI_RETURN_CONTINUATION=off
# Supply explicit measured CPU, GPU, memory, and walltime flags on each sbatch.
sbatch --no-requeue --export=ALL,AMBI_RETURN_MODE=smoke \
  slurm/run_ambi_reward_training_oscar.sbatch
# Submit after the matching gate completes successfully.
sbatch --no-requeue --export=ALL,AMBI_RETURN_MODE=train \
  slurm/run_ambi_reward_training_oscar.sbatch
```

The gate receipt binds the continuation and replay policies. A CPU exact-resume
gate cannot authorize this path, or vice versa. Model checkpoints and metadata
remain at the configured cadence (25,000 decisions for these recipes), below
`$AMBI_RETURN_OUTPUT/training/`. Training claims that directory exclusively and
refuses to restart into it. No lineage directory, drain budget, or periodic
replay generations are used. Model checkpoints support evaluation/weight
transfer, not exact recovery of interrupted training.

The launcher saves an external generated manifest alongside its launch receipt.
It preserves the complete scientific configuration, replaces the `exact-resume`
W&B tag with `continuation-off` and `auto-replay-placement`, corrects the study
note, and records the execution policy in W&B run parameters. The receipt binds
both the canonical source configuration and the generated manifest's hash.

This option changes replay RNG placement and can change sampled minibatches.
It is a throughput option, not a learning-equivalence claim. Benchmark mature
replay and end-to-end updates before selecting it for a future campaign. This
does not change any existing experiment configuration or running job.
