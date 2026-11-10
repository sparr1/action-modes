# Humanoid Stand: BOOM coefficient 0.0042 | mean execution

Fresh training counterpart of [source run 99b2e280](https://forge.coreweave.com/wandb/rwgao_b-brown-university/ambi/runs/99b2e2808d2742b9a548a5f58bad062c).
The only scientific change is the task, now `humanoid-stand`. The algorithm file
`algs/ambi_boom_x0p01_humanoid_stand_mean.json` preserves every source learner parameter except the
W&B group, display name, and tags. The matching experiment preserves seed 55,
one million agent decisions, state observations, H3/J8/K1/N128/C16/A4/B256,
post-tanh alpha 0.0021, reward-only critics, all 40 checkpoints every 25k,
and critic geometry every 1,000 outer updates. This is fresh training.

Run through `slurm/run_humanoid_stand_mean_oscar.sbatch` from a clean pinned Git checkout.
Its preparation command verifies unchanged learner sources and persisted CUDA
parity evidence, then rebinds the evidence to the new task. For mean BOOM this
also verifies both mean-action GPU tests. A fresh full-size Stand GPU startup,
checkpoint and continuation checks, geometry diagnostics, and measured timing
must pass before online training. All checks keep W&B disabled in the environment
and learner. Training publishes to the same shared workspace as the other Stand arm.

Each independent arm uses one L40S, one physical CPU, and 8 GiB. Existing one/two-CPU
scaling evidence shows no useful benefit from the second core. The 59-hour limit
uses measured 50–51-hour million-decision runtime, 15% margin, and 30 minutes for
startup/I/O; the startup gate verifies the estimate. Both arms may run concurrently.
The publisher updates only its named workspace, verifies panels through readback,
and exposes publication failures. Set `AMBI_STAND_WORKSPACE_STATE` to the shared
campaign submission JSON and the runtime/source variables required by the launcher.

Source learner commit: `7427e1c681cbc3f2b2fbc3109855afc7d720e57b`.
