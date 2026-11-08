# Post-tanh outer-adapt recipe on Humanoid Run

`ambi_posttanh_outer_adapt_humanoid_run.json` trains the configuration from
W&B run `rwgao_b-brown-university/ambi/fc7c433ab4734a5aa7eb1717e2165969`
on `humanoid-run`. Its learner source is pinned to
`5f6089fe70441b968ed814d1b1e73ddbd3f08a71`.

The task changes from Humanoid Walk to Humanoid Run. All learning settings
match the source: fresh seed 55, one million decisions, sampled real and
imagined actions, H3/J8/K1, N128/C16/A4/B256, post-tanh alpha 0.0021,
reward-only critics and action-local adaptive Q scaling. All 40 checkpoints
are retained at 25k intervals. Strict compilation, whole-inner CUDA graphs,
vmap, TF32-high and periodic critic geometry remain enabled.

`slurm/run_posttanh_humanoid_run_oscar.sbatch` uses one L40S and one physical
CPU core. The source completed in about 49.5 hours; the initial 58-hour request
includes approximately 15% margin. Existing source-bound CUDA parity and
1/2-core scaling evidence are attributed and verified before launch. A fresh
production-shape Humanoid Run startup check verifies throughput, checkpoint
roundtrips, geometry diagnostics and graph continuation before full training.
Startup checks disable W&B in both the environment and learner configuration.
The idempotent workspace publisher installs and verifies progress panels.
