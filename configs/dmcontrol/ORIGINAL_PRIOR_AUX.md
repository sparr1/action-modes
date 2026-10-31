# Historical prior with a detached auxiliary reward critic

`experiments/ambi_original_prior_aux_detached_target21_1p5m.json` adds one
reward-only auxiliary critic to the historical `u13m14st` recipe at
`253e89b3896ca3891844d797573b00ade338165d`. It retains seed 55, Humanoid Walk
state observations, H3, 1.5 million decisions, GPU replay capacity one million,
the five-head distributional primary critic, min-pair primary actor/target Q,
historical network initialization, and direct-clamp log standard deviation
bounds [-20, 2]. The primary automatic temperature starts at 1 and has explicit
target entropy -21. There is no extra actor, inner optimization or evaluation.

The independent five-head auxiliary critic estimates reward-only return under
the same stochastic SAC prior. Its loss coefficient is 0.1 and learning rate
is 3e-4. Its latent inputs are detached, blocking auxiliary gradients into the
encoder and dynamics. World, primary-critic and auxiliary-critic gradients
share the established clipping operation, so the primary parameter updates
are not guaranteed identical to a run without the auxiliary critic. Auxiliary
initialization and stochastic computations use isolated RNG state; historical
primary initialization and its RNG stream are preserved. See the core README
for checkpoint identity and limitations of this SAC-only backport.

All 60 portable model checkpoints and metadata sidecars are retained every
25,000 decisions. These include auxiliary online/target weights and provenance;
they are not full environment/replay resume snapshots.

## Hydra launch gate

Submit `slurm/run_original_prior_aux_hydra.sbatch benchmark` from a clean,
tested checkout. It uses the existing locked runtime and one L40 on gpu2301,
four physical CPU cores, 16 GiB host memory, and a one-hour pilot allocation.
It creates separate, clearly labeled online W&B benchmark runs for 1, 2 and
4 CPU cores. Each executes the original 2,500-decision warmup and 2,500-update
pretraining. It times full training iterations at decisions 3001–4000.

At decision 4001, the disposable benchmark fills replay to its full one-million
row capacity with repetitions of a real 501-row episode, assigning distinct
episode IDs through the native replay loader. It warms the resulting sampler
for 500 decisions, then times decisions 4501–5000 with online logging. This
measures full-capacity repeated-episode throughput, not a mature trained policy.
The fixture is never used by production. The gate also checks finite metrics,
zero compile fallbacks/inner updates, and native auxiliary checkpoint reloads.

The recommendation chooses the smallest CPU count within 5% of the fastest
measured case. Walltime uses the slower early/full-capacity rate, 20% throughput
margin, measured startup and all 60 checkpoint writes, plus five minutes for
final I/O. Training requires the exact source/config/lock/runtime gate binding.

For production, set `AMBI_AUX_REPRO_GATE` to the completed `benchmark.json`,
choose a verified-unused `WANDB_RUN_ID`, and override `--cpus-per-task` and
`--time` using that recommendation. The launcher uses the installed W&B SDK's
bundled core uploader for compatibility with the current server. Both pilot
and production set Torch intra-op threads to the chosen CPU count and inter-op
threads to one. Outputs go outside source under
`/cs/home/rgao48/projects/ambi-runs/original-prior-aux-reproduction-20261030`.
The storage preflight reserves both retained banks plus 4 GiB margin; source,
existing artifacts, and the running baseline are never overwritten.
