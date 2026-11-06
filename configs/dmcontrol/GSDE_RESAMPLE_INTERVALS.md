# gSDE real-step resampling intervals

`experiments/ambi_gsde_resample_intervals.json` contains four single-seed
Humanoid Walk runs, each changing one interval from the existing squashed
gSDE `real_steps` reference (W&B `590be7c27a9b40eaa9a68218cdb74d0a`).

| Config suffix | Imagined interval | Execution interval |
| --- | ---: | ---: |
| `i4_e8` | 4 | 8 |
| `i16_e8` | 16 | 8 |
| `i8_e4` | 8 | 4 |
| `i8_e16` | 8 | 16 |

The reference `(8, 8)` is reused. Both intervals count real decisions; imagined
matrices are independent per trajectory and separate from execution noise.
Current actor features and scales apply to retained base matrices. Actor and
critic training draws remain fresh. Episode boundaries clear persistent noise.

All runs retain `scope=real_steps`, actor features, log-scale initialization -3,
mean clip 2, squashed entropy, fixed alpha 0.0021, seed 55, reward-only critics,
and H3/J8/K1/N128/C16/A4/B256. BOOM is not enabled. The budget remains one million
real decisions with all forty checkpoint/metadata pairs retained every 25k.
Strict compilation, vmap, whole-inner CUDA graphs and critic geometry logging
match the reference.

`slurm/run_gsde_resample_intervals_oscar.sbatch INDEX` runs a disabled-W&B GPU
preflight before online training. It verifies full production shapes and update
dose, 96 steady-state graph decisions, independent interval refresh boundaries,
zero compiler fallbacks, geometry diagnostics and checkpoint round-tripping.
Original CUDA parity evidence is reused only after matching its runtime,
manifest, JUnit checksum and unchanged learner/runtime/test source paths.
Each new preflight receipt binds the exact new commit and all four configs.

One physical CPU per L40S comes from the reference's measured 1-vs-2-core gate.
Walltime is supplied at submission from measured full-decision throughput, with
a modest margin. The preflight rejects an unexpected slowdown or insufficient
walltime. Jobs use independent output directories and run identities, reject
requeue/overwrite, and install verified progress/result panels before training.
