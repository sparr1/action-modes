# gSDE noise-persistence screen

The six configurations in `experiments/ambi_gsde_noise_scopes.json` cross two
reference AMBI recipes with imagined-noise scopes `round`, `solve`, and
`real_steps`. This is a single-seed exploratory screen (seed55), with1M real
Humanoid Walk decisions per run and checkpoints every25k decisions.

- [Scaled reference](https://forge.coreweave.com/wandb/rwgao_b-brown-university/ambi/runs/d9c2c24cfd00449ca0efd75252f2fc59): literal `tdmpc2_scaled`, alpha0.0001, outer and inner.
- [Squashed reference](https://forge.coreweave.com/wandb/rwgao_b-brown-university/ambi/runs/c7315843d48e49c58364dbc5686a4bd2): `squashed`, alpha0.0021, outer and inner.

The first reference's display name says pretanh; this screen preserves its
actual scaled statistic, including the internal action-dimension multiplier.
The analytic `pre_tanh` option is not used. Inner temperature inherits the fixed
outer coefficient. Q normalization resets from the outer prior each decision
and adapts per actor update. Both critics have reward-only targets.

All arms use actor noise features, log-scale initialization-3, mean clipping2,
and resampling intervals8 for both real-step imagined collection and sampled
execution. Every imagined trajectory has an independent base matrix. Current
features/scales apply to retained matrices after parameter updates. SAC
optimization samples remain fresh. Gaussian std-head mapping/bounds inherited
from the reference metadata do not apply to gSDE.

H3/J8/K1/N128/C16/A4/B256, finite-horizon bootstrapping, full prior-initialized
clones, model/critic architecture, update budget, replay, and diagnostic cadence
match the references. Strict compilation, vmap, TF32-high, and whole-inner CUDA
graphs are enabled. Every1000 updates, unregularized critic geometry remains
observational, with64 samples and observation/action noise0.01/0.02.

`slurm/gsde_noise_scopes_oscar.py` provides CUDA parity, per-arm full-training
gates, CPU sizing, and production entry points. Gates use the production
architecture/update dose, shorten seeding/pretraining and measured duration,
disable W&B, check zero compiler fallbacks, and exercise checkpoint and geometry
paths. They compare1 and2 physical CPUs and a native reference. Production
requires exact-source passing receipts. Runtime estimates use measured complete
decisions with15% margin and30minutes for startup/I/O. No gate implicitly submits
production. `slurm/gsde_noise_scopes_workspace.py` installs and verifies the
comparison panels and pending/running status in a dedicated W&B workspace.
