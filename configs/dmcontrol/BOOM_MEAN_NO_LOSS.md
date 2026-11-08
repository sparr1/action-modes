# Mean-action training without BOOM alignment

`ambi_boom_mean_no_loss.json` is a single-seed ablation of W&B run
`rwgao_b-brown-university/ambi/a23ba4087b6b4847995e9998fa4936c6`, using its
learner source at `450aa82ed65eeb887c8e8072a0bf9402a2ae8ef0`.
The only scientific change is `outer_boom_coef: 0.0042 -> 0.0`.

The objective selector remains `boom` to preserve the source's H-state actor
loss, Q scaling, behavior-label collection, compilation and RNG conventions.
The zero coefficient removes the alignment contribution and its gradients.
Changing the selector to `q_maximization` would also change the actor horizon
to H+1 states, so this experiment deliberately does not make that second change.

Training starts fresh with seed 55 on state-observation Humanoid Walk for one
million decisions. Real actions use the adapted policy mean; imagined actions
remain stochastic. H3/J8/K1, N128/C16/A4/B256, alpha 0.0021, reward-only critics,
all 25k checkpoints, whole-inner CUDA graphs, strict compilation, vmap,
TF32-high and periodic critic geometry match the source.

The Oscar launcher is `slurm/run_boom_mean_no_loss_oscar.sbatch`. Its helper
verifies that learner sources and the dependency lock match the source run,
reuses explicitly attributed CUDA parity and 1/2-core sizing evidence, and
requires a fresh full-size coefficient-zero startup check before production.
Startup benchmarks have W&B disabled. The existing idempotent workspace
publisher installs and verifies the comparison and progress panels before
online training begins. Allocation time is based on measured source throughput
with approximately 15% margin, and startup timing must fit that allocation.
