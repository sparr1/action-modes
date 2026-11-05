# BOOM entropy ablation

`boom_upstream_reference.py --entropy released` preserves the pinned released
BOOM equations. `--entropy ambi-posttanh` changes only the actor entropy term
to AMBI's joint Gaussian log probability with the exact stable tanh Jacobian.
For Humanoid Walk it uses alpha = 21 × 0.0001 = 0.0021, matching AMBI's
dimension-matched post-tanh configurations. This is an upstream-code ablation,
not an unmodified official run.

The alignment term still calls the released Gaussian helper, including its
extra dimension multiplier. It retains detached actor standard deviations,
the 0.05 denominator floor, detached softmax(Q/S) weights, lambda = 0.42,
and the strict S > 2 gate. Sampling, MPPI, replay, world-model losses, native
optimizers, clipping, seed 55, evaluation, and checkpoints are preserved.
Only actor-update calls substitute the entropy log probability; planner and
model-target calls continue to use the released policy sampler.

For Oscar use `REFERENCE_ENTROPY=ambi-posttanh` with
`REFERENCE_ACCELERATION=model-planner-graph`, `REFERENCE_STEPS=400000`, and
`BOOM_REFERENCE_METHOD=boom`. A new scratch `BOOM_REFERENCE_ROOT` keeps all
outputs separate. `BOOM_REFERENCE_UPSTREAM` can name the existing clean
checkout at BOOM commit `b3b2f09c03533c15b804ced9abc6767e73ff8e1e`.

The gate checks identical actions, alignment gradients and RNG for fixed
weights; observer parity; and accelerated planner/model parity. Its disabled
W&B benchmark sizes CPU and walltime before automatic scientific submission.
The comparison workspace includes the existing released BOOM control; no
duplicate control training is needed. Stable entropy and the fraction of
coordinates below the alignment standard-deviation floor are additional
observational metrics for this ablation.
