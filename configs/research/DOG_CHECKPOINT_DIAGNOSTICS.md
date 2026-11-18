# Dog Stand checkpoint diagnosis

`diagnose_dog_checkpoints.py` implements the checkpoint tests around the
seed-55 and seed-56 return collapses, with step-matched seed 57 controls.
The exact checkpoint and measurement panel is in `dog_checkpoint_diagnostics.json`.
The checkpoints are frozen policy/value snapshots, not full training resumes.
This campaign cannot reconstruct the replay batch or update that initiated a collapse.

1. **Behavior:** three 500-decision episodes per checkpoint/controller, paired
   by environment seed. Compare sampled prior execution with the saved H3/J8
   inner recipe, using the same CUDA graph implementation as training. Private
   controller noise is reset per episode and paired across checkpoints. Prior
   and inner controllers have distinct noise streams. There are no outer updates.
2. **Action preferences:** six identical physical roots, two each from successful
   prior behavior and the two collapsed inner behaviors at decision 200. Test
   positive and negative pre-tanh steps of length 0.1 along inherited Q,
   differentiated one/three-step model objectives, and a random control. Also
   evaluate both before/after prior mean actions under each checkpoint. Each
   checkpoint always retains its own encoder, model, critic, and continuation prior.
3. **Model error:** at rounds 0 and 8, generate a three-action imagined sequence
   and replay those exact actions in the real simulator. Use the real endpoint's
   sampled continuation action in both terminal Q evaluations. Total error splits
   additively into reward error, model endpoint value error, and real-endpoint
   critic error. Latent RMSE is also retained. This is a controlled model test;
   it does not claim that an inner policy would execute that open-loop sequence.
4. **Inner rounds:** capture all nine actor boundaries (initialization and rounds
   1–8), with full value/gradient/entropy trace events. Execute each frozen actor's
   sampled first action in the simulator, then the checkpoint's frozen prior.
   This measures local action quality against the critic's continuation semantics;
   the separate full episodes test repeated closed-loop adaptation.

All candidates at a root share two sampled continuation-noise streams. The real
simulation continues for H+1000 decisions without the 500-decision timeout or a
terminal Q bootstrap. At gamma 0.99 and reward bound 2, omitted root return is
at most 0.00838; paired gain bounds add the two omitted tails. Monte Carlo
uncertainty is separate and can be much larger. This is an initial diagnostic
panel, not a high-precision effect-size or across-training-seed estimate.

Diagnostic Q uses the mean of all heads, equal to the expectation of the saved
actor's random-pair mean. Optimization retains the checkpoint's pair sampling
and minimum-pair TD targets. Per-head values are recorded. Round traces run the
native implementation with observational probes and independent RNG; full
episodes retain the training CUDA graph. These use different random sequences
but the same optimization objectives and budget.

Dog snapshots preserve full MuJoCo integration state, task and wrapper RNG/clocks,
reward constants, and acceleration/contact sensor readings. Restoration requires
identical runtime/model fingerprints. Integration tests cover exact subsequent
observations/rewards and continuation across both episode clocks. Outer model,
optimizer, alpha, and scale digests must remain unchanged throughout every test.

Each command requires a fresh output directory and writes JSON plus progress to
stdout. The `bank` command combines complete episode shards; all probes use its
content hash. Slurm checks a clean pinned Git commit. Smoke/sizing runs use the
same commands with short budgets and separate output directories. W&B is disabled;
these diagnostic bundles are not episode training curves.
