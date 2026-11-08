# Post-tanh policy-prior fixes

`experiments/ambi_posttanh_prior_fixes.json` trains six Humanoid Walk agents.
Each of `outer_adapt` and `outer_frozen` is crossed with an ordinary SAC prior,
corrected TD-M(PC)² (beta 1), and corrected BOOM (lambda 0.42). There is no
BOOM lambda-zero control. The reference backbones are the corresponding
`ambi_inner_entropy_posttanh_*` configurations; only the outer prior objective
and its execution/publication settings change.

All cells retain seed 55, one million agent decisions, fixed joint post-tanh
entropy coefficient 0.0021, H3/J8/K1, N128/C16/A4/B256, reward-only primary
critics, and 40 portable checkpoints at 25k intervals. Both inner scales start
from the current outer scale. `outer_adapt` updates its local scale after each
actor update; `outer_frozen` keeps that scale fixed for the current solve.
Both adapt their inner actor and critic normally.

The TD-M(PC)² penalty uses the corrected squashed joint cross entropy and
coefficient beta*d/(61*S), gated by S>2 and sufficient valid replay labels.
The implementation's per-dimension CE diagnostic is multiplied by d exactly
once to recover the joint density. BOOM aligns sampled actions to the saved
behavior mean after tanh, using detached current pre-tanh standard deviation
floored at 0.05 and detached softmax(Q/S) weights. Its alignment contribution
is lambda/S when S>2. BOOM uses H nonterminal states for all actor terms;
ordinary SAC and TD-M(PC)² retain H+1 actor states. Thus this six-cell screen
compares the methods as implemented, without isolating BOOM's horizon change
through a lambda-zero control. Neither method relabels collection-time policies.

Strict compilation, vectorized critics, GPU replay, TF32-high, and whole-inner
CUDA graphs are enabled. The two added losses use their reduce-overhead
compiled regions. Graph replay has its own RNG contract. Sparse native inner
diagnostics remain supported, and observational critic geometry runs every
1,000 outer updates in every cell, independent of regularization.

`slurm/run_posttanh_prior_fixes_oscar.sbatch` invokes the matching launcher.
Its disabled-W&B gates verify CUDA numerical/gradient behavior, end-to-end
training with valid behavior labels at production shapes, checkpoint I/O,
graph/native/graph continuation, and unchanged parameters/RNG after geometry
observation. Separate 1/2-CPU measurements choose the smallest allocation within
5% of the fastest measurement. Walltime is measured throughput times one million
with 25% margin plus half an hour, rounded up. Source/config/runtime receipts
must match before training. All scientific training runs publish into a dedicated
six-run workspace through the idempotent, serialized workspace publisher.

## Outer-adapt regularization strength

`experiments/ambi_posttanh_prior_strength.json` selects four fresh outer-adapt
runs with regularization multipliers 0.1 and 0.01 relative to the original
campaign. BOOM uses `outer_boom_coef=0.042` or `0.0042`; TD-M(PC)² uses
`outer_behavior_policy_kl_coef=0.1` or `0.01`. All other scientific parameters,
including the S>2 gate, sampled-action objectives, entropy, seed, one-million
decision budget, and checkpoint/diagnostic cadences match the original
outer-adapt arms. There are no new baseline or outer-frozen runs.

`slurm/run_posttanh_prior_strength_oscar.sbatch` executes one arm per GPU with
`AMBI_PRIOR_VARIANT=strength`. Prepare the campaign using
`slurm/posttanh_prior_strength_oscar.py prepare`. Preparation verifies that the
learner, environment, dependencies, and CUDA parity tests are byte-identical to
validated commit `342c7f1ec4d55fe18257499431eed335226e335a`, checks the original
receipt and JUnit hashes, and records the provenance of reused GPU parity and
1/2-CPU sizing evidence. It never represents reused checks as newly executed.

Every job first performs a fresh, disabled-W&B check at its exact coefficient:
production-size Humanoid training, 128 timed graph decisions, behavior labels,
checkpoint roundtrip, critic geometry and graph/native/graph continuation.
Only success permits scientific training in the same allocation. The measured
runtime must fit the allocated walltime. All four jobs can execute concurrently;
one CPU per L40S follows the measured sizing evidence. The dedicated workspace
contains all four run IDs with explicit progress and method-specific panels.

## BOOM x0.01 Humanoid Stand and Run

`experiments/ambi_boom_x0p01_humanoid_stand.json` and
`experiments/ambi_boom_x0p01_humanoid_run.json` retain the complete learning
recipe of Humanoid Walk run `03bc3869a52b4a69a64d180fc28f4d1e` at
`bcaa1f59b8e513d7095b1276d6cd2f34144ab18a`. Only the task and publication
identity change: seed 55, one million decisions, BOOM coefficient 0.0042,
sampled final actions, H3/J8/K1, fixed squashed entropy coefficient 0.0021,
all 25k checkpoints, and critic geometry every 1,000 updates remain intact.

`slurm/boom_humanoid_oscar.py prepare --task stand` (or `run`) uses
`AMBI_PRIOR_VARIANT=humanoid_stand` (or `humanoid_run`). It verifies unchanged
learner/environment/lock/parity-test sources against the linked run, checks the
saved CUDA evidence and 1/2-CPU scaling results, and records their reuse.
`slurm/run_boom_humanoid_oscar.sbatch stand` (or `run`) then performs a fresh
task-specific production-shape GPU smoke check with W&B disabled, including
128 timed decisions, checkpoint roundtrip, geometry observation and
graph/native/graph continuation, before starting fresh scientific training.
Each task owns a separate output directory and W&B run. Workspace publication
is idempotent and a publication failure stops the launcher visibly.

One CPU per L40S follows the 1/2-core benchmark (0.1732 versus 0.1735 seconds
per decision, approximately one core actually used). The 59-hour reservation
comes from the completed reference's 50.29 hours plus 15% and half an hour for
startup. A fresh smoke measurement must fit that reservation. Both jobs may
run concurrently, subject to the live account limits and GPU availability.
