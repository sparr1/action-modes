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
