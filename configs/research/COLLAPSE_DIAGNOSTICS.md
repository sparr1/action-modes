# Policy-collapse checkpoint diagnostics

`diagnose_policy_collapse.py` measures three hypotheses without outer learning:

1. Matched full-episode prior and inner-controller mean versus sampled actions.
2. Q/S and alpha-H actor-gradient norms and directions on one immutable upright
   observation bank, before and after fresh local adaptation at each root.
3. Frozen gSDE prior mean, independently refreshed matrices, and matrices held
   for the checkpoint's configured execution interval.

This protocol supports Humanoid Walk state observations, single-component Q,
primary actors/critics, action-scoped inner SAC, and squashed-action entropy.
It preserves saved inner horizons, batches, update counts, alpha, scale update,
critic dropout during adaptation, and gSDE imagined-noise scopes. There are no
entropy or scale interventions and no outer updates or checkpoint writes.

The shared bank contains 32 upright states selected deterministically from a
specified frozen prior's mean-action episodes with environment seeds 201–210.
Candidates occur every 25 decisions from 25 onward and must have head height
at least 1.2 and torso upright at least 0.8. Selection spans all eligible
candidates. The bank records its source checkpoint, environment, and content
hash; every checkpoint must use exactly that bank.

Gradient measurements use 32 independent marginal action samples at each root
and three paired repeats. Actor parameters are the coordinates; Q/S and alpha-H
are ascent contributions. Shared noise and critic-head pairs are identical
across checkpoints and before/after adaptation. Encoder inputs are detached;
critic dropout is disabled only for the measurement. Autograd does not write
parameter `.grad` buffers or change optimizers. Zero gradients have explicit
validity flags and null ratios/cosines. Conditional action spread is recorded
separately from the historical training minibatch R; the two are not equivalent.

Full episodes use seeds 101–105, at most 500 decisions, and paired controller
seeds. Prior parameters are frozen. Inner episodes retain the saved fresh SAC
recipe at every decision. A private CUDA graph produces both sampled action and
final mean; the mean arm discards the sampled output, keeping the graph learning
RNG sequence paired. gSDE collection matrices retain their native scopes. The
unused execution draw in the mean arm is independent of collection noise. This
is an explicitly identified frozen-evaluation RNG protocol, not an exact replay
of the original online training stream. GPU validation compares captured and
uncaptured execution and checks equal adapted policies between mean/sample arms
on the same root. All paths verify complete outer-state immutability.

Checkpoint timing must be reported honestly: 25k checkpoint spacing may miss a
10k training-drop window. Pre/drop/recovery designations refer to observed
training windows; a bracketing snapshot is not an exact saved collapse state.
Neither passing a fixed-state probe nor a few paired episodes proves causality.

Example commands (fresh output directories are required):

```bash
python diagnose_policy_collapse.py bank --checkpoint STABLE --output BANK
python diagnose_policy_collapse.py parity --checkpoint CHECKPOINT --output GPU_CHECK
python diagnose_policy_collapse.py episodes --checkpoint CHECKPOINT --output EPISODES
python diagnose_policy_collapse.py gradients --checkpoint CHECKPOINT --bank BANK/bank.json --output GRADIENTS
```

The launcher consumes an explicit campaign manifest outside the source tree.
Compute writes atomic result/progress JSON and never initializes W&B. Smoke and
resource measurements also keep W&B disabled. Existing checkpoint bytes,
metadata, and unrelated runs are preserved.
