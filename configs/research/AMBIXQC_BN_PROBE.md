# Fixed-root AMBI-XQC critic BatchNorm probe

This diagnostic uses the shared auxiliary UTD2 backbone at 475k decisions,
with the same pinned checkpoint and sidecar as the H1 screen. It studies
consistency with the learned model and frozen return critic. Its MSE is **not
an estimate of true environment value accuracy** and is not an episode-return
comparison.

`run_ambixqc_bn_probe.py --checkpoint PATH --output NEW_DIRECTORY --device cuda
--mode production` runs the complete probe. `--mode smoke` uses one root, one
solver seed and inspection steps 0, 1 and 3; production uses 32 roots, three
solver seeds and steps 0, 1, 3 and 12. Both modes require clean committed source,
verify checkpoint/sidecar hashes, and reject existing output directories.

## Paired construction

Roots come from the frozen persistent actor's deterministic mean trajectories
on environment seeds 101–105. The 32 positions are spread evenly from decision
0 through 499, with 7, 7, 6, 6 and 6 roots per trajectory. Capturing these roots
creates observations needed by the diagnostic; these episodes are not new
performance-curve points. Their returns are saved only as capture provenance.

For each root and solver seed 12345, 23456 or 34567, an independent CPU generator
creates 256 H1 training transitions, an independent 256-action held-out set,
12 minibatch index vectors sampled with replacement, bootstrap noises and
actor-step noise. The stochastic behavior policy is the unchanged persistent
actor. World-model transitions, predicted rewards and frozen reward scale are
shared exactly across all three critic conditions:

- `batch_update`: fit using joined current/next-policy batch statistics and
  update running buffers.
- `batch_no_update`: fit using those same batch statistics without changing
  running buffers.
- `running`: fit using inherited running statistics without changing buffers.

Every condition starts from the online auxiliary return critic, copying it
into online and target critics. Critic learning rate is 5e-5. The actor and
entropy temperature remain fixed during critic fitting. Targets are reward-only
categorical XQC targets using the frozen persistent actor and auxiliary critic
at the H1 boundary. Target BN behavior and parameter-only Polyak updates retain
XQC's existing rules. No training replay archive is sampled.

## Inspection

At 0, 1, 3 and 12 critic updates, read-only queries compare running-statistic Q
against joined-batch Q with `batch_no_update`. Queries use the identical held-out
actions and corresponding next-policy actions. Reported quantities include:

- Running twin-mean and twin-min Q MSE against the expectation of the actual
  reward-only categorical target: select the lower-mean frozen auxiliary head,
  transform every atom by model reward divided by the frozen scale plus its
  discounted support value, then clip and project onto XQC's support. This is
  generally different from clipping the scalar Bellman expectation. The
  unprojected scalar-target MSE and categorical clipping are separate diagnostics;
  selection uses the projected target that the critic is trained to predict.
- Running versus joined Q discrepancy, and cosine/norm ratio of action
  gradients. The gradient is the derivative of the sum of twin-min values with
  respect to the held-out action batch. Joined BN couples rows, so this is a
  batch-objective gradient, not isolated per-row Jacobian diagonal entries.
  Zero-norm cosine and zero-denominator ratios are recorded as null with
  explicit validity flags, rather than assigned artificial finite values.
- Each running-mean/variance buffer's RMS drift from inheritance, plus running
  action-gradient comparisons against the `batch_update` control.
- One actor step on a separate fresh clone, always initialized from the same
  inherited actor and alpha, with actor BN in running mode, actor LR 5e-5 and
  identical actor noise. Its resulting deployed KL and mean-action displacement
  are measured at the root. This disposable diagnostic step never feeds back
  into critic fitting or subsequent inspections.

Full persistent learner state, paired-input tensors, global RNG and diagnostic workspace hashes
provide mutation guards. Training actor/temperature counters remain zero;
target BN buffers remain inherited. Temporary network clones reuse copied
parameters rather than random initialization. Generated inputs and their
hashes are saved for reproducibility; expect roughly a few hundred MB for the
full 96 root/solver input bundles, depending on tensor storage overhead.

## Selection artifact

The alternative is chosen between `batch_no_update` and `running` by the mean
held-out **running twin-mean Q MSE after three critic updates**, weighting each
root/solver pair equally. Exact ties prefer `running`. `batch_update` remains
the reference even if its MSE is lowest. This chooses an alternative for the
next controlled episode experiment, not a proven best controller.

`selection.json` uses schema `ambixqc-bn-probe-selection-v1`, records the exact
source/checkpoint, root/seed/inspection grid, selected mode, rule, scores and
artifact hashes. `validate_selection(path, source_sha=...)` checks the complete
production grid, frozen-state evidence, paired input files and hashes, and
recomputes the selection. Smoke outputs cannot authorize the next production
stage. No W&B run or episode-evaluation registry is created by this diagnostic.
