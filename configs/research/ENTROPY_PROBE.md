# Frozen inner-SAC entropy diagnosis

`ambi_entropy_probe.json` pins two checkpoint files and sidecars: the SAC-aux
100k checkpoint from `ambi/full7391933x4`, and the MPPI-aux 400k checkpoint from
`ambi/mppi7530927x0`. Each uses seeds 101–105 and roots immediately before
decisions 0, 10, 50, 100 and 250. One worker owns one checkpoint and one seed.

The source controller is the existing `early_h3_j8/mean` recipe, whose matrix
file hash is pinned. Each real decision starts fresh from the checkpoint actor,
auxiliary return critic and outer temperature. Eight rounds each collect 128
H=3 trajectories, then perform 16 critic updates followed by four actor and
four temperature updates with batch size 256. All three learning rates are
3e-4. The reward-only inner target uses the auxiliary return critic at the
horizon, while the actor remains entropy regularized. The evaluator executes
the adapted mean; `inner_execution_action=policy_sample` remains in the recipe
and the wrapper's deterministic evaluation flag selects mean execution.

Historical early-checkpoint traces did not save actions or root observations.
These workers therefore collect **new protocol-matched trajectories**, not
exact historical replays. Each worker runs one unscored warmup, resets the
private controller RNG using the evaluator's episode seed scheme, then runs
the ordinary H3/J8 controller. Reaching root 250 takes exactly 250 source
controller decisions, with no action executed at root 250. The complete bank,
including observations, actions and rewards, is saved before any diagnostic
executes. This ordering prevents diagnostics from changing the source episode.
Premature episode termination fails the cell instead of dropping requested
roots. Roots from the two backbones are their respective on-policy roots,
not identical simulator states.

At each saved root, `utils/ambi_entropy_probe.py` creates independent private
solver state, preserving the continuing controller and global RNG. Its seed is
`solver_seed(55, "entropy_probe", environment_seed, decision_index)`, paired
across backbones. The diagnostic solve is a new draw at the same observation,
not the original trajectory solve. It records an instrumented exact H3/J8
solve, gradients before/after the first critic phase, and fixed-replay actor
arms with inherited automatic alpha, one-tenth inherited automatic alpha and
zero actor entropy weight. The fixed replay and critic in these arms isolate
the actor update mechanism; they are not closed-loop performance evaluations.
Root and imagined-depth anchors remain separately labeled. The default uses
32 actor updates and 128 anchor samples per group. Results preserve signed
entropy-gradient and Q-gradient effects, minibatch and anchor entropy,
temperature movement, and pre/post-update quantities from the core probe.

The CLI verifies the entire outer model, auxiliary critic, optimizers,
temperature and training counters before collection, after collection, and
after every diagnostic. Every completed root is saved atomically. A final
`result.json` is written only on success; failures leave `failure.json` plus
already durable artifacts. Source code, checkpoint, sidecar, resolved config,
root bank and each root record have SHA256 identities. Existing output
directories are never overwritten. A failed diagnostic can reuse its saved
bank with `--root-bank`; checkpoint, code, protocol, seed and root coverage must
all match. No GPU worker initializes W&B or updates the outer model.

Example (checkpoint and metadata paths are in the profile inventory):

```bash
python evaluate_ambi_entropy_probe.py \
  --checkpoint-id sac_aux_100k --checkpoint "$CHECKPOINT" --metadata "$METADATA" \
  --environment-seed 101 --output "$OUTPUT" --device cuda --torch-threads 1
```

For a smoke check, add `--decisions 0 10`; for metadata/config preflight add
`--plan-only`. `slurm/run_ambi_entropy_probe_oscar.sbatch` requires an exact clean
Git commit and the locked DMControl interpreter. GPU, CPU, memory, walltime and
array concurrency are supplied at submission after live capacity checks and
short throughput measurements; the launcher imposes no arbitrary reservation.

Compare paired seed-level summaries, retaining the five roots as a cluster
within each seed. Fifty roots are not fifty independent environment seeds.
Differences between these two checkpoint/backbone choices establish a mechanism
diagnostic, not a controlled causal comparison of training algorithms.
