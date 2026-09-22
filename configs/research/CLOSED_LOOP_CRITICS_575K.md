# Closed-loop soft versus return-only critics at 575k

This exploratory screen uses the auxiliary-return SAC backbone
`rwgao_b-brown-university/ambi/aux6428346x0` (target entropy −10.5,
shared representation gradients), at 575,000 training decisions. The checkpoint
was selected because the existing MPPI comparison had a large paired critic gap;
this is a diagnostic case, not an unbiased estimate across training checkpoints.

At every real observation, the evaluator creates fresh actor, critic, target,
optimizers, and replay from the frozen checkpoint. It collects imagined
transitions and performs the complete inner solve, executes the adapted actor's
mean action, and repeats for the full 500-decision Humanoid Walk episode. There
is no outer learning or writeback. This is full-episode closed-loop refinement,
not a model-only probe or a prefix followed by prior-policy continuation.

The six conditions cross J={1,2,4} with two coherent critic strategies:

| Arm | Initial critic | Inner fitting | Frozen terminal bootstrap |
|---|---|---|---|
| Soft | SAC soft Q | Entropy augmented | Soft Q with outer entropy correction |
| Return-only | Auxiliary return Q | Reward only | Return Q, no entropy correction |

Both retain the same SAC actor and inherited adaptive temperature with target
entropy −10.5. Critic initialization, fitting, and terminal bootstrap change
together; this comparison does not isolate their individual effects.

Each round uses N128/H3, C16/A4, B256, replay capacity 3072, critic-first updates,
learning rates 0.0003 and target tau 0.01. Replay survives across rounds within
one decision and resets at the next observation. Compilation is strict. Five
paired environment seeds 101–105 use controller seed 55. Compatible prior
episodes are reused with their original provenance and immutable hashes.

Episode return is raw undiscounted environment reward. Report paired gains over
the prior and direct return-only minus soft differences at each J. Five seeds
make this a preliminary screen; bootstrap intervals are exploratory. Retain 32
independent model-return probes at initialization and after each round, together
with all actor/critic update traces. Model-return probes are separate from real
episode returns and exclude an explicit entropy bonus.

## Publication

`slurm/ambi_closed_loop_critics.py` prepares explicit New performance identities
and evaluates complete five-seed settings. `slurm/ambi_closed_loop_publish.py`
is the single CPU publication owner. It publishes performance and inner-training
histories plus a comparison overview containing real returns, paired gains,
critic differences, per-seed outcomes, settings, and links. Pending conditions
remain missing rather than zero. The overview updates as conditions finish.

GPU workers do not initialize W&B. Publication failure does not require rerunning
episodes; recover from the complete local bundles and inspect the publication
journal before resuming an uncertain upload. All W&B staging and compiler caches
use scratch through `slurm/run_ambi_closed_loop_critics_oscar.sbatch`.
