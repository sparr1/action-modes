# H4 extension through J18 at the 575k shared checkpoint

The `ambi_closed_loop_critics_h4_j16_575k.json` and
`ambi_closed_loop_critics_h4_j18_575k.json` matrices extend the completed H4
return/return mean-action curve with one new setting each. Each matrix selects
only its new return-only setting; the schema's prior variant remains unselected.
Reuse the hash-validated completed frozen-prior episodes for paired gains.

The checkpoint remains `ambi/aux6428346x0` at 575,000 training decisions:
target entropy −10.5, shared representation, checkpoint SHA256
`0c6955db7cb8555a67d7863344b70be68f4b3250814d131e647ee6f9ef01a042`.
Both the adapted critic and terminal bootstrap use the auxiliary return critic;
one-step targets contain reward only and terminal entropy is disabled. Actor
entropy remains adaptive and initialized from the checkpoint. Real execution
uses the final adapted actor's `tanh(mu)`; imagined actions remain sampled.

The only learning changes from the completed H4/J14 setting are round count and
replay capacity:

| J | Model transitions / replay capacity | Critic updates | Actor / temperature updates |
|---|---:|---:|---:|
| 16 | 8,192 | 256 | 64 / 64 |
| 18 | 9,216 | 288 | 72 / 72 |

Every real decision starts fresh from the frozen checkpoint, with N128 imagined
rollouts per round, H4, C16/A4, B256, and replay retained across rounds. Five full
500-decision episodes use environment seeds 101–105 and controller seed 55,
with action-repeat two and summed rewards. Retain 32 independent model-return
probes at initialization and after each round. H4 continues the existing
intentional rollout beyond the checkpoint's training unroll horizon of three.

Use `slurm/ambi_closed_loop_critics.py prepare` with the selected matrix to create
a separate campaign and publication identity for each J. The established
`slurm/run_ambi_closed_loop_critics_oscar.sbatch` launcher supports CPU preparation,
one-setting GPU workers, and the CPU publisher. Run the J18 three-decision smoke
before either full evaluation. Preserve existing results and source identity;
no other horizons, soft critics, prior reruns, or checkpoint sweep are selected.
