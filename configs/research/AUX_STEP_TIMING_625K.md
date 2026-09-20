# Matched-budget stepwise collection and learning at 625k

Two new full-episode closed-loop refinement settings:
soft/soft H3/J8/C16/A4/B256 at N32 and N128, on checkpoint625k of aux6434715x3.
Use environment seeds101–105/controller55, 500 real decisions per episode.
Reuse completed round-timing N32 and N128 controls, including their original
performance and training W&B IDs; do not rerun the prior or these controls.

After each parallel imagined step, append its N transitions and perform:

| Depth | Critic updates | Actor and alpha updates |
|---|---:|---:|
| 1 | 5 | 1 |
| 2 | 5 | 1 |
| 3 | 6 | 2 |

Within each depth, critic updates precede actor/alpha updates. Continue the same
branches with the updated actor; each new round starts again at the real root.
The general component schedule allocates `floor(K*d/H)-floor(K*(d-1)/H)` updates
at depth d, for per-round component count K. This keeps 128 critic, 32 actor
and 32 temperature updates per real decision; target tau.01 updates follow all
128 critic steps. B256 means40960 combined actor/critic replay draws. N32 uses
768 model transitions per decision; N128 uses3072. Replay capacity3072 retains
all transitions from all rounds; inner state resets at every real action.

Keep entropy-augmented critic fitting, frozen soft outer terminal bootstrap
with outer entropy, inherited adaptive alpha, actor/critic LR3e-4, strict
compilation, horizon conditioning off and original diagnostic probes. Timing
changes the available replay and subsequent imagined actions, so equal work
counts do not imply identical sampled data or target trajectories.

The N32 historical source is1f748ab3201df1b1e5c4d416faafc093949df692;
N128 is the reused fe87ae07b2a7ed751cd6865a3f60b2eae88e6abb reference.
Preparation verifies their checkpoint, protocol, exact planner identity apart
from timing, source identity, manifest and trace hashes, and complete coverage.
The new overview contains two fresh rows and two links to existing reference
runs. Only production indices0–1 may be submitted. Paired prior gains and
`comparison/round_timing_gain_*` are published along with full raw traces and
per-decision/update diagnostics. Trace validation checks collection depth,
retained replay, critic-first ordering within every depth, exact doses and
final work counts. Probes remain at initialization and after complete rounds.

H1 round/step parity is an implementation check (actions, parameters,
optimizers, replay, private RNG and update counts), not a new full-episode
campaign. Run CPU parity locally and compiled CUDA parity in the GPU smoke.
The smoke also evaluates both actual H3 checkpoint configurations for three
real decisions before production is released.

Prepare via `slurm/ambi_aux_hj_sweep.py prepare --matrix
configs/research/ambi_aux_step_timing_625k.json --baseline-campaign <completed
N/B campaign>` plus the usual checkpoint, inventory, prior and registry paths.
The new production campaign consists of10 full episodes; the overview includes
10 historical control episodes. Mixed-device historical comparisons remain
exploratory. No learning-rate, batch-size, rollout-count or J/H sweep is added.
