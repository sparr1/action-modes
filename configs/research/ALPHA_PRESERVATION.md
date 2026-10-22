# Preservation of policy improvement with temperature carry

This separate extension tests the MPPI-trained auxiliary-return checkpoint at
500,000 decisions, source `rwgao_b-brown-university/ambi/mppi7530927x0`. It is a
positive control: the existing sampled fresh-alpha H3/J8 solve returned 316.37
on average versus 174.60 for the sampled base policy, a paired gain of 141.77
over five full episodes. The original result artifacts are revalidated before
reuse. The purpose is to check whether adapting alpha across episodes retains
useful policy improvement, in addition to changing entropy.

The scientific protocol remains `run-temperature-chain-v1`: five independent
streams, each containing three full 500-decision episodes. Streams use the same
rotated ordered seeds as the earlier chain experiment: 101/102/103,
102/103/104, 103/104/105, 104/105/101 and 105/101/102. Controller seed 55 is paired
per environment seed. Only alpha and its Adam state carry across all decisions
and episode boundaries. Each stream starts from the checkpoint alpha, which is
verified against its portable state and pinned profile. Actor, critics, replay
and their optimizers reset at every action. The backbone remains frozen.

Settings are H3/J8/C16/A4/N128/B256, reward-only auxiliary return Q for inner
learning and terminal bootstrap, sampled action execution, target entropy
-10.5, and actor/critic/temperature learning rates 0.0003. There is one alpha
step per actor step, no KL, no Bernoulli copying and no Retrace. All three
episode positions are retained. The primary endpoint is episode three:

- Paired return improvement over the sampled prior.
- Paired return difference from the sampled fresh-alpha inner SAC control.

A positive difference from fresh alpha means improvement; a negative difference
means loss. Suppressing entropy is not sufficient evidence of success if policy
improvement disappears. No noninferiority threshold is selected after seeing
the results, and five streams provide an exploratory preservation check rather
than proof of equivalence. Report uncertainty across independent streams and
individual paired outcomes. Episodes within a stream are dependent, and the
same historical reference episode reused at different positions is not a new
independent observation.

The prior and fresh-alpha results are reused from
`/oscar/scratch/rgao48/ambi/entropy-kl-mppi500k/20261020-v2/episodes/campaign.json`,
production modes `prior` and `entropy`. Preparation checks the exact checkpoint,
original matrix/profile hashes, complete five-seed/500-decision coverage,
sampled execution, update budgets, entropy settings, fresh-state scopes, frozen
outer state, finite diagnostics and complete trace hashes. It copies compact
reference summaries to the new campaign and pins all source artifacts. No new
prior, fresh-alpha, within-episode carry, KL or mean-action controls are run.

The new matrix is `configs/research/ambi_alpha_preservation.json`; its checkpoint
list contains only MPPI500k. It does not relabel the earlier two-backbone chain
campaign. The existing chain validators check all 48,000 alpha/Adam steps per
stream and continuity across its two episode boundaries.

```bash
python slurm/ambi_alpha_preservation.py prepare --root "$PRESERVATION_ROOT"
python slurm/ambi_alpha_preservation.py run \
  --campaign "$PRESERVATION_ROOT/campaign.json" --index 0
```

Indices 0–4 are independent streams, totaling 15 complete episodes and 7,500
decisions. Before production, stream 0 runs two eight-decision episodes using
`--smoke-steps 8 --smoke-root "$PRESERVATION_ROOT/hardware/GPU_TYPE/smoke"` for
each selected GPU class. Runtime and memory follow the measured chain procedure:
one CPU, 4 GiB host memory, cold solves separated from warm decisions, and a
25% plus two-minute walltime margin. The scheduler records hardware identity and
does not launch until the dedicated publication workspace is installed.

Each stream owns exclusive results, trace bundle, `chain-summary.json` and a
validated receipt. The publisher only forms complete-checkpoint performance
estimates after all five streams validate; incomplete or nonfinite output stays
a progress/failure state. Primary episode-three results and all three-position
trajectories are published in a separate preservation workspace.
