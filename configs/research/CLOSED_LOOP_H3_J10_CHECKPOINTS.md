# H3/J10 closed-loop checkpoint curve

The target −10.5/shared backbone (`ambi/aux6428346x0`) is evaluated at 100k,
200k, 300k, 400k and every 25k from 500k through 1M. This sparse early grid
retains dense coverage where the existing prior and MPPI curves improve; it
does not assume refinement cannot help earlier checkpoints.

The 25 points use return-only inner and terminal critics (`aux_return`), SAC
actors, one-step reward-only finite-horizon targets, no terminal entropy, H3,
J10, N128, B256, C16/A4, adaptive actor entropy and final `tanh(mu)` execution.
Each real decision starts fresh actor/critic/target/replay/optimizer state.
Replay capacity 3,840 retains every imagined transition. Each checkpoint keeps
its own saved temperature and weights; no numeric alpha or Q scale is copied
from 575k. This backbone disables actor Q normalization. CPU preparation
verifies saved checkpoint alpha against its completed prior reference.

Five full 500-decision episodes use environment seeds 101–105 and controller
seed 55, with action repeat two and summed rewards. Each decision retains 32
sampled model-return probes at initialization and after each inner round.
All outer state remains frozen. Model diagnostics are distinct from full
undiscounted real-episode return.

All 25 frozen-prior baselines are reused from the verified checkpoint bank.
The completed H3/J10 evaluation at 575k is also reused without republishing its
original performance or diagnostic IDs. Therefore only 24 checkpoints and
120 new full episodes are executed. Historical scientific identities remain
explicit; matching checkpoint hashes, seeds, protocol and canonical planner
identities are required. Prior and reused bundle hashes are pinned in
`ambi_closed_loop_h3_j10_checkpoint_refs.json`.

Prepare on a CPU allocation:

```bash
python slurm/ambi_closed_loop_checkpoint_sweep.py prepare \
  --root /scratch/new-campaign --inventory /scratch/checkpoint-manifest.json \
  --registry /scratch/evaluation-series/runs
```

The companion Oscar launcher accepts `EVAL_MODE=prepare`, `worker` or `watch`,
plus `CAMPAIGN_ROOT` and `EXPECTED_ACTION_MODES_SHA`. Preparation allocates one
new evaluation-series performance run shared by the 24 checkpoints, with
metadata-only New/Append identity validation. Separate per-checkpoint training
diagnostics preserve full traces. A CPU publisher serializes writes to the
shared performance run and assembles the 25-point overview, including the
original 575k result. The overview reports return, frozen-prior paired gain,
95% paired bootstrap intervals, source alpha and completion status. Detailed
performance and diagnostic runs retain runtime measurements.

Indices 0 and 24 (100k and 1M) are three-decision GPU smoke checks. Production
uses `campaign.production_indices`, excluding reused index 7. Gate production
on both smoke checks; request concurrent independent checkpoints within live
account limits. Workers cannot evaluate the reused checkpoint. GPU workers
never initialize W&B. The CPU publisher and Slurm jobs continue independently
of the submitting laptop.
