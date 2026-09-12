# MPPI and inner-SAC action boundary measurements

`evaluate_mppi_saturation.py` is a separate, offline diagnostic on the frozen
`mey3rxj8` 200k backbone. It reuses the complete mean-prior simulator-root bank
from experiment 1: seeds 101–120 and decisions 0/100/200/300/400. Root-bank
checkpoint, scientific implementation, protocol, integration snapshots and
observations must pass the existing calibration loader.

```bash
python evaluate_mppi_saturation.py \
  --checkpoint /absolute/checkpoint \
  --root-bank /absolute/real/bundle/artifacts/root-bank.json \
  --output-dir /absolute/new-saturation-output --device cuda
```

Use `--smoke` for only the first root and one solver repetition. It retains
the complete model, all four MPPI iterations, SAC update budget and 1,024 policy
draws. Normal execution uses all 100 roots and three independent solves each.
No simulator continuation or episode-return comparison runs in this diagnostic.

The unchanged AMBI MPPI helper runs H1/N128/E16, six prior trajectories, four
iterations, temperature 0.5, standard-deviation bounds 0.05–2, and the checkpoint
discount and frozen online-Q reduction. A scoped observer clones each candidate
population and predicted values without consuming RNG or changing its action.
The observer reconstructs the existing elite fit after the solve and verifies
the final mean and returned action exactly. Exception cleanup removes the
observer and restores global RNG streams and module modes.

For every iteration, measurements cover all candidates, prior candidates,
Gaussian proposal candidates, unweighted elites, the categorical elite mixture,
and the optimized mean. Exact saturation means an action component equals ±1;
near saturation means its magnitude is at least 0.99. Mean absolute action is
also recorded. Fractions are over action components, not whole action vectors.

**The elite mixture and optimized mean are different distributions.** The
mixture statistic computes the expected saturation when selecting an elite
according to its fitted weight. The optimized-mean statistic measures the
deterministic action returned by `mppi_plan(eval_mode=True)`. The official
TD-MPC2 implementation samples an elite even during evaluation. This observer
does not run a separate official TD-MPC2 policy or add an execution-noise draw.

On exactly the same roots, the diagnostic reruns the existing inherited
H1/N128/J4/C32/A4/B256, fixed-zero-temperature SAC setting. Private solver seeds
use the original `real-calibration-solve` namespace. Immutable actor snapshots
at rounds 0/1/2/4 are evaluated using 1,024 fixed private Gaussian noise rows,
paired across rounds and solver repetitions within each root, plus zero noise
for the mean action. The checkpoints, outer optimizers and saved Q scale remain
frozen. This does not change full-episode evaluation or the J1/J2/J4 campaign.

The output is immutable and contains a compact `report.html`, `results.json`,
per-root `measurements.jsonl.gz`, complete observed MPPI populations in
`mppi-populations.jsonl.gz`, and artifact checksums. Statistics average solver
repetitions within root, roots within source episode, then episodes equally.
95% episode-cluster intervals use 2,000 seeded bootstrap resamples; a one-episode
smoke does not receive a population interval. GPU evaluation never publishes
online. The bundle also contains the source matrix, checkpoint metadata and
complete root bank, with hashes linking them to the recorded protocol.

Publish a completed result from a CPU allocation using a deliberately chosen
new attempt label and unique W&B run ID:

```bash
python publish_mppi_saturation.py --bundle /absolute/completed-output \
  --attempt-label mppi-saturation-200k-example --run-id UNIQUE_RUN_ID
```

Publication uses `ambi-inner-bench` and preserves the source protocol, hashes,
complete raw measurements and compact HTML as an `action-saturation` artifact.
Native curves use `saturation/round` with separate MPPI distributions and SAC
sample/mean metrics; their final-round values also appear in scalar summaries.
Missing SAC round 3 is omitted. Completed publication is idempotent. An uncertain
network failure retains a journal and requires inspection before any retry.
Smoke publication requires an explicit `--allow-smoke` and never masquerades as
the complete panel.

`slurm/run_mppi_saturation_oscar.sbatch` requests one L40S, six CPUs and 32 GiB,
pins a clean tested Git SHA, and uses the locked DMControl runtime. Set
`AMBI_MPPI_CHECKPOINT`, `AMBI_MPPI_ROOT_BANK`, and `AMBI_MPPI_OUTPUT_ROOT`;
`AMBI_MPPI_SMOKE=1` runs the focused GPU checks and writes `OUTPUT_ROOT/smoke`,
while normal execution writes `OUTPUT_ROOT/production`. Schedule
`slurm/run_mppi_saturation_publish_oscar.sbatch` after successful production,
with `AMBI_MPPI_BUNDLE`, `AMBI_MPPI_ATTEMPT` and `AMBI_MPPI_RUN_ID`. Both scripts
require `EXPECTED_ACTION_MODES_SHA`; the GPU worker always remains offline.
