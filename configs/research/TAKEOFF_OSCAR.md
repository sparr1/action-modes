# Scratch SAC takeoff screen on Oscar

`ambi_scratch_takeoff_h1.json` is an exploratory frozen-checkpoint screen on
the reward/Q-scale backbone `rwgao_b-brown-university/ambi/mey3rxj8`. It selects
100k, 125k, 150k, 200k, 300k, 500k, and 2M decisions. Its H1 inner solve collects
128 fresh action samples with the actor fixed, performs 32 critic updates and
then four actor updates, and repeats for four rounds. Every real decision starts
fresh actor/critic weights and replay. See the matrix for the complete contract.

The ordinary episode evaluator retains mean-action execution, five episodes
with seeds 101–105, and a 500-decision cutoff. Its model probes use 32 sampled
rollouts at initialization and every completed round. Real calibration is a
separate sampled-action measurement on 25 shared prior roots, three independent
solves per root, four simulator rollouts per saved actor, five actor snapshots,
and 1,000 prior-continuation decisions. These scopes get separate diagnostic
series; sampled continuation returns are not relabeled as ordinary episode
returns. Each checkpoint freezes its own saved Q scale during adaptation.

## Inventory and explicit publication attempt

Prepare a JSON inventory outside the checkout with `source_run` equal to the
full run path above and exactly seven `checkpoints`, ordered as listed. Each
checkpoint row contains:

```json
{
  "step": 100000,
  "path": "/absolute/oscar/path/to/checkpoint",
  "sha256": "verified weight SHA256",
  "metadata_sha256": "verified adjacent sidecar SHA256",
  "prior_reference_bundle": "/absolute/path/to/completed/prior/bundle",
  "prior_reference_manifest_sha256": "verified prior manifest SHA256"
}
```

The worker verifies both checkpoint file hashes, the sidecar step, and the
prior reference manifest hash before reuse. The evaluator also
enforces the matrix's pinned checkpoint hashes, architecture, and objective.
The existing prior bundle must contain one completed prior-only run with the
same checkpoint, environment/action/controller-seed protocol and episode seeds.
This is checked by the evaluator; never modify historical reference metadata
to force compatibility. Real calibration generates its own simulator-state bank
and sampled prior references once per checkpoint, retaining both for later reuse.

Create a **new** checkpoint-axis evaluation attempt using metadata-only
`evaluate_ambi_checkpoint.py --eval-series-spec-dir` followed by
`eval_series.py create --spec ... --attempt-label ... --owner oscar-rgao48`.
Validate the same run against specifications from every selected checkpoint.
Save the explicit selector-to-run-directory mapping as `EVAL_RUN_MAP`:

```json
{"initialization/scratch": "/oscar/scratch/rgao48/ambi/evaluation-series/runs/NEW_ID"}
```

Do not append this new recipe to historical inherited or scratch curves. The
seven checkpoint results share this one new episode curve; model and real
diagnostics each use one checkpoint-specific series with the explicit attempt.

## Guarded smoke, then production

Use a clean Git-synchronized checkout at the tested commit and the existing
locked DMControl Python environment. The maintained launcher requires:

```bash
export EXPECTED_ACTION_MODES_SHA=<tested-commit>
export AMBI_DMC_PYTHON=/absolute/path/to/locked/dmcontrol/python
export AMBI_TAKEOFF_MANIFEST=/absolute/path/to/seven-checkpoint-inventory.json
export AMBI_TAKEOFF_ROOT=/absolute/path/to/fresh-campaign
export AMBI_TAKEOFF_ATTEMPT=<explicit-new-attempt>
export EVAL_RUN_MAP=/absolute/path/to/eval-run-map.json
```

Supply Slurm stdout/stderr paths outside the source checkout. The launcher has
no implicit array or concurrency cap. First submit smoke indices `3,10`, which
select the 200k checkpoint for ordinary episodes and real calibration:

```bash
sbatch --export=ALL,AMBI_TAKEOFF_SMOKE=1 --array=3,10 \
  --output="$AMBI_TAKEOFF_ROOT/slurm/smoke-%A_%a.out" \
  --error="$AMBI_TAKEOFF_ROOT/slurm/smoke-%A_%a.err" \
  slurm/run_ambi_takeoff_oscar.sbatch
```

The smoke explicitly fails when CUDA is unavailable. It runs analytic and real
simulator restoration tests, a CUDA RNG/outer-state preservation test, and the
actual full-size checkpoint. Ordinary evaluation uses two decisions. Real
calibration uses one prior root, one solve, all five actor snapshots, four
rollouts per snapshot, and the full 1,000-decision prior continuation, crossing
the original episode cutoff. Its benchmark records seven warmed, alternating
probes-off/probes-on timing pairs and verifies their actions match. Inspect
test outcomes, timing, complete bundles and model/real work before production.
Neither smoke task queues or publishes scientific results.

After both smoke cells pass, submit `--array=0-13%CONCURRENCY` with
`AMBI_TAKEOFF_SMOKE=0`. Indices 0–6 are full episode evaluations; indices 7–13
are separate real calibration jobs in the same checkpoint order. Set concurrency
from live GPU/CPU/memory allowances across all active jobs. The default per-cell
request is one L40S, six CPUs, 32 GiB, and four hours; override scheduling
resources only after examining the smoke and live queue. All output cells are
fresh, and existing results are never overwritten or automatically reevaluated.

## CPU publication and recovery

GPU workers stage completed checkpoint records without starting W&B. Use the
existing `run_eval_series_publisher_oscar.sbatch` once for the new episode run,
with `EVAL_RUN_DIR`, `EVAL_COMPUTE_JOBS`, and owner `oscar-rgao48`. It may watch
while GPUs run if CPU headroom permits, or drain after the compute array ends.

Separately, submit `run_ambi_takeoff_publisher_oscar.sbatch` with
`--dependency=afterany:COMPUTE_ARRAY`. An optional `--array=0-13` publishes the
14 independent model/real diagnostic bundles in parallel. Without an array,
one CPU job drains all 14 sequentially. Only complete bundles are published;
missing, incomplete, and failed uploads are recorded in fresh publication
summary receipts, and do not prevent publication of other completed cells.
The publisher returns a failing status if any selected bundle remains missing
or incomplete. Successful diagnostic receipts prevent duplicate uploads;
uncertain interrupted uploads require inspection before retrying.

Reports live under `production/step_<STEP>/episodes/{report.html,model-series}`
and `production/step_<STEP>/real/bundle/report.html`. The real directory also
retains simulator roots, prior-continuation cache, work shards and timing
benchmarks. `worker-completion.json` records total worker wall time separately
from the evaluator's optimization, probes, simulation, serialization, and
publication timings. The full panel is exploratory evidence from one trained
backbone; a frozen evaluation does not establish faster online outer learning.

Local launcher validation requires no scheduler or GPU:

```bash
environments/dmcontrol/.venv/bin/python -m pytest -q tests/test_ambi_takeoff_launcher.py
```
