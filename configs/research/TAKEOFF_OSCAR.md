# Frozen inner-SAC screens on Oscar

`ambi_scratch_takeoff_h1.json` is an exploratory frozen-checkpoint screen on
the reward/Q-scale backbone `rwgao_b-brown-university/ambi/mey3rxj8`. It selects
100k, 125k, 150k, 200k, 300k, 500k, and 2M decisions. Its H1 inner solve collects
128 fresh action samples with the actor fixed, performs 32 critic updates and
then four actor updates, and repeats for four rounds. Every real decision starts
fresh actor/critic weights and replay. See the matrix for the complete contract.

The same launcher supports the explicit single-checkpoint
`ambi_prior_refinement_h1_200k.json` reference. Its default
`initialization/inherited` copies the prior actor and online critic, preserves
the prior standard deviation, and uses alpha zero. Only 200k is selected;
the update schedule and evaluation coverage remain unchanged. Always pass the
same selected matrix to smoke, production, identity creation, and publication.

`ambi_prior_refinement_h1_parallel.json` expands that reference to six
checkpoints: 125k, 150k, 200k, 300k, 500k, and 2M. Only the checkpoint list and
description change. With twelve available GPUs, its twelve independent episode
and real-calibration cells can run in one wave; verify live account capacity
before choosing concurrency. The 200k-only matrix remains available for narrow
iterations.

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
full run path above and exactly the matrix's `checkpoint_contract.checkpoints`,
in the same order. The default scratch matrix requires seven entries; the
200k-only prior matrix requires one, and the parallel prior matrix requires the
six checkpoints above. Each checkpoint row contains:

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
For compatible real calibration reuse, the inventory row may additionally
specify all three fields: `real_root_bank`, `real_root_bank_sha256`, and
`real_reference_cache`. The bank must be an existing absolute file with the
verified hash, and the cache must be an existing absolute directory. Stage a
copy of the reusable cache in the new campaign to preserve historical results.
Production passes `--root-bank` and the selected cache; smoke always generates
fresh roots and references. The evaluator still verifies the bank's runtime,
science and protocol identity and every cached reference before reuse.

Create a **new** checkpoint-axis evaluation attempt using metadata-only
`evaluate_ambi_checkpoint.py --eval-series-spec-dir` followed by
`eval_series.py create --spec ... --attempt-label ... --owner oscar-rgao48`.
Validate the same run against specifications from every selected checkpoint.
Save the explicit selector-to-run-directory mapping as `EVAL_RUN_MAP`:

```json
{"initialization/scratch": "/oscar/scratch/rgao48/ambi/evaluation-series/runs/NEW_ID"}
```

For the prior refinement matrix, the mapping key is `initialization/inherited`.

Do not append this new recipe to historical inherited or scratch curves. The
seven checkpoint results share this one new episode curve; model and real
diagnostics each use one checkpoint-specific series with the explicit attempt.

## Guarded smoke, then production

Use a clean Git-synchronized checkout at the tested commit and the existing
locked DMControl Python environment. The maintained launcher requires:

```bash
export EXPECTED_ACTION_MODES_SHA=<tested-commit>
export AMBI_DMC_PYTHON=/absolute/path/to/locked/dmcontrol/python
export AMBI_TAKEOFF_MANIFEST=/absolute/path/to/selected-checkpoint-inventory.json
export AMBI_TAKEOFF_ROOT=/absolute/path/to/fresh-campaign
export AMBI_TAKEOFF_ATTEMPT=<explicit-new-attempt>
export EVAL_RUN_MAP=/absolute/path/to/eval-run-map.json
```

The default matrix remains `configs/research/ambi_scratch_takeoff_h1.json`.
To launch a controlled variant, set `AMBI_TAKEOFF_MATRIX` to the selected
versioned matrix for smoke, production, and the diagnostic publisher. Both Python
subcommands also accept
`--matrix`, which takes precedence over that environment variable. Relative
matrix paths resolve from the repository root. The launcher validates the matrix
before any compute or publication: it must use the pinned backbone, list distinct
supported checkpoint steps with SHA256 pins, and select exactly one default
preset. Worker and publisher derive their checkpoint panel and selector from
that matrix. Receipts record its path, hash, selector, and selected steps;
the evaluators retain their existing checkpoint and identity validation. Create
and validate a new evaluation attempt using that same selected matrix; do not
reuse another recipe's run map.

Supply Slurm stdout/stderr paths outside the source checkout. The launcher has
no implicit array or concurrency cap. First submit smoke indices `3,10`, which
select the 200k checkpoint for ordinary episodes and real calibration:

```bash
sbatch --export=ALL,AMBI_TAKEOFF_SMOKE=1 --array=3,10 \
  --output="$AMBI_TAKEOFF_ROOT/slurm/smoke-%A_%a.out" \
  --error="$AMBI_TAKEOFF_ROOT/slurm/smoke-%A_%a.err" \
  slurm/run_ambi_takeoff_oscar.sbatch
```

For the **200k-only prior refinement**, export
`AMBI_TAKEOFF_MATRIX=configs/research/ambi_prior_refinement_h1_200k.json`
and use **`--array=0-1`** for smoke and production. Index 0 evaluates episodes;
index 1 runs real calibration. Every other index is rejected before compute.

For the **six-checkpoint prior panel**, use the same fresh-root and inventory
setup with its selected matrix. Smoke indices **2,8** select 200k episodes and
real calibration. After smoke passes, production indices **0–11** cover the
six episode cells followed by six real-calibration cells in checkpoint order:

```bash
export AMBI_TAKEOFF_MATRIX=configs/research/ambi_prior_refinement_h1_parallel.json
sbatch --export=ALL,AMBI_TAKEOFF_SMOKE=1 --array=2,8 \
  --output="$AMBI_TAKEOFF_ROOT/slurm/smoke-%A_%a.out" \
  --error="$AMBI_TAKEOFF_ROOT/slurm/smoke-%A_%a.err" \
  slurm/run_ambi_takeoff_oscar.sbatch
# After smoke passes, use live resource headroom for the production concurrency.
sbatch --export=ALL,AMBI_TAKEOFF_SMOKE=0 --array=0-11%12 \
  --output="$AMBI_TAKEOFF_ROOT/slurm/production-%A_%a.out" \
  --error="$AMBI_TAKEOFF_ROOT/slurm/production-%A_%a.err" \
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
In general, a selected panel of N checkpoints has episode indices 0 through
N−1 and real-calibration indices N through 2N−1. Inventory entries outside that
selected panel are rejected; the worker never expands a smaller selection.

## CPU publication and recovery

GPU workers stage completed checkpoint records without starting W&B. Use the
existing `run_eval_series_publisher_oscar.sbatch` once for the new episode run,
with `EVAL_RUN_DIR`, `EVAL_COMPUTE_JOBS`, and owner `oscar-rgao48`. It may watch
while GPUs run if CPU headroom permits, or drain after the compute array ends.

Separately, submit `run_ambi_takeoff_publisher_oscar.sbatch` with
`--dependency=afterany:COMPUTE_ARRAY`. An optional `--array=0-13` publishes the
14 independent model/real diagnostic bundles in parallel. Without an array,
one CPU job drains all selected bundles sequentially. For the 200k-only matrix,
use `--array=0-1` or a single sequential publisher; both visit exactly two
bundles. Set `AMBI_TAKEOFF_MATRIX` for the publisher as well as the workers.
For the six-checkpoint prior matrix, the diagnostic publisher uses
`--array=0-11` or one sequential publisher covering exactly twelve bundles.
Only complete bundles are published;
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
