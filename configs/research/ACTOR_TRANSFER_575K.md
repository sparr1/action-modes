# Actor-only transfer at the frozen 575k checkpoint

This study tests whether retaining the adapted feedback actor between real
consecutive decisions reduces the compute needed for useful refinement. It is
`actor-transfer-v2`, distinct from the fresh-state closed-loop refinement recipe.

## Scientific panel

The matrix `ambi_actor_transfer_575k.json` evaluates **H={1,2,3}**,
**J={1,2,4,6,8,10}**, and **cold versus actor-only warm initialization**:
36 settings, five ordinary full episodes each (180 episodes). Environment
seeds 101–105 and controller seed 55 are paired. **Every real decision,
including the first decision of each episode, uses the selected J.** There is
no first-action budget override. The paired arms differ only in whether actor
weights reset at each real decision.

The superseded `actor-transfer-v1` campaign used first J10 and subsequent J.
Its results remain separately identified and are unsuitable as J<10 controls
for this study. Its six J10 cells have identical numerical solve budgets and
can be explicitly reused after the checks below; their original artifacts,
publication identities, and URLs are preserved. The default corrected group
is `actor-transfer-v2-575k-20260925`.

The source is `rwgao_b-brown-university/ambi/aux6428346x0`, checkpoint step 575000,
SHA-256 `0c6955db7cb8555a67d7863344b70be68f4b3250814d131e647ee6f9ef01a042`.
The hash-pinned frozen-prior mean episodes are reused from
`ambi_actor_transfer_575k_refs.json`, with their historical scientific identity
preserved. Metadata, saved alpha, runtime, episode protocol, and source hashes
are verified before preparing a campaign.

Each round collects N128 H-step imagined trajectories, performs C16 critic
updates, then A4 actor/temperature updates, with minibatch B256. The inner and
terminal critics use the auxiliary return pathway with one-step finite-horizon
reward-only targets. Alpha resets from the saved checkpoint then adapts within
each solve. Inner target reduction is min-pair; actor and terminal reductions
retain their checkpoint-resolved settings. Real execution uses `tanh(mu)` at
every observation, with action repeat two and the ordinary 500-decision timeout.

Warm cells retain **actor weights only**, within one episode. The critic, target
critic, replay, temperature, optimizers, and optimizer moments reset at every
real decision. The cold arm resets the actor too. All state resets at episode
boundaries; no outer updates or writeback occur. Replay capacity is
`max(3072, 128*H*J)`, enough to preserve all solve
transitions; matched cold and warm cells have identical capacities. Full frozen
outer-state hashes are checked before and after evaluation.

## Logging and interpretation

The transfer diagnostic snapshots cover initialization, immediately before the
first actor-update block (after critic fitting), immediately after that block,
and completed rounds. They record the inherited actor relative to the frozen
prior at the **same current observation**, including policy changes and the
model/critic scores provided by the engine. These are diagnostic predictions,
not real-return calibration. Fresh independent diagnostic randomness must not
alter collection, learning, or execution randomness.

The 32-rollout to-go probes occur at initialization, immediately before and
after the first actor block, and after each completed round (J+3 stages). Critic and actor update positions, actual first/steady budgets, transfer
flags, alpha, replay size, frozen state, compile fallback, and raw decision
metrics accompany every episode. Trace validation checks the actual chronology
of critic-first updates, zero initial replay, exact update/model-step counts,
selected J at every decision, and warm transfer only after the first decision.

CPU publication creates one authoritative performance curve per setting, with
complete manifest and raw traces attached. The campaign overview exposes:

- Return versus J and measured controller seconds per decision, by horizon.
- Warm-minus-matched-cold return differences and exploratory paired 95%
  bootstrap intervals (2,000 resamples, seed 20260912).
- Per-phase and per-round model/policy diagnostics, split between the first
  decision and subsequent decisions. Root measurements are first averaged
  within each episode, then environment seeds are weighted equally.
- First/steady latency and probe-cost fields from evaluator runtime records.

Only five environment seeds from one trained backbone are evaluated. The
intervals are exploratory and unadjusted for the full comparison grid. Later
state distributions diverge across arms despite seed pairing. Distinguish
nominal round reductions from observed latency. Controller time subtracts
measured CUDA probe duration from prediction wall time, retaining host trace
overhead; it is not pure uninstrumented latency. Report the full instrumented
prediction time and diagnostic duration too, with first versus steady decisions
separated. Model initialization and warmup compilation are separate overheads.

## Execution

Prepare and launch from a clean Git-synchronized checkout at the exact tested
commit. `slurm/run_ambi_actor_transfer_oscar.sbatch` accepts three explicit modes:
`prepare` (CPU), `worker` (GPU array), and `watch` (CPU publication). Set
`EXPECTED_ACTION_MODES_SHA`, `CAMPAIGN_ROOT`, `EVAL_MODE`, and `PYTHON_BIN`.
Preparation additionally requires `CHECKPOINT_INVENTORY` and
`EVAL_REFERENCES_PATH`; `EVAL_MATRIX_PATH` may explicitly select the matrix.
Set `EVAL_REUSE_J10_FROM` (or pass `--reuse-j10-from` to CPU preparation) to
an original v1 campaign directory to reuse all six J10 cells. Preparation
requires each original cell to be fully evaluated and published. It verifies
only the redundant first-action override differs in requested and resolved
configs and scientific implementation fingerprint, checks the checkpoint,
runtime, source commit, L40S hardware receipt,
all five 500-decision episode traces, artifact hashes, diagnostics, and original
publication identities. It pins the source campaign hash and keeps the original
first-override diagnostic flag; source traces are never relabeled or rewritten.
A missing/incomplete J10 cell or any scientific config mismatch aborts reuse.
Without this explicit option, all 36 settings are newly evaluated.

The worker enforces L40S hardware for timing comparability. CPU prepare loads
checkpoint metadata/temperature only; evaluation and compilation run on GPUs.
Per-job caches and W&B staging live in scratch. Smoke indices in `campaign.json`
cover cold and warm H1/J1, H2/J4, and H3/J8, each with two seeds and
three decisions to test uniform J, transfer, and episode reset. Production
indices exclude reused J10 cells; submit the indices from `campaign.json`.
GPU workers never initialize W&B. One CPU watcher bounds publication concurrency
and drains finished workers; `submission.json` supplies its `gpu_job_ids` list.

The CPU watcher also installs explicit Results panels in the authenticated
user's existing personal **project workspace** through
`utils/wandb_results_layout.py`. A native UI edit verified that the current
single-run page reads this project's `project-view`; the similarly named
legacy viewer-owned `run-view` does not control these panels. Logging metrics,
setting `hidden=False`, or logging `wandb.plot` objects alone is insufficient
when panels are absent. Future campaign publishers must install and verify the
results layout as part of publication, and verify browser rendering before
reporting visibility.

The helper preserves unrelated sections, settings, associations, and all other
project workspaces; only its stable owned section IDs are inserted or updated.
The explicit personal view defaults to `nw-nwuserrwgao_b-w` and can be selected
with `WANDB_RESULTS_VIEW_NAME`. No personal username is guessed from a team
entity. The scope is this project's workspace and run pages. The delivered URL
selects the exact overview run; generic workspace filters can exclude that run.
Close the workspace browser tab during an external layout update, then open a
fresh tab: an already-open UI can autosave stale state over the API patch.

Before/proposed/after snapshots and verification receipts are retained under
`results-layout/`. Readback verifies the saved schema, not browser rendering.
A layout failure is logged explicitly in the watcher and run summary; it does
not discard results, interrupt computation, or trigger checkpoint reevaluation.
Stable IDs make retries idempotent, including recovery from an uncertain
response. The helper reads fresh state immediately before mutation, but W&B
view updates in this helper do not perform an atomic compare-and-swap. A
concurrent edit in the final read/write gap can still be overwritten without
readback detecting it.
Use live Oscar allocation/QOS headroom to choose array concurrency. Do not reuse
an arbitrary historical throttle.

Example CPU preparation (inside a Slurm allocation):

```bash
python slurm/ambi_actor_transfer_campaign.py prepare \
  --root /oscar/scratch/rgao48/ambi/actor-transfer-575k/DATE/campaign \
  --inventory /path/to/checkpoint-manifest.json \
  --references configs/research/ambi_actor_transfer_575k_refs.json \
  --registry /oscar/scratch/rgao48/ambi/evaluation-series/runs
```

Do not overwrite completed bundles or rerun completed GPU cells for a publication
failure. Performance publication is reconciled through the existing immutable
record journal; the overview refuses an automatic second owner until its
publication state has been inspected.
