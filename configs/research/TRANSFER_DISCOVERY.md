# Inner-SAC transfer discovery at the frozen 575k checkpoint

This exploratory campaign tests transfer between successive decisions within
an episode across **H = 1, 2, 3** and **J = 1, 2, 4, 6, 8, 10**. Each of the
252 mechanism/H/J cells runs development seeds 101–103, for 756 episodes.
Every episode has up to 500 real decisions. Every decision solves with the
selected J, C16/A4/N128/B256, then executes the adapted actor's mean action.
The backbone, control priors and all outer optimizers remain frozen.

The configuration is [ambi_transfer_discovery_575k.json](ambi_transfer_discovery_575k.json).
It pins step 575000 and checkpoint SHA256
`0c6955db7cb8555a67d7863344b70be68f4b3250814d131e647ee6f9ef01a042`,
using the existing `return_return/fresh` checkpoint recipe: SAC actor, auxiliary
reward critic, reward-only inner target and auxiliary reward terminal critic.
The actor retains the recipe's learned temperature and entropy term; a
reward-only critic does not mean entropy-free actor optimization.

| Arms | Carried information |
| --- | --- |
| `rho_a{0,05,1}_c{0,05,1}` (nine combinations) | Actor and online critic independently initialize as prior + rho × (previous final − prior). Rho 0 is the pretrained prior, not a random network. |
| `behavior_only` | Fresh learner networks; the preceding final actor is held fixed for imagined collection. The first decision uses ordinary fresh collection. |
| `fresh_replay25` | Fresh learner, 25% minibatch rows from the preceding solve's imagined replay. |
| `joint_replay25` | Full actor/online-critic weights plus the same 25% preceding-solve replay. |
| `full_state_replay25` | Joint weights, target critic, Adam, temperature and lifetime state, plus 25% preceding-solve replay. |
| `joint_prior_anchors` | Joint weights, actor KL(current‖frozen prior) coefficient 0.1, critic normalized parameter penalty coefficient 0.1. |

The critic anchor averages across parameter tensors the squared deviation from
the frozen prior divided by that tensor's mean squared prior value plus
`1e-6`. Both coefficients are exploratory fixed settings. The full-state arm
changes several coupled state variables together; success would motivate a
subsequent decomposition of target, Adam and temperature effects.

Except for full-state carry, targets are explicitly copied from the selected
starting online critic, and Adam, temperature and replay initialize afresh.
For replay arms, the fixed requested minibatch fraction is `round(0.25 × B)`
old rows and the remaining current rows, sampled with replacement. Replay
comes from the immediately preceding solve, not an accumulating episode-wide
buffer. It preserves the horizon/boundary metadata and cannot cross horizons
or critic objectives. The first solve has no donor or old replay. Replay
capacity is `max(3072, H × J × 128)`, matching the earlier uniform-J convention.
Episode reset clears every donor.

Run one cell from the repository root:

```bash
python evaluate_ambi_transfer_campaign.py \
  --checkpoint /path/to/575k.pt \
  --horizon 2 --rounds 8 --arm rho_a05_c05 \
  --device cuda --output-dir /path/to/new/h2_j8_rho_a05_c05
```

`--list-cells` prints `{index,name,H,J,arm}` JSON rows without loading a
checkpoint, ordered high J first. `--dry-run` checks the checkpoint contract
and prints the fully resolved cell without creating output. `--smoke` runs
two decisions by default; an explicit `--max-steps 3 --seeds 101 102` can
exercise two transfer boundaries in a labeled smoke. A shorter production
episode is rejected unless it is explicitly a smoke. `--no-compile` is a
recorded eager override; normal GPU cells enable reviewed compiled kernels.

Each output directory is created exclusively and is never reused implicitly.
It contains an immutable `manifest.json`, `runtime.json`, live atomically
updated `progress.json`, lean per-decision JSONL, one immutable JSON per
completed episode, and final `results.json`. Failures write `failure.json`
and update progress. The manifest records source-file hashes, Git HEAD,
checkpoint/configuration hashes, resolved settings, timing semantics and seeds.
Results are published separately by the campaign orchestrator; the evaluator
does not initialize W&B or train the outer model.

All arms use the established private RNG protocol
`solver_seed(55, "episode", environment_seed)`, carried forward within each
episode. Environment seeds and action selection rules match across arms;
their later real states differ. High-level timing includes donor construction
and export, with explicit CUDA synchronization at measured boundaries. There
are no model calibration probes, learner snapshots or full update traces in
the default campaign. First-decision compile time is retained and reported
separately from subsequent decision latency. Historical timing with diagnostic
probes is not interchangeable.

This is a development screen, not a confirmatory claim. There are three
independent environment episodes per cell; decisions and replay rows are not
additional trials. Use paired episode returns, failure frequency and cost to
shortlist mechanisms, then use the separately implemented saved-root
[diagnostics](TRANSFER_DIAGNOSTICS.md) to attribute their effects. Changing H
also changes the number of imagined transitions at fixed J/C/A/N, so this
grid makes no equal-compute comparison across horizons.

The Oscar entry point is `slurm/run_ambi_transfer_discovery_oscar.sbatch`, with
an explicitly pinned clean commit and `EVAL_MODE=prepare`, `worker` or `publish`.
Preparation verifies the checkpoint and sidecar hashes and writes an immutable
campaign manifest. Nineteen GPU smoke cells cover every mechanism at H3/J2,
full-state carry at H1/J2 and H2/J2, and fresh H1/2/3 at J10. Each smoke uses
two seeds and three decisions. Every production worker requires all smoke
completion receipts from the same campaign and commit. One worker owns all
three episodes for a configuration; concurrent workers never share outputs.
The publisher exposes complete three-seed results and separate live progress,
verifies output receipts and installs its own W&B sections without replacing
existing workspace content. Scheduler concurrency is selected from live
account GPU/CPU/memory limits, independently of the scientific grid.

Reporting defaults to `rwgao_b-brown-university/ambi-inner-bench`. The CPU
publisher creates a dedicated saved workspace filtered to the run's
`config.publication_id`, preserving existing personal and saved workspaces.
It accepts explicit `--entity` and `--project` overrides:

```bash
python slurm/ambi_transfer_discovery_publish.py \
  --root /path/to/existing/campaign \
  --publication-root /path/to/new/publication-inner-bench \
  --entity rwgao_b-brown-university --project ambi-inner-bench \
  --gpu-job-id 1234567
```

Publication state binds the W&B entity/project, run ID, campaign file hash,
evaluation source commit and campaign path. A restart resumes that same run
only when these bindings match. Legacy state without an explicit destination
binding is rejected; changing projects requires a new publication root and
creates a new reporting run. The reporting checkout may be updated separately
while the evaluator remains pinned to its original source. Existing campaign
manifests, GPU jobs, result bundles and scientific identities are preserved.
Local campaign/state reads, receipt snapshots and atomic publication writes
retry only `ESTALE` (stale file handle), for at most five attempts with delays
of 1, 2, 4 and 8 seconds. Persistent stale handles and all other errors remain
explicit failures; retrying reporting never resubmits evaluation work.
