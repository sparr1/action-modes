# Auxiliary-return SAC backbone training horizons

`experiments/ambi_aux_return_sac_horizons.json` launches three fresh Humanoid
Walk state backbones with training unroll horizons 1, 5, and 7. The existing
H=3 reference is W&B run `rwgao_b-brown-university/ambi/aux6428346x0`, trained
for two million environment decisions at source commit
`6f0caf4acb417cc514f27d5d6e8cad17a205031f`. The source branch for this
campaign starts from that historical revision. Its 575K snapshot was an
intermediate checkpoint, not the training budget.

| Array index | Config suffix | Training unroll horizon |
|---|---|---:|
| 0 | `ambi_aux_return_sac_clip_target10p5_shared_h1` | 1 |
| 1 | `ambi_aux_return_sac_clip_target10p5_shared_h5` | 5 |
| 2 | `ambi_aux_return_sac_clip_target10p5_shared_h7` | 7 |

Each complete recipe is identical to
`algs/ambi_aux_return_sac_clip_target10p5_shared.json` except
`train_unroll_horizon` and descriptive W&B group/tags. Seed 55, two million
decisions, 2,500 warmup decisions, 2,500 pretraining updates, UTD 1, batch size
256, five distributional critic heads, automatic alpha initialized at 1,
squashed target entropy -10.5, directly clamped actor log standard deviation
[-10, 2], actor learning rate 3e-4, and no actor Q normalization are preserved.
One SAC actor serves both the soft critic and the auxiliary reward-only critic.
Auxiliary gradients remain attached to the shared representation. Behavior uses
the stochastic SAC prior with no inner adaptation or MPPI.

The inactive inner rollout and outer planning horizons remain 3. Existing
configuration warnings about inner H=3 exceeding training H=1 are therefore
expected, although the inner controller is disabled. Original temporal loss
weighting is unchanged: rho=0.5, model/critic weighted sums divided by H, and
actor loss over H+1 latent states. Changing H consequently also changes these
loss scales; this campaign does not normalize them away. At H=1 the auxiliary
critic's current-state loss reaches the encoder but has no dynamics gradient
path. For H>1 both encoder and dynamics receive its gradients.

All 80 scheduled checkpoints are retained every 25K decisions for each new
backbone (240 total), including 575K. Checkpoint sidecars preserve the recipe.
These are portable model snapshots, not restartable whole-training snapshots.
The study is exploratory and uses one training seed per H. No inner-SAC
evaluations or warm-start transfer experiments are launched by this manifest.

## Oscar launch and validation

`slurm/run_ambi_aux_return_horizons_oscar.sbatch` uses the existing locked
DMControl interpreter and installs no dependencies. Before submission, verify
live account/QOS limits, GPU availability, output quota, and a clean checkout
at the tested source SHA. L40S placement is supplied via the Oscar `sbatch`
resource override. Defaults are one GPU, eight CPUs, 32 GiB and 48 hours;
choose the wall time from measured H-specific throughput with ample margin.
There is no arbitrary concurrency cap; all three useful workers may run
concurrently within the allocation.

Required environment variables are:

- `AMBI_PROJECT_DIR`: absolute clean Git root at the tested source commit.
- `AMBI_PYTHON`: existing absolute locked DMControl interpreter.
- `EXPECTED_ACTION_MODES_SHA`: full tested commit SHA.
- `AMBI_AUX_OUTPUT_ROOT`: absolute scratch/project output storage outside source.
- `AMBI_AUX_CAMPAIGN`: unique campaign directory basename.
- `AMBI_AUX_MODE`: `smoke` or `train`.

Create scheduler log directories and override `--output` and `--error` as needed.
Submit smoke with `--array=0-2` and an appropriate short allocation (two hours
is a conservative allowance). Submit production with `--array=0-2` and
`--dependency=aftercorr:<smoke-array-id>`. Each production task starts only
when the corresponding smoke task succeeds, then independently validates that
H's receipt. An `afterok` dependency on the complete smoke array is also valid.
If any gate fails, diagnose it before resubmission with a fresh campaign.
No production output directory or receipt can be silently overwritten.

Every horizon's GPU gate performs cold and warm full-size strict compile
updates, checks all six fallback flags and finite metrics, verifies the exact
portable checkpoint roundtrip, restores exact learner/RNG state and reproduces
the next update, checks expected representation gradients and stochastic
prior-only actions, and executes a fresh 512-decision training smoke including
real replay and scheduled checkpoint sidecars. Smoke-only overrides are W&B
off, 500 warmup decisions, two pretraining updates, replay capacity 4096 and
256-decision checkpoint intervals. The full production warmup is not executed
by the short gate. All overrides are recorded in the receipt.

Receipts are created exclusively at
`<output-root>/<campaign>/gate/<config>/receipt.json`. They bind the tested
source commit, complete manifest, all three recipe hashes, environment lock,
per-H cold/warm and fresh-training evidence, and Python/PyTorch/CUDA/GPU runtime.
Training requires the same interpreter, versions, GPU model and capability.
Receipt creation fails if tests are missing, skipped, incomplete, or report a
compile fallback. The checkpoint size estimate in each receipt covers that
horizon's 80-checkpoint bank.

Each training cell writes immutable `launch.json` at
`<output-root>/<campaign>/<config>-seed55/` with the resolved settings, full
source/recipe binding, receipt hash, Slurm job information, exact command and
W&B identity. W&B uses group `ambi-aux-return-sac-horizons-20261002` and unique
IDs `auxh<array-job-id>x<task-index>`, with resume disabled. Checkpoints and
metadata live in the normal timestamped directory beneath each cell. Temporary
files, CUDA compilation caches and W&B caches are directed to campaign scratch
storage; the source tree and home directory are not used for these outputs.

The launcher rejects scheduler restarts/requeues because exact training resumes
are unsupported for these portable snapshot banks. A requested restart requires
a deliberate recovery decision rather than silently starting the same cell over.
