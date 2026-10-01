# Staged AMBI-XQC critic BatchNorm study

This exploratory experiment separates three decisions: diagnose the local critic
BatchNorm mismatch, compare full-episode controllers, then vary actor learning
rate and the number of adaptation rounds. It uses the shared auxiliary
representation, XQC UTD2 Humanoid Walk checkpoint at **475,000 decisions**:
SHA256 `00d9347f00eb7719f55d4e3ef7abff8c24d7f2e214c923cdeba7b8e34afa1368`.
The world model remains UTD1. One training seed does not establish robustness
across training runs.

Every full episode uses seeds 101–105, controller seed 12345, and 500 real
control decisions. At every real decision the solver resets the inherited actor,
selected online critic, target critic, optimizers, temperature, and imagined
replay. Execution uses the adapted actor mean. The entire persistent model,
optimizers, BN state, reward normalizer and training RNG remain frozen.
Archived training replay is not added to the inner loss.

## Fixed adaptation settings

| Setting | Value |
|---|---|
| Imagination horizon / branches / sampled batch | H1 / N256 / B256 |
| Critic update slots per round / actor delay | G3 / delay 3 |
| Inner actor BN | Frozen inherited running statistics |
| Inner critic BN | Stage-specific; see below |
| Replay capacity / sampling | 1,024 / uniform with replacement |
| Critic learning rate | 5e-5 |
| Temperature learning rate | **5e-5, independently fixed across every cell** |
| Temperature initialization / objective | Inherit checkpoint alpha; automatic XQC objective |
| Reward scale | Frozen checkpoint real-data scale |
| Horizon bootstrap | Frozen persistent actor and selected online outer critic |
| World model / actor / target semantics | Inherit checkpoint; no architecture or target-support overrides |

`inner_critic_bn_mode` controls only the inner online critic's training forward:
`batch_update` uses the joined replay/policy batch and updates its running
statistics, `batch_no_update` uses the same batch statistics without changing
buffers, and `running` uses inherited running statistics. Target-critic BN,
actor queries of the critic and frozen outer-tail queries retain their XQC rules.
Actor weights and BN affine parameters learn with actor BN in running mode.

The two critic routes are pure **soft/soft** (XQC initialization and outer tail,
entropy-augmented critic targets) and **return/return** (auxiliary reward-return
initialization and outer tail, reward-only targets). Both retain actor entropy
regularization. This is an XQC protocol, not the native SAC
`closed-loop-refinement-v1` reference recipe.

## Stages and declared selection

1. **Same-root diagnostic.** `run_ambixqc_bn_probe.py` compares all three critic
   BN modes at 32 fixed real roots and three independently paired solver seeds.
   Its immutable `selection.json` selects `batch_no_update` or `running` using
   the predeclared held-out running-Q MSE criterion at three critic updates.
   `validate_selection` recomputes the decision and verifies complete coverage,
   input/artifact hashes, frozen outer state and RNG ownership. This model-only
   diagnosis is not a full-episode return result.
2. **Full-episode BN × critic-route comparison.** Four J1 controllers:
   historical `batch_update` and the selected alternative, each with soft/soft
   and return/return routes. Actor, critic and temperature LRs are all 5e-5.
   Twenty new full episodes. The winner is the largest mean raw episode return;
   exact ties use the recorded condition order: baseline soft, baseline return,
   alternative soft, alternative return. Paired gains and uncertainty are
   reported, but do not silently change this exploratory selection rule.
3. **Winning route/mode, rounds × actor LR.** A 2×2 grid of J in {1, 4} and actor
   LR in {1.25e-5, 5e-5}. Critic and temperature LRs remain fixed at 5e-5.
   The identical J1 / actor-LR5e-5 winner is reused from Stage 2; only the other
   three conditions run, adding fifteen episodes. J changes data collection
   and update counts; the factorial separates it from actor LR, not from all
   sources of extra compute.

The initial launch includes **Stages 1–2 only: 20 new full episodes**, plus the
same-root diagnostic and short GPU smokes. The coordinator stops after Stage 2
by default. Stage 3 is a deferred extension, requires an explicit
`--include-stage3`, and would raise the total to 35 new full episodes.
J1 uses 256 model transitions, three critic updates and one actor and
one temperature update per real decision; J4 uses 1,024 transitions, twelve
critic updates and four actor and four temperature updates. Optimizer delay
accepts slots 0, 3, 6, …; there is no initial three-slot wait.

Reuse the checkpoint-matched prior bundle through the pinned reference index.
Historical mean returns at this checkpoint are prior **551.9365**, soft-critic
MPPI **622.2512**, and return-only MPPI **621.1313**. These are existing reference
measurements, not additional study jobs. MPPI used H3 and a different compute
budget. Reuse requires checkpoint/sidecar, seed, action, protocol and immutable
artifact validation; matching a displayed mean alone is insufficient.

## CPU orchestration and immutable plans

`run_ambixqc_bn_study.py` never submits jobs. Every GPU task has an explicit
plan and index. Generated matrices are artifacts beside their plans, outside
the checkout; their `default_presets` is empty. No training preset is changed.
Commands such as `prepare`, `inspect`, `spec` and `collect` perform CPU
administration only. Use the existing locked DMControl interpreter.

```bash
python run_ambixqc_bn_study.py prepare --stage smoke \
  --source-sha "$SOURCE_SHA" --output-plan "$RESULTS/smoke-plan.json"
python run_ambixqc_bn_study.py prepare --stage stage2 \
  --source-sha "$SOURCE_SHA" --selection "$PROBE/selection.json" \
  --output-plan "$RESULTS/stage2-plan.json"
python run_ambixqc_bn_study.py inspect --plan "$RESULTS/stage2-plan.json"
python run_ambixqc_bn_study.py spec --plan "$RESULTS/stage2-plan.json" \
  --index 0 --manifest "$MANIFEST" --spec-dir "$RESULTS/stage2-specs/0"
python run_ambixqc_bn_study.py collect --plan "$RESULTS/stage2-plan.json" \
  --result-root "$RESULTS/stage2" --output "$RESULTS/stage2-results.json"
python run_ambixqc_bn_study.py prepare --stage stage3 \
  --source-sha "$SOURCE_SHA" --stage2-results "$RESULTS/stage2-results.json" \
  --output-plan "$RESULTS/stage3-plan.json"
```

A plan records the exact source SHA, checkpoint, generated matrix hash, condition
settings and pinned selection/result dependencies. An unchanged preparation may
be repeated; a conflicting artifact is never overwritten. `collect` rejects
partial or duplicate results and recomputes all episode/trace validations before
selecting a winner. Stage 3 verifies every Stage 2 result again and records the
hash-bound reused winning bundle instead of scheduling a duplicate evaluation.

The run map for each stage is explicit:

```json
{
  "schema": "ambixqc-bn-study-run-map-v1",
  "plan_sha256": "digest from that stage's plan",
  "runs": {
    "controller/condition_selector": "/absolute/registered/evaluation/run"
  }
}
```

It must assign every new condition exactly once to distinct registry directories.
GPU workers disable W&B, validate their completed bundles, then stage them for
the existing single-owner CPU publisher. Publisher receipt acknowledgement and
native panel verification remain part of campaign orchestration.

### Continue after a completed Stage 2 launch

`continue_ambixqc_bn_study.py` launches only the three new Stage 3 conditions.
Run it with `slurm/continue_ambixqc_bn_study_oscar.sbatch` on a CPU allocation,
using a separate extension result directory. It verifies the completed parent,
all four publication acknowledgements and the original six GPU smoke checks.
The existing J1 / actor-LR5e-5 winner is reused without another evaluation.

The continuation records two commits: its own tooling commit and the original
experiment commit. It imports the evaluator, controller, GPU launcher and
publication helpers from the original clean experiment checkout, preserving
the source identity of the paired study. The parent completion, plans, results
and publication records remain unchanged. The working named dashboard uses
the alphanumeric token `xqcbn290926`; the continuation verifies its exact
extension panel specification before submitting GPU jobs.

The required arguments are `--execution-root`, `--source-sha`, `--tooling-sha`,
`--parent-root`, `--result-root`, `--workspace-spec` and `--progress-run-id`.
The parent evidence supplies checkpoint, prior-reference and smoke paths.
`--gpu-type` defaults to `nvidia_rtx_a5000`. This is an explicitly authorized
extension, not an automatic consequence of completing Stage 2.

### Critic learning rate follow-up

`run_ambixqc_critic_lr_study.py` extends the completed Stage 3 study with two
critic-only learning-rate changes: `1e-4` and `2e-4`. It reuses the completed
J4 / actor-LR5e-5 / critic-LR5e-5 return-only, running-BN result as the baseline.
J4/H1/N256/B256/G3, delay 3, replay capacity 1024, actor and temperature rates
5e-5, frozen real reward scale, and action-local resets remain fixed. The same
five paired seeds 101–105 each run 500 decisions, adding ten full episodes.

The operator and GPU wrapper live in a separately pinned tooling checkout.
Evaluation, scientific identities, validation and publication use the original
clean experiment checkout. Both source identities and the completed baseline
artifacts are recorded. A short smoke for each new rate precedes production;
the baseline and prior episodes are reused. Each completed condition is
validated and published individually, with cumulative progress advancing from
seven conditions / 35 episodes to nine / 45. This is exploratory tuning;
fresh-seed confirmation is a separate experiment.

Use `slurm/orchestrate_ambixqc_critic_lr_study_oscar.sbatch` for the CPU
coordinator and `slurm/run_ambixqc_critic_lr_study_oscar.sbatch` for the two
GPU workers. The coordinator takes `--execution-root`, `--source-sha`,
`--tooling-sha`, `--parent-root` (the completed Stage 3 directory),
`--result-root`, `--workspace-spec`, `--progress-run-id`, and
`--worker-launcher`, with optional `--gpu-type`.

## Oscar launcher and production gate

`slurm/run_ambixqc_bn_study_oscar.sbatch` takes the existing exact-source,
locked-Python, scratch-results, checkpoint-manifest and prior-reference variables,
plus `AMBIXQC_BN_PLAN` and `AMBIXQC_STUDY_STAGE` (`smoke`, `stage2`, `stage3`).
Production also requires `EVAL_RUN_MAP` and `AMBIXQC_SMOKE_ROOT`.

Smoke array indices 0–5 cover **all three critic BN modes × both routes**, using
J4, actor LR1.25e-5, independently fixed temperature LR5e-5, two seeds and three
real decisions. The first task runs focused core and campaign tests. Each smoke
must provide exact-source, immutable matrix/inventory, trace/work/frozen-state,
allocated CUDA runtime and successful job evidence. Production revalidates all
six smokes; old four-condition smoke receipts do not satisfy this gate.

Stage 2 uses indices 0–3 and Stage 3 uses 0–2. Each task requests one GPU, six
CPUs and 32 GB; the operator chooses GPU type and concurrency from live cluster
capacity. Source travels through Git, and the clean committed source and locked
runtime are checked before and after each task. Outputs and private caches are
never reused. Compile settings remain checkpoint-derived; the driver neither
enables fallback nor overrides strict compilation.

## Rounds × updates per round extension

`run_ambixqc_update_sweep.py` evaluates J in {1, 2, 4, 6} crossed with
`inner_updates_per_round` G in {3, 6, 9}. Actor delay stays 3; all actor,
critic and temperature learning rates remain 5e-5. The checkpoint, return-only
initialization and outer tail, running actor/critic BN, frozen real reward scale,
H1/N256/B256 and full-episode protocol above are unchanged. This tests additional
optimization before the next collection round, separately from increasing J.
The XQC ordering accepts actor/temperature updates at slots 0, 3, 6, …; it does
not introduce a critic-only warmup block.

The J1/G3 and J4/G3 completed conditions are reused, with their episode,
checkpoint, source and publication evidence verified. The default grid therefore
adds ten conditions and fifty episodes. `--rounds` accepts a nonempty subset of
1, 2, 4, 6; reuse and submission counts follow that explicit selection. Each new
cell has its own immutable plan entry, registry, output directory and GPU task.
It must pass a short exact-configuration smoke before its five full episodes.
Each validated condition is published independently, without waiting for the
slowest cell. Metadata-only label updates preserve completed summaries.

Per real decision, model transitions equal 256J, critic updates JG, and actor and
temperature updates JG/3. Replay capacity is max(1024, 256J), retaining every
collected transition: original XQC outer-terminal provenance does not support
eviction within an inner solve. Only J6 needs capacity 1536; J1/J2/J4 retain 1024.
Thus each G comparison holds the collection budget and capacity fixed. Changing
G can still change the actual transitions collected by later adapted actors.

Execution stays pinned to the original clean `10852a8` scientific checkout;
new orchestration and publication tooling runs from a separate verified Git
checkout. The locked DMControl runtime and compiled loss policy remain intact.
Oscar workers use one GPU, six CPUs, 32GiB, and a six-hour limit. The launch
uses `--gpu-type prefer_l40s`: Slurm first tries L40S and falls back to A5000
when the preferred resources are unavailable. Each worker records its actual
GPU model; wall-clock comparisons must account for mixed hardware. The eight-hour
CPU coordinator uses an explicit `--max-concurrent` selected from live quota and
availability, scheduling larger JG cells first. The default ten-cell launch can
use ten concurrent GPUs. No backbone training or archived-replay mixing is
introduced.


## J8 and G12 extension

`run_ambixqc_update_extension.py` extends a completed rounds/update-dose sweep
to J in {1, 2, 4, 6, 8} and G in {3, 6, 9, 12}. It runs exactly eight new
conditions: G12 at J1/J2/J4/J6 and J8 at G3/G6/G9/G12. Each receives the same
five paired full episodes. The twelve existing grid results are reused only
after validating their completed parent plans, worker bundles and publication
receipts. No existing condition is allocated or evaluated again.

All scientific settings above remain fixed, including the original `10852a8`
evaluator and checkpoint. Replay follows max(1024, 256J), so J8 retains 2048
rows. J8/G12 performs 96 critic and 32 actor/temperature updates per real
decision, with 2048 imagined transitions. Stage 6 adds 40 episodes; the J/G
grid totals 20 conditions / 100 episodes and the whole BN/LR campaign totals
27 conditions / 135 episodes.

Submit the extension coordinator with `--completed-sweep-root` pointing to the
completed Stage 5 result directory and a separate `--result-root`. Use
`slurm/orchestrate_ambixqc_update_extension_oscar.sbatch` with
`slurm/run_ambixqc_update_extension_oscar.sbatch`. Workers retain the six-hour
limit and exact-configuration CUDA smoke gates. Choose `--max-concurrent` from
live combined quota; eight workers use 48 CPUs and eight GPUs.
`--gpu-type prefer_l40s` preserves the L40S preference with A5000 fallback.
The live J-axis comparison reads published summaries directly and includes
four curves, one for each G, at J1/J2/J4/J6/J8.

### Single actor learning-rate follow-up

The actor-rate comparison changes only `inner_actor_lr` from `5e-5` to
`1e-4` at J2/G6. Reuse the completed J2/G6 baseline from the Stage 5 sweep;
run exactly one new condition with five full paired episodes. Critic and
temperature learning rates remain `5e-5`, including the explicit temperature
rate override. The frozen 475k shared-representation UTD2 checkpoint,
original `10852a8` scientific execution, H1/N256/B256, replay1024, return-only
initialization and outer tail, running BN, frozen reward scale, seeds101–105,
controller seed12345, and 500-decision episodes remain unchanged.

`run_ambixqc_actor_lr_study.py` validates the completed parent and baseline
publication, runs an exact-condition CUDA smoke, then evaluates and publishes
only the new cell. Each decision performs512 model transitions,12 critic,
4 actor and4 temperature updates. Its `stage7` outputs are separate from prior
campaigns, and campaign publication advances from27conditions/135episodes to
28conditions/140episodes. Use the corresponding actor-LR worker and coordinator
Slurm wrappers, with90minutes for the GPU task and3hours for the CPU coordinator.

The new curve uses `actor_lr_study_id=ambixqc-actor-lr-20261001` and explicitly
labels the distinct learning rates. It must not receive `update_sweep_id`,
which would mix the changed actor rate into the existing fixed-rate J/G curves.
Evaluate paired environment return first; increased predicted Q or policy KL
alone is not evidence of improvement. This remains an exploratory comparison
on one backbone and reused tuning seeds.
