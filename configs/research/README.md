# Frozen-checkpoint AMBI research

`ambi_checkpoint_graph_eval_consecutive.json` selects ten consecutive full
episodes per checkpoint, matching TD-MPC2's episode sampling and return
averaging convention. The environment starts with `reset(seed=55)` exactly
once, followed by nine ordinary `reset()` calls. The controller stream is also
seeded once with 55 and advances across episode boundaries, including the
private CUDA graph stream. Episode-local controller state still resets, and
each real decision uses fresh prior-initialized inner adaptation. Compilation
and warmup use the initial observation without stepping the environment; their
controller draws are discarded before scoring.

The prior and inner SAC each receive their own environment initialized with
the same seed. Episode indices (0–9 in artifacts, 1–10 in plots) identify the
paired observations; hashes of the initial observations must agree before
paired gains can be published. The report averages all ten returns, records
sample SD and retains every episode point. Each Humanoid Walk episode has 500
decisions, preserving the action repeat and bounded 0–1000 return convention.

Each checkpoint restarts these streams, so independent checkpoint tasks remain
parallel and repeatable. This matches the standalone evaluation convention;
it does not reconstruct the historical RNG position of the official online
trainer, which shares its environment with training. Our prior and H3/J8 SAC
continue to execute actor means; this change does not replace them with the
TD-MPC2 planner.

The frozen graph accepts checkpoints trained with either sampled or mean final
action execution. It retains that saved configuration and explicitly executes
means during evaluation; the training graph's execution restrictions are unchanged.

Select the new matrix for direct evaluation:

```bash
python evaluate_ambi_checkpoint.py \
  --matrix configs/research/ambi_checkpoint_graph_eval_consecutive.json \
  --checkpoint /path/to/model --device cuda \
  --bundle-dir /path/to/new-evaluation --output /path/to/new-evaluation.json
```

For `slurm/graph_checkpoint_campaign.py plan` and `prepare`, add
`--episode-sampling consecutive` to the usual campaign arguments. The chosen
matrix and its hash are pinned in the campaign; workers and publishers recover
that choice automatically. Smoke checks shorten it to two consecutive episodes
without W&B. Direct `--num-episodes` and `--environment-seed` overrides apply
only to a consecutive matrix and change the scientific protocol identity.
`--seeds` is rejected for this mode. It supports return-only prior/SAC mean
execution, including the graph backend; banks and temperature chains are
separate protocols. An external prior reference must match the entire stream
protocol and all starting-observation hashes.

The ten-episode protocol requires new evaluation series. It cannot append to
or reuse the old five-seed results. The legacy matrix and existing campaigns
retain their settings:

`ambi_checkpoint_graph_eval.json` compares the saved controller with its frozen
mean-action prior using five paired full episodes (seeds 101–105, controller
seed 55). It inherits the checkpoint's inner dose, entropy and normalization.
The SAC arm explicitly selects `frozen_mean_cuda_graph_v1`: one complete captured
fresh solve per decision, strict compilation, vectorized critics and mean-action
execution. Return-only bundles retain per-episode returns, scalar controller
metrics, pairing and frozen-state checks, without inner traces or model probes.
Its isolated graph RNG is a separate evaluation identity from native execution.
Warmup/capture is reported separately and discarded before the seeded episodes.
Unsupported lifetimes, sampled execution, traces and probes fail explicitly.
`slurm/graph_checkpoint_campaign.py` prepares a pinned checkpoint inventory and
two new evaluation series, runs checkpoint tasks, and publishes through the
single-owner registry. Its CPU publisher installs and verifies a dedicated W&B
workspace before uploading results, and reports missing/failed cells.
The return chart shows all measured episode returns at each checkpoint
with a mean line for each controller. Dots have small horizontal offsets so
overlapping episodes remain visible; their returns are unchanged. The private
custom-chart preset queries an episode table maintained in each run summary.
Updating that table adds no scalar history rows and preserves result identity.

The [joint entropy and prior-KL evaluation](ENTROPY_KL_EVAL.md) keeps automatic
temperature learning enabled while testing two fixed prior-KL penalties on
the same saved roots and sampled full episodes.

The [inner temperature learning-rate diagnostic](ALPHA_LR_PROBE.md) tests faster
log-alpha adaptation on identical saved roots and paired sampled episodes while
keeping actor/critic learning rates, initialization and update counts fixed.

The [prior-KL entropy mitigation diagnostic](PRIOR_KL_PROBE.md) compares
entropy-regularized inner SAC with KL-to-base refinement on identical saved
roots, followed by a separately identified sampled-action episode evaluation.

The [inherited-critic gradient diagnostic](BOOTSTRAP_DIAGNOSTICS.md) compares
reward-Q gradients, differentiable model gradients, and paired real-simulator
action interventions at common states across backbones. It supports the
auxiliary SAC-return critic and complements full-episode refinement evaluation.

The frozen evaluator supports finite-horizon inner SAC and transition-based
update counts. It rejects `inner_outer_replay_fraction > 0` because these model
snapshots do not include real replay. Test replay mixing through a populated
training agent. See the [inner SAC options](../../RL/tdmpc2_core/README.md#finite-horizon-inner-sac-and-real-replay-mixing)
for the target convention and schedule restrictions.

The active end-to-end configs live directly under `configs/ambi/`.
The frozen-checkpoint matrix is
`configs/research/ambi_inner_decoupling.json`, outside the active AMBI training
tree so it cannot be confused with the independently trained branch and horizon
comparisons.

### Critic-only LoRA-RL comparisons

The [LoRA-RL implementation contract](../../RL/tdmpc2_core/README.md#critic-only-lora-rl)
keeps the actor dense and applies low-rank updates only to selected critic
matrices. The initial reference is rank 96, direct scale one, and AdamW adapter
weight decay `2e-4`. Output heads, biases, and normalization parameters remain
trainable. All scientific inner state resets at each real decision.

The frozen-checkpoint matrix `ambi_inner_decoupling.json` provides these
explicit selectors without changing its default operator comparison:

- `critic_lora_placement/clone`: full actor and critic adaptation.
- `critic_lora_placement/input_hidden_r96`: LoRA on the critic input and hidden
  matrices; the default placement for AMBI's learned-latent input.
- `critic_lora_placement/hidden_r96`: LoRA on hidden matrices only, closer to
  the paper's placement, with a normally trainable input projection.
- `critic_lora_rank/rank_64`, `rank_96`, and `rank_128`: input-and-hidden
  placement with identical scale, decay, actor adaptation, and SAC budgets.

These replace the old `adaptation_mechanism` and `lora_capacity` axes. Old
selectors and active `"lora"` configurations do not silently select the new
method. Reproducing a historical run requires its historical source; start a
new evaluation attempt when adopting `"lora_rl"`.

Four matching algorithm/experiment pairs under `configs/dmcontrol/` provide
single-seed Humanoid Walk training screens:

- `ambi_humanoid_walk_base_v2_lora_rl_input_hidden_r64`
- `ambi_humanoid_walk_base_v2_lora_rl_input_hidden_r96`
- `ambi_humanoid_walk_base_v2_lora_rl_input_hidden_r128`
- `ambi_humanoid_walk_base_v2_lora_rl_hidden_r96`

Each retains the base-v2 recipe: seed 55, 14 million real decisions, J=8, N=32,
H=3, G=1, replay capacity 768, and the original evaluation/checkpoint settings.
Only critic adaptation and descriptive run identity change. These are full
training configurations, not smoke tests; no sweep runs by adding them. Rank 96
and decay `2e-4` borrow the paper's DMC SimbaV2 settings, while the rank and
placement alternatives test the transfer to AMBI. The trained prior and zero-B
initialization deliberately differ from the paper's main setup. Neither a
return improvement nor a speedup has been established for these configurations.

# Frozen-checkpoint preset workflow

`ambi_inner_decoupling.json` is a compact matrix of one-axis-at-a-time overrides
based on `configs/algs/AntAMBITDMPC2.json`. The canonical AMBI reference is
fresh, action-local, fully cloned inner SAC.
The remaining operators, LoRA and persistence settings below deliberately test
auxiliary ablations or comparators; they are not alternative definitions of
AMBI. The matrix covers:

- none, SAC, TD3, and compute-matched MPPI inner operators;
- the reference five-head distributional-Q model and a scalar twin-Q ablation;
- actor, critic, and temperature adaptation controls;
- train-only actor, online-critic, and joint prior writeback at beta 0.01, 0.1,
  and 1.0, plus the no-writeback reference;
- temperature, imagined behavior, and returned-action exploration;
- explicit J/N/H/G collection and joint-update schedules;
- action, episode, and run lifecycles;
- replay sampling, bootstrap source, rollout horizon, critic-only LoRA-RL placement and rank,
  and outer-policy anchoring controls.

Within a variant's `alg_params`, `null` removes an inherited base key. The
operator comparison uses this to keep SAC's J/N/G controls out of MPPI and
no-improvement configurations.

List the matrix without importing the training stack:

```bash
python3 evaluate_ambi_checkpoint.py --list-presets
```

Materialize ordinary algorithm configs that can be referenced by the existing
`main.py --run ... --alg-dir ...` workflow:

```bash
python3 evaluate_ambi_checkpoint.py \
  --comparison inner_operator \
  --materialize-dir configs/algs/generated
```

Materialization also writes `AMBIResearchExperiment.json`, preserving the
matrix's environment parameters. Run it with both paths pointed at that output:

```bash
python3 main.py \
  --run configs/algs/generated/AMBIResearchExperiment.json \
  --alg-dir configs/algs/generated
```

Evaluate the default operator comparison from one frozen checkpoint:

```bash
python3 evaluate_ambi_checkpoint.py \
  --checkpoint /path/to/ambi-checkpoint.pt \
  --device cuda \
  --seeds 101 102 103 104 105 \
  --output logs/ambi_inner_operator_eval.json
```

Select individual presets with repeatable `--preset comparison/variant`, or an
entire axis with repeatable `--comparison comparison`. The evaluator creates a
fresh model for every preset, uses paired environment seeds, defaults to the
policy mean in the real environment, never calls the outer update, and hashes
outer model/optimizer/temperature state before and after each run. Its output
contains per-episode real returns and all finite model-predicted inner metrics.
Output is written atomically, and an existing `--output` path is preserved
unless `--overwrite` is supplied explicitly.
When a comparison's reference preset is selected, it also reports seed-paired
return deltas for every selected variant.

Q representation is part of the checkpoint architecture. The reference
checkpoint uses five distributional heads. It can compare checkpoint-compatible
inner operators and controls, but it cannot be evaluated as a scalar twin model
(or vice versa). Train and supply a matching checkpoint for each side of the
Q-representation comparison, using one preset per invocation:

```bash
python3 evaluate_ambi_checkpoint.py --checkpoint distributional.pt \
  --preset q_representation/distributional_five --output distributional-eval.json
python3 evaluate_ambi_checkpoint.py --checkpoint scalar.pt \
  --preset q_representation/scalar_twin --output scalar-eval.json
```

The evaluator rejects a mixed-architecture selection before running either
side. It also rejects the train-only `execution_noise` and `prior_writeback`
axes. Deterministic evaluation returns the policy mean, which collapses the
execution-noise variants. Prior writeback is deliberately disabled outside
training, so its variants would likewise collapse under a frozen outer
checkpoint. Materialize and train those axes instead.

## Early H3/J8 checkpoint diagnosis

`ambi_early_h3_j8.json` applies the production fresh H3/J8 inner SAC recipe
(C16/A4/N128/B256, auxiliary return critic initialization and horizon tail,
reward-only targets, and automatic temperature initialized from the saved
outer temperature) to checkpoint-derived configurations. The campaign pins
the MPPI-aux and SAC-aux source runs and checkpoint hashes externally; the
matrix preserves each checkpoint's outer networks, optimizer state, learned
temperature and environment. MPPI schedule controls are explicitly removed
before selecting inner SAC. This is frozen evaluation, with no outer learning
or weight writeback.

The `early_h3_j8/mean` and `early_h3_j8/sample` selectors differ only in real
action execution. The latter explicitly sets variant
`evaluation.execution_action="sample"`, which invokes the public prediction
API with `deterministic=False`; imagined learning still uses the same recipe.
Sampled execution uses the actor's squashed Gaussian, as in training. Protocol
identity records `tanh_sample`, keeping these results distinct from
`tanh_mean`. Evaluate the selectors in separate invocations and bundle
directories because one benchmark bundle has one action protocol. This
sampled variant is an explicit extension to the mean-action closed-loop
refinement procedure.

Five paired environment seeds (101–105), controller seed 55, and complete
500-decision episodes are the defaults. Bundles retain optimizer and decision
metrics throughout each episode. At decisions 0, 100, 200, 300 and 400, 32
paired model rollouts measure reward prefix, auxiliary online-Q horizon tail,
total predicted return and change from initialization, initially and after
each of the eight rounds. The probe uses independent fixed noise and critic
pairs; its time and model steps are reported separately. The generic
`evaluation.togo_return_every` option defaults to 1, preserving older matrix
behavior. Model predictions are diagnostics, not measured real policy gains.

`ambi_entropy_probe.json` pins a smaller follow-up diagnosis on SAC-aux 100k
and MPPI-aux 400k. It first saves fresh mean-action H3/J8 trajectory prefixes,
then measures exact inner-solver entropy dynamics and paired fixed-replay actor
updates at roots 0, 10, 50, 100 and 250. See
[ENTROPY_PROBE.md](ENTROPY_PROBE.md) for checkpoint hashes, source/probe RNG
separation, frozen-state checks, output ownership and the per-seed worker CLI.

## Calibrated BOOM likelihood checkpoint campaign

`ambi_boom_likelihood_calibrated.json` trains the same 400k-decision, seed55
outer-adapt H3/J8/K1 recipe as the likelihood pilot, changing only the fixed
BOOM coefficient and publication labels. The gate remains S>2, the detached
weights use Q/S, and alignment is divided by S. Choose the coefficient from
valid diagnostic batches and retain the supporting history and selection
receipt beside the launch manifest; gradient balance is not a return optimum.

The November 7 pilot (`ambi/8b8130452c1741e580fd69487eb7f7c2`, lambda0.0042)
supplied 71 valid probes from decisions 3,100–10,100. Thirteen had S>=1.5,
including three with S>2. The selected fixed lambda0.000042 is the largest
logged candidate that keeps every observed near-threshold prospective
alignment/SAC gradient ratio below one: median0.240, maximum0.735. The three
active probes alone project to ratios0.0097–0.0122. Including batches within
25% below the gate protects against its sharp activation transients; this
analysis window does not change the S>2 training gate. These are fixed-state
counterfactual gradient ratios, not predictions of Adam steps or later returns.

`slurm/boom_likelihood_calibrated_oscar.py gate` runs fresh CUDA correctness
checks and pairs a prior with the graph controller after checkpoint reload.
The evaluator sizing benchmark uses 1 and 2 physical CPUs with W&B disabled;
its untrained snapshot never enters scientific result series. The prior
selector removes inactive SAC compute controls while retaining BOOM's outer
checkpoint identity. Training retains all acceleration and critic geometry.

For every 25k checkpoint in reverse order, invoke
`slurm/graph_checkpoint_campaign.py plan` with `--max-step 400000
--checkpoint-every 25000 --descending`. This creates a pending workload and
workspace without fabricating weights, metrics, or W&B result runs. Run
`prepare --available-only` with the same arguments once at least one checkpoint
exists; it verifies available checkpoint/metadata pairs and allocates two
evaluation series for the full grid. Array index 0 remains 400k and index 15
remains 25k, including while later checkpoints are missing. Without
`--available-only`, preparation still requires the complete grid.

Submit the GPU array with `--hold`, then run the finite CPU `release` command
with `--compute-job ARRAY_ID --training-job TRAINING_ID --timeout-seconds N`.
It inventories new checkpoints and releases only available tasks, newest first,
once a minute. Set its deadline and CPU walltime from remaining training time
plus a measured margin. Missing checkpoints hold no GPU resources. On training
failure or its deadline it cancels only unreleased held tasks, leaving completed
and running evaluations intact. Independent checkpoints can run concurrently;
choose the array cap from live QOS headroom.

Each evaluates five paired full episodes (101–105; controller seed55), prior
versus saved H3/J8 inner SAC, with mean real actions and a frozen outer model.
GPU workers pin a separate immutable checkpoint inventory per result, so later
checkpoint discovery cannot change publication fingerprints. Start one finite
CPU `publish` process per selector alongside the release controller, passing the
same array ID. Results and progress panels update after each checkpoint rather
than waiting for training or the whole array to finish. Publishers verify all
16 points before exiting. No recurring agent is required.
