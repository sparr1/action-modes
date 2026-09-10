# Frozen-checkpoint AMBI research

The four TD-AMBI prior banks have dedicated [MPPI and J5 C1/A1 evaluation
matrices](TD-AMBI-PRIOR-EVAL.md), preserving each saved critic objective and
checkpoint alpha/Q scale. They provide fixed-scalar and adaptive-scalar SAC
comparisons, with paired prior references and explicit evaluation curves.

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
fresh model for every preset, uses paired environment seeds, always returns the
policy mean to the real environment, never calls the outer update, and hashes
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
