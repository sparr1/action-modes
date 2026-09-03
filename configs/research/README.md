# Frozen-checkpoint AMBI research

The active end-to-end configs live directly under `configs/ambi/`.
The frozen-checkpoint matrix is
`configs/research/ambi_inner_decoupling.json`, outside the active AMBI training
tree so it cannot be confused with the independently trained branch and horizon
comparisons.

## Humanoid Walk AMBI Search training matrix

`ambi_search_humanoid_walk_v2.json` is the compact training matrix for the
finite-horizon, outer-blueprint-tail AMBI Search variants. It resolves from
`configs/dmcontrol/algs/ambi_humanoid_walk_base_v2.json` and leaves the frozen
five-cell `configs/ambi/` suite unchanged. Its 35 recipes include:

- no-inner and bit-for-bit legacy-continuing controls;
- the full three critic layouts by five Q-return estimators;
- fresh-round, explicitly uncorrected retained replay, PDIS, model-resimulated,
  and Retrace recipes;
- online, frozen, optimizer-step EMA, round-end EMA, and hard depth-stage target
  strategies;
- frozen outer-target-Q and outer-online-Q leaves; and
- all three V-trace value layouts with round- and action-retained replay.

The matrix is a set of alternatives, not a Cartesian product across its named
comparisons. Every selector is a complete valid configuration. It has generated
W&B metadata enabled: materialization replaces the base's fixed run name with a
unique comparison/variant name and adds a unique selector tag. Expanded JSONs
are intentionally not checked in. To inspect or materialize it through the
generic preset API:

```bash
python3 - <<'PY'
from utils.ambi_research import list_preset_selectors, load_preset_matrix, materialize_presets

path = "configs/research/ambi_search_humanoid_walk_v2.json"
matrix = load_preset_matrix(path)
selectors = list_preset_selectors(matrix)
print(*selectors, sep="\n")
materialize_presets(path, "/tmp/ambi-search-humanoid-v2", selectors=selectors)
PY
```

Materialization writes ordinary algorithm JSONs and a matching experiment
manifest to the requested output directory. Review the selected recipes and
production budget before launching them; the checked-in matrix itself neither
submits jobs nor enables a special execution path.

## Literature-shaped twelve-job campaign

`ambi_search_humanoid_walk_literature12.json` narrows the larger implementation
matrix to twelve single-seed Humanoid Walk jobs. Every cell uses eight rounds,
512 length-three rollouts per round, a 512-sample minibatch, an outer-online
blueprint leaf, fresh round-local replay sampled without replacement, 2.5e-4
inner actor and critic learning rates, and fixed inherited outer entropy
temperature. Both the inner policy-to-prior KL and the persistent outer
behavior-policy KL are explicitly disabled. This is 12,288 imagined model
transitions per real action.

The cells compare complete suffix targets, finite lambda returns, V-trace, hard
depth propagation, actor-update dose, and one explicitly labeled inner-Polyak
ablation. They contain no no-inner, legacy, or KL-regularized control. To
materialize all twelve ordinary algorithm configs and their matching experiment
manifest:

```bash
python3 - <<'PY'
from utils.ambi_research import materialize_presets

materialize_presets(
    "configs/research/ambi_search_humanoid_walk_literature12.json",
    "/tmp/ambi-search-humanoid-lit12",
)
PY
```

Each materialized config has a unique generated W&B run name and tag. The
matrix itself does not submit jobs. Because this campaign was explicitly
selected for execution, its reviewed expansion is tracked under
`ambi_search_humanoid_walk_literature12/`: `algs/` contains twelve ordinary
algorithm configs and `experiments/` contains twelve one-config, one-trial
exact-resume manifests.

The Oscar launcher maps array tasks 0 through 11 to those manifests in the
matrix order. It requires both the exact pushed commit and a unique campaign
name, verifies that `HEAD` and `origin/ambisearch` match, and places durable
resume state in the approved scratch allocation:

```bash
sbatch \
  --export=ALL,AMBI_EXPECTED_COMMIT=<full-sha>,AMBI_SEARCH_CAMPAIGN=<unique-name> \
  slurm/run_ambi_search_humanoid_walk_literature12_oscar.sbatch
```

The array launcher requests twelve concurrent L40S slots. Submission is an
operator action and is never performed by loading or materializing the matrix.

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
- replay sampling, bootstrap source, rollout horizon, clone/LoRA, LoRA rank,
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
