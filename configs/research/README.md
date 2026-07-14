# AMBI inner-loop research presets

`ambi_inner_decoupling.json` is a compact matrix of one-axis-at-a-time
overrides based on `configs/algs/AntAMBITDMPC2.json`. It covers:

- none, SAC, TD3, and compute-matched MPPI inner operators;
- scalar twin-Q and five-head distributional-Q models;
- actor, critic, and temperature adaptation controls;
- temperature, imagined behavior, and returned-action exploration;
- independent model-step and optimizer-step budgets;
- action, episode, and run lifecycles;
- replay sampling, bootstrap source, rollout horizon, clone/LoRA, LoRA rank,
  and outer-policy anchoring controls.

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
When a comparison's reference preset is selected, it also reports seed-paired
return deltas for every selected variant.

Q representation is part of the checkpoint architecture. A scalar checkpoint
can compare checkpoint-compatible inner operators and controls, but it cannot
be evaluated as a distributional model (or vice versa). Train and supply a
matching checkpoint for each side of the Q-representation comparison, using
one preset per invocation:

```bash
python3 evaluate_ambi_checkpoint.py --checkpoint scalar.pt \
  --preset q_representation/scalar_twin --output scalar-eval.json
python3 evaluate_ambi_checkpoint.py --checkpoint distributional.pt \
  --preset q_representation/distributional_five --output distributional-eval.json
```

The evaluator rejects a mixed-architecture selection before running either
side. It also rejects `execution_noise`, because deterministic evaluation must
return the policy mean and would collapse those training-only variants.
