# Auxiliary reward critic with the original AMBI prior settings

`experiments/ambi_aux_return_original_prior_study.json` defines two fresh,
single-seed Humanoid Walk state runs based on
[`u13m14st`](https://wandb.ai/rwgao_b-brown-university/ambi/runs/u13m14st),
`AMBITDMPC2-humanoid-walk-outer-prior-no-inner-checkpoint-bank-1p5m-seed55`.
Its original source was `253e89b3896ca3891844d797573b00ade338165d`, with recipe
`configs/dmcontrol/algs/ambi_humanoid_walk_outer_prior_no_inner_checkpoint_bank_1p5m.json`.

The new runs match that prior's active training settings except for retaining
the current critic initialization and adding an independent reward-only
auxiliary critic. Both critics bootstrap from the stochastic SAC actor; there
is no return actor. The acting SAC policy still learns from the main soft
critic. No inner optimization or MPPI runs.

| Array index | Config | Auxiliary gradients into encoder/dynamics |
|---|---|---|
| 0 | `ambi_aux_return_original_prior_shared` | Enabled |
| 1 | `ambi_aux_return_original_prior_detached` | Detached |

Both use seed 55 and 1.5 million decisions, with 2,500 random warmup decisions,
2,500 pretraining updates, and then UTD 1. Checkpoints are retained every 25,000
decisions: 60 per run, 120 total. They are portable model checkpoints, not exact
interrupted-training resume snapshots. This is an exploratory single-seed pair.

## Deliberate settings

- Actor Q and target Q both use the minimum of a sampled pair from five heads.
- SAC uses squashed entropy, automatic temperature initialized to 1, and target
  entropy -21. These explicitly resolve the original run's `auto` settings.
- Policy log standard deviations use direct clamping to [-20, 2].
- Actor, primary critic, and auxiliary critic learning rates are 3e-4. Both
  critic coefficients are 0.1. Q normalization and behavior regularization are
  disabled.
- Model size 5, distributional Q with 101 bins over [-10, 10], H3, rho 0.5,
  batch 256, replay capacity 1M, dropout 0.01, and target tau 0.01 every update
  match the original active recipe. At H3, the current temporal weighting
  matches the original `reference_weighted_mean` with reference horizon 3.
- The current critic initialization remains untouched: Linear constructor
  samples and biases are retained, then output weights are zeroed. The original
  run instead reinitialized critic weights with a small truncated normal and
  zeroed all biases. This pair does **not** reproduce that initialization or
  its subsequent random-number stream.

Relative to the September 15 auxiliary SAC study's -21 pair, only actor
`mean_pair` becomes `min_pair`, the policy lower bound changes from -10 to -20,
and the budget changes from 2M to 1.5M, together with descriptive run metadata.
The inactive inner actor reduction/bounds and absent auxiliary actor's resolved
reduction are aligned with those choices. Existing recipes remain unchanged.

In the detached arm, only the auxiliary loss is blocked from the representation;
the primary critic still updates the encoder and dynamics. Auxiliary critic
parameters still participate in joint gradient clipping. This is not a
representation freeze or a guarantee of bitwise equivalence to a run without
the auxiliary ensemble.

Current strict compilation, early and periodic outer-policy diagnostics,
event-indexed W&B logging, and checkpoint sidecars remain enabled. These differ
from the historical run's execution and observability settings. W&B project
is `ambi`, group is `ambi-aux-return-original-prior-20260916`, and each run name
contains its config and `seed55`.

## Launch and validation

Use `slurm/run_ambi_aux_return_sac_oscar.sbatch`, setting
`AMBI_AUX_RECIPE=original_prior` for both the smoke and training jobs, with an
independently verified GPU smoke receipt for these exact recipes and source commit. The
campaign retains the strict compilation, gradient-routing, checkpoint and
real-environment checks described in
[the auxiliary SAC study](AUX_RETURN_SAC_STUDY.md#oscar-launcher-and-validation).
Its two-cell array is `--array=0-1`; choose resources and concurrency from live
Oscar capacity. The original campaign's four-cell receipt is not valid here.

`tests/test_ambi_aux_return_original_prior_configs.py` checks the complete
recipe differences, resolved actor/critic semantics, pair isolation, budget,
checkpoint count, and unchanged current network initialization.
