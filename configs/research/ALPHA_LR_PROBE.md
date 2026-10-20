# Inner temperature learning-rate diagnosis

This follow-up changes only `inner_temperature_lr` in the original
entropy-regularized H3/J8 inner SAC objective. It does not combine the prior-KL
intervention with temperature tuning. Actor and critic learning rates remain
`0.0003`; each decision still starts from the checkpoint's inherited outer
temperature, with a fresh log-alpha parameter and optimizer. The target entropy,
critic routing, actor/critic update counts, outer state and all other scientific
settings remain unchanged.

The paired root diagnostic compares learning rates `0.0003`, `0.003`, `0.03`,
and `0.3`. The large final value is an aggressive diagnostic arm, not a proposed
training default. With a roughly constant positive entropy residual, 32 Adam
updates of log-alpha suggest alpha multipliers near `exp(-32 * learning_rate)`:
0.990, 0.908, 0.383 and 0.0000677. These are illustrations rather than bounds:
changing residuals, Adam momentum, epsilon and gradient clipping affect the
actual updates. Low initial entropy can first increase alpha; momentum can
delay its subsequent decrease. Both phases must be measured.

`ambi_alpha_lr_probe.json` retains the original entropy-probe root bank IDs,
checkpoint/metadata hashes and source provenance. There are two checkpoints
(SAC-aux100k and MPPI-aux400k), five environment seeds (101–105), and roots at
decisions 0/10/50/100/250. No source trajectory is recollected. Four full solves
share initialization, first replay, first critic phase and fixed entropy-anchor
noise. Subsequent replay may diverge as the policies adapt. Record temperature
before/after each update, entropy residual, Adam moments, actual actor gradient
balance, and fixed real/imagined-anchor entropy. Probe timings include observer
overhead and are not controller-speed measurements.

`ambi_alpha_lr_episodes.json` and `slurm/ambi_alpha_lr_episodes.py` define six
full-episode cells: the two checkpoints crossed with the three increased rates.
Each cell runs five paired sampled-action episodes of 500 decisions, matching
training execution. Compatible prior and `0.0003` baseline episodes are reused.
This is distinct from the named mean-action `closed-loop-refinement-v1` recipe.
All 128 critic, 32 actor and 32 temperature updates per decision must execute,
with strict compilation, finite metrics, complete traces and unchanged outer
state. A worker owns all five seeds and its result directory. No outer training
is launched.

The native W&B comparison shows returns, sample standard deviation, paired
improvement over prior, controller time, entropy, and temperature change.
Full traces stay on Oscar; only bounded summary artifacts are uploaded.
Use the final publisher's `--once --require-complete` after all workers terminate
so failed or missing results cannot remain silently pending.
