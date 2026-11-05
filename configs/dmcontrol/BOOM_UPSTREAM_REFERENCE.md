# Released BOOM threshold reference

This diagnostic runs the official BOOM implementation, rather than the AMBI
BOOM adaptation, on DMC Humanoid Walk. Its purpose is to observe what policy
performance and update magnitudes accompany the first `S > 2` crossing.

The upstream source is a separate clean checkout of
`https://github.com/molumitu/BOOM_MBRL` at
`b3b2f09c03533c15b804ced9abc6767e73ff8e1e`. No upstream source is modified.
`slurm/boom_upstream_reference.py` invokes its environment, agent, replay, and
online trainer. Runtime wrappers only observe policy outputs and the existing
gradient-clipping result. The original Gaussian/entropy arithmetic, seed-data
pretraining and replay prefetch behavior are preserved, including their quirks.

Use the upstream default config, changing task to `humanoid-walk` and seed to
55. Keep the released 2,001,000-decision budget, action repeat two, summed
rewards, ten evaluation episodes for both planner and prior every 50k decisions,
and original S > 2 gate. The released action-dimension heuristic gives lambda
0.42 for the 21-dimensional Humanoid. This differs from the BOOM paper's written
DMC coefficient d/1000. This reference follows the released code.

The entrypoint uses EGL and scratch output paths and replaces offline-only
upstream publication with a dedicated online scientific run. It reproduces
upstream's one Torch thread, TF32 and cuDNN benchmark settings. Upstream itself
compiles the neural modules and vectorizes the critic ensemble. It does not
use AMBI's whole-inner CUDA graph because there is no AMBI inner learner.

Use an isolated Python 3.10 environment with the DMC dependencies listed in
`slurm/boom_reference_requirements.txt`. The Torch CUDA 12.1 build is supported.
The MuJoCo 3.1.6 / dm-control 1.0.20 versions follow upstream; AMBI's runtime
uses older versions, which is a comparison limitation. JAX/HumanoidBench
dependencies are unused and omitted from this DMC-only runtime.

All update metrics are saved in `updates.jsonl`. Visible panels aggregate 100
updates, with min/max and gate-active fraction so a mean below two cannot hide
activation. Entropy diagnostics remove the upstream extra Gaussian multiplier
while retaining its floored tanh Jacobian; they do not change the objective.
The observer saves model/scale snapshots before and after the first active
actor update, every 25k decisions and at completion. These are diagnostic
checkpoints, not exact-resume snapshots. The stock training and evaluation
schedule is unchanged; crossing snapshots can be evaluated separately.

Before production, run the `validate` mode on a scheduled GPU. It verifies
exact equality of model parameters, optimizer states, losses and all RNG states
with observation disabled/enabled, both below and above the gate. The
`benchmark` mode disables W&B in both config and environment and measures
one/two-CPU affinity windows of complete online training on the same GPU.
Its shortened budget/evaluation only apply to the timing run. Choose production
CPU and time reservations from this measurement and current Slurm limits.

The Slurm launcher requires pinned clean observer and upstream source, unique
output paths, and a scientific W&B run ID. `boom_reference_workspace.py`
idempotently installs and reads back a separate workspace with progress,
returns, threshold, objective, gradient and entropy panels.
