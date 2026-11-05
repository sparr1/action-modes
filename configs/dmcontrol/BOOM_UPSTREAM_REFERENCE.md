# Released BOOM threshold reference

This diagnostic runs the official BOOM implementation, rather than the AMBI
BOOM adaptation, on DMC Humanoid Walk. Its purpose is to observe what policy
performance and update magnitudes accompany the first `S > 2` crossing.

The upstream source is a separate clean checkout of
`https://github.com/molumitu/BOOM_MBRL` at
`b3b2f09c03533c15b804ced9abc6767e73ff8e1e`. No upstream source is modified.
`slurm/boom_upstream_reference.py` invokes its environment, agent, replay, and
online trainer. Runtime wrappers observe policy outputs and the existing
gradient-clipping result. Optional CUDA graph execution captures the planner
iterations while retaining its original sampling and final action selection. The original Gaussian/entropy arithmetic, seed-data
pretraining and replay prefetch behavior are preserved, including their quirks.

Use the upstream default config, changing task to `humanoid-walk` and seed to
55. Use the requested 200,000-decision initial diagnostic budget, action repeat two, summed
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
checkpoints, not exact-resume snapshots. The stock update and evaluation cadence is retained; crossing snapshots can be evaluated separately.

Before production, run the `validate` mode on a scheduled GPU. It verifies
numerical agreement of model parameters, optimizer states and losses, and exact equality of all RNG states
with observation disabled/enabled, both below and above the gate. The
`benchmark` mode disables W&B in both config and environment and measures
one/two-CPU affinity windows of complete online training on the same GPU.
Its shortened budget/evaluation only apply to the timing run. Choose production
CPU and time reservations from this measurement and current Slurm limits.

The Slurm launcher requires pinned clean observer and upstream source, unique
output paths, and a scientific W&B run ID. `boom_reference_workspace.py`
idempotently installs and reads back a separate workspace with progress,
returns, threshold, objective, gradient and entropy panels.

## Paired TD-M(PC)² reference

`slurm/tdmpc_square_upstream_reference.py` also supports the official
`DarthUtopian/tdmpc_square_public` commit
`d1c2632c36effd2f7b661bfe5f822a3db8054d40`. Use the same DMC dependencies and
seed 55, with the native `residual` actor objective, beta=1 and scale threshold
2. Use the requested 200,000-decision initial diagnostic budget, vectorized
critic heads, native TF32 defaults and complete replay/trainer behavior. The
launcher restricts Torch threads for measured CPU sizing. Its original
evaluations use one planner episode, one prior episode and 100 Monte Carlo
prior episodes every 50k decisions. Disable video capture as an output choice.
The Monte Carlo and predicted values appear in the shared reference dashboard.

The TD adapter changes only call signatures and adds metric aliases for the
common observer. Validate it with the same numerical model/optimizer and exact RNG parity
test. Validation restores parameter gradients as well as model and optimizer
states: both released implementations include retained previous actor gradients
in the model-wide gradient clipping operation. This behavior is preserved.

Set `BOOM_REFERENCE_METHOD=tdmpc_square` and point `BOOM_REFERENCE_ROOT` to its
own scratch directory containing the clean official `upstream` checkout. Set
`BOOM_WORKSPACE_ROOT` to the common campaign root. Both roots have separate
submission records and output ownership. The common root's `references` list
selects the exact two W&B identities without changing the AMBI comparison.

GPU validation includes a repeated unobserved update as a numerical control.
Floating outputs use rtol=2e-5 and atol=1e-7; RNG and integer states must match
exactly. The receipt records both native-repeat and observed-update differences.

`REFERENCE_AUTOLAUNCH=1` makes a successful gate submit its scientific job with
an `afterok` dependency on the gate itself. It requires both parity receipts
and a completed timing checkpoint, selects one CPU unless two improve speed by
more than 5%, and reserves measured training plus evaluation time with a 25%
margin and half an hour for startup. TD sizing retains its full native Monte
Carlo evaluation. Submission failures propagate; failed gates never launch
training. The scientific job ID and measured resource plan are written to the
campaign submission record before the shared dashboard is updated.

`REFERENCE_STEPS=200000` selects the agreed diagnostic budget.
`REFERENCE_ACCELERATION=planner-graph` captures the complete MPPI iteration
region, reusing the original neural modules. BOOM retains native module
compilation; TD retains native module arithmetic. A GPU test compares native,
translated eager and captured planners for both episode reset modes and
training/evaluation actions, including exact Python, NumPy and Torch RNG state.
Graph construction restores RNG and planner state after warmup. Sampling of
critic head pairs retains the original NumPy stream, and final categorical
action selection remains outside the graph with each method's original API.
