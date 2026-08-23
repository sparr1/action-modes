# XQC compatibility port

This repository contains a state-observation PyTorch port of **XQC:
Well-Conditioned Optimization Accelerates Deep Reinforcement Learning**. The
behavioral reference is the authors' official implementation at commit
`9a6832bb742ef01bbe9f1e06153a9338e612dae5`:

- Paper: <https://arxiv.org/abs/2509.25174>
- Official code: <https://github.com/danielpalenicek/xqc>
- Upstream license: [MIT](licenses/XQC-LICENSE.txt)

The port intentionally follows released-code behavior where it differs from a
more conventional SAC implementation. In particular, it uses population-
variance Flax-style BatchNorm, categorical twin critics, post-update unit-row
projection, joined current/next critic batches, non-committed target BatchNorm
statistics, and delayed actor/temperature optimizer steps. The released code
uses the actor learning rate for temperature, always projects final kernels,
and optimizes `alpha * (entropy - target_entropy)`; those details are retained
and covered by tests.

The Action Modes runtime stays PyTorch-only. JAX and Flax are needed solely to
regenerate the small test oracle from the pinned official checkout. This first
port supports feature vectors, including DMControl state observations and
future AMBI latent states. Pixels, CrossQ/MSE ablations, Hessian analysis, and
AMBI integration are outside its scope.

## Reproduction profiles

The official Walker Walk profile uses 500,000 agent decisions, corresponding
to one million raw frames at action repeat two:

```bash
environments/dmcontrol/.venv/bin/python main.py \
  --run configs/dmcontrol/experiments/xqc_walker_walk_state.json \
  --alg-dir configs/dmcontrol/algs
```

`xqc_walker_walk_state_smoke.json` stops at decision 5,020. Because updates
begin only after decision 5,000 and UTD is two, it performs exactly 40 learner
updates. This smoke checks execution and state invariants; it does not compare
learning curves across PyTorch and JAX PRNGs.

The Humanoid Walk comparison uses the same full learner profile for seeds 0
and 1:

```bash
environments/dmcontrol/.venv/bin/python main.py \
  --run configs/dmcontrol/experiments/xqc_humanoid_walk_state.json \
  --alg-dir configs/dmcontrol/algs
```

This is a two-seed implementation-parity check, not a statistical reproduction
of the paper. The paper curve aggregates ten seeds (0 through 9), whereas this
profile deliberately runs only the first two. The pinned official repository's
`results/xqc.csv` provides the released Humanoid Walk reference: at one million
raw frames its ten-seed mean return is 713.05, with individual seed returns
ranging from 574.95 to 895.44.

The profile logs online to the `ambi_humanoid` W&B project. It intentionally
does not set a W&B group or tags: the Hydra launcher supplies one shared,
launch-specific group and implementation tags to both comparison jobs.
Run `slurm/run_xqc_humanoid_walk_hydra.sbatch` exactly twice with the same
comparison group, once with `IMPLEMENTATION=official` and once with
`IMPLEMENTATION=action`. Each allocation requests one GPU for up to 30 days;
the preferred Hydra node is selected when the jobs are submitted, not pinned
in the tracked launcher. Evaluation CSVs are written alongside the durable
W&B run data as a recovery and post-processing fallback.

Action Modes records evaluation steps in agent decisions. For comparison with
the released raw-frame axis, multiply decision steps by the action repeat of
two, except label the evaluation after decision one as frame zero. This gives
the paper grid `0, 100000, ..., 1000000`. Different PyTorch/JAX random-number
streams and the separately locked DMControl/MuJoCo versions preclude matching
individual trajectories; compare the curve shape and aggregate returns.

Hydra validation is submitted from a clean checkout whose exact commit was
pushed to GitHub. See `slurm/run_xqc_validation_hydra.sbatch`; the job also
requires a separate clean official XQC checkout pinned to the commit above.
Prepare the two isolated, lock-resolved environments before submission:

```bash
uv sync --project environments/dmcontrol --locked
git -C /absolute/path/to/xqc checkout --detach 9a6832bb742ef01bbe9f1e06153a9338e612dae5
git -C /absolute/path/to/xqc submodule update --init --recursive
uv sync --project /absolute/path/to/xqc --locked
```

The launcher refuses dirty or mismatched checkouts and places run artifacts in
the job's `SLURM_TMPDIR`, falling back to the compute node's `/tmp` scratch. It
counts completed updates around the unmodified official learner and requires
exactly 40 in both the JAX and PyTorch smoke runs. Both smokes reject non-finite
learned state and require projected-kernel norm residuals no larger than
`1e-6`. Regenerated oracle metadata and structure must match exactly; numeric
leaves use `atol=1e-6` and `rtol=1e-5` to accommodate CPU-XLA differences
across platforms.
