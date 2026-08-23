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
exactly 40 in both the JAX and PyTorch smoke runs. Regenerated oracle metadata
and structure must match exactly; numeric leaves use `atol=1e-6` and
`rtol=1e-5` to accommodate CPU-XLA differences across platforms.
