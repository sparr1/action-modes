# TD-MPC2 core

This package vendors the algorithmic core from the official TD-MPC2 implementation
at upstream commit `8bbc14ebabdb32ea7ada5c801dc525d0dc73bafe`.

Copied/adapted here:

- `agent.py`
- `common/buffer.py`
- `common/init.py`
- `common/layers.py`
- `common/math.py`
- `common/scale.py`
- `common/world_model.py`

Intentionally not copied yet:

- official environment factory
- Hydra training entrypoint
- official trainer loop
- official logger / W&B setup

AMBI should construct the environment and own the top-level training harness. A `RL/TDMPC2.py` wrapper should import this package and implement AMBI-compatible `learn`, `predict`, `save`, and `load` methods.

## Compatibility scope

The port preserves the single-task, state-observation TD-MPC2 planning and
training equations. The compatibility layer deliberately differs in these ways:

- critics use ordinary `ModuleList` ensembles instead of TensorDict-vectorized
  modules;
- CPU and explicit device selection are supported;
- arbitrary finite Box action bounds are normalized to `[-1, 1]` by the wrapper;
- `torch.compile` is not supported by the compatibility ensemble and is rejected;
- the wrapper rejects pixel observations rather than silently flattening them;
- current official vectorized-critic checkpoints are converted on load.

`common/soft_world_model.py` and `ambi_agent.py` are AMBI extensions, not
upstream TD-MPC2. They retain the encoder, latent dynamics, reward model,
SimNorm representation, and multi-step consistency/reward training, but replace
TD-MPC2's policy prior with a squashed-Gaussian SAC actor and soft Bellman
objectives. The reference AMBI model uses TD-MPC2's model-size-driven
distributional ensemble (five Q heads at model size 5); scalar twin critics
remain an explicit ablation. AMBI defaults to per-root inner SAC adaptation and
retains a matched MPPI inner operator for comparison.

Model saves are suitable for evaluation and weight transfer. They do not include
replay, environment state, or all trainer counters, so they are not exact
mid-run resume checkpoints.
