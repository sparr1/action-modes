# TD-MPC2 Meta-World v2 runtime

The four AMBI runs use goal-observable Assembly, Hand-Insert, Push, and
Stick-Pull from Meta-World commit
[`04be337a12305e393c0caf0cbf5ec7755c7c8feb`](https://github.com/Farama-Foundation/Metaworld/tree/04be337a12305e393c0caf0cbf5ec7755c7c8feb).
TD-MPC2 used these tasks with MuJoCo **2.1.0**, `mujoco-py==2.1.2.14`, and
`gym==0.21.0`. AMBI converts the legacy Gym API to Gymnasium 0.29.1 at
the environment boundary. The learner uses the existing Python 3.10,
PyTorch 2.3.1 (CUDA 12.1), and TorchRL/TensorDict 0.4 stack.
See the [upstream dependencies](https://github.com/nicklashansen/tdmpc2/blob/8bbc14ebabdb32ea7ada5c801dc525d0dc73bafe/docker/environment.yaml)
and [upstream wrapper](https://github.com/nicklashansen/tdmpc2/blob/8bbc14ebabdb32ea7ada5c801dc525d0dc73bafe/tdmpc2/envs/metaworld.py).

## Install on Linux x86-64

From the repository root, on a host with `uv` (lock generated with 0.9.9),
a C compiler with OpenMP, `patchelf`, and OpenGL and OSMesa development
headers/libraries, run:

```bash
environments/metaworld/install.sh
```

On Debian/Ubuntu the native prerequisites include `build-essential`,
`patchelf`, `libgl1-mesa-dev`, and `libosmesa6-dev`. A system Python 3.10
needs its development headers. The installer also uses Git, curl, tar, and
GNU coreutils. It verifies the official MuJoCo 2.1.0 SDK archive against its
published SHA-256, extracts it under `.mujoco/` in this directory and creates
`.venv-tdmpc2/` alongside it. Both directories are local to this setup.
The older `.venv/` from the native-MuJoCo prototype does not provide this
protocol. The Linux SDK targets x86-64 and the lock selects Linux CUDA wheels.

Gym 0.21 contains a malformed optional dependency in its published metadata.
The lock records its base dependencies. The installer syncs the remaining
locked packages with uv, then installs the checksummed original Gym sdist with
the pinned versions of pip, setuptools, and wheel. It finishes by compiling
and loading `mujoco-py`, verifying the environment pins, and comparing a
100-decision episode on each task against the raw simulator with W&B disabled.
It also checks that restoring each task reproduces two subsequent episodes.
Re-run the installer after a failed
build once the missing prerequisites are available.

Use the `environments/metaworld/python` wrapper to run scripts from the
repository root. It executes the isolated interpreter with the SDK's shared
libraries on the loader path and selects the OSMesa CPU renderer for headless
state observations. PyTorch can still use CUDA for learning. Install once in
the checkout where training will run so compute jobs reuse the compiled
binding.

## Verify the four environments

```bash
WANDB_MODE=disabled AMBI_RUN_REAL_METAWORLD_TESTS=1 \
  environments/metaworld/python -m pytest -q tests/test_metaworld_env.py
```

The registered Gymnasium entry point is `MetaWorld-v0`. At construction the
adapter checks the installed Meta-World Git commit, Gym and mujoco-py versions,
and the version returned by the loaded MuJoCo library. The experiment manifests
record the protocol as `metaworld-tdmpc2-04be337-state-final-success-v1`.

Each episode has 100 decisions. Each action advances two raw steps and the
agent receives their summed reward. Timeouts are truncations. Evaluation uses
the success flag from the last raw transition of the episode. Reset retains
TD-MPC2's zero-action settling step and returns the observation taken just
before it. The settling step is outside the interaction budget. The adapter
isolates each environment's reset randomness from the learner's NumPy stream.

The four configurations use 1,000,000 training decisions, or 2,000,000 raw
training steps per task. The four tasks support exact training continuation
at episode boundaries. The resume generation saves the goal sampler, stacked
observation history, mutable simulator model positions, and learner/replay state.
Run the full two-episode continuation test on the target simulator:
`AMBI_RUN_REAL_METAWORLD_TESTS=1 environments/metaworld/python -m pytest -q tests/test_metaworld_resume.py`.
See the [experiment configurations](../../configs/metaworld/README.md).
