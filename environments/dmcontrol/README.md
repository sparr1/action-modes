# Isolated DMControl runtime

This nested `uv` project is the supported runtime for AMBI's single-task
DMControl experiments. It deliberately does not participate in the repository's
root `requirements.txt`: the existing Gymnasium Robotics environment requires
MuJoCo `<3.0`, while the pinned DMControl stack requires MuJoCo `>=3.1.1`.

The lock targets CPython 3.10 on Linux x86-64 and macOS arm64. Its environment-
defining versions match the TD-MPC2 setup used by this integration:

- `dm-control==1.0.16`
- `mujoco==3.1.2`
- `gymnasium==0.29.1`
- `numpy==1.24.4`
- `torch==2.3.1`, using CUDA 12.1 wheels on Linux x86-64

## Create or verify the environment

Run all commands from the repository root. Install `uv`, ensure a CPython 3.10
interpreter is available, then create exactly the locked environment:

```bash
uv python install 3.10
uv sync --project environments/dmcontrol --locked
uv run --project environments/dmcontrol python -m pytest --version
```

`uv sync --locked` must be run before submitting compute jobs. Compute jobs
should use the already-created interpreter and must not resolve dependencies:

```bash
environments/dmcontrol/.venv/bin/python main.py --help
```

The project is intentionally non-installable. Keep the current working
directory at the repository root so imports resolve to this checkout.

## Run the examples

State observations are the TD-MPC2-compatible default. This manifest runs the
TD-MPC2 comparator followed by AMBI-TD-MPC2 with full training budgets:

```bash
environments/dmcontrol/.venv/bin/python main.py \
  --run configs/dmcontrol/experiments/walker_walk_state.json \
  --alg-dir configs/dmcontrol/algs
```

The RGB manifest is only a functional smoke test. It uses the exact 3-frame,
64-by-64 pixel observation format but intentionally tiny training and replay
budgets:

```bash
environments/dmcontrol/.venv/bin/python main.py \
  --run configs/dmcontrol/experiments/walker_walk_rgb_smoke.json \
  --alg-dir configs/dmcontrol/algs
```

Use `--num-runs 1` to run only the first algorithm in either comparison
manifest, or `--alg-index 1 --num-runs 1` to run only AMBI-TD-MPC2.

The adaptive parameter-noise D512 example is a full-cost, single-seed Humanoid
Walk screen. It runs `J=8`, `N=512`, and `H=3`; each real decision therefore
uses 12,288 imagined transitions, split into 256 clean rollouts and 256 noisy
rollouts from four perturbed actors (64 each). Its configured budget is 14
million decisions, so inspect the config and submit it through the normal
scheduled-compute workflow rather than treating it as a local smoke test:

```bash
environments/dmcontrol/.venv/bin/python main.py \
  --run configs/dmcontrol/experiments/ambi_humanoid_walk_base_v2_adaptive_param_noise_d512_k4.json \
  --alg-dir configs/dmcontrol/algs
```

## Rendering backends

Choose the MuJoCo rendering backend in the job environment **before** Python
starts. Shared Python code intentionally does not force one backend.

Headless Linux with an NVIDIA GPU:

```bash
MUJOCO_GL=egl environments/dmcontrol/.venv/bin/python main.py \
  --run configs/dmcontrol/experiments/walker_walk_rgb_smoke.json \
  --alg-dir configs/dmcontrol/algs
```

CPU-only Linux rendering tests:

```bash
AMBI_RUN_REAL_DMCONTROL_TESTS=1 MUJOCO_GL=osmesa \
  environments/dmcontrol/.venv/bin/python -m pytest -q \
  tests/test_dmcontrol_env.py tests/test_tdmpc2_pixels.py
```

On macOS, leave `MUJOCO_GL` unset so MuJoCo uses the platform default. A Linux
host must provide the corresponding system OpenGL/EGL libraries in addition to
the locked Python packages. EGL requires a working NVIDIA driver and EGL
installation; OSMesa requires the distribution's OSMesa runtime. Those system
libraries cannot be made portable by a Python lock.

## Pixel replay memory

Pixel replay is intentionally stored exactly as TD-MPC2 stores it: complete,
overlapping `uint8` frame stacks, with no lazy-frame compression. One
`(9, 64, 64)` observation is 36,864 bytes. One million observation rows
therefore require about 36.9 GB (34.3 GiB) before replay metadata and allocator
overhead. The RGB smoke config uses 2,000 rows, about 73.7 MB of raw observation
storage, and is not a benchmark configuration. Choose a production replay
budget explicitly after accounting for host memory; the learner will not
silently shrink it. Use `log_type: "summary"` for RGB runs unless retaining
every image observation in detailed trajectory logs is explicitly intended.

Random-shift augmentation remains active during RGB acting and evaluation to
match TD-MPC2. A deterministic evaluation request selects the deterministic
policy action; it does not freeze the image crop.

## Updating the lock

Dependency changes are deliberate compatibility work. Edit `pyproject.toml`,
then regenerate and review the complete lock on a networked development host:

```bash
uv lock --project environments/dmcontrol
uv sync --project environments/dmcontrol --locked
```

Do not update the root requirements as part of a DMControl-only dependency
change.
