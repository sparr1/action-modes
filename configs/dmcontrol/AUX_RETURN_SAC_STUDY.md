# Auxiliary reward critic with a SAC prior

`experiments/ambi_aux_return_sac_study.json` defines four fresh, single-seed
Humanoid Walk state runs. Each trains a standard entropy-augmented SAC critic
and an independently parameterized reward-only auxiliary critic. Both bootstrap
from the same stochastic SAC actor. There is no additional return actor and no
inner optimization or MPPI in this campaign.

| Array index | Algorithm config | SAC target entropy | Auxiliary gradients into encoder/dynamics |
|---|---|---:|---|
| 0 | `ambi_aux_return_sac_clip_target10p5_shared` | -10.5 | Enabled |
| 1 | `ambi_aux_return_sac_clip_target10p5_detached` | -10.5 | Detached |
| 2 | `ambi_aux_return_sac_clip_target21_shared` | -21 | Enabled |
| 3 | `ambi_aux_return_sac_clip_target21_detached` | -21 | Detached |

Each run uses seed 55, two million decisions, 2,500 warmup decisions followed
by 2,500 pretraining updates, then UTD 1. All 80 scheduled checkpoints per run
are retained at 25,000-decision intervals (320 total). These are portable model
checkpoints, not interrupted-training resume snapshots. This is an exploratory
single-seed comparison.

Both critic coefficients are 0.1. The SAC actor and both critic learning rates
are 3e-4. Actor Q uses mean-pair reduction, target Q uses min-pair reduction,
and both ensembles have five distributional heads. SAC starts at alpha 1 with
automatic temperature tuning, squashed entropy, direct-clamp log standard
deviation bounds [-10, 2], and no Q normalization. Both critics see the same
outer replay samples and update schedule. The detached cases stop only the
auxiliary critic loss from updating the representation.
The detached arms still share the joint gradient clipping step, so they do
not fully isolate SAC optimization or freeze the representation.

The full recipes preserve the original SAC prior observability, including
training/episode metrics, early and periodic outer-policy diagnostics, and
event-indexed W&B logging. W&B project is `ambi`, group is
`ambi-aux-return-sac-20260915`, and run names include the config and `seed55`.
Training uses strict compilation: a compiled-region failure stops the run.

## Oscar launcher and validation

Use `slurm/run_ambi_aux_return_sac_oscar.sbatch`. It installs no dependencies
and requires these explicit environment variables:

- `AMBI_PROJECT_DIR`: absolute clean Git checkout at the intended commit.
- `AMBI_PYTHON`: absolute executable for the existing locked DMControl runtime.
- `EXPECTED_ACTION_MODES_SHA`: full 40-character source commit.
- `AMBI_AUX_OUTPUT_ROOT`: absolute scratch or project storage outside source.
- `AMBI_AUX_CAMPAIGN`: unique directory basename.
- `AMBI_AUX_MODE`: `smoke` or `train` (default).
- `AMBI_AUX_GATE_RECEIPT`: absolute successful receipt, required for training.

Choose the account, resource overrides, and array concurrency from current
Oscar limits and availability. The script defaults to one generic GPU, eight
CPUs and 32 GiB, with no account or array throttle. Create scheduler output
directories first and pass `--output`/`--error` when using locations other than
the script's defaults. Use one non-array smoke job (allow two hours), then the
four training cells with `--array=0-3` and an `afterok` dependency on the smoke.
The launcher rejects requeues/restarts and existing per-cell output directories.

The smoke checks all four full-sized networks on a seed-55 real-transition
replay fixture, including two initially cold strict CUDA updates, finite
losses, all six compile fallback flags, paired replay hashes, isolated attached
or detached representation gradients, portable checkpoint state, exact learner
state plus reproducible next update, and stochastic SAC action selection with
zero inner updates and no return actor. It also runs the auxiliary checkpoint
regression suite.

A separate fresh 512-decision training smoke exercises the real replay,
training harness, scheduled checkpoints, and checkpoint sidecars. Its explicit
overrides are warmup 500, pretraining 2, replay capacity 4096, W&B off and a
256-decision checkpoint interval. It preserves production network and batch
sizes. The short smoke does not execute production's entire warmup/pretraining
schedule; its overrides are recorded in the receipt.

Successful smoke reports create
`<output-root>/<campaign>/gate/receipt.json`, bound to the source commit,
manifest, every algorithm config and environment lock. Missing/skipped tests,
fallbacks, mismatched config hashes or incomplete reports cannot produce a
valid receipt. It records cold/warm update timing and estimates the 320-model
bank size from actual checkpoint bytes; check scratch quota before launching
the production array.
The receipt requires the same Python/PyTorch/CUDA runtime for training; generic
GPU models may differ and are recorded, with strict compilation on every run.

Each training cell writes `launch.json` under
`<output-root>/<campaign>/<config>-seed55/`, recording the full resolved config,
input recipes, source/lock hashes, GPU/runtime, Slurm allocation, exact command,
receipt hash, and W&B ID/path/URL. The normal timestamped experiment artifacts
and checkpoint sidecars are nested beneath that cell directory. Credentials
are consumed from the existing environment or netrc and never written into
launch metadata. W&B uses a unique Slurm-array-derived ID with resume disabled.
