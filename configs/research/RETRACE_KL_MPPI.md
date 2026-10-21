# Retrace with prior KL on the MPPI-trained backbone

`ambi_retrace_kl_mppi.json` evaluates two methods at each of the same frozen
MPPI auxiliary-return backbone checkpoints, 400k and 500k training decisions:
Retrace KL-only and Retrace entropy plus KL. This is four cells, five paired
seeds 101–105 per cell and 500 decisions per episode. Controller seed is 55;
execution samples the adapted actor. Existing sampled-prior episodes and the
five completed comparison curves are reused without reevaluation.

The actor, critic, learning rates and H3/J8/C16/A4/N128 recipe match the existing
KL comparisons. The critic uses finite-horizon Retrace with lambda 1 and 86
complete trajectories per update: 258 critic rows, versus 256 sampled rows in
the one-step baseline. Actor and temperature batches remain 256. There are
128 critic updates and 32 actor updates per decision. The KL-only arm performs
zero temperature updates; the joint arm performs 32 with learning rate 0.0003.
The critics retain reward-only targets and auxiliary-return initialization/tail.
Beta is fixed at the saved outer alpha: 0.013081684708595276 at 400k and
0.023066775873303413 at 500k. There is no carryover between real decisions.

The companion profile pins checkpoint/sidecar hashes and exact sampled-prior
manifest/result hashes. Preparation reads actual checkpoint alpha and resolves
each publication identity using the actual reference bundle. Full evaluations
require matched episode and solver seeds, unchanged outer state, complete
optimizer traces and finite diagnostics. At decisions 0, 100, 200, 300 and 400,
32 independent probe rollouts retain the same initial/round-boundary value
diagnostics; their time is separately reported and excluded from controller time.

Launch preparation and worker/publication interfaces:

```bash
python slurm/ambi_retrace_kl_mppi.py prepare --root ROOT
python slurm/ambi_retrace_kl_mppi.py run --campaign ROOT/campaign.json --index 0
python slurm/ambi_retrace_kl_mppi.py publish --campaign ROOT/campaign.json --require-complete
```

Cell order is 400k KL-only, 400k joint, 500k KL-only, 500k joint. Workers run in
Slurm GPU allocations and never initialize W&B. With `RETRACE_CPU_GATE=1`, the
launcher runs separate 32-decision 400k KL-only smoke measurements on one and
two CPUs, then an eight-decision 500k joint smoke. Smoke paths contain cell,
length and CPU count and cannot collide with production output. Receipts include
controller/probe time, process wall time, child CPU time and peak child RSS.
Resource sizing and concurrency are chosen from this gate and current limits.

Preparation extends the retained five-method profile to seven independent
curves. Install and verify the resulting native layout before the first CPU
publication. Calling `publish` before results finish exposes pending counts.
Complete cells append paired gain, variance, return, runtime, entropy, alpha,
KL and Retrace trace/correction diagnostics. The three Retrace panels augment
the prior 17 panels. Effective trace length averages all valid suffixes: at H3,
the maximum mean is 2, even though a first-step trace can span three steps.

Full gzip optimizer traces remain on Oscar, with their hashes and locations
in compact W&B artifacts. Uploads are limited to 16 MiB of summaries per cell.
Publication rejects changed results or references and is idempotent through
the existing evaluation-series journal. No full training run is launched by
these commands, and no prior checkpoint result is replaced.
