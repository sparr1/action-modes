# Actor updates at longer horizons

The frozen 625k auxiliary-return checkpoint is evaluated with soft inner
initialization and either soft or return-only outer bootstrap. The matrix
`ambi_aux_actor_budget_625k.json` crosses H=2/3 with A=4/8/16 at J=1.
N128, B256, C32, tau=.01, full replay, entropy-augmented inner fitting and
inherited adaptive alpha are fixed. Each actor step also updates temperature;
this tests the existing actor/temperature schedule, not actor-only updates
with a fixed temperature-update budget. Actor states remain uniform replay.

Each of 12 conditions runs environment seeds 101–105, controller seed 55,
for 500 real decisions. This is a five-seed exploratory screen on one trained
checkpoint. Paired A8/A16 minus A4 differences use fresh baselines from this
campaign. A new deterministic prior supplies the common prior improvement.
All original historical bundles and W&B results remain immutable.

Execution enables deterministic PyTorch algorithms and sets the cuBLAS
workspace before CUDA initialization. Every job uses L40S and an independent
compiler cache, with runtime versions, precision, driver/device information,
cache paths and generated Python/JSON hashes saved in `execution.json`.
Two independent smoke processes check all eight H/critic/A4-or-A16 cases,
one seed and three decisions each. Their non-timing traces, episode fields
and model metrics must agree exactly before the full campaign starts. This
short gate is not a guarantee across all software, hardware or full episodes.

`slurm/ambi_aux_hj_sweep.py` validates actual model work, replay size and every
critic, actor and temperature update. Its publisher retains raw traces,
per-update curves, per-decision/per-seed metrics and model probes. W&B training
runs log `comparison/a4_gain_mean`, uncertainty and paired coverage, alongside
the normal prior-paired performance metrics. A4 rows are zero by construction.

Source is synchronized by Git before launch. Run the two control-array tasks,
then the gate/prior job, then the 12 production tasks, using Slurm `afterok`
dependencies and fail-closed control validation. The overview publisher starts
at submission and publishes only complete, validated condition bundles.
