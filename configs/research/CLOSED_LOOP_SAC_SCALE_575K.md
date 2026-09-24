# SAC rollout and update scheduling at 575k

This campaign evaluates the frozen target-entropy −10.5/shared backbone
`aux6428346x0` at 575,000 decisions. It tests the article-inspired N1024/B4096
recipes with 20/4/10 and 20/20/10 critic/actor/target updates per collection
round. The source is [Speeding Up SAC with Massively Parallel
Simulation](https://arthshukla.substack.com/p/speeding-up-sac-with-massively-parallel).
The selected hyperparameters test transfer to fresh learned-model inner solves;
they do not reproduce the article's persistent simulator training protocol.

The grid is H1/H2/H3 × J1/2/4/6/8/10 × the two schedules: 36 settings and
180 full episodes. Each setting uses environment seeds 101–105, controller
seed 55, and 500 Humanoid Walk decisions. DMControl repeats each action twice
and sums its rewards. Every real decision starts a fresh inherited actor,
online critic, target critic, temperature, optimizers, and replay; the final
adapted mean action is executed. All outer parameters remain frozen.

Both critic initialization and frozen terminal bootstrap use auxiliary return
Q. Critic targets remain reward-only and one-step. The actor retains adaptive
squashed entropy, inherits initial alpha approximately 0.0046039033, and uses
target entropy −10.5. Actor/critic/temperature learning rates remain 0.0003;
target Polyak coefficient remains 0.01. The checkpoint uses no actor Q
normalization. These are deliberate retained AMBI settings, not additional
article hyperparameter substitutions.

The scheduler uses G20, actor interval P5 or P1, and target interval T2.
Every slot performs a critic update from one replay minibatch, then a due actor
and temperature update on the same minibatch, then a due target update.
Intervals count completed critic updates across rounds and reset each decision.
Targets at H1 always use the frozen terminal critic; the target-copy cadence
only affects bootstrapped learning targets for nonterminal depths at H2/H3.

Each solve collects J×1024×H transitions and retains all of them, with
capacity max(3072, J×1024×H) and sampling with replacement. It performs 20×J
critic, (20/P)×J actor and temperature, and 10×J target updates. Total replay
draws are 20×J×4096; actor updates share the critic minibatches. There is no
per-round replay reset or outer-replay mixing.

The primary outcomes are raw full-episode return and paired improvement over
the existing hash-pinned prior. Existing prior episodes are reused without
republishing or rerunning them. The campaign preserves all update traces and
32 sampled model-return probes at initialization and after every round. New
performance and training identities belong to each setting; the overview
compares J curves grouped by H and schedule, with pending outcomes left empty.
Five paired environment seeds and one backbone seed provide exploratory
evidence; confidence intervals use 2,000 paired environment-seed resamples.

GPU smoke evaluations cover H1/H3, both actor intervals, and J10 at the full
N/B/update budget. They verify checkpoint identity, frozen state, finite
metrics, strict compilation without fallbacks, replay retention, exact update
counts, actor-update chronology, and probe coverage. Target ordering is covered
by scheduler tests; the GPU traces verify target totals. Warmup/compilation,
control time, model-probe time, and serialization time are recorded separately
to estimate the full sweep. Production resources and time limits are selected
from those measurements and the live Oscar allowance.

Use `slurm/ambi_closed_loop_sac_scale.py prepare` and `worker`, and
`slurm/ambi_closed_loop_sac_scale_publish.py watch`. The launcher
`slurm/run_ambi_closed_loop_sac_scale_oscar.sbatch` requires the exact tested
commit and a clean Git checkout. GPU evaluation does not publish; the CPU
watcher publishes complete verified bundles and records failures explicitly.

`slurm/submit_ambi_closed_loop_sac_scale.py` provides separate `prepare`,
`smoke`, and `production` dispatch phases. Each requires `--root` and the full
`--sha`, records a submission intent before calling Slurm, and refuses an
existing intent journal. Production verifies all four smoke receipts before
estimating time limits and submitting the 36 cells. It permits the 12-GPU
account allowance, groups equal J/actor-interval settings for submission,
and starts one CPU watcher with three publication subprocesses. Recheck live
account use before dispatch; do not blindly retry an uncertain submission.
