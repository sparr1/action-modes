# Actor updates in closed-loop prior refinement

This experiment asks whether more optimization of the inherited actor improves
real return after a fixed amount of critic fitting. It uses
[closed-loop refinement eval](CLOSED_LOOP_REFINEMENT_EVAL.md): a fresh solve at
every real decision, followed by mean-action execution for a normal episode.

The frozen backbone is `rwgao_b-brown-university/ambi/mey3rxj8`, checkpoint
200,000, SHA256
`909e5c1d125aecc952e0544d802b4b946ae5089909c7a55885f272c32ed3e12f`.
Gamma remains 0.99 and the saved Q divisor remains 11.182598114013672. No outer
learning or writeback occurs. The actor's learned mean, standard deviation,
log-standard-deviation mapping and bounds are inherited unchanged.

Each decision starts from the inherited actor and online critic, with fresh
optimizers and replay. One round collects 128 sampled H1 transitions, performs
32 critic updates, then performs A actor updates. B256, replay capacity 2,048,
sampling with replacement, fixed alpha zero, learning rates, Adam settings,
clipping, Q reductions and the fixed saved Q scale are unchanged. The sole
changed learner parameter is `inner_actor_updates_per_round`.

| A | Critic updates | Actor minibatch rows | New refinement episodes |
|---|---:|---:|---:|
| 1 | 32 | 256 | 60 |
| 2 | 32 | 512 | 60 |
| 4 | 32 | 1,024 | 0; reuse the existing 60 |
| 8 | 32 | 2,048 | 60 |

At H1, every replay current-state input is the encoded root. Actor minibatches
therefore repeat that root while drawing fresh policy noise for each row. More
actor rows are additional stochastic optimization samples, not new imagined
transition coverage. Every setting collects the same 128 transitions and fits
the critic 32 times at a matched root. The actor then optimizes against that
fitted critic. Its Q evaluation remains stochastic under the existing dropout
and critic-pair protocol; no extra critic optimizer steps occur during the actor
phase. At a matched seed/root, shorter actor budgets share the longer budget's
initialization, collection, critic fit and actor-update prefix.

The H1 target uses the frozen outer online Q at a sampled outer-policy terminal
action. Terminal actions and Q pairs are sampled during fitting; a frozen outer
function does not imply one fixed scalar target. Mean real execution and sampled
imagined actions retain their distinct meanings. Ordinary episode time limits
remain enabled.

## Coverage and interpretation

Evaluate environment seeds 101–120 with controller seeds 55–57 and at most 500
real decisions per episode. There are 12 cells and 240 refinement episodes:
180 new A1/A2/A8 episodes and 60 historical A4 episodes. A4 must not be rerun.
The existing sealed-shard merger requires inline prior references, so new
workers also execute 180 cheap prior-only episodes with no inner optimization.
The reporter verifies their pairing against the historical prior values.

The primary outcome is undiscounted full-episode real return, with paired gains
against A4 and the frozen prior. Average the three controller repetitions within
each environment seed, then weight the 20 environment seeds equally. Report
pointwise 95% paired environment-seed percentile bootstrap intervals using
2,000 resamples and bootstrap seed 20260912. These are exploratory intervals for
this frozen backbone, not independent policy-training confidence intervals.
Once controllers choose different real actions, later states and replay differ;
training curves averaged along those trajectories are descriptive dynamics.

Record actual actor and critic update counts and measured control runtime.
Total optimizer work per decision is 32+A, while model collection remains 128
transitions. Keep model-return probes at initialization and after the completed
round, using 32 paired samples. Their axes are actor updates 0/A and critic
updates 0/32; totals decompose into raw model reward plus discounted frozen
outer bootstrap, without an entropy bonus or Q normalization. These model
predictions do not replace real returns.

Retain per-update actor loss, Q, entropy, gradient, action-change and available
saturation statistics. Training metrics are sampled pre-update minibatches;
actor sample saturation is not automatically saturation of the executed mean.
Stream complete critic and actor traces, preserve missing values, and keep
source hashes and sealed bundles. Historical A4 timing was measured under its
original runtime load and is not a simultaneous hardware-load control.

## Configuration and execution

`actor_steps_a{1,2,8}_j1_h1_200k.json` derive from
`round_scaling_j1_h1_200k.json`, changing only actor update count in the learner.
A4 uses the unchanged historical matrix. The immutable campaign validates the
checkpoint, source identity, full coverage, selector and pinned historical
manifest/seal file hashes. Historical cells also pass the current full-trace
parser before they can be accepted into the complete comparison.

`slurm/ambi_actor_steps_campaign.py` exposes `metadata`, `worker`, and `merge`.
The manifest declares `actor_updates: [1,2,4,8]`, `critic_updates: 32`, `rounds: 1`,
and cells ordered by A then controller seed, named `a{A}-s{seed}`. Each cell has
four five-seed shards. The 36 new GPU tasks run A8 first, then A2, then A1;
A4 has no worker tasks. With 12 available L40S allocations, the tasks use three
waves. Set concurrency using live combined GPU, CPU and memory headroom.

Use `run_ambi_actor_steps_oscar.sbatch`,
`run_ambi_actor_steps_merge_oscar.sbatch`, and
`run_ambi_actor_steps_report_oscar.sbatch` from a clean Git checkout at the
explicit tested SHA. They use `AMBI_ACTOR_STEPS_CAMPAIGN` and the locked DMControl
runtime. GPU workers and mergers stay offline; the comparison reporter owns one
explicit new run in `ambi-inner-bench`, together with standalone HTML and paired
artifacts. Existing checkpoint-axis curves and historical bundles are preserved.
Incomplete cells must remain visible and cannot become a completed comparison.

This launch intentionally skips a prelaunch GPU smoke at the user's request.
The optional `--smoke` flag remains available for a separately requested short
check; it is not invoked implicitly or required as a production launch gate.
Local analytic and launcher tests do not constitute a new GPU runtime test.
