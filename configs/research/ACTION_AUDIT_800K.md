# Matched-state action audit at 800k

This diagnostic locates where H1 inner SAC loses action quality at the frozen
800,000-step `aux6428346x0` checkpoint. It does not train the outer learner and
is not a full-episode policy improvement evaluation.

The versioned recipe is `ambi_800k_action_audit.json`. It pins both checkpoint
and metadata hashes. The existing return/return fresh preset supplies the
architecture and SAC conventions, with an explicit 800k checkpoint contract.
Every inner solve uses H1, C16/A4/N128/B256, action-local networks, replay,
optimizers and temperature, and J4/J6/J8. Eager execution avoids compilation
side effects in instrumentation; timings are diagnostic costs, not comparable
controller benchmark latency.

## State and candidate pairing

Three source histories (prior mean, fresh H1/J6 SAC, and H3 return-Q MPPI)
each visit seeds 101–105. Capture states before decisions 25, 100, 250 and 400.
This yields at most 60 roots, with one independently owned history/seed shard.
If a trajectory terminates before a root, report that root as unreachable.
The source prefix stops at decision 400; its reward is not an episode return.

All candidates at a root share the exact simulator and encoded state. Capture
the simulator state, observation, source identity, frozen-outer digest and
source RNG checks. Counterfactual audits restore the simulator and global RNG
and cannot consume the source controller's private RNG.

The bank contains the prior mean, final SAC means and 16 policy samples per
prior/adapted policy; every actual replay action from J4/J6/J8; cold-start MPPI
H1/H3 proposal means; 64 uniform actions; and 32 pre-tanh perturbations around
the prior mean at each scale 0.5, 1 and 2. Replay checks enforce H1 root-only
states, boundary flags, complete data and no overflow. SAC J variants share a
named initial solver stream. MPPI uses 512 samples, 64 elites, 24 policy
trajectories, 8 iterations, temperature 0.5, and standard deviation 0.05–2.
The MPPI source history shifts the previous proposal mean, with a separate
named seed per real decision. It is a diagnostic state source, not a replay
of a historical benchmark's exact solver RNG sequence.

## Quantities being compared

Score every bank action with the initial and final learned critics (expected
mean-pair values and head spread) and independent H1/H3 frozen-model returns.
All scoring forces the candidate first action, then follows the frozen SAC
prior and uses the frozen auxiliary return critic at the boundary. H3 scoring
therefore differs from MPPI's optimized multi-action sequence objective.
Report root-only relative error, ranking and empirical bank regret; do not
average these with untrained successor states.

Choose broad-bank and replay-bank winners independently for H1 and H3 using
32 selection draws. Re-score those four selections plus the six controller
means with 32 fresh validation draws. Preserve selected action provenance.
For each of these ten actions, use eight further paired continuation draws to
compare:

| Quantity | Meaning |
|---|---|
| A | Model prefix plus frozen terminal Q |
| B | Real simulator prefix plus the same frozen terminal Q |
| C | Real prefix plus a finite 500-decision frozen-prior continuation |
| A−B | Model-prefix error under the shared terminal estimate |
| B−C | Terminal estimate discrepancy against the finite real continuation |

Real branches continue beyond the episode time limit for calibration while
respecting true termination. C has an explicit finite cutoff and no appended
bootstrap; B−C includes omitted tail value and is not proof of infinite-horizon
Q bias. Each H has H+500 real decisions. Candidate gains subtract the prior
action using common noise. Within-root Monte Carlo SEs describe simulation
noise; they are not uncertainty over independent episodes. Aggregate roots
within a seed first, and keep the three source histories separate.

These comparisons can distinguish action coverage from critic ranking,
mean-action/actor optimization effects, model errors, and terminal calibration.
They do not by themselves establish that a different full replanning controller
will improve total episode return, or isolate every entropy/gradient mechanism.

## Execution and output

Use the locked DMControl runtime and a clean, pinned Git revision. Prepare a
new scratch campaign directory with `evaluate_ambi_action_audit.py --prepare`,
then run tasks 0–14 through `slurm/run_ambi_action_audit.sbatch`. `--smoke` uses
a distinct campaign identity, one root per source, two draws and four tail
steps, while retaining the real checkpoint and J4/J6/J8 update counts.

Each task writes a started manifest, exact simulator snapshots, full local
model-bank scores, root JSON records, progress, and a completion manifest.
Existing task outputs cannot be overwritten. W&B publication uses compact
tables and a separate saved view in `ambi-inner-bench`; raw banks and simulator
branches remain on Oscar. The CPU publisher retries transient publication
errors and verifies chart installation. Progress and failures remain explicit.
