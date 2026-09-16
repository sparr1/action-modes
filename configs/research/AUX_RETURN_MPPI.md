# Frozen auxiliary-return MPPI comparison

`ambi_aux_return_mppi.json` evaluates the seed-55, target-entropy -21,
detached-representation auxiliary SAC backbone `ambi/aux6434715x3`.
Use its 40 scheduled checkpoints from 25,000 through 1,000,000 decisions.

The two arms differ only in `inner_horizon_critic_source`: `sac` selects
the frozen online entropy-augmented critic; `aux_return` selects the frozen
online reward-only critic. Both use mean-pair reduction and the same frozen
stochastic SAC actor for policy trajectories and the horizon action. Scores
sum raw predicted rewards and the discounted terminal Q, with no explicit
entropy correction. No actor, critic, temperature or representation learns.

MPPI uses horizon 3, 512 candidates, 64 elites, 24 policy trajectories,
eight iterations, temperature 0.5, standard deviation bounds [0.05, 2],
and a shifted previous proposal mean within each episode. Each decision
uses 12,336 model transitions. Evaluation executes the optimized proposal
mean. This is the maintained AMBI MPPI controller; it differs from the older
native TD-MPC2 evaluation controller's sampled weighted-elite action rule.

Use environment seeds 101–105, controller seed 55 with independent
SHA256-derived episode RNG, and complete 500-decision Humanoid Walk episodes.
The campaign contains 400 episodes. Eager evaluation disables training-only
outer-policy diagnostics and compilation; network weights and scientific
settings remain those of the checkpoint. Both complete auxiliary and primary
outer states are hashed before and after evaluation.

`slurm/run_ambi_aux_return_mppi_oscar.sbatch` runs both arms at one checkpoint
per array task, indices 1–40. It requests one L40S, four CPUs and 24 GiB;
choose concurrency from current account headroom. `--smoke` uses two seeds
and three decisions at the same production MPPI budget. Run smoke tasks at
the first and final checkpoints before production. Full episodes are saved
with per-decision metrics, paired return differences, and HTML reports.

Create two new evaluation-series identities using `--eval-series-spec-dir`,
assign the resulting owner directories with `--eval-run-map`, and run one
CPU publisher per curve in `ambi-inner-bench`. The labels identify soft Q
versus return-only Q. Prior-only episodes are not part of this request.
