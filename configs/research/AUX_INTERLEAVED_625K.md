# Matched-budget interleaved actor and critic updates at 625k

`ambi_aux_interleaved_625k.json` repeats the twelve settings from
`ambi_aux_actor_budget_625k.json`, changing only component update order:

| Critic steps per round | Actor and alpha steps per round | Interleaved schedule |
| --- | --- | --- |
| 32 | 4 | 8 critic, 1 actor/alpha, repeated 4 times |
| 32 | 8 | 4 critic, 1 actor/alpha, repeated 8 times |
| 32 | 16 | 2 critic, 1 actor/alpha, repeated 16 times |

Both soft-inner/soft-terminal and soft-inner/return-terminal settings use H2/H3,
J1, N128, B256, full replay, target tau .01, an inherited adaptive alpha and
entropy-augmented interior fitting. Terminal entropy follows the original
setting: outer alpha with the soft terminal critic, no entropy with the return
terminal critic. Five full closed-loop episodes use environment seeds 101–105
and controller seed 55. Each real decision resets adaptation from the frozen
625k checkpoint and executes the refined policy mean.

Collection still precedes updates within each round. The actor sees progressively
fitted critics, and subsequent interior critic targets see the changed actor and
alpha. The horizon boundary remains frozen. Critic and actor steps retain their
own minibatches, drawn in the same componentwise RNG sequence as critic-first
execution; targets update once per critic step. This tests order at matched
compute, not a simultaneous change to a 1:1 update ratio.

`inner_component_update_order` defaults to `critic_first`; the explicit default
preserves historical planner identity. The opt-in `interleaved` mode requires
canonical component SAC with round timing, trainable actor and critic, no
explorer or outer replay mixing, and fixed positive counts C >= A with C divisible
by A. Ordinary shared-batch update schedules remain unchanged.

The launcher uses deterministic compiled L40S execution and independent compiler
caches. Production depends on eight exact repeated interleaved controls and
eight exact default-path comparisons against the saved critic-first controls
(both arms, both horizons, A4/A16; one seed and three decisions each). These are
bounded reproducibility checks, not a guarantee for all possible trajectories.

Historical comparison is pinned to commit
`d02e78fe5fe8a29b6898dcbe8e93ac09a547613e`. Preparation requires complete frozen
baseline bundles with matching checkpoint, environment, seeds, planner settings
apart from order, and deterministic runtime identity. Default-path parity must
pass before those baselines are reused; global identity compatibility is not
relaxed. The completed deterministic prior is reused.

Workers validate every decision's update chronology and optimizer totals.
W&B training runs retain complete trace artifacts, update curves, decision
curves, fixed-noise probes, paired gains against the prior, and
`comparison/phased_gain_*` against the matched critic-first run. Five episodes
at one checkpoint form an exploratory paired screen; they do not establish
robustness across training seeds or tasks.

Use `slurm/ambi_aux_hj_sweep.py prepare --baseline-kind interleaved` with the
completed actor-budget campaign, then `slurm/run_ambi_aux_interleaved_oscar.sbatch`:
control array indices 0/1 repeat interleaving, index 2 checks the default path;
`--gate` validates their results; production indices 0–11 each evaluate one panel.
