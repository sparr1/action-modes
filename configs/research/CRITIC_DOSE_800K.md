# Critic update budget at 800k

This experiment asks whether additional critic fitting improves the fresh
H1/J6 inner SAC controller at the same frozen 800k auxiliary-return checkpoint
used by the [critic-bypass evaluation](CRITIC_BYPASS_800K.md).

The new conditions use 32 or 64 critic updates per round, giving 192 or 384
critic optimizer steps per real decision. Actor updates remain four per round
(24 total), and each of six rounds collects 128 imagined transitions (768
total). Batch size, learning rates, target-update rule, entropy settings,
initialization, and every other solver setting remain unchanged. Actor, critic,
optimizers, replay, and temperature reset at every real decision; the backbone
and its optimizers remain frozen.

Each dose is evaluated with actor-mean execution and learned-Q selection from
the complete 770-action bank. There are 80 new full episodes: two doses by two
selectors by 20 paired environment seeds, 101–120, using controller seed 55 and
500 real decisions. The existing C16 actor-mean, learned-Q, direct-model, prior,
and MPPI H3 results are reused through a pinned completed snapshot. No old
episode or reference is recomputed.

The actor-update count and rollout budget stay fixed, but the resulting actor
and later candidate banks can change when the critic receives more updates.
Additional critic updates also consume additional learning randomness. Named
initial solve seeds match across conditions; this is not an assertion of
identical minibatches or candidate banks across doses. The experiment measures
the effect of increasing the existing critic budget, including its effect on
subsequent actor updates and data collection.

At decisions 0, 25, 100, 250, 400, and 499, shadow diagnostics score actor mean,
prior mean, learned-Q choice, and direct-model choice on the same local bank.
Selection uses 32 common terminal-action draws; independent validation uses
32 other draws and never changes learning or execution. These are model-target
diagnostics, not simulator ground truth. Compare held-out selection quality
and Q ranking with full-episode return, paired gain versus the corresponding
C16 controller and prior, seed variability, win rate, and controller latency.

All new controllers use eager execution on RTX A5000 GPUs, matching the C16
campaign. Synchronized controller timing includes fitting and action selection;
shadow diagnostics are timed separately. Additional compute is an outcome of
the intervention, not a matched resource budget. Do not compare these timings
directly with historical compiled controllers.

The GPU smoke run covers all four new conditions with the full update budget
and four real decisions. It verifies realized update counts, complete candidate
banks, frozen outer state, scoring parity, and unchanged actions when shadow
diagnostics are enabled. Production output has separate immutable task owners;
the CPU publisher validates results and installs a separate W&B comparison
view in `ambi-inner-bench`. Partial progress is visible; final arm summaries
require all 20 seeds.
