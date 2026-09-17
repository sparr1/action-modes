# Automatic inner temperature at 625k

`ambi_aux_closed_loop_625k_auto_alpha.json` unfreezes the inner temperature in
the matching inherited-alpha condition. Every real decision starts alpha at
the checkpoint's saved value, 0.004345251712948084, and creates a fresh inner
Adam state. The saved outer optimizer and coefficient remain frozen.

Use `inner_temperature_mode=auto`, initialization and target entropy
`inherit_outer`, and temperature learning rate 0.0003 (the same resolved rate
as the fixed condition, now active). The inherited target is -21 and uses
squashed-action entropy. The canonical component schedule performs C32 critic
updates, then A4 actor updates, each with one temperature update: T4 per solve.
No state from this solve carries to the next real decision.

Everything else matches: J1/N128/H1/B256, soft-Q critic initialization and frozen
soft-Q outer bootstrap, mean-action execution, five 500-decision episodes with
environment seeds 101–105 and controller 55, and 32 independent paired model
probes before and after the round. At H1 the finite-horizon critic boundary
target has no additional entropy bonus. Alpha controls the inner actor's
entropy term; the probes keep the reward-plus-soft-Q score.

Set `AMBI_ALPHA_MODE=auto` for the GPU and CPU launchers and create a new
evaluation-series attempt. Reuse verified prior, alpha-zero, and inherited
fixed-alpha results. Workers and the merger validate a saved-value alpha reset
at every decision, exactly four temperature steps, positive updated alpha,
actual coefficient changes, complete episode/probe coverage and unchanged
outer state. Receipts distinguish initial alpha from mean final alpha.
