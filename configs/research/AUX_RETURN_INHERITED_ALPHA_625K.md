# Inherited outer temperature at 625k

`ambi_aux_closed_loop_625k_inherited_alpha.json` changes only
`inner_entropy_enabled` from the matching alpha-zero matrix. The same checkpoint,
soft-Q critic initialization and frozen soft-Q terminal bootstrap are retained.
The saved outer log-temperature is -5.438671588897705, giving alpha
0.004345251712948084. Its optimizer state is also present in the checkpoint.

The inner actor uses the squashed-policy objective `E[Q_inner - alpha*log pi]`.
`inner_temperature_mode=inherit_outer` copies the saved coefficient at every real
decision and holds it fixed: zero temperature optimizer steps. This differs from
`inner_temperature_mode=auto` with `inner_temperature_initialization=inherit_outer`,
which would initialize from the same value and then fit a separate inner alpha.
The latter is not part of this condition.

At H1 all modeled transitions use the frozen outer boundary, with no additional
entropy bonus in that critic target. The actor entropy term is now enabled.
Model probes retain their original reward-plus-soft-Q score without an entropy
bonus, so they remain comparable to the alpha-zero diagnostics.

Use J1/N128/H1/C32/A4/B256, five 500-decision episodes with environment seeds
101–105 and controller55, and32 probes at initialization and after the round.
Reuse the matching prior and completed alpha-zero panel. Workers verify both
initial and final inner-alpha statistics against the saved checkpoint value,
full probe coverage, no temperature updates and unchanged frozen outer state.

Set `AMBI_ALPHA_MODE=inherit_outer` for the existing GPU and CPU launchers.
Allocate a new evaluation-series attempt; do not append to the alpha-zero run.
