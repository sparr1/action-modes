# Return-only inner critic and terminal bootstrap at 625k

Use the existing `ambi_aux_closed_loop_625k_auto_alpha.json` matrix with
`--preset critic/return_q`. This changes both `inner_critic_source` and
`inner_horizon_critic_source` from `sac` to `aux_return`. The checkpoint's
auxiliary critic estimates reward return under the SAC prior. The actor copied
into the solve and the frozen horizon actor both remain `sac`.

Keep the completed soft-Q auto-alpha condition's other settings: fresh J1/N128/
H1/C32/A4/B256 adaptation at every real decision; alpha initialized from saved
outer 0.004345251712948084, four temperature updates at learning rate0.0003,
squashed target entropy-21, fresh optimizers/replay, mean-action execution, and
five full500-decision episodes (environment seeds101–105, controller55).

The critic targets are reward-only with a frozen return-Q boundary. The actor
still has its explicit entropy bonus because automatic alpha remains enabled.
The32 independent probes before and after refinement use reward-plus-return-Q
scores; their absolute values have different semantics from soft-Q-tailed
probes. Compare controller performance using paired real-episode returns.

Set `AMBI_ALPHA_MODE=auto` and `AMBI_CRITIC_MODE=return_q` for both launchers.
The Python campaign helper exposes the same selection as `--critic-mode return_q`.
Generate a new series specification using the return-Q selector and a new
attempt labeled `aux625k-return-init-return-tail-j1-auto-alpha-20260916`.
Workers, merge receipts and the publisher bind the critic selection explicitly.
Reuse the existing prior and soft-Q auto-alpha episodes after checkpoint,
protocol and implementation checks. Only the five return-Q episodes are new.
