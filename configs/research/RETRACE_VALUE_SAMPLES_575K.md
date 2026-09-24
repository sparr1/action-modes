# Retrace value-action averages at the 575k checkpoint

[`ambi_closed_loop_retrace_values_575k.json`](ambi_closed_loop_retrace_values_575k.json)
defines 48 fresh full-episode closed-loop settings: H2/H3, J1/2/4/6/8/10,
and independent interior/boundary action sample counts `(1,1)`, `(4,1)`,
`(1,4)`, `(4,4)`. Every setting uses Retrace **lambda 1.0**. The previous
lambda-0.9 campaign remains separate; its results are not reused as controls.

The frozen checkpoint is step 575,000 from
`rwgao_b-brown-university/ambi/aux6428346x0`, with SHA-256
`0c6955db7cb8555a67d7863344b70be68f4b3250814d131e647ee6f9ef01a042`.
It is the target-entropy -10.5/shared-representation SAC backbone with auxiliary
return critics. Every real decision initializes a fresh inner actor from SAC
and inner critic from the auxiliary return critic. The horizon boundary uses
the frozen SAC policy and auxiliary return critic. Both critic targets contain
reward only; actor entropy and inherited automatic temperature adaptation remain
active. Execution samples the final adapted squashed Gaussian.

The fixed compute recipe is N128 imagined trajectories per round, C16 critic
and A4 actor/temperature updates, a 256-transition actor minibatch, and
`ceil(256/H)` complete trajectories per critic minibatch. The critic supervises
every valid suffix with Retrace. Replay retains every round: capacity 3072 for
J1–J8 and 3840 for J10. All component lifetimes remain action-local. Increasing
the value sample counts adds batched policy/Q evaluations without changing the
rollout, replay, or optimizer-step budgets.

Each setting owns five complete 500-decision Humanoid Walk state episodes,
with environment seeds 101–105 and controller seed 55. This gives 240 episodes;
32 independent to-go probes run initially and after each inner round. Pairing
uses equal environment and solver seeds. Candidate returns are compared with
the freshly evaluated `(1,1)` control at the same H/J and lambda, only after
both sides finish. This is a five-episode screen on one trained checkpoint.

The controller is `slurm/ambi_closed_loop_retrace_values.py`; its companion
Oscar wrapper selects separate CPU preparation/publication and GPU workers.
Preparation creates 48 distinct generic evaluation-series identities and no
historical reference bundles. The worker order is J10, J8, J6, J4, J2, J1;
within each round budget it runs H2 then H3 and the four pairs in the order
above. Array indices 3 and 7 are the three-decision GPU smokes for H2/H3,
J10, `(4,4)`; production uses all indices 0–47 after smoke validation.

Preparation and workers require a clean checkout at the tested commit. Workers
verify the checkpoint/source, matrix hash, sample counts, replay retention,
exact update budgets, sampled execution, absence of compilation fallback,
finite results, frozen outer state, complete episode pairing identifiers,
and raw trace/probe coverage before writing completion receipts. Failures
produce separate failure receipts and cannot be published as completed results.

The CPU publisher owns performance, optimizer/temperature diagnostics, and the
campaign overview. It exposes eight return-versus-J curves and paired gains
against `(1,1)`, preserving missing results as missing. Resource requests,
array concurrency, and publisher concurrency are chosen at submission from
live Oscar capacity; the launcher contains no fixed array throttle.
