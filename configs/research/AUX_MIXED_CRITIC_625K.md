# Soft inner critic initialization with return-only terminal bootstrap

Select `critic/soft_init_return_tail` in the 625k closed-loop matrices. Each
real decision starts the inner critic from the saved SAC soft-Q weights
(`inner_critic_source=sac`), while the frozen terminal bootstrap uses the
auxiliary return-only critic (`inner_horizon_critic_source=aux_return`).
The inner target critic starts from the selected online critic. Both actor
sources remain the saved SAC policy.

The September 17 comparison uses the `aux6434715x3` seed55 backbone at625k,
with the same J1/N128/H1/C32/A4/B256 recipe, fresh learners and replay at every
real decision, mean-action execution, and five500-decision episodes with
environment seeds101–105 and controller55. Reuse the verified frozen-prior
episodes and earlier same-checkpoint conditions.

| Policy entropy | Matrix | `AMBI_ALPHA_MODE` | Temperature updates |
|---|---|---|---|
| Disabled, alpha0 | `ambi_aux_closed_loop_625k.json` | `zero` | 0 |
| Automatic, initialized from saved outer alpha | `ambi_aux_closed_loop_625k_auto_alpha.json` | `auto` | 4 |

Set `AMBI_CRITIC_MODE=soft_init_return_tail` for both GPU workers and the CPU
publisher. Generate a separate New curve specification and attempt per alpha
condition using the explicit selector. The inherited fixed-alpha matrix also
exposes this critic selector, but it is not part of this two-condition launch.

Automatic alpha starts at0.004345251712948084 at every decision, with a fresh
optimizer, learning rate0.0003 and squashed target entropy−21. Disabling
entropy forces effective alpha to zero and performs no temperature updates.

The inner critic's **initialization** is soft Q. Its subsequent fitting still
uses the established reward-only finite-horizon targets and return-Q boundary;
turning policy entropy on does not change those critic targets. Retain32
independent reward-plus-return-Q probes at initialization and after refinement.
Validate full outer-state immutability, actual source routing, alpha/update
counts, complete paired coverage and strict compilation before publication.
