# Soft inner critic with return-only terminal bootstrap at 575k

The selected backbone is `ambi/aux6428346x0`, target −10.5/shared, checkpoint
575,000 with SHA256
`0c6955db7cb8555a67d7863344b70be68f4b3250814d131e647ee6f9ef01a042`.
This mixed-objective screen changes the inner critic initialization to the SAC
soft critic and uses entropy-augmented in-horizon critic targets. The terminal
critic remains the frozen `aux_return` critic, with no explicit terminal entropy:

| Setting | Value |
| --- | --- |
| `inner_critic_source` | `sac` |
| `inner_horizon_critic_source` | `aux_return` |
| `inner_sac_critic_target` | `entropy_augmented` |
| `inner_terminal_entropy` | `none` |
| actor and horizon actor | `sac` |

The sampled matrix selects **36 settings**: H1/H2/H3 × J1/J2/J4/J6/J8/J10 ×
one-step/Retrace(λ=0.9). Every setting runs five full 500-decision episodes,
environment seeds101–105 and controller seed55. Adaptation starts fresh from
the inherited actor/critic/temperature at every real decision; outer checkpoint
state remains frozen. Final execution samples `tanh(mu + std * epsilon)` with
unit standard-deviation scale from the final adapted actor and its isolated
execution RNG. Actor entropy stays adaptive, inherited α=0.004603903274983168
and target −10.5.

All other controls match the reward/reward screen: N128, C16/A4 per round,
critic-first updates, actor/temperature batch256, retained replay within each
decision, and capacity3072 for J≤8 or3840 for J10. Retrace critic fitting uses
256/128/86 whole trajectories at H1/H2/H3 (256/256/258 nominal rows); one-step
fitting uses256 independent transitions. Trajectory sampling and RNG behavior
are part of the estimator recipe. H1 has no intermediate imagined transition
before the return-only terminal boundary; do not infer full-run equivalence
from a target-level simplification.

The versioned reference file pins54 completed reward/reward results: 18
one-step mean, 18 one-step sampled and18 Retrace sampled settings. Preparation
verifies source commits, manifest/trace hashes, publication IDs, complete paired
episodes and resolved scientific controls. Historical results and prior-only
episodes are never rerun. Inactive historical Retrace fields normalize only to
their documented defaults.

The principal comparison is **soft-inner/return-terminal minus reward/reward**
at identical H, J, estimator, execution mode and episode seeds. It changes the
inner initialization and in-horizon target objective together; it does not
isolate just one critic term. Within the new campaign, Retrace-minus-one-step
is reported separately. Missing outcomes remain absent until both complete
paired results exist. The overview retains full episode provenance, partial
progress and exact training/probe artifacts. Error bars use episode sample SD;
paired intervals use2000 seed bootstrap resamples. This remains exploratory
evidence from one trained backbone.

`slurm/ambi_closed_loop_soft_return.py prepare` accepts `--root`, `--checkpoint`,
`--inventory`, `--references`, `--registry`, optional `--matrix`, `--group`,
`--label`, and `--execution policy_sample` (default). The matching sampled JSON
is selected automatically. `worker --root --index` owns all five episodes for
one setting. Indices0–5 are J10 boundaries for all H/estimator pairs and support
`--smoke` for one seed and three decisions. The Oscar launcher supports the same
administrative environment as the reward screen, plus `EVAL_EXECUTION`.

An explicit `--execution both` selects the separate72-setting matrix, including
mean execution `tanh(mu)` for every cell and12 boundary smokes. This capability
does not add mean runs to the default launch. If used, the overview also pairs
sampled-minus-mean within each mixed-critic estimator. There is no historical
reward/reward mean Retrace reference, so its cross-critic comparison remains
absent rather than using a different estimator or execution mode.
