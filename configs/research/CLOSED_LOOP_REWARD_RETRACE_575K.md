# Reward/reward sampled execution and Retrace at 575k

This screen uses the target −10.5/shared backbone `ambi/aux6428346x0` at
575,000 training decisions, checkpoint SHA256
`0c6955db7cb8555a67d7863344b70be68f4b3250814d131e647ee6f9ef01a042`.
Both the initial critic and the finite-horizon terminal critic use `aux_return`;
critic targets are reward-only and contain no explicit terminal entropy.
The actor retains inherited adaptive entropy, initially α=0.004603903274983168.

`ambi_closed_loop_reward_retrace_575k.json` selects 27 new settings:

- Sampled one-step execution at J6/J8/J10 for each H1/H2/H3: nine settings.
- Sampled Retrace with λ=0.9 at J1/J2/J4/J6/J8/J10 for each H1/H2/H3: 18 settings.

Every setting runs seeds101–105, controller seed55, for five full 500-decision
episodes. At each real decision the actor, critic, target, temperature,
optimizers and imagined replay reset to their inherited initialization. The
final action is an unscaled sample `tanh(mu + std * epsilon)` from the final
adapted policy, using the isolated execution RNG. The frozen outer checkpoint
and its training state remain unchanged. This is a sampled-execution variant of
the full-episode closed-loop refinement procedure.

All settings use N128, C16/A4 per round, critic-first updates, B256 for actor
and temperature fitting, and retained imagined replay within each decision.
One-step critics sample 256 individual transitions per update. Retrace critics
sample whole trajectories with replacement: 256/128/86 trajectories for H1/H2/H3,
respectively, giving 256/256/258 nominal critic rows. This trajectory batching and
collection RNG behavior are part of the estimator recipe; H1 targets reduce to
one-step targets, but this does not imply identical full closed-loop runs.

Replay capacity stays matched to the historical round budget: 3072 transitions
for J≤8 and 3840 for J10, for both estimators and all horizons. All collected
rounds are retained; H3/J10 exactly fills capacity. Per decision, J10 performs
160 critic, 40 actor and 40 temperature updates, and collects 1280/2560/3840
transitions at H1/H2/H3. Existing compiler, frozen-state, exact-work, replay and
32-rollout probe guards remain active.

The versioned references file pins 18 historical mean-execution results and
nine completed one-step sampled J1/J2/J4 results. Preparation verifies manifest
and trace hashes, source commits, publication IDs, full episodes and scientific
settings before creating new performance identities. No historical episode or
prior-only baseline is rerun. Inactive historical Retrace defaults normalize
only to estimator `one_step`, λ=1 and unspecified trajectory batch.

The combined overview includes all historical points and validated new points.
Its paired comparisons are distinct:

- **Retrace minus sampled one-step**, matched by H, J, environment seed and
  solver seed, compares estimator recipes. A pairing remains absent until both
  complete results exist, including one-step results produced by this campaign.
- **Sampled estimator minus historical one-step mean** also changes execution
  for Retrace; it is explicitly labeled separately and is not a prior-relative
  improvement.

Error summaries use episode sample standard deviations and paired bootstrap
intervals from five seeds. These are exploratory results from one trained
backbone. Returns alone do not identify a mechanism such as model error.

`slurm/ambi_closed_loop_reward_retrace.py prepare` accepts a new `--root`,
`--checkpoint`, `--inventory`, and evaluation-series `--registry`; `--matrix`
and `--references` default to these versioned files. `worker --root --index`
owns all five episodes of one setting. Indices0–5 are the six H/estimator J10
boundaries; each supports `--smoke` for one seed and three decisions.
`slurm/ambi_closed_loop_reward_retrace_publish.py watch --root` is the single
CPU overview owner and publishes completed performance runs plus complete
inner-training/probe artifacts. Its progress table keeps missing evaluations
and publication failures explicit. Existing campaign helpers are unchanged.
