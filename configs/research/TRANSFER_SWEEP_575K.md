# H3 actor-only and critic-only transfer sweep at 575k

This full-episode evaluation crosses J={1,8}, fresh/actor-only/critic-only
initialization, soft/soft versus return/return critics, and solve intervals
1 and H=3. It contains **24 distinct configurations**, each with five paired
environment seeds 101–105 and controller seed 55: 120 ordinary 500-decision
episodes before any verified result reuse. A separately evaluated frozen-prior
reference adds five episodes, giving **25 configurations and 125 episodes** in
the full campaign. The eight introductory critic
examples in [CRITIC_TRANSFER_575K.md](CRITIC_TRANSFER_575K.md) remain unchanged.

The backbone is `rwgao_b-brown-university/ambi/aux6428346x0` at 575,000 training
decisions. All matrices require its checkpoint and sidecar through the existing
`checkpoint_contract`: step 575000 and SHA-256
`0c6955db7cb8555a67d7863344b70be68f4b3250814d131e647ee6f9ef01a042`.
Actor parameterization, target entropy −10.5, representation and outer model
are inherited from this same trained checkpoint.

## Reproducible matrices

| Matrix | Protocol | Default selected modes | Default configurations |
|---|---|---|---:|
| `ambi_critic_transfer_sweep_575k.json` | `critic-transfer-v1` | Fresh, critic-only; solve every decision | 8 |
| `ambi_critic_transfer_hold_h_sweep_575k.json` | `critic-transfer-hold-h-v1` | Fresh, critic-only; solve every H decisions | 8 |
| `ambi_actor_transfer_sweep_575k.json` | `actor-transfer-v2` | Actor-only; solve every decision | 4 |
| `ambi_actor_transfer_hold_h_sweep_575k.json` | `actor-transfer-hold-h-v1` | Actor-only; solve every H decisions | 4 |
| `ambi_transfer_prior_reference_575k.json` | Ordinary frozen evaluation | Frozen SAC prior mean, no inner optimization | 1 |

Each matrix has groups `soft_soft_j1`, `soft_soft_j8`, `return_return_j1` and
`return_return_j8`, with `fresh` as the reference in each group. Critic matrices
select both `fresh` and `critic_warm`; actor matrices select only `actor_warm`.
For example, `return_return_j8/critic_warm` identifies the carried return critic
at J8 in the relevant cadence matrix.

Fresh selectors also exist in the actor matrices so their comparisons can be
evaluated independently. They are **excluded from those matrices' defaults**:
the sweep evaluates each fresh condition once through the critic matrix and
uses that same matching result for both warm-start comparisons. A campaign
must enumerate `evaluation.default_presets`, not every available selector,
to preserve the 24-configuration workload.

The separate reference matrix selects `reference/prior` on the same checkpoint,
five environment seeds and controller seed. It uses mean actions, action-local
state and interval1, with no transfer or to-go probes. Evaluating this inexpensive
reference under the current source provides a matching baseline without
assuming that an older implementation's artifact is compatible.

## What changes and what stays matched

| Mode | Actor at each solve | Online critic at each solve | Target at each solve |
|---|---|---|---|
| Fresh | Checkpoint prior | Selected checkpoint critic | Copy of starting online critic |
| Actor-only | Previous adapted actor; checkpoint prior at episode start | Selected checkpoint critic | Copy of starting online critic |
| Critic-only | Checkpoint prior | Previous adapted online critic; selected checkpoint critic at episode start | Copy of starting online critic |

Only one online network is retained in either warm arm. Replay, temperature,
optimizer moments and optimizer step counts reset at every solve. Carried
network lifetime counts accumulate within the episode as diagnostics; the new
critic-only target-update cadence starts at zero on each solve. Both networks
are dense clones; rebasing and outer writeback are disabled. The outer model
and terminal actor/critic remain frozen, and episode/checkpoint resets clear
all carried values and held-policy state.

Soft/soft selects the SAC critic for both inner initialization and the terminal
bootstrap, uses entropy-augmented inner targets and the outer entropy correction
at the horizon. Return/return selects the auxiliary return critic for both,
uses reward-only inner targets and no terminal entropy correction. Both use
the same SAC actor, squashed-action entropy, and temperature reset to the
checkpoint value followed by adaptation toward the inherited target.

All settings use H3, N128, C16/A4 per round, B256, learning rates 0.0003,
critic-first updates, target tau 0.01, sampling with replacement, replay capacity
3072 and strict compilation. J is uniform at every solve, including the first;
there is no special initialization dose. J1 collects 384 imagined transitions
and performs 16 critic plus four actor/temperature updates per solve. J8
collects 3072 transitions and performs 128 critic plus 32 actor/temperature
updates; its replay exactly accommodates all eight rounds without eviction.

Real execution uses mean feedback actions. Interval1 solves at all 500
decisions. Interval3 solves at 0,3,6,…,498 (167 solves); the fixed adapted actor
evaluates each fresh observation between solves without learning. Its last
block contains two decisions. Report both per-solve and amortized per-decision
time, since holding changes the amount of optimization over the episode.

## Results and comparisons

The primary measurement is raw undiscounted full-episode reward. Compare each
warm arm with the **matching fresh condition at the same critic type, J and
cadence**, then compare actor-only with critic-only. Report J1 versus J8 and
every-decision versus hold-H as distinct compute comparisons, with paired
per-seed returns, gains over the separately evaluated frozen prior and measured
control time. Five environment seeds and one
trained backbone make this an exploratory study; individual real decisions
are not independent replicates.

Every inner-SAC condition enables neutral transfer diagnostics and 32 paired
model-return probe rollouts. Retain initialized, first-critic, first-actor and
post-round measurements, transfer/lifetime counters, held-decision counts and
separate diagnostic time. Probe values are model predictions rather than
calibrated full-episode values. The study does not train the outer model or
replace its terminal critic.

GPU smoke checks use short episodes and separate output directories; they are
not substituted for full results. The CUDA fixture gate separately checks
eager/compiled numerical parity with critic dropout disabled and exact
reproducibility between two compiled controllers with critic dropout enabled.
Eager CUDA and Inductor may generate different dropout masks from the same seed,
so stochastic eager/compiled weight equality is not a valid parity criterion.
Both fixture paths check critic carry-over, target and optimizer resets,
allocation reuse, held decisions, episode boundaries and checkpoint loads.
These fixture choices do not change the production dropout configuration.

Resume only complete, identity-verified
results or previously submitted work for the same campaign, preserving outputs
and avoiding duplicate evaluations. The historical actor-transfer launchers and
publishers remain unchanged. New campaign outputs use their own directories;
the generic evaluator writes bundles and JSON without automatically publishing
W&B results.
