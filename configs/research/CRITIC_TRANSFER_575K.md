# Critic-only transfer at 575k

The opt-in [hidden-layer transfer variant](CRITIC_HIDDEN_TRANSFER_575K.md)
retains critic hidden layers while randomly resetting its output head at each
solve. It uses separate protocols and matrices; these full-critic examples
continue to retain the complete critic.

The separate [H3 transfer sweep](TRANSFER_SWEEP_575K.md) extends these examples
to J1/J8 and actor-only controls using 24 distinct default configurations.
The eight examples and their defaults below remain unchanged.

These frozen-checkpoint examples compare a fresh inner critic with an online
critic carried between solves. The actor always starts each solve from the
frozen SAC prior. They use the auxiliary-return SAC backbone
`rwgao_b-brown-university/ambi/aux6428346x0` at 575,000 training decisions:
target entropy −10.5, directly clamped actor log standard deviation, and shared
representation gradients during backbone training. The checkpoint and its
metadata sidecar must match the matrix's `checkpoint_contract`: step 575000 and
SHA-256 `0c6955db7cb8555a67d7863344b70be68f4b3250814d131e647ee6f9ef01a042`.

The two matrices contain eight examples in total:

| Matrix | Protocol | Solve cadence |
|---|---|---|
| `ambi_critic_transfer_575k.json` | `critic-transfer-v1` | Every real decision |
| `ambi_critic_transfer_hold_h_575k.json` | `critic-transfer-hold-h-v1` | Every H real decisions |

Each matrix has the same four selectors:

| Selector | Online critic at solve start | Inner fitting | Frozen terminal bootstrap |
|---|---|---|---|
| `soft_soft/fresh` | Frozen SAC soft critic | Entropy augmented | SAC soft critic with outer entropy correction |
| `soft_soft/critic_warm` | Previous solve's adapted SAC critic | Entropy augmented | SAC soft critic with outer entropy correction |
| `return_return/fresh` | Frozen auxiliary return critic | Reward only | Auxiliary return critic, no entropy correction |
| `return_return/critic_warm` | Previous solve's adapted return critic | Reward only | Auxiliary return critic, no entropy correction |

Every matrix defaults to **only `return_return/critic_warm`**. Selecting a whole
comparison runs its fresh and carried-critic arms and reports their paired
differences. Soft and return comparisons have separate fresh references: their
critic initialization, training objective and terminal continuation change
together, while the actor and entropy settings remain matched.

## State lifetime

For `critic_warm`, `inner_actor_scope="action"` and
`inner_critic_scope="episode"`. At the first solve of an episode, the online
critic starts from its selected frozen checkpoint critic. At later solves,
only that adapted online critic is retained. The **target critic is copied
from the current online critic at every solve**, then follows the ordinary
within-solve target updates. The previous solve's lagged target is discarded.

Replay, actor/critic optimizer moments and step counters, temperature and
temperature optimizer reset at every solve. The critic's cumulative update
count is retained within the episode for diagnostics; per-solve update counters
and target-update cadence restart at zero. The actor is freshly initialized from the
frozen SAC prior even when the critic is retained. Actor and critic are dense
clones; persistent rebasing, joint actor-and-critic transfer, replay carry-over
and prior writeback are outside this protocol. The outer encoder, dynamics,
reward model, actors and critics remain frozen, including the terminal critic.
Episode boundaries, evaluation resets and checkpoint loads clear transferred
state and any held policy. Evaluation may reuse tensor allocations to preserve
compiler identities, but all expired scientific values and counters reset.

The hold-H matrix uses `inner_solve_interval=H=3`. Solves occur at decisions
0, 3, 6, … . Between solves, the adapted feedback actor evaluates its mean action
at each fresh real observation, with no collection or learning. At the next
solve, its actor weights reset and the online critic is retained only in the
`critic_warm` arm. These are feedback actions, not repeated actions or an
imagined open-loop action sequence. The final block of a 500-decision episode
contains two decisions.

## Matched evaluation settings

All examples use H3, J1 at every solve including the first, C16/A4/N128/B256,
replay capacity 3072, sampling with replacement, learning rates 0.0003,
critic-first updates, target tau 0.01 and strict compilation. Actor entropy is
squashed-action entropy; temperature starts from the checkpoint value at each
solve and adapts toward the inherited target −10.5. Retaining critic weights
does not retain temperature. Each solve collects 384 imagined transitions and
performs 16 critic and four actor/temperature updates.

Five paired environment seeds 101–105 use controller seed 55 and ordinary
500-decision Humanoid Walk episodes. Real execution uses the final actor's
mean action. The primary outcome is raw undiscounted full-episode reward.
`evaluation.transfer_diagnostics=true` retains initialization, first-critic,
first-actor and post-round probes, with 32 paired model-return rollouts. These
probes are diagnostics, not calibrated predictions of full-episode reward.
Bundle results distinguish solve and held decisions and separate diagnostic
time from control time.

The legacy `evaluation.actor_transfer_diagnostics` spelling remains supported.
For critic-only runs, `inner_critic_transferred` is one only on solves after the
first solve of an episode; `inner_critic_target_reinitialized` is one on every
solve. Both are zero on held decisions. `inner_critic_updates_initial` records
the cumulative critic updates before an actual solve (zero on held decisions),
and `inner_critic_lifetime_updates` reports the cumulative episode count after
the decision. Existing optimizer-step metrics report only that solve's work.

These examples test the online-critic initialization change. They do not
establish improved return or speed. Paired uncertainty must treat environment
seeds as the independent clusters; five seeds and one trained backbone support
an exploratory comparison only. Existing results may be reused only after
checkpoint, protocol, configuration, seeds and scientific implementation pass
the normal compatibility checks.

## Use through the frozen evaluator

List selectors without loading the checkpoint or running an episode:

```bash
python3 evaluate_ambi_checkpoint.py \
  --matrix configs/research/ambi_critic_transfer_575k.json --list-presets
```

On an authorized compute allocation, supply the matching checkpoint and adjacent
metadata sidecar and a new output directory. For example, evaluate the matched
return-only pair:

```bash
python evaluate_ambi_checkpoint.py \
  --matrix configs/research/ambi_critic_transfer_575k.json \
  --checkpoint /path/to/step_575000.pt \
  --comparison return_return \
  --device cuda \
  --bundle-dir /path/to/new-critic-transfer-bundle \
  --output /path/to/new-critic-transfer-results.json
```

Use `--preset return_return/critic_warm` for only the default carried-critic
example, or `--comparison soft_soft` for the soft-critic pair. Substitute
`ambi_critic_transfer_hold_h_575k.json` for the hold-H comparison. Full-episode
transfer diagnostics require a bundle; shared-root model-only bank evaluation
is a different procedure and is not accepted by these protocols.

These commands write local result bundles and JSON. The historical campaign
launchers and publishers are unchanged; these examples do not automatically
submit a job, create a campaign, or publish W&B results.
