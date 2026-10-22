# Ten-episode temperature persistence test

This experiment tests whether the relative performance of cross-episode alpha
carry keeps falling over a longer sequence. It runs ten fresh sampled-action
episodes for both persistent temperature and a matched episode-reset control.
It is not a mean-action experiment and does not extend an old trace by attaching
new episodes to a reconstructed temperature state.

| Arm | Temperature and Adam state |
|---|---|
| `episode_reset` | Carry across decisions; initialize checkpoint alpha and fresh Adam at every episode |
| `run_carry` | Carry continuously across all decisions and ten episodes; initialize only at the first episode |

Each arm has five independent initialized histories. The MPPI-trained
auxiliary-return checkpoint at 500,000 decisions, source
`ambi/mppi7530927x0`, remains frozen. Settings retain H3/J8/C16/A4/N128/B256,
sampled imagined behavior and real actions, target entropy -10.5, and actor,
critic and temperature learning rates 0.0003. Each decision performs 128 critic,
32 actor and 32 temperature updates. Actor, critic, their optimizers and replay
reset every decision. There is no KL, Bernoulli copying, Retrace, additional
temperature work, policy writeback or change to the engine/evaluator.

The protocol is `alpha-long-episodes-v1`, campaign
`alpha-long-episodes-20261022`. Selectors are
`alpha_long_episodes/episode_reset` and `alpha_long_episodes/run_carry`. New cells
0–4 are episode reset, and 5–9 are continuous carry. A paired GPU allocation
runs reset cell `i` followed by carry cell `i+5` on the same GPU and host.

## Seeds and workload

Every history contains ten distinct environment seeds. Its first five seeds
are a rotation of 101–105, and its last five a rotation of 106–110, using the
same stream offset in both blocks:

| History | Episodes 1–5 | Episodes 6–10 |
|---|---|---|
| 0 | 101, 102, 103, 104, 105 | 106, 107, 108, 109, 110 |
| 1 | 102, 103, 104, 105, 101 | 107, 108, 109, 110, 106 |
| 2 | 103, 104, 105, 101, 102 | 108, 109, 110, 106, 107 |
| 3 | 104, 105, 101, 102, 103 | 109, 110, 106, 107, 108 |
| 4 | 105, 101, 102, 103, 104 | 110, 106, 107, 108, 109 |

The first three seeds preserve the earlier experiment's prefix for each
history, but all ten episodes are newly evaluated. Each episode runs all 500
decisions. Controller seed 55 and SHA256-v1 RNG initialization remain paired
by environment seed. Total new work is 10 histories, 100 full episodes and
50,000 decisions. The carry arm executes 160,000 temperature updates per
history; the reset arm executes 16,000 before each episode reset.

## Analysis

Episode ten is the primary endpoint. At each of the ten episode positions,
report the five paired carry-minus-reset return differences, their mean,
sample standard deviation, individual outcomes and wins/losses. Plot both
arms' return and temperature/entropy trajectories over all ten episodes.
Report a descriptive late contrast for each paired history:

`mean(carry - reset, episodes 6–10) - mean(carry - reset, episodes 1–5)`.

A negative contrast indicates a smaller relative carry advantage in the later
block. The two blocks use different environment-seed sets; pairing controls
for their direct baseline difficulty, but heterogeneous treatment effects
still limit a causal interpretation of the time trend. Episodes within a
history are dependent, and seeds recur across different histories. Do not
pool 50 returns as independent trials or claim formal noninferiority from
five initialized histories.

The previous three-episode results are suggestive rather than proof of harm:
episode-three reset minus carry was about +27.2 return with paired sample
standard deviation 79.9, with only two of five pairs favoring reset. Equivalent
first-episode protocols also produced substantial long-horizon return
variation despite nearly identical early decisions. The new analysis reports
first-episode matching differences descriptively; numerical identity of full
episode returns is not a validation requirement.

This is frozen-backbone policy-improvement evaluation. It cannot establish
whether exploration improves subsequent outer-model training. Near-target
entropy also does not establish preserved policy improvement; return and
entropy must be assessed together.

## Validation, resources and publication

Preparation verifies the checkpoint and inherited alpha, matrix/profile hashes,
and the completed five-history short preservation campaign at
`/oscar/scratch/rgao48/ambi/alpha-preservation/20261022-b315ca33/campaign.json`.
Those old results are independently revalidated and pinned only as checkpoint
provenance and measured runtime sources. No old three-episode performance
curve is expanded into a ten-episode control. Both long arms are new.

The shared full-result validator accepts ten ordered unique seeds and checks
all actual optimizer work, probes, finite metrics, sampled execution, episode
lengths and unchanged outer state. Existing lifecycle validators check every
alpha and Adam boundary. Summaries retain the actual lifecycle protocol
`episode-temperature-reset-v1` or `run-temperature-chain-v1` and add the long
campaign identity; receipts use `alpha-long-episodes-v1`. Protocol constants
are never patched globally.

Alpha, actor-update minibatch entropy, reward and runtime curves use existing
recorded traces. Optional root-to-prior KL, actor tanh-saturation, critic-head
disagreement and TD-error metrics are copied only where already measured,
with explicit coverage counts and no zero imputation. Actor-update entropy
is not the final executed root-policy entropy. No new probes or RNG draws are
added.

Each selected GPU class first runs both arms for two eight-decision episodes.
Production timing uses those measurements plus pinned observed three-episode
runtime scaled to ten episodes, with an explicit margin. Five paired GPU
allocations use one CPU and 4 GiB each. Both cells of a pair retain separate
exclusive output directories and validated receipts.

```bash
python slurm/ambi_alpha_long_episodes.py prepare --root "$ALPHA_LONG_ROOT"
python slurm/ambi_alpha_long_episodes.py run \
  --campaign "$ALPHA_LONG_ROOT/campaign.json" --index 0
python slurm/ambi_alpha_long_episodes.py run \
  --campaign "$ALPHA_LONG_ROOT/campaign.json" --index 5
```

The dedicated comparison has two actual sampled-execution methods. Final
paired performance curves wait for all ten cells to validate, while pending
histories and failures remain visible. The complete ten-episode trajectory,
episode-ten primary comparison and late-minus-early paired advantage are
published without changing the earlier workspaces.
