# Split-value prior backbone screen

This exploratory Humanoid Walk state screen trains three independent prior-only
backbones from seed 55. Each runs for 2 million agent decisions and keeps all
80 scheduled checkpoints at 25,000-decision intervals, with adjacent metadata
sidecars. It supplies checkpoint banks for later frozen-backbone evaluations.

| Index | Algorithm basename | Outer alpha | Return scale S |
|---|---|---|---|
| 0 | `ambi_prior_split_clip_fixed0p0021` | Fixed 0.0021 | Moving percentile range |
| 1 | `ambi_prior_split_clip_target10p5` | Automatic, initial 0.005; target −10.5 | 1 |
| 2 | `ambi_prior_split_clip_target21` | Automatic, initial 0.005; target −21 | 1 |

The two automatic-alpha arms differ only in their target entropy. Comparing
either against the fixed-alpha arm also changes the normalization system and
initial temperature. This single-seed screen does not establish a statistically
confirmed winner. The fixed arm retains an inactive `target_entropy=-21` field;
it does not optimize a temperature loss.

## Shared learning recipe

- The actor clips predicted log standard deviations to `[-10, 2]` and uses the
  joint entropy of the squashed action distribution. There are no smooth-map
  arms and no entropy decay schedules.
- `critic_value_mode="return_entropy"` gives each of five independent ensemble
  members a shared MLP trunk and separate ordered `return`, `entropy` categorical
  heads. A sampled pair selects one member by composed policy value, then uses
  both of that member's components. All configured reductions are `min_pair`.
- Reward and value components use mean-preserving symexp two-hot regression
  with 101 bins over symlog support coordinates `[-10, 10]`. The critic loss
  averages its two cross-entropies. Thus `critic_coef=0.2` produces
  **`0.1 * return_loss + 0.1 * entropy_loss`** under the existing temporal and
  ensemble reductions. Both heads train the encoder and dynamics.
- Return and entropy targets contain no alpha or normalization coefficient.
  The actor minimizes `alpha * log_pi - Q_return / S - alpha * Q_entropy`.
  Alpha and scale are frozen at each update's entry and refreshed for the next
  update. Only the fixed-alpha arm updates the return scale; automatic-alpha
  arms use `S=1`.
- The 5M model starts from scratch, with zero final split-head weights and
  biases. Training retains H=3, batch size 256, UTD=1, discount 0.99, rho=0.5,
  replay capacity one million, and explicit 2,500-decision warmup and 2,500
  pretraining updates. Production uses CUDA and strict compilation; compilation
  fallback is treated as a failure.
- Collection samples the outer prior. Inner optimization, MPPI, writeback,
  explorers, auxiliary value-equivalence training, and online policy evaluation
  are disabled. Dormant inner settings are not a frozen-checkpoint adaptation
  recipe. Later evaluation can independently select the supported inner entropy
  and return/soft initialization semantics.

The fixed coefficient 0.0021 reflects the 21-dimensional action scale in the
TD-MPC2 entropy convention (`21 * 1e-4`). It is an exploratory choice, not an
exact equivalence: this screen uses squashed-action entropy and explicitly
models future entropy.

## Diagnostics and entry point

Outer-policy diagnostics use a private RNG and the existing fixed-observation
bank: 32 states, 32 samples per state, every 100 decisions through decision
10,000 and every 1,000 thereafter. W&B uses event-indexed logging in project
`rwgao_b-brown-university/ambi` and group
`ambi-prior-split-value-20260915`. Config-derived names identify the arm and
seed. Split training also reports component losses, decoded values, target
clipping, composed policy value, and the coefficients actually used.

The manifest is `configs/dmcontrol/experiments/ambi_prior_split_study.json`.
It overrides only top-level fields and preserves each algorithm's full
`alg_params` mapping. For one scheduled cell, use:

```bash
environments/dmcontrol/.venv/bin/python main.py \
  --run configs/dmcontrol/experiments/ambi_prior_split_study.json \
  --alg-dir configs/dmcontrol/algs \
  --alg-index 0 --trial-index 0 --num-runs 1
```

Indices 0–2 follow the table above. Run the command on an allocated compute
node, after the focused configuration and CUDA validation gates. The Oscar
launcher requests an L40S per cell; keep checkpoints and their metadata in
designated output storage and record the tested source commit with the run.

The [implementation guide](../../RL/tdmpc2_core/README.md#split-return-and-entropy-critics)
describes the equations, strict checkpoint semantics, and supported inner
adaptation choices. These production recipes are distinct from the smaller
`ambi_split_value_entropy_*` demonstration configs.
