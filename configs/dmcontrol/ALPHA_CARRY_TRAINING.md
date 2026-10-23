# Full-training temperature-carry factorial

`experiments/ambi_alpha_carry_training.json` defines eight fresh Humanoid Walk
runs. This is a single-seed exploratory screen of whether sustained inner
temperature adaptation improves end-to-end learning with the soft primary and
reward-only auxiliary critics.

| Factor | Settings | Meaning |
|---|---|---|
| Inner temperature lifetime | `episode`, `run` | Both carry log-alpha and its Adam state across decisions. `episode` resets both at episode boundaries; `run` carries both across episodes. |
| Auxiliary return representation gradients | shared, detached | Detached blocks the auxiliary return TD loss from updating the encoder and latent dynamics; its critic heads still train. |
| Target entropy | −10.5, −21 | The selected value applies to both the outer and inner SAC actors. Their learned temperatures remain independent. |

Every cell uses H3/J8/C16/A4/N128/B256, model size 5, distributional critics,
vmap, strict CUDA compilation, seed 55, 2,500 random warmup decisions and 2,500
pretraining updates. The actor, critic, their Adam states and imagined replay
reset to fresh prior initialization at each decision. Inner temperature starts
from the current outer temperature at the beginning of its lifetime. An episode
reset therefore uses the **current** outer alpha, not a fixed initial constant.

Outer SAC retains its entropy-augmented critic. Inner SAC uses reward-only
targets and the auxiliary return critic for both initialization and the terminal
bootstrap. Its terminal actor is the frozen SAC prior. Training executes sampled
actions, and imagined rollouts also sample actions. There is no Bernoulli
transfer, prior KL, Retrace or actor/critic writeback. All actor, critic and
temperature learning rates stay at 3e−4.

Detachment affects the auxiliary TD training loss only. It does not detach Q
from action during inner actor optimization. The shared outer optimizer and
joint gradient clipping remain unchanged, so this factor does not isolate all
possible indirect interactions with the auxiliary critic.

## Budget, storage and reporting

Each run has 1,000,000 **decisions**, equivalent to 2,000,000 raw environment
steps at action repeat 2. Episodes last at most 500 decisions. The manifest
retains all 40 periodic model checkpoints, every 25,000 decisions, with their
metadata sidecars and a **complete resident replay snapshot at that same
decision** (`checkpoint_replay: true`). Each retained model has its own
step-matched replay; these are not references to a mutable latest replay file.
The replay option is a top-level checkpoint setting in both the algorithms and
manifest. This retained checkpoint bank is distinct from exact training resume
state, which additionally requires the environment, wrapper and optimizer state
handled by the training-state mechanism.

The requested Oscar allocation is one L40S for four days per run. CPU and memory
requests must be selected from the campaign's training benchmark and current
account limits; the algorithm configurations do not prescribe resource counts.
Four days is the allocation limit, not a guarantee that 1M decisions finishes
within it. The launcher drains before that limit and verifies a resumable
handoff if the target has not been reached; it does not automatically requeue.

This campaign stores large outputs under Oscar scratch at
`/oscar/scratch/rgao48/ambi/alpha-carry-training/<campaign>/`, including native
training-state generations and all model/replay snapshots. This differs from the
older home-only launcher's storage policy because home quota is insufficient
for the eight checkpoint banks. The launcher checks the scratch quota and
filesystem writes. Scratch is subject to the cluster's retention policy; these
files must be preserved elsewhere before their retention window expires.
Native resume generations publish separately every 15 minutes at an episode boundary,
and W&B publishes the committed metric prefix. Thus dashboard data can lag
training by approximately that interval. The same run ID is retained on exact
continuation. To continue, use the same source/configuration and campaign path
with `AMBI_RESUME_MODE=required`. These rolling resume generations do not replace
the full replay snapshot retained at **every** 25,000-decision model checkpoint.

W&B project `ambi`, group `ambi-alpha-carry-full-20261023`, contains native
per-run curves with all three factors in the display names and tags. Existing
training returns, outer/inner alpha, entropy and optimization metrics remain
enabled. Root entropy uses 128 paired samples before/after inner adaptation at
the existing 1,000-decision diagnostic cadence. Root entropy and sampled
minibatch entropy measure different state distributions and must be interpreted
separately. Frozen-policy evaluation remains disabled, matching the source
full-training recipe; training return is not a paired estimate of improvement
over the same checkpoint's base policy.

The eight settings have one shared training seed. Use them to identify promising
mechanisms and failure modes, then repeat selected settings with independent
seeds before making a variance or robustness claim. A common seed does not keep
the trajectories identical once the controllers make different decisions.

## Manifest ordering

Indices 0–3 use episode reset and 4–7 use run carry. Within each block, shared
representation gradients precede detached gradients; target −10.5 precedes
−21. Config basenames encode those factors:

```text
ambi_alpha_carry_h3_j8_{episode|run}_{shared|detached}_ent{10p5|21}
```

The manifest is used with `--alg-dir configs/dmcontrol/algs`. Production launch
must use the campaign GPU validation gate rather than invoking all eight cells
implicitly from a login shell.
