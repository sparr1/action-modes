# Prior-KL entropy mitigation diagnostic

`evaluate_ambi_prior_kl.py` compares complete frozen H3/J8 inner solves and
fixed-critic actor interventions on the 50 saved roots from the entropy probe.
The source checkpoint pair is SAC-aux 100k and MPPI-aux 400k. Each has seeds
101–105 and pre-action roots 0, 10, 50, 100 and 250. No source episodes are
recollected and no outer training or W&B publication occurs in this evaluator.

The baseline retains reward-only inner critic targets, auxiliary return critic
initialization and terminal bootstrap, inherited automatic temperature, and
32 actor/temperature updates. The mitigation replaces the entropy bonus with
`beta * KL(adapted policy || frozen inherited primary actor)`, with beta equal
to the checkpoint's inherited outer alpha and held fixed. It disables inner
entropy and allocates no inner temperature parameter or optimizer. Critic
learning, H3/J8, C16/A4, N128/B256, model and all outer parameters remain fixed.

KL is the analytic joint diagonal-Gaussian KL before tanh. The shared invertible
tanh map makes this equal to the corresponding squashed-policy KL. The anchor
is the inherited actor at each state, not a uniform distribution. Both arms
still sample policy actions internally and return the adapted mean for root
diagnostics. These root probes do not measure closed-loop return.

Each pair uses the same diagnostic seed and verifies identical initial actor,
first replay, post-first-16-update critic, fixed anchors and anchor noise. Later
full-solve replay differs as policies adapt. A separate pair freezes that first
replay and post-16-update critic, disables critic dropout, and applies 32 actor
updates with identical minibatches, Gaussian noise and critic head pairs.

The full solve records entropy, analytic KL, Gaussian standard deviation,
means and saturation on the same real root and fixed imagined-depth anchors
after every actor update (including every round and final policy). Gradient
projections use the actual stochastic Q/entropy forward plus analytic
deterministic KL, followed by the actual Adam parameter change. They are local
first-order measurements, not counterfactual return guarantees.

`ambi_prior_kl_probe.json` pins the original bank content IDs, checkpoint and
metadata hashes, original profile hash, and original source commit b62b0c5.
The evaluator checks the original adjacent `source.json` and retains original
identity separately from the new science identity. It never rewrites an old
bank's source hash to force reuse. MPPI seed 102 uses its completed `seed102-retry`
bank. Use the original full bank even for `--decisions 0` smoke evaluation.

```bash
python evaluate_ambi_prior_kl.py \
  --checkpoint-id sac_aux_100k --checkpoint /path/to/checkpoint \
  --metadata /path/to/checkpoint.metadata.json --environment-seed 101 \
  --root-bank /path/to/original/seed101/root-bank.json \
  --output /new/immutable/result/directory --device cuda --torch-threads 1
```

`--plan-only` validates checkpoint and bank provenance without loading the model.
Result bundles preserve per-root records and verify outer model/optimizer and
global CPU/CUDA RNG preservation. Aggregate roots within each seed before
computing uncertainty across the five environment seeds. The pair consists of
two independently trained backbones and is a mechanism diagnostic rather than
a comparison across independent training seeds.

## Sampled full-episode follow-up

`ambi_prior_kl_episodes.json` and `slurm/ambi_prior_kl_episodes.py` define the
separate behavior test: five seeds 101–105, controller seed 55, 500 decisions,
H3/J8, and sampled action execution matching training. The two selectors are
restricted by the campaign launcher to the pinned SAC-aux 100k and MPPI-aux
400k checkpoints. Their fixed coefficients are 0.003365497337654233 and
0.013081684708595276 respectively, the saved outer temperatures verified by
the root probes. This is an explicit experiment coefficient choice; it does
not automatically track alpha if used in a changing training backbone.

Compatible sampled prior and entropy-baseline episodes from the early H3/J8
campaign are reused. The evaluator checks prior reference compatibility and
stores paired improvements. The validation receipt requires 128 critic and
32 actor updates per decision, zero temperature updates, complete streamed
traces, unchanged outer state, and no compilation fallback. Timing excludes
separately measured model-probe overhead. This sampled protocol is distinct
from the named mean-action `closed-loop-refinement-v1` recipe.
