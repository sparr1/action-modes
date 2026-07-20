# Native SAC distributional-Q notes

The active comparator is
`configs/ambi/algs/native_sac_distributional_five_q.json`.

The native SAC implementation supports a distributional Q ensemble. The active
comparator uses five heads to match the size-5 TD-MPC2 and AMBI critics. A
random pair of two heads is selected per Q query, and `min_pair` is applied to
the Bellman target and actor objective.

The distributional critic is selected with:

```json
{
  "q_representation": "distributional",
  "num_q": 5,
  "q_pair_size": 2,
  "q_target_reduction": "min_pair",
  "q_actor_reduction": "min_pair",
  "q_num_bins": 101,
  "q_vmin": -10,
  "q_vmax": 10
}
```

`q_vmin` and `q_vmax` bound the **symlog-transformed** target. Targets outside
the support are clipped, and `q_target_clip_fraction` reports how often this
happens. `q_distribution_entropy` and `q_distribution_max_probability` help
diagnose categorical collapse. All critic distributions are decoded to scalar
values before the SAC target minimum and actor loss are evaluated.

Run the short consolidated smoke experiment with:

```bash
python3 main.py \
  --run configs/ambi/experiments/smoke/smoke_algorithm_wiring.json \
  --alg-dir configs/ambi/algs
```

Run the canonical one-seed exploratory comparator with:

```bash
python3 main.py \
  --run configs/ambi/experiments/canonical/native_sac.json \
  --alg-dir configs/ambi/algs
```

This checked-in run uses seed 55 only and is labeled exploratory in its
manifest. The older scalar-versus-twin-distributional ablation remains under
`configs/experiments/` for historical use and explicitly pins `num_q=2`; it is
not the active five-head comparator.

This is a categorical/two-hot Q parameterization trained from the usual scalar
soft-SAC target. It should not be interpreted as a calibrated distribution over
future returns.
