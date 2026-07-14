# Native SAC distributional-Q ablation

The native SAC implementation can keep its standard twin-critic structure while
changing each critic from a scalar output to categorical logits over symlog Q
bins. The soft Bellman target, clipped double-Q minimum, entropy temperature,
and actor objective are otherwise unchanged.

The distributional critic is selected with:

```json
{
  "q_representation": "distributional",
  "q_num_bins": 101,
  "q_vmin": -10,
  "q_vmax": 10
}
```

`q_vmin` and `q_vmax` bound the **symlog-transformed** target. Targets outside
the support are clipped, and `q_target_clip_fraction` reports how often this
happens. `q_distribution_entropy` and `q_distribution_max_probability` help
diagnose categorical collapse. Both critic distributions are decoded to scalar
values before the SAC target minimum and actor loss are evaluated.

Run the short smoke experiment with:

```bash
python3 main.py \
  --run configs/experiments/AntNativeDistributionalSACDebug.json \
  --alg-dir configs/algs
```

Run the matched scalar-versus-distributional experiment with:

```bash
python3 main.py \
  --run configs/experiments/AntNativeSACQRepresentation.json \
  --alg-dir configs/algs
```

This comparison runs both variants on the same five seeds (55 through 59).

The paired experiment deliberately uses two critics in both variants. Increasing
the distributional ensemble size should be tested separately so representation
and ensemble capacity are not changed in the same ablation.

This is a categorical/two-hot Q parameterization trained from the usual scalar
soft-SAC target. It should not be interpreted as a calibrated distribution over
future returns.
