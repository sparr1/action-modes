# Legacy frozen-checkpoint presets

`ambi_inner_decoupling.json` is the previous one-axis matrix used by
`evaluate_ambi_checkpoint.py`. It evaluates checkpoint-compatible inner-loop
changes against a frozen outer model and is intentionally separate from the
new end-to-end suite.

Do not use this matrix for the primary horizon, breadth/depth, optimizer, batch,
or round comparisons. Those trials must start independently from
`configs/ambi/experiments/` so their actions, replay, world models, critics, and
policies may diverge naturally.

Legacy usage remains:

```bash
python evaluate_ambi_checkpoint.py --list-presets
```
