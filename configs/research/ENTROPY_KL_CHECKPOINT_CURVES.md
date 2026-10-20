# Entropy and KL checkpoint curves

The MPPI auxiliary-return backbone comparison has four persistent W&B runs:
sampled prior, entropy-only, KL-only, and entropy plus KL. Their IDs and owner
directories are stored in
`/oscar/scratch/rgao48/ambi/evaluation-series/mppi-entropy-kl-v1/catalog.json`.
The saved workspace remains
<https://wandb.ai/rwgao_b-brown-university/ambi-inner-bench?nw=3w61ds5ze92>.
Each method is a native solid line against `checkpoint/training_decisions`,
with a stable color and normal per-run hover/legend interaction.

## Scientific compatibility

`ambi_mppi_entropy_kl_curves.json` pins the complete identity for each curve:
backbone `ambi/mppi7530927x0`, scientific-source fingerprint, episode protocol,
and resolved planner. All adapted evaluations use sampled actions, H3/J8,
C16/A4/N128/B256, five seeds101–105, controller seed55 and500 decisions.

KL beta follows the rule **one times the checkpoint's saved outer alpha**, fixed
throughout adaptation. This rule is the sole normalization of the source
identity: numeric beta remains recorded per point and in its source reference.
Every other identity field must match exactly. Different backbone runs,
temperature learning rates, beta multipliers, action rules, seeds, or controller
settings require a separate comparison. Duplicate attempts at one checkpoint
are rejected instead of averaged.

Historical sampled-prior bundles supply100k,150k,200k and400k references.
Their return semantics match the500k prior, but their compilation and diagnostic
settings differ; the combined prior curve contains return and paired-gain
metrics only. Adapted gains are recomputed from the same checkpoint and paired
environment/solver seeds. Existing deltas are checked when present.

## Publication

Raw campaign runs and artifacts are retained. The persistent curves refer back
to their original record IDs, identity hashes, checkpoint hashes, journals and
bundle receipts. Only compact provenance and normalized scalar results are
uploaded; large diagnostic traces are not duplicated.

Prepare the catalog once, from a clean synchronized checkout on Oscar:

```bash
python slurm/ambi_entropy_kl_curves.py prepare
```

Repeated prepare calls reuse exactly the same four run IDs. Install the native
layout from the catalog using the isolated `wandb==0.28.1` and
`wandb-workspaces==0.4.5` runtime:

```bash
python slurm/ambi_entropy_kl_curves_layout.py --catalog CATALOG --receipt WORKSPACE_RECEIPT
```

Copy this generated receipt to the catalog root's `workspace.json`. Publish
through a short CPU allocation; no evaluation needs rerunning:

```bash
python slurm/ambi_entropy_kl_curves.py publish --require-complete
```

For subsequent compatible checkpoints, explicitly register their raw campaign
sources, then append to these same curves:

```bash
python slurm/ambi_entropy_kl_curves.py publish --campaign CAMPAIGN_JSON
```

The MPPI500k campaign publisher performs this registration and append after
publishing its raw results. Its CLI requires the persistent catalog, preventing
another per-checkpoint-only dashboard. New campaign launchers should use this
same final publication call and catalog instead of allocating display runs per
checkpoint. Layout verification is mandatory before publishing; missing or
changed panels cause a visible error. The existing `eval_series` journal and
owner lock provide idempotent uploads and interrupted-publication recovery.
