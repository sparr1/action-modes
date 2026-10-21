# Entropy and KL checkpoint curves

The MPPI auxiliary-return backbone comparison has four original persistent W&B runs:
sampled prior, entropy-only, KL-only, and entropy plus KL. A fifth, orange MPPI
curve can be explicitly added with its own pinned planner identity. Their IDs and owner
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

Repeated prepare calls reuse exactly the same run IDs. An extended profile must
retain the four original identities exactly and add `identities.mppi`; prepare
then appends only the missing MPPI run. Existing sources, records, curve IDs and
colors remain unchanged. Use that extended profile for later staging/publication;
the historical four-method profile cannot silently remove the added curve.
Install the native
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

For a matched MPPI comparison, generate a campaign-local extended profile from
the immutable original profile and the MPPI source registry. Register campaign
cells with `mode="mppi"`, `step`, `run_id` and `run_dir` before compute finishes;
no `outer_alpha` is required. For example:

```bash
python slurm/ambi_entropy_kl_curves.py prepare --profile EXTENDED_PROFILE --campaign MPPI_CAMPAIGN
python slurm/ambi_entropy_kl_curves_layout.py --catalog CATALOG --receipt WORKSPACE_RECEIPT
python slurm/ambi_entropy_kl_curves.py publish --profile EXTENDED_PROFILE
```

Install the new catalog receipt at its root's `workspace.json` before publishing.
MPPI pending checkpoint counts are visible immediately; only complete published
five-seed source records become measurements. Gains are recomputed against the
existing sampled prior with exact checkpoint hashes and matching episode RNG
seeds. No prior episode is rerun. MPPI's own source fingerprint and action rule
remain explicit; they are not relabeled as the SAC protocol. Return, paired-gain
variability and controller timing share the existing panels. Inner actor,
temperature and prior-KL diagnostics are omitted for MPPI, rather than shown as
zero, and its raw results and compact provenance are retained.

## Retrace comparisons

The `retrace_kl` (purple) and `retrace_both` (green) modes add separately
identified Retrace SAC curves to the same workspace. They preserve every
existing run ID, measurement, color and panel. A campaign-local extended profile
retains all existing identities and adds each Retrace identity after applying
`canonical_identity(raw_identity, mode, outer_alpha)`. Campaign source cells
include `mode`, `step`, `run_id`, `run_dir` and `outer_alpha`.

The fixed beta rule remains one times checkpoint outer alpha; estimator,
Retrace lambda, trajectory batch size, entropy setting, actor and critic update
counts, seeds, execution rule and source fingerprint remain part of the exact
identity. Retrace curves cannot accept one-step SAC records. KL-only disables
inner entropy, while the joint curve retains it. Sampled-prior returns at the
same checkpoint and paired seeds are reused to compute gains.

If a Retrace curve is present, the layout retains the existing 17 panels and
adds three measured critic diagnostics: successor trace coefficient, effective
trace length and absolute target correction. They are averaged over critic
updates, then decisions and evaluation episodes. The exact normalized keys are
`diagnostics/inner_retrace_trace_coefficient_mean_mean`,
`diagnostics/inner_retrace_effective_trace_length_mean`, and
`diagnostics/inner_retrace_correction_abs_mean_mean`. Other controllers have no
synthetic zeros in these panels. Entropy, alpha, policy KL, return, paired-gain
variability, controller timing and available Retrace work totals remain in the
published records. `panel_count(catalog)` reports 20 for the extended layout;
four- and five-controller catalogs retain 17 panels.
