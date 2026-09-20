# Soft/soft rollout-count and minibatch-size screen at 625k

Run H={1,3}, J8/C16/A4 with (N,B)={(32,256),(128,64),(32,64)}.
These are six new conditions, five paired full 500-decision episodes each.
Reuse the H1/H3 J8/C16 N128/B256 panels from the completed critic-budget campaign
at fe87ae07b2a7ed751cd6865a3f60b2eae88e6abb, including their existing W&B IDs.
The eight-condition overview links those references without rerunning episodes,
creating replacement run IDs, or uploading duplicate diagnostic artifacts.

Only rollout count and the shared actor/critic batch size vary. Retain checkpoint
625k of aux6434715x3, seeds101–105/controller55, all replay (capacity3072),
round collection then critic-first updates, soft inner and outer critics,
entropy-augmented fitting, outer terminal entropy, inherited adaptive alpha,
learning rates3e-4 and tau.01. Horizon conditioning and its optional diagnostics
stay off. Use original strict compiled execution; historical mixed-device
comparisons remain exploratory. No J/H extension or stepwise update test is
part of this campaign.

`ambi_aux_rollout_batch_625k.json` contains only the six new evaluator selectors.
The shared H/J launcher's preparation dispatch verifies complete reference
bundles, checkpoint/protocol/planner identities and trace hashes. It allows only
N/B differences and explicitly pins the historical scientific source. Six new
indices0–5 are followed by two reused overview rows6–7. Workers reject reused
production cells; the smoke selects the three new H3 shapes only.

Keep per-update/per-decision diagnostics and frozen-model probes. Publish paired
prior improvement plus `comparison/n128_b256_gain_*` on training diagnostic runs.
Record actual N/B, optimizer totals, model transitions and sampled replay rows.
C/(NH) measures critic updates per imagined transition; CB/(NH) measures sample
presentations per imagined transition. B changes both actor and critic batches.
Neither ratio alone measures compute or guarantees sufficient data diversity.

Prepare with `slurm/ambi_aux_hj_sweep.py prepare --matrix
configs/research/ambi_aux_rollout_batch_625k.json --baseline-campaign <completed
C8/C16 campaign>`, plus the established checkpoint/inventory/reference/registry
arguments. Submit only the generated `production_indices`. No new prior runs.
