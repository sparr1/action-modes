# Critic LoRA at 575k: horizons and inner-round budgets

This frozen-checkpoint screen compares critic-only LoRA with the existing dense
inner-SAC controller. It uses the same checkpoint and protocol as the September
2026 ERE screen, but retains uniform replay sampling so the LoRA effect is
separate from ERE. The actor continues to adapt densely.

The matrix is `ambi_closed_loop_critic_lora_575k.json`; immutable dense controls
are pinned in `ambi_closed_loop_critic_lora_575k_refs.json`. The driver verifies
their hashes, configurations, episode coverage, and publication identities
before creating new evaluation runs.

| Setting | Value |
|---|---|
| Backbone | `ambi/aux6428346x0`, shared return auxiliary critic, target entropy −10.5 |
| Checkpoint | 575,000 decisions; SHA256 `0c6955db7cb8555a67d7863344b70be68f4b3250814d131e647ee6f9ef01a042` |
| Horizons H | 1, 2, 3 |
| Inner rounds J | 1, 2, 4, 6, 8, 10 |
| LoRA ranks | 16, 96 |
| Placement | Input and hidden critic matrices; output heads, biases, normalization train normally |
| Adapter | Direct scale 1, AdamW factor decay 0.0002, inherited critic learning rate 0.0003 |
| Initialization | Inherited critic plus fresh random A and zero B at every real decision |
| Actor | Fresh inherited dense actor, learning rate 0.0003 |
| Collection / updates | N128, B256, C16 then A4 per round |
| Replay | Uniform, retained throughout each solve; capacity max(3072, 384J) |
| Critic objective | One-step finite-horizon reward-only targets; auxiliary return critic for both adaptation and frozen terminal bootstrap |
| Entropy | Adaptive temperature initialized from the checkpoint; squashed actor entropy |
| Execution | Adapted actor mean, fresh solve at every real decision |
| Evaluation | Five paired environment seeds 101–105, controller seed 55, full episodes up to 500 decisions |

The 36 new settings require 180 full episodes. All 18 matching dense settings
and the frozen-prior episodes are reused; they retain their original run
identities. Each new controller has its own performance curve and diagnostic
run. The campaign overview compares return and paired gain over both dense
adaptation and the prior as J increases, separately for each H and rank.

Rank 16 tests a strong restriction on critic updates; rank 96 tests the broader
adapter already supported by the implementation. This screen holds learning
rate and adapter scaling fixed. It therefore compares these concrete recipes,
not independently optimized ranks. These five evaluation seeds on one trained
checkpoint are exploratory evidence, not independent training replications.

Use `slurm/ambi_closed_loop_critic_lora.py prepare` on a CPU allocation to verify
the checkpoint and references and create a new campaign. Four short GPU smoke
cells cover both ranks at H1/J1 and H3/J10 before production is released. Submit
workers with `slurm/run_ambi_closed_loop_critic_lora_oscar.sbatch`, pinning the
tested source SHA. GPU workers only write local results; one CPU publisher
owns publication through `slurm/ambi_closed_loop_critic_lora_publish.py`.
Both outer state and frozen checkpoint remain unchanged.
