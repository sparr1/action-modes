# Critic learning-rate screen at 625k

Four new soft/soft full-episode closed-loop refinement settings on checkpoint
625k of aux6434715x3: H3/J8/N128/B256/A4, C8 or C16 crossed with critic learning
rates 0.0006 and 0.001. Reuse both matching completed 0.0003 controls from the
critic-budget campaign at fe87ae07b2a7ed751cd6865a3f60b2eae88e6abb, preserving
their result bundles and original W&B performance/training IDs.

Actor and temperature LR stay 0.0003. Keep inherited adaptive alpha, soft inner
critic initialization, entropy-augmented critic fitting, frozen soft terminal
bootstrap with outer entropy, tau0.01, critic-first round timing, strict
compilation and full replay capacity3072. Every real decision starts a fresh
prior-initialized learner. No horizon conditioning or writeback is introduced.

Each setting runs seeds101–105/controller55 for five full500-decision episodes:
20 new episodes total, plus10 reused control episodes and the existing prior.
Each decision collects3072 imagined transitions, performs32 actor/temperature
updates and64 or128 critic/target updates. Minibatch presentations total24576
at C8 and40960 at C16. Learning rate does not change these counts or the target
interpolation schedule. Compare each rate against the control with the same C.

The campaign verifies checkpoint, protocol, historical source, immutable bundle
and trace hashes, complete pairing, and planner equality apart from critic LR.
W&B retains paired improvement versus the prior and adds
`comparison/critic_lr_3e4_gain_*`, full raw traces, per-update/per-decision
training diagnostics and frozen-model probes at initialization and every round.
Mixed-device historical comparisons remain exploratory.

Prepare through `slurm/ambi_aux_hj_sweep.py prepare --matrix
configs/research/ambi_aux_critic_lr_625k.json --baseline-campaign <completed
critic-budget campaign>` with the standard checkpoint/inventory/prior/registry
paths. Only production indices0–3 are new work. The shared GPU launcher's
`--smoke-index` mode checks one selected cell in each task of a parallel0–3
smoke array. Release production only after all four actual-checkpoint checks
pass. Reused controls are excluded from the smoke.
