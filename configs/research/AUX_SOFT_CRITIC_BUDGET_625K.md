# Soft/soft horizon and critic-budget comparison at 625k

This extends the original closed-loop H/J table with H={1,2,3}, J={2,4,8},
and C={8,16} critic updates per round. Eighteen new settings use five full
500-decision episodes each, environment seeds101–105 and controllerseed55.
The question is whether a longer horizon can reach useful returns with less
critic fitting or fewer refinement rounds, and how lower C changes H/J scaling.

Use the existing seed55 auxiliary-return backbone checkpoint625k, SHA256
`b91650597b585805e5e2c28fffb721e5aa3737b2a032360304e86aa81ff02ca3`.
Both inner initialization and outer bootstrap use soft Q. Interior fitting is
entropy-augmented, outer terminal entropy is enabled, and actor alpha is
inherited and adaptive. Keep N128/B256/A4, critic-first order, actor and critic
learning rates3e-4, target tau.01, mean-action execution, original compiled
execution, and all imagined data across rounds. Capacity3072 accommodates H3/J8.
Every real decision starts fresh from the checkpoint actor/critic and alpha.

The existing C32 results are historical references, not repeated experiments.
J2/J4 references come from the original H/J campaign at ea4892b; J8 references
come from the original-protocol extension at28964ee. Strict identity checks
permit only the critic update count and its resolved total, plus replay capacity
where all data are retained on both sides. Historical science identities and
paired environment/controller seeds are preserved. The prior is also reused.

Tau stays fixed deliberately. At H>1, decreasing C also reduces target-network
updates per round. The screen estimates that complete practical intervention,
not isolated critic approximation error. Learning rates, tau compensation,
actor update changes, return-only critics, and J16 are outside this screen.

Retain real returns and paired gains over the prior and matched C32 reference,
full update traces, per-decision summaries, exact work counts, alpha, and the
32-trajectory fixed-noise model probes before adaptation and after every round.
Model probes remain predictions rather than observed MC continuation returns.
GPU workers save immutable bundles; independent CPU publishers log performance
and training runs, including `comparison/c32_gain_*` and true C metadata.

Use `ambi_aux_soft_critic_budget_625k.json` with the shared campaign launcher,
`--baseline-kind critic_budget`, `--baseline-campaign` pointing to the original
H/J campaign, and `--baseline-extension-campaign` pointing to the J6/J8 campaign.
Smoke checks cover H3/J8 at both C values before production. Preserve the original
numerical mode and use available supported GPUs; this is not a hardware latency
optimization study. Schedule longer panels first with concurrency based on live
quota. Compare real return versus J, total critic updates J*C, actor updates4*J,
and imagined transitions128*H*J; no single axis represents all changed work.

Five evaluation seeds, one controller seed and one training checkpoint provide
exploratory comparisons. Historical mixed-device execution variability remains
a limitation, especially for small differences. Confirm promising effects in a
separate matched repeat before treating them as established scaling behavior.
