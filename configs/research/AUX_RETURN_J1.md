# Auxiliary-return J1 checkpoint benchmark

`ambi_aux_return_j1.json` evaluates the auxiliary SAC target -21 detached,
seed55 backbone (`rwgao_b-brown-university/ambi/aux6434715x3`) on the normal
checkpoint-curve protocol: checkpoints every25k through1M, environment seeds
101–105, controller seed55, and full500-decision mean-action episodes.

The inner setting is J1/N128/H1/C32/A4/B256: each decision begins with fresh
copies of the checkpoint actor and selected online critic, a target critic
initialized from that online critic, fresh optimizers and replay. Collect128
one-step imagined transitions, perform32 critic updates followed by4 actor
updates, and execute the adapted actor mean. Both learning rates are0.0003;
replay capacity is2048 with replacement. Alpha is fixed at zero, outer learning
and writeback are disabled, and critic dropout remains enabled. Targets are
finite-horizon reward-only targets with frozen outer terminal bootstrapping.
The current auxiliary engine represents exact alpha zero with
`inner_entropy_enabled=false`; the inherited temperature is inactive and no
temperature optimizer is created. The launcher verifies measured alpha and
temperature updates are exactly zero.

Two variants change both the inner critic initialization and terminal critic:

- `critic/soft_q`: SAC soft-Q for initialization and terminal bootstrap.
- `critic/return_q`: auxiliary return-only Q for initialization and bootstrap.

Both use the same frozen SAC actor for initialization and terminal sampling.
The soft-Q variant still has a soft-Q terminal value, despite reward-only inner
fitting and zero inner alpha; do not describe that terminal value as return-only.
The checkpoint's clipped log-std parameterization and unscaled Q are inherited.
No historical TD-MPC2 numeric scale or tanh log-std mapping is substituted.

This borrows the inner optimization setting from the historical refinement
study; it uses the five-seed checkpoint benchmark, without the study's20×3
seed panel or32-sample per-round model-return probes. Standard inner traces,
control timings, standalone HTML and full-episode returns remain available.

Pass each checkpoint's existing verified frozen-prior bundle through
`--reference-bundle`. This records seed-paired AMBI-minus-prior gain in the
initial W&B publication. Reuse compatible prior episodes. Create one new
evaluation curve per critic setting, then assign both explicitly with
`--eval-run-map`. GPU workers stage completed results; existing CPU publishers
own W&B sessions. `slurm/run_ambi_aux_return_j1_oscar.sbatch` supports a short
strict-CUDA smoke via `--smoke`, followed by the full paired evaluation.
