# Inner entropy and Q normalization comparison

`experiments/ambi_inner_entropy.json` defines six Humanoid Walk training runs.
All use seed 55, 1,000,000 agent decisions, the TD-MPC2 policy parameterization,
one five-head reward-only critic ensemble, and identical outer learning.

| Actor entropy | Normalized coefficient | Inner Q scale |
|---|---:|---|
| Native TD-MPC2 pre-tanh statistic | 0.0001 | Copy outer, freeze |
| Native TD-MPC2 pre-tanh statistic | 0.0001 | Copy outer, adapt |
| Native TD-MPC2 pre-tanh statistic | 0.0001 | Carry across decisions, adapt |
| True joint post-tanh entropy | 0.0021 | Copy outer, freeze |
| True joint post-tanh entropy | 0.0021 | Copy outer, adapt |
| True joint post-tanh entropy | 0.0021 | Carry across decisions, adapt |

The post-tanh coefficient is 21 times 0.0001, matching the native entropy
statistic's extra action-dimension multiplier. Inner and outer entropy modes
match. Temperature is fixed; no entropy target or scale compensation is used.

Every real decision performs an H3/J8/K1 solve: eight rounds, 128 imagined
rollouts per round, 16 critic and four actor updates per round, batch 256,
and 3,072 replay entries. Both targets remain reward-only. The finite-horizon
inner protocol and frozen-prior terminal bootstrap are retained. Actor,
critic, replay and their optimizers are freshly inherited at each solve.

Outer normalization is P95-P5 with floor one and EMA rate 0.01. The adapting
inner arms update the same statistic once per actor update, also at 0.01.
`inner_actor_loss_scale_scope=action` resets from the latest outer estimator;
`run` initializes from outer only on the first solve and then retains its own
estimate across episode boundaries. Inner adaptation never writes the outer
estimator. Exact training-state serialization includes the carried scalar;
evaluation reset starts a fresh chain. Held decisions, when used outside this
K1 campaign, do not update it.

All 40 portable model checkpoints per run are retained at 25,000-decision
intervals, with outer normalization and metadata sidecars. These model banks
support evaluation and transfer rather than exact mid-run continuation.

`slurm/run_inner_entropy_oscar.sbatch` runs on Oscar L40S GPUs. Each arm first
passes a W&B-disabled GPU gate with the actual model and inner budget, comparing
one and two CPUs, checking normalization behavior, finite diagnostics,
compilation, GPU replay, and checkpoint save/reload. The gate sizes the pending
production job from measured throughput with a 25% plus 30-minute margin,
updates all CPU resource fields while held, verifies them, and releases the
job. Its `afterok` dependency still prevents training until validation exits
successfully. No arbitrary training concurrency cap is imposed.

Compilation, vectorized critics and TF32 matmul are enabled. The experimental
whole-solve CUDA graph is not used: it does not support this Q-normalization
protocol. Scratch quota is checked against the full six-run checkpoint bank
plus 30% and 2 GiB; caches and W&B staging also stay off the home filesystem.
