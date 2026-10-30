# Critic smoothness: matched H3/J10/K3 training

This four-cell exploratory screen trains fresh humanoid-walk agents for
1,000,000 decisions (2,000,000 raw control steps), using seed 55. Each baseline
has an MPR-only cell and an action-gradient-consistency-only cell.

| Baseline | Reference W&B run | Critic receiving the penalty | Inner temperature lifetime |
|---|---|---|---|
| Single return critic | `de37a4eec8094c67a2a691db04e47a2b` | Primary reward-return critic | Reset from outer temperature each solve |
| Detached auxiliary return critic | `774f452fdda045aaa835dec670cb3584` | Both primary soft critic and auxiliary reward-return critic | Temperature and Adam state persist across solves |

Both retain H3/J10/K3, C16/A4/N128/B256, five distributional critic heads,
stochastic training actions, target entropy -10.5, 2,500 warmup decisions,
2,500 pretraining updates, and one outer update per decision. The first
reference retains its 500,000-row replay capacity; the second retains
1,000,000 rows, its soft primary SAC critic, and representation-detached
auxiliary return learning. The second initializes inner critics and horizon
continuation from the auxiliary return critic. Actors and critics initialize
fresh each solve in both baselines. There is no transfer of their parameters
or optimizer state between solves.

The frozen source run parameters and original commits are recorded in
`critic_smoothness_sources.json`. The first source predates root-entropy
diagnostics, so that observation is explicitly disabled there; the second
retains its 128-sample diagnostic. Other non-smoothness algorithm parameters
are preserved. Training starts from scratch; reference checkpoints are not
loaded.

## Losses and perturbations

MPR compares critic action gradients at the original observation and at one
Gaussian-perturbed observation with per-coordinate standard deviation 0.01.
Perturbations occur before encoding. Action smoothness compares gradients at
the same latent state and two actions: the recorded replay action and that
action plus Gaussian noise with per-coordinate standard deviation 0.02,
clamped to [-1,1]. Both action inputs are independent autograd leaves.

Each active term has coefficient 0.004 per selected critic; all other PAVE terms
are zero. The auxiliary baseline regularizes both its soft primary critic and
its auxiliary return critic. TD's
coefficient is 0.1 and its head reduction is a mean; the regularizer sums its
five per-head losses. Therefore the coefficient ratio per head is
`0.004 / (0.1 / 5) = 0.2`. This is a coefficient convention, not a measured
loss-value or parameter-gradient ratio. TD retains rho 0.5 and divide-horizon
temporal weighting; regularizer rows are weighted uniformly.

Use every transition in the TD replay sample: 256 sequences times 3 steps,
or 768 rows (`outer_pave_batch_size=null`). Gradients from these extra losses
update only the selected outer critic parameters (both ensembles for the
auxiliary baseline). Observation encodings and replay
actions are detached. Losses use decoded scalar Q and are not normalized by
noise variance or Q scale. They run on every outer update, including
pretraining; inner losses are unchanged.

## Launch and validation

`experiments/ambi_critic_smoothness_training.json` lists all four cells for
benchmarking. Each cell also has a single-configuration manifest suitable
for exact resume. `slurm/benchmark_ambi_smoothness.py` measures complete
training decisions on 1 and 2 physical CPU cores with the production loss,
CPU replay, and full regularizer batch. W&B, checkpoint I/O, and production
pretraining are disabled only in this short gate. Require finite losses,
positive active penalties, 768 samples, and no compilation fallback.

`slurm/run_ambi_smoothness_hydra.sbatch` supports gate and training modes.
Training checks the passed gate against the exact source commit,
configuration hashes, dependency lock, and GPU runtime. Set production CPU
and walltime from measured throughput, including a margin for checkpoints,
logging, and runtime variation. Four independent GPUs permit four cells in
parallel. These are single-seed exploratory comparisons.

All runs retain model checkpoints every 25,000 decisions. The auxiliary
baseline also retains the reference's complete replay companions. Large
checkpoint banks are on node-local `/ltmp` on `gpu2501`; these are scratch
artifacts and must be retrieved before node cleanup. Console logs, the
snapshot-location pointer, and the latest two exact-resume generations are
kept separately under the campaign's shared home directory. A continuation
uses the same clean source, configuration, lineage, and node with explicit
`--resume-mode required`. Unexpected failures are never automatically
requeued. A saved W&B workspace shows progress and actual training returns;
empty initial curves mean pending uploads, not zero performance.
