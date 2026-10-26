# Selected return-only checkpoint comparison

`return_prior_selected_eval.json` compares each checkpoint's frozen prior mean
policy with its original inner-SAC controller. It inherits the checkpoint's
architecture, H3 rollout horizon, J/K schedule, reward-only targets, and automatic
inner temperature initialized from the saved outer temperature. K3 holds the
adapted policy and solve-time encoder for two further observations between solves.

Both controllers execute mean actions in five complete Humanoid Walk episodes:
environment seeds 101–105, controller seed 55, and at most 500 decisions. Inner
imagined collection and optimization remain stochastic. The outer model, actor,
critic, temperature, optimizers, and update counters remain frozen. No to-go
probes or additional controller repetitions are enabled.

The campaign runner consumes a separately recorded inventory of three source
runs, selecting `min(500000, latest complete portable checkpoint)` once when
that inventory is created. The inventory records absolute checkpoint and sidecar
paths, content hashes, selected training step, source run, J, and K. Production
validates that both controllers completed all five episodes and that outer state
remained unchanged. A short unpublished smoke uses one seed and six decisions.

Use `slurm/ambi_return_prior_eval.py prepare` to resolve publication identities
without running episodes. Allocate explicit New evaluation-series attempts before
launching workers with `slurm/run_ambi_return_prior_eval_oscar.sbatch`. Resource
requests and concurrency are selected from current scheduler availability and
measured throughput. GPU workers only save results; the campaign CPU publisher
validates, stages, and publishes complete paired bundles. Each source/controller
has its own evaluation run, and the named workspace shows pending status until
the corresponding result is available.

The paired return difference measures the benefit of refinement at evaluation
time for that frozen checkpoint. It does not isolate how refinement affected
data collection and learning during training. These source runs share one
training seed; variation across evaluation episodes is not training-seed
uncertainty.
