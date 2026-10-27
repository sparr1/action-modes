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

## Cadence swaps on the same checkpoint

`return_prior_cadence_swap.json` isolates evaluation cadence by evaluating the
J8/K1-trained checkpoint with K3 and the J8/K3-trained checkpoint with K1.
Both checkpoints remain at 500k decisions, and only `inner_solve_interval`
changes from the saved SAC configuration. J8, H3, mean execution, paired seeds
101–105, controller seed55, and full500-decision episodes remain fixed.

The two missing cells reuse the completed prior-only and original-cadence
results from the selected-checkpoint comparison. They do not rerun either
prior or the original SAC controller. The publisher validates the reused
bundles and pairs by environment seed, reporting improvement over the same
checkpoint's prior and the change relative to its original evaluation cadence.
A dedicated view displays both prior controls and all four train-K/eval-K
combinations. Each new evaluation has its own New publication identity.

Use `slurm/ambi_return_cadence_eval.py` with the frozen inventory and the
original campaign root as `--reference-root`. Its two selected indices map to
K1-trained/eval-K3 and K3-trained/eval-K1. This intervention measures the effect
of evaluation cadence within each learned checkpoint; it does not isolate
training-seed variation or hold inference compute equal between K1 and K3.

## H4 refinement with K1 execution

`return_prior_h4_k1.json` evaluates the same two J8 checkpoints at 500k with
K1 and H4. It preserves the saved controller settings except for the requested
solve interval and rollout horizon, and expands action-local replay from 3,072
to 4,096 transitions to retain all `J8 * N128 * H4` imagined transitions. Each
solve still performs 128 critic, 32 actor, and 32 temperature updates.

The five environment seeds, controller seed, mean actions, frozen outer state,
and full 500-decision episodes match the existing evaluations. Only the two H4
settings are newly evaluated. Prior controls are reused from the original
comparison; the H3/K1 reference for the K1-trained checkpoint comes from that
comparison, while the H3/K1 reference for the K3-trained checkpoint comes from
the completed cadence swap. A separate six-run view reports H4 return, paired
gain over the prior, and H4-minus-H3 return at fixed K1.

Use `slurm/ambi_return_h4_eval.py` with the original campaign as
`--reference-root`. Publication uses `slurm/ambi_return_h4_publish.py` with both
`--reference-root` and `--cadence-root`. Each new H4 evaluation gets its own New
publication identity; existing controls and their workspaces remain reusable.

## H4 refinement with K4 execution

`return_prior_h4_k4.json` uses the same two 500k checkpoints and the same H4/J8
recipe, changing evaluation cadence to K4. The adapted actor and its solve-time
encoder are held between solves, but each action is recomputed from the new
observation. Every episode starts with a fresh solve. Five 500-decision episodes
therefore use 625 solves, versus 2,500 at K1; each H4 solve still generates
4,096 imagined transitions with 128 critic, 32 actor, and 32 temperature updates.

Only the two K4 arms are new. Original prior controls and completed H4/K1
results are reused, with mean execution, environment seeds 101–105, controller
seed 55, and frozen outer state unchanged. The separate comparison workspace
displays each checkpoint's prior, H4/K1, and H4/K4 outcomes and controller cost.
Use `slurm/ambi_return_h4_k4_eval.py` and
`slurm/ambi_return_h4_k4_publish.py`; the publisher's `--h4-root` identifies the
existing H4/K1 campaign. Each K4 arm has its own New publication identity.

## J10 refinement at each checkpoint's original K

`return_prior_j10.json` evaluates the two J8-trained 500k checkpoints with ten
inner rounds, retaining H3 and their original K1 and K3 solve intervals.
Action-local replay capacity increases from 3,072 to 3,840 to retain all
`J10 * N128 * H3` transitions. Each solve performs 160 critic, 40 actor, and
40 temperature updates. No other saved controller settings change.

The comparison also includes the J10/K3-trained source at its explicitly
selected 500k checkpoint, evaluated with its saved H3/J10/K3 controller and
its own prior. Its earlier 325k results are a separate checkpoint and are not
used as a paired baseline here. The two J8-trained sources reuse their existing
500k prior and J8 controller results. This requires 20 new complete episodes:
five for each J8-to-J10 evaluation, plus five prior and five SAC episodes for
the J10-trained source. All use mean actions, environment seeds 101–105,
controller seed 55, and frozen outer state.

Use `slurm/ambi_return_j10_eval.py` and `slurm/ambi_return_j10_publish.py` with
the original selected campaign as `--reference-root`. The eight-run workspace
distinguishes training J/K from evaluation J/K. J10-minus-J8 deltas pair the
same checkpoint and environment seed only; comparing the two trained J values
is a separate backbone comparison, not a paired refinement effect.
