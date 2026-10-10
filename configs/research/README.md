# Frozen-checkpoint AMBI research

The [800k matched-state action audit](ACTION_AUDIT_800K.md) compares fresh SAC
J4/J6/J8, MPPI H1/H3, replay and broad action candidates on identical saved
states. It separates learned-Q ranking, held-out model scores, real prefixes
and finite prior-tail calibration, with compact W&B progress and result panels.

[The spectral transfer campaign](SPECTRAL_TRANSFER_575K.md) implements plain,
activation-weighted and gradient-scored SVD initialization, with per-layer
norm-matched dense controls and actor/critic/joint variants. It also supports
gradient-gated coordinate copying with per-layer norm controls and
signed projection of the donor delta onto the new decision's initial surrogate
gradient, with one coefficient and a global norm-matched control per component.
This projection uses no SVD in the controller and has no matrix-rank parameter.
It records sampled
spectral energy, directional usefulness, independent held-out initialization
and post-J objectives, and all required transfer costs. Its 575k example matrix
and Oscar wrapper are configuration for a future reviewed launch.

The [H1 gradient-alignment screen](ambi_gradient_transfer_h1_575k.json) crosses
J1/2/4/6 with gradient-selected SVD ranks 1/4, gated copying and signed gradient
projection at strengths 0.5/1, including actor/critic/joint and matched controls.
Its 232 settings retain all spectral and held-out diagnostics, with separate
dashboard panels for each component and transfer strength.

[The Bernoulli J12 extension](ambi_bernoulli_j12_575k.json) adds nine p=0.5 cells
at H1/2/3, preserving the three transfer mechanisms, paired full episodes and
sampled diagnostics. Pass all four historical reference roots, including
`--bernoulli-j10-reference-root`; three H3/J12 smoke cells gate production.
The existing replay-capacity rule retains all 4608 imagined transitions at H3/J12.
Fresh controls stop at J10: the J12 comparison reports return and controller
time, with no matched fresh J12 gain or newly scheduled fresh controls.

[The Bernoulli J10 extension](ambi_bernoulli_j10_575k.json) adds nine p=0.5 cells
at H1/2/3 with the same three transfer mechanisms and paired episode protocol.
It reuses the completed lower-J and J8 Bernoulli campaigns plus fresh controls
through J10. Pass all three reference roots, including
`--bernoulli-j8-reference-root`; three H3/J10 smoke cells gate production.
The existing replay-capacity rule retains all 3840 imagined transitions at H3/J10.

[The Bernoulli J8 extension](ambi_bernoulli_j8_575k.json) adds nine p=0.5 cells:
actor-only, critic-only and joint transfer at H1/2/3, with three paired full
episodes each. It retains the sampled diagnostics, reuses p=0.5 J1/2/4/6 results
and fresh J1/2/4/6/8 controls, and gates production on three H3/J8 smoke cells.

[The Bernoulli probability extension](BERNOULLI_PROBABILITY_575K.md) adds 25%
and 75% retention to the same H/J grid, with audited reuse of the completed 50%
screen and historical fresh controls.

[The Bernoulli p=0.5 transfer screen](BERNOULLI_TRANSFER_575K.md) tests actor,
critic and joint random copying across H1/2/3 and J1/2/4/6 at the frozen 575k
checkpoint, with verified sampled-root diagnostics and separate probe timing.

[The 575k transfer discovery campaign](TRANSFER_DISCOVERY.md) screens 14
mechanisms across H=1,2,3 and J=1,2,4,6,8,10 with three paired full episodes
per configuration. It includes independent actor/critic retention strengths,
rollout behavior, recent imagined replay, complete learner state and prior
anchoring. Its Oscar launcher gates production on real-checkpoint smoke runs.

[Inner SAC transfer diagnostics](TRANSFER_DIAGNOSTICS.md) provides same-state
actor/critic/joint forks, independent critic references, target initialization
crossings, horizon/support checks, stationary learning tests, and optional real
simulator interventions. Horizons are configurable; the default grid is
H=1,2,3,5,7. This is a separate diagnostic protocol from production evaluation.

The [575k critic hidden-layer transfer examples](CRITIC_HIDDEN_TRANSFER_575K.md)
retain trainable critic hidden layers and normalization while creating a fresh
Xavier-uniform, zero-bias output head at every solve, including the first.
Separate protocols cover soft/soft and return/return critics with every-decision
or hold-H execution; existing full-critic examples remain unchanged.
The [matched hidden-layer sweep](CRITIC_HIDDEN_TRANSFER_SWEEP_575K.md) adds
eight J1/J8 conditions and an Oscar smoke/production gate without rerunning
the historical controls.

The [H3 transfer sweep](TRANSFER_SWEEP_575K.md) crosses J1/J8, fresh/actor-only/
critic-only initialization, soft/soft versus return/return critics, and
every-decision/hold-H cadences. Its four matrices select 24 unique conditions
by default, evaluating each shared fresh reference once. A fifth matrix adds
the same-checkpoint frozen-prior mean reference on the five paired seeds.

The [575k critic-only transfer examples](CRITIC_TRANSFER_575K.md) compare fresh
and carried online critics for both soft/soft and return/return objectives,
with every-decision and hold-H solve cadences. The actor resets at each solve;
the target is freshly copied from the retained online critic. Each matrix
defaults to only the return-only carried-critic example and uses the generic
frozen evaluator with local output bundles.

The [H-decision actor-hold study](ACTOR_TRANSFER_HOLD_H_575K.md) solves at
t=0,H,2H,… and evaluates the fixed adapted feedback actor on fresh observations
between solves, across the same H/J cold-versus-warm panel.

The [575k actor-only transfer study](ACTOR_TRANSFER_575K.md) compares cold and
warm actor initialization across H1/2/3 and J1/2/4/6/8/10, using selected J at
every decision, with per-phase transfer diagnostics and return-versus-compute reporting.

The [575k closed-loop critic comparison](CLOSED_LOOP_CRITICS_575K.md) evaluates
soft and return-only critics with C16/A4, H3 and J={1,2,4}, and publishes an
explicit W&B comparison of full-episode returns, paired gains and diagnostics.

The frozen evaluator supports finite-horizon inner SAC and transition-based
update counts. It rejects `inner_outer_replay_fraction > 0` because these model
snapshots do not include real replay. Test replay mixing through a populated
training agent. See the [inner SAC options](../../RL/tdmpc2_core/README.md#finite-horizon-inner-sac-and-real-replay-mixing)
for the target convention and schedule restrictions.

The active end-to-end configs live directly under `configs/ambi/`.
The frozen-checkpoint matrix is
`configs/research/ambi_inner_decoupling.json`, outside the active AMBI training
tree so it cannot be confused with the independently trained branch and horizon
comparisons.

### Critic-only LoRA-RL comparisons

The [LoRA-RL implementation contract](../../RL/tdmpc2_core/README.md#critic-only-lora-rl)
keeps the actor dense and applies low-rank updates only to selected critic
matrices. The initial reference is rank 96, direct scale one, and AdamW adapter
weight decay `2e-4`. Output heads, biases, and normalization parameters remain
trainable. All scientific inner state resets at each real decision.

The frozen-checkpoint matrix `ambi_inner_decoupling.json` provides these
explicit selectors without changing its default operator comparison:

- `critic_lora_placement/clone`: full actor and critic adaptation.
- `critic_lora_placement/input_hidden_r96`: LoRA on the critic input and hidden
  matrices; the default placement for AMBI's learned-latent input.
- `critic_lora_placement/hidden_r96`: LoRA on hidden matrices only, closer to
  the paper's placement, with a normally trainable input projection.
- `critic_lora_rank/rank_64`, `rank_96`, and `rank_128`: input-and-hidden
  placement with identical scale, decay, actor adaptation, and SAC budgets.

These replace the old `adaptation_mechanism` and `lora_capacity` axes. Old
selectors and active `"lora"` configurations do not silently select the new
method. Reproducing a historical run requires its historical source; start a
new evaluation attempt when adopting `"lora_rl"`.

Four matching algorithm/experiment pairs under `configs/dmcontrol/` provide
single-seed Humanoid Walk training screens:

- `ambi_humanoid_walk_base_v2_lora_rl_input_hidden_r64`
- `ambi_humanoid_walk_base_v2_lora_rl_input_hidden_r96`
- `ambi_humanoid_walk_base_v2_lora_rl_input_hidden_r128`
- `ambi_humanoid_walk_base_v2_lora_rl_hidden_r96`

Each retains the base-v2 recipe: seed 55, 14 million real decisions, J=8, N=32,
H=3, G=1, replay capacity 768, and the original evaluation/checkpoint settings.
Only critic adaptation and descriptive run identity change. These are full
training configurations, not smoke tests; no sweep runs by adding them. Rank 96
and decay `2e-4` borrow the paper's DMC SimbaV2 settings, while the rank and
placement alternatives test the transfer to AMBI. The trained prior and zero-B
initialization deliberately differ from the paper's main setup. Neither a
return improvement nor a speedup has been established for these configurations.

# Frozen-checkpoint preset workflow

`ambi_inner_decoupling.json` is a compact matrix of one-axis-at-a-time overrides
based on `configs/algs/AntAMBITDMPC2.json`. The canonical AMBI reference is
fresh, action-local, fully cloned inner SAC.
The remaining operators, LoRA and persistence settings below deliberately test
auxiliary ablations or comparators; they are not alternative definitions of
AMBI. The matrix covers:

- none, SAC, TD3, and compute-matched MPPI inner operators;
- the reference five-head distributional-Q model and a scalar twin-Q ablation;
- actor, critic, and temperature adaptation controls;
- train-only actor, online-critic, and joint prior writeback at beta 0.01, 0.1,
  and 1.0, plus the no-writeback reference;
- temperature, imagined behavior, and returned-action exploration;
- explicit J/N/H/G collection and joint-update schedules;
- action, episode, and run lifecycles;
- replay sampling, bootstrap source, rollout horizon, critic-only LoRA-RL placement and rank,
  and outer-policy anchoring controls.

Within a variant's `alg_params`, `null` removes an inherited base key. The
operator comparison uses this to keep SAC's J/N/G controls out of MPPI and
no-improvement configurations.

List the matrix without importing the training stack:

```bash
python3 evaluate_ambi_checkpoint.py --list-presets
```

Materialize ordinary algorithm configs that can be referenced by the existing
`main.py --run ... --alg-dir ...` workflow:

```bash
python3 evaluate_ambi_checkpoint.py \
  --comparison inner_operator \
  --materialize-dir configs/algs/generated
```

Materialization also writes `AMBIResearchExperiment.json`, preserving the
matrix's environment parameters. Run it with both paths pointed at that output:

```bash
python3 main.py \
  --run configs/algs/generated/AMBIResearchExperiment.json \
  --alg-dir configs/algs/generated
```

Evaluate the default operator comparison from one frozen checkpoint:

```bash
python3 evaluate_ambi_checkpoint.py \
  --checkpoint /path/to/ambi-checkpoint.pt \
  --device cuda \
  --seeds 101 102 103 104 105 \
  --output logs/ambi_inner_operator_eval.json
```

Select individual presets with repeatable `--preset comparison/variant`, or an
entire axis with repeatable `--comparison comparison`. The evaluator creates a
fresh model for every preset, uses paired environment seeds, by default returns the
policy mean to the real environment, never calls the outer update, and hashes
outer model/optimizer/temperature state before and after each run. Its output
contains per-episode real returns and all finite model-predicted inner metrics.
Output is written atomically, and an existing `--output` path is preserved
unless `--overwrite` is supplied explicitly.
When a comparison's reference preset is selected, it also reports seed-paired
return deltas for every selected variant.

For stochastic execution, set `inner_eval_execution_action="policy_sample"`
in a SAC or prior-only preset. Each real decision draws from the final actor's
learned squashed Gaussian at unit standard-deviation scale, using the isolated
execution RNG. Evaluation mode, fresh adaptation and outer-state freezing stay
in force. The default `"mean"` preserves historical behavior. Training-time
execution noise and standard-deviation scaling do not affect this override.
Bundles and curve identities record `squashed_gaussian_sample`; they cannot
reuse a mean prior as if it had the same action protocol. Historical mean
outcomes require an explicitly labeled, seed-paired execution comparison.

An existing evaluation curve that omitted its prior reference can be backfilled
without repeating planner episodes. Evaluate the frozen SAC policy mean on the
same checkpoint files, environment seeds, controller seed, and full-episode
protocol. For the auxiliary-return MPPI matrix, use
`ambi_aux_return_prior_reference.json` and
`slurm/run_ambi_aux_return_prior_oscar.sbatch`. Wait for the original curve's
publisher to finish, then validate the complete reference grid:

```bash
python backfill_eval_series.py /path/to/authoritative/run \
  --prior-bundles /path/to/prior/step_*/bundle \
  --checkpoint-inventory /path/to/checkpoint-manifest.json \
  --owner oscar-rgao48
```

Add `--publish --receipt /path/to/backfill-receipt.json` to append verified
`eval/paired_gain_mean`, sample standard deviation, paired episode count, and
prior return statistics to that same run. Gains are planner return minus prior
return, matched by seed. The adapter verifies checkpoint hashes, backbone,
scientific implementation, episode protocol, solver seeds, and frozen-state
checks. Original return/runtime rows and checkpoint artifacts remain unchanged;
supplemental rows share their checkpoint x coordinate and have separate
`evaluation-reference` artifacts. Publication uses the existing exclusive-owner
lock, immutable journal, and remote acknowledgement checks. Repeating the same
backfill is idempotent; a different reference for an already paired point is
rejected.

Q representation is part of the checkpoint architecture. The reference
checkpoint uses five distributional heads. It can compare checkpoint-compatible
inner operators and controls, but it cannot be evaluated as a scalar twin model
(or vice versa). Train and supply a matching checkpoint for each side of the
Q-representation comparison, using one preset per invocation:

```bash
python3 evaluate_ambi_checkpoint.py --checkpoint distributional.pt \
  --preset q_representation/distributional_five --output distributional-eval.json
python3 evaluate_ambi_checkpoint.py --checkpoint scalar.pt \
  --preset q_representation/scalar_twin --output scalar-eval.json
```

The evaluator rejects a mixed-architecture selection before running either
side. It also rejects the train-only `execution_noise` and `prior_writeback`
axes. Deterministic evaluation returns the policy mean, which collapses the
execution-noise variants. Prior writeback is deliberately disabled outside
training, so its variants would likewise collapse under a frozen outer
checkpoint. Materialize and train those axes instead.

The [625k matched-budget interleaving screen](AUX_INTERLEAVED_625K.md) compares
actor/alpha updates spaced through the critic steps with the completed
critic-first H2/H3, J1, C32, A4/A8/A16 screen.

The original 625k H/J table extends to J6/J8 for its two soft-inner arms in
[`AUX_J68_625K.md`](AUX_J68_625K.md) and `ambi_aux_j68_625k.json`. Existing J4
results are reused, and replay capacity grows to retain all H3/J8 transitions.

The [soft/soft critic-budget comparison](AUX_SOFT_CRITIC_BUDGET_625K.md) adds
C8/C16 at H1/H2/H3 and J2/J4/J8, with fixed tau and learning rates, full replay,
and strict pairing to the corresponding historical C32 results.

The [all-checkpoint transfer evaluation](TRANSFER_CHECKPOINT_CURVES.md) compares
three selected warm starts with fresh SAC at J2/J4 and the frozen base actor
across the auxiliary-return backbone's 25k–2M checkpoint bank. It uses five
paired seeds, fixed colors, return/gain/variance/runtime curves, and sampled
transfer diagnostics with verified RNG isolation.
