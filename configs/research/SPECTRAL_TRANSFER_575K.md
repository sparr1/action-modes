# Spectral transfer between successive inner SAC solves

This campaign evaluates successive-decision spectral transfer using the audited
575k checkpoint and the same return/return controller, full episodes, and paired
seeds as the transfer
campaigns. The frozen prior never changes. At decision zero every arm is fresh;
later decisions may use only the immediately preceding solve as donor.

For each selected actor or critic matrix, define
`D = W_previous_final - W_frozen_prior`. Initialization is
`W_frozen_prior + strength * filter(D)`. SAC then trains the full dense network
for J rounds. This does not constrain the subsequent optimization to low rank.
Each round retains C=16 critic updates, A=4 actor updates, N=128 imagined
branches and B=256. H changes imagination depth; J changes the number of rounds.

| Method | Filter | Interpretation |
| --- | --- | --- |
| `svd` | Largest r singular components of D | Best rank-r approximation in parameter Frobenius norm; magnitude is not evidence of transfer usefulness. |
| `activation` | Truncate `D L`, then solve `R L = (D L)_r`, with `C=L Lᵀ` | Minimizes `||(D-R)L||²` for the damped prior-input covariance. Preserves layer preactivations on that distribution, not full-network behavior. |
| `gradient` | Score each donor mode `D_i` by `-<G_new,D_i>` and keep up to r positive modes | Selects locally beneficial directions for a specified frozen-reference surrogate. It does not guarantee improvement after J updates or in real return. |
| `gradient_gate` | Retain each donor coordinate `D_j` only when `G_j D_j < 0` | Copies individually descent-aligned changes; zero products reset. There is no rank constraint or SVD in the controller. |
| `gradient_projection` | Project the full matrix-parameter delta vector onto the initial new-decision gradient: `R = (<D,G> / ||G||²) G` | Preserves the signed component along the gradient line. This is one direction in parameter space, not a rank-one constraint on each matrix. |

The activation covariance is the uncentered second moment `XᵀX/n`, plus
`covariance_damping * mean(diag(C)) * I`; the scale is one when all inputs are
zero. Cholesky and a triangular solve avoid an explicit inverse. The default
damping is 1e-4. Requested rank is clamped to each layer's dimensions. Thus
rank32 is a different fraction in a hidden layer and an output layer.
Gradient selection within a repeated-singular-value subspace can depend on
the SVD basis. All decompositions are exact rather than randomized SVD.

### Gradient-gated copying

`gradient_gate` replaces a Bernoulli mask with the deterministic sign test
`G_j D_j < 0` at each matrix coordinate. The gate is computed before strength
scaling, with the same initial new-decision frozen-reference gradient used by
gradient-selected SVD and gradient projection. Zero-gradient and zero-delta
coordinates reset. Numeric ranks are rejected; `rank` is null or omitted.
Only the retained coordinates carry their donor values at strength one.

The retained sum has nonpositive first-order selection loss change, but this
does not guarantee an improvement for a finite transfer or after SAC training.
It can discard groups of weights whose coordinated change would be useful.
Its dense control matches the retained norm per layer. Decision metrics record
retained counts/fractions, donor-gradient alignment, actual predicted benefit
and displacement norms. Selection probes are controller work; full spectra are
computed only by sampled held-out diagnostics.

### Initial-gradient line projection

`gradient_projection` concatenates all eligible matrix parameters conceptually
and uses one projection coefficient per component: one for the actor, and one
for the complete online critic (including all its heads). It does not project
each layer independently. `G` is evaluated once at the frozen-prior
initialization on the new decision's selection bank, using the same fixed
surrogate as gradient-selected SVD below. It is not an Adam-preconditioned step
or a gradient from the first SAC replay minibatch.

The projection is signed. If `<D,G> > 0`, it preserves the uphill component;
it does not clip to the descent ray. A zero gradient produces zero transfer.
At strength one and before floating-point rounding, `<G,R> = <G,D>`,
`<G,D-R> = 0`, and `||R|| <= ||D||`. Consequently this tests the effect of
discarding orthogonal donor changes while preserving the first-order selection
loss change; it does not guarantee a favorable change. Strength scales `R`
before initialization, and ordinary dense SAC follows.

This method accepts `rank: null` (or no rank). Numeric ranks are rejected.
Mixed campaign grids enumerate gate and projection once per strength/component,
without duplicating them for each SVD rank. Grids containing only these
nonrank methods have `ranks: []`.
The controller computes dot products and norms without SVD. Required selection
probe/backward-pass time remains part of controller time; sampled spectral and
held-out diagnostics remain separately timed. Decision metrics record the
projection coefficient, donor-gradient inner product, gradient squared norm,
actual transferred norm and predicted benefit. Unselected mode/rank quantities
are null rather than implying a rank-one weight matrix.

The gradient-alignment campaign also aggregates computed selection metrics over
applicable transfer decisions in each episode, excluding the donorless first
decision. These means retain their contributing-decision counts and remain
separate from sampled held-out measurements. Retention, signed projection
coefficient and predicted-benefit charts use this selection bank; undefined or
uncomputed fields remain absent. Their raw per-decision values remain in JSONL.

## Matched comparisons

Actor-only, critic-only and joint transfer are independent choices. Each
SVD-method or gated-copy candidate has a paired dense control: for each layer, replace its
filtered update R by `D * ||strength * R|| / ||D||`. This matches transferred
norm layer by layer. It tests direction selection against attenuation with the
same norm, while still paying the selection cost. Zero donor deltas remain
zero. Activation-weighted transfer can have greater parameter norm than D, so
its dense norm-matched coefficient can exceed one.

For `gradient_projection` only, the dense control uses a single coefficient
matching the **whole component's** transferred norm. A global projection can
introduce changes in a layer whose donor delta was zero, so per-layer matching
cannot always be satisfied. This control is labeled separately from the
per-layer controls used by the three SVD methods and gated copying.

Additional controls are fresh initialization, full matrix carry, deterministic
50% matrix blending, and Bernoulli 50% matrix copying. All new arms transfer
only named 2D parameters. Biases, normalization affine parameters and buffers
reset to the frozen prior. Historical Bernoulli experiments transferred all
parameters, so their results are not silently reused as matched controls.
Optimizers, replay, temperature and counters reset each decision; the target
critic is copied from the selected online critic. Episode boundaries clear
the donor and private mask streams.

## Selection and held-out probes

Probe states are the new real root plus independent one-step successors under
the frozen prior. Candidate actions and rollout labels also use the frozen
prior, independently of the donor or candidate. The actor gradient is for
negative frozen-prior Q at the candidate's deterministic mean action, without
entropy. It intentionally does not score log-standard-deviation output rows.
The critic gradient is for decoded squared Q error averaged across all heads,
against fixed prior-continuation model-return labels. These are explicitly
fixed-model surrogates, not the moving SAC objective or environment ground
truth. Deeper imagined-state distributions are not fitted by this initial probe.
Selection builds Monte Carlo labels only for critic-gradient scoring;
activation filters and actor-only gradient filters skip that work. Held-out
evaluation always builds the complete reference bank. Uncomputed measurements
are explicitly absent rather than represented by zero.

The selection and held-out contexts have different named RNG streams.
Observational probes run on disposable copies and preserve global RNG, learner
RNG, model parameters, gradients and module modes. Held-out data evaluates the
prior, donor, initialization and actual post-J learner using identical labels
and the same reference critic. Post-J means after the configured SAC solve,
not the separate four-step stationary diagnostic fit.

The JSONL decision artifacts contain sampled layer spectra and raw diagnostics:

* Singular values, cumulative squared energy, energy at ranks
  1/2/4/8/16/32/64/128, and ranks capturing 50/90/95/99% energy.
* Stable rank, effective rank, energy-effective rank, mode benefit scores,
  positive-benefit energy fraction and energy/benefit correlation when defined.
* Donor and initialization norms, squared-norm ratio, residual energy,
  donor/update cosine, first-order benefit and prior-input preactivation error.
* Held-out actor/critic losses at prior/donor/initial/post-J, gains over prior,
  and improvement within the solve.
* Existing fixed-target critic, action-gradient, policy, feature and stationary
  fit diagnostics.

Energy fractions of donor singular values are bounded by one. The actual
transfer/donor squared-norm ratio need not be. Selected rank counts filter
slots, not numerical rank of the resulting dense norm-matched control. Undefined
correlations stay null; they are not reported as measured zero. Donor geometry
at the first decision is excluded from episode averages, with per-metric sample
counts retained. Objective comparisons at that decision remain valid.

Full diagnostics are sampled at decisions 0,1,25,100,250,499 (bounded by actual
episode length). Controller selection probes are computed whenever the active
filter needs them. `control_seconds = prediction_seconds + transfer_seconds`;
transfer includes required probe construction, covariance/SVD/gradient work,
initialization and donor export. Probe/filter/export subtimers are included in
that total, not added again. Sampled held-out diagnostics and snapshot time are
reported separately. A fresh control's diagnostic-only donor export is also
diagnostic cost. L40S timing is a future GPU measurement, not inferred from CPU
unit tests.
The controller path records aggregate transfer metrics without constructing
full layer spectra; activation transfer performs one weighted decomposition per
matrix. Detailed donor spectra are computed only by sampled diagnostics.

## Configure and prepare a campaign

`ambi_spectral_transfer_575k.json` selects the initial truncated-SVD sweep:
H1/2/3, J1/2/4/6, rank32, strength1, plain SVD and its dense norm-matched control,
all three component choices plus the matrix controls. That is 16 arms and 192
cells, each with three paired full episodes (576 episodes). The rank and strength
are held fixed in this screen; these settings are not a claim of optimality.

The generator supports all five methods, explicit horizon subsets of H1/2/3,
and finer rank/strength/J choices.
For example, this writes a configuration only:

```bash
python slurm/ambi_spectral_transfer_campaign.py generate \
  --methods svd activation gradient --ranks 16 32 --strengths 0.5 1 \
  --rounds 1 2 4 6 --components actor critic joint \
  --output /path/to/reviewed-matrix.json
```

Each added method/rank/strength/component combination also adds its matched
control; gradient gating and projection have no rank axis. For example, select
`--methods gradient_projection --strengths 1 --components actor critic joint`
to generate the new candidate and global norm controls with the same H/J and
paired-episode protocol. Review the printed cell count before selecting a launch grid.
Use `evaluate_ambi_transfer_campaign.py --campaign PATH --list-cells` to inspect
the exact high-J-first task ordering without loading a checkpoint.
For this family the named base matrix is resolved inside the checked-out
repository's `configs/research` directory, so generated matrix files remain
portable between local and cluster checkouts.

After review, commit and synchronize through Git before preparing on Oscar.
The prepare command pins clean source, checkpoint and metadata hashes, matrix
and base-matrix identities, arms, probes and smoke coverage. A new output root
is required. The Oscar wrapper has separate prepare, worker and publish modes;
it does not submit jobs itself. GPU workers require L40S and a real CUDA
arithmetic preflight. Production requires bound, validated smoke receipts for
all mechanisms at maximum H/J plus the smallest fresh cell. Smoke episodes
span two transfer boundaries and compare diagnostics-on/off actions, rewards,
training metrics and learner/RNG state exactly.

Publication uses `ambi-inner-bench`, component-separated views and distinct
method/control colors. Pending panels, result panels and spectral diagnostics
are installed and verified by the existing idempotent publisher. Original
episode returns remain the unit of aggregation; probe roots are not independent
return samples. Select mechanisms using both return and total controller time,
with immediate and post-J surrogate behavior as explanatory evidence.

No commit, push, cluster synchronization, submission or online publication is
performed by generating or reviewing these files.

## H1 gradient-alignment screen

`ambi_gradient_transfer_h1_575k.json` selects H1 and J1/2/4/6 for
gradient-selected SVD at ranks 1 and 4, gradient-gated copying, and signed
gradient-line projection. Every method uses strengths 0.5 and 1.0 with
actor-only, critic-only and joint variants. Each candidate has its appropriate
norm-matched dense control, plus shared matrix-only fresh/full/half-blend/
Bernoulli-50% controls. This is 58 arms, 232 settings and 696 full episodes on
paired seeds 101–103 at the audited 575k checkpoint.

All controls execute under this implementation; historical rank-extension
reuse remains a separate strict contract. Preparation binds 59 GPU smoke cells:
all 58 arms at H1/J6 plus fresh H1/J1, each with two seeds and three decisions.
Every worker verifies all smoke receipts before production. The same C16/A4,
N128/B256, 500-decision, fresh optimizer/replay/temperature protocol and
independent selection/held-out streams remain in force.

The dedicated `ambi-inner-bench` workspace separates actor/critic/joint and
strength 0.5/1.0 panels. Each panel repeats the shared controls and displays
only candidates at its selected strength. Gate, projection and the two
gradient-SVD ranks have distinct colors; lighter dashed curves identify dense
norm controls. The norm-matching scope is recorded in result tables. Return
and controller time remain primary outcomes, with spectral geometry and
initial/post-J probe losses as explanatory metrics.

## Rank1/rank4 extension with audited rank32 reuse

`ambi_spectral_transfer_r1_r4_575k.json` adds rank1 and rank4 at strength1 for
actor-only, critic-only and joint transfer, each with its dense norm-matched
control. H remains 1/2/3 and J remains 1/2/4/6. This schedules **144 new cells /
432 full episodes**. The comparison also reads the 192 completed rank32 and
matrix-control cells from the original campaign, giving 336 comparison cells.
The original rank32 configuration and result files are unchanged.

Prepare this matrix with `--spectral-reference-root` pointing to the completed
original campaign. The Oscar wrapper forwards `SPECTRAL_REFERENCE_CAMPAIGN_ROOT`
to that argument. Preparation fails before creating output if references are
missing or incompatible; it does not fall back to rerunning controls.

Reuse pins the original campaign, matrix, source commit, checkpoint, metadata,
and base-matrix hashes. All shared learner settings, paired seeds, full-episode
lengths, normalized diagnostic settings, compiled CUDA execution, L40S hardware,
and Python/Torch/NumPy versions must match. A broad Git-blob fingerprint requires
identical runtime source and dependency locks, allowing only the reviewed
orchestration/publication changes, new rank matrix, tests and Markdown. Each
original result is checked against its manifest and bound worker receipt, and
its complete resolved learner configuration is compared with the new matrix.

Prepared `cells` contain only the 144 new cells with contiguous worker indices.
Twelve H3/J6 smoke cells cover every new rank/component/control arm before
production. Original reference cell indices, episode data, source and timing
remain attached to the historical records; no result is copied or relabeled.
Workers reject reference-cell execution. The publisher rechecks pinned evidence
and reports new-work progress separately from the combined comparison.
