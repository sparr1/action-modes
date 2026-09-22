# Closed-loop soft versus return-only critics at 575k

This exploratory screen uses the auxiliary-return SAC backbone
`rwgao_b-brown-university/ambi/aux6428346x0` (target entropy −10.5,
shared representation gradients), at 575,000 training decisions. The checkpoint
was selected because the existing MPPI comparison had a large paired critic gap;
this is a diagnostic case, not an unbiased estimate across training checkpoints.

At every real observation, the evaluator creates fresh actor, critic, target,
optimizers, and replay from the frozen checkpoint. It collects imagined
transitions and performs the complete inner solve, executes the adapted actor's
mean action, and repeats for the full 500-decision Humanoid Walk episode. There
is no outer learning or writeback. This is full-episode closed-loop refinement,
not a model-only probe or a prefix followed by prior-policy continuation.

The six conditions cross J={1,2,4} with two coherent critic strategies:

| Arm | Initial critic | Inner fitting | Frozen terminal bootstrap |
|---|---|---|---|
| Soft | SAC soft Q | Entropy augmented | Soft Q with outer entropy correction |
| Return-only | Auxiliary return Q | Reward only | Return Q, no entropy correction |

Both retain the same SAC actor and inherited adaptive temperature with target
entropy −10.5. Critic initialization, fitting, and terminal bootstrap change
together; this comparison does not isolate their individual effects.

Each round uses N128/H3, C16/A4, B256, replay capacity 3072, critic-first updates,
learning rates 0.0003 and target tau 0.01. Replay survives across rounds within
one decision and resets at the next observation. Compilation is strict. Five
paired environment seeds 101–105 use controller seed 55. Compatible prior
episodes are reused with their original provenance and immutable hashes.

Episode return is raw undiscounted environment reward. Report paired gains over
the prior and direct return-only minus soft differences at each J. Five seeds
make this a preliminary screen; bootstrap intervals are exploratory. Retain 32
independent model-return probes at initialization and after each round, together
with all actor/critic update traces. Model-return probes are separate from real
episode returns and exclude an explicit entropy bonus.

## Publication

`slurm/ambi_closed_loop_critics.py` prepares explicit New performance identities
and evaluates complete five-seed settings. `slurm/ambi_closed_loop_publish.py`
is the single CPU publication owner. It publishes performance and inner-training
histories plus a comparison overview containing real returns, paired gains,
critic differences, per-seed outcomes, settings, and links. Pending conditions
remain missing rather than zero. The overview updates as conditions finish.

GPU workers do not initialize W&B. Publication failure does not require rerunning
episodes; recover from the complete local bundles and inspect the publication
journal before resuming an uncertain upload. All W&B staging and compiler caches
use scratch through `slurm/run_ambi_closed_loop_critics_oscar.sbatch`.

## H2 follow-up

`ambi_closed_loop_critics_h2_575k.json` repeats the six conditions with H2.
All other settings remain matched to H3, including the checkpoint, paired
seeds, optimizer counts, replay capacity, and entropy settings. Each round now
collects 256 imagined transitions instead of 384. This tests a shorter model
rollout and earlier frozen-critic bootstrap while retaining C16/A4 per round.
H2 and H3 have separate campaign and performance identities; completed H3 and
prior episodes are reused for comparison.

For preparation, select the matrix, group, and descriptive label through
`EVAL_MATRIX_PATH`, `EVAL_CAMPAIGN_GROUP`, and `EVAL_CAMPAIGN_LABEL`. Unset values
retain the original H3 defaults. GPU workers and the publisher derive H from
the prepared campaign, including model-probe work counts and W&B metadata.

## H1 follow-up

`ambi_closed_loop_critics_h1_575k.json` adds the same six conditions at H1.
It keeps the H2/H3 checkpoint, critics, inherited adaptive actor entropy, paired
seeds, N128/B256/C16/A4, replay capacity3072, and full-episode reset procedure.
H1 collects128 imagined transitions per round, or128/256/512 per decision at
J1/J2/J4. Sampling with replacement supports B256 even in the first round's
128-transition replay. The frozen terminal critic is reached after one model
step; each32-rollout model probe therefore uses32 transitions, with64 at
initialization to include the paired frozen-prior probe.

Select this matrix through `EVAL_MATRIX_PATH` and use a separate campaign group
and label. Existing H2/H3 and prior results retain their original identities.
The default matrix remains H3; adding H1 does not rerun an existing condition.

## J6 extension

`ambi_closed_loop_critics_h1_j6_575k.json`,
`ambi_closed_loop_critics_h2_j6_575k.json`, and
`ambi_closed_loop_critics_h3_j6_575k.json` each select only the two J6 critic arms
at the named horizon. Existing J1/J2/J4 matrices and defaults are unchanged.
Each extension uses a separate campaign and New performance identities while
reusing the same prior episodes; it does not repeat a completed condition.

J6 retains C16/A4 per round, giving 96 critic updates and 24 actor/temperature
updates per real decision. N128 yields 768 imagined transitions at H1, 1536 at
H2, and 2304 at H3. All fit within the unchanged replay capacity of 3072, so no
imagined transitions are evicted. Actor/critic initialization, objectives,
adaptive entropy, pairing, diagnostics, and full-episode execution remain the
same as the corresponding earlier horizon screen.

Both extension smoke indices exercise J6, the largest budget in those matrices,
for one seed and three decisions. Production evaluates five complete paired
episodes per arm. Model probes retain the initial snapshot and all six round
endpoints, with 32 rollouts each and work counts derived from the selected H.

## J8 extension

`ambi_closed_loop_critics_h1_j8_575k.json`,
`ambi_closed_loop_critics_h2_j8_575k.json`, and
`ambi_closed_loop_critics_h3_j8_575k.json` each add only the two J8 critic arms
at the named horizon, with all other controls matched to J6. Existing matrices,
defaults, and completed results are preserved; each new campaign reuses the
same prior and allocates separate performance identities.

J8 gives 128 critic updates and 32 actor/temperature updates per real decision.
N128 yields 1024, 2048, and 3072 imagined transitions for H1, H2, and H3,
respectively. H3 exactly fills the unchanged replay capacity of 3072; none of
the eight rounds' transitions are evicted. Each smoke arm runs J8 for one seed
and three decisions before production's five complete paired episodes. Probes
retain the initial snapshot and all eight round endpoints.

## J10 extension

`ambi_closed_loop_critics_h1_j10_575k.json`,
`ambi_closed_loop_critics_h2_j10_575k.json`, and
`ambi_closed_loop_critics_h3_j10_575k.json` each select only the two J10 critic
arms at the named horizon. J10 gives 160 critic updates and 40 actor/temperature
updates per real decision, with 1280, 2560, and 3840 imagined transitions at
H1, H2, and H3. The ten rounds retain the same N128/B256/C16/A4 settings.

All J10 matrices use replay capacity 3840, increasing the earlier capacity of
3072 to preserve the established requirement that action-local replay retains
every round. H3 exactly fills this capacity; H1/H2 have unused space. Keeping
3072 would violate configuration validation at H3/J10, so no ring eviction or
change to that validation is introduced. Existing matrices remain unchanged.

Checkpoint, critic objectives, inherited adaptive entropy, seeds, diagnostics,
and full-episode execution remain matched. Each campaign allocates two new
performance identities and reuses the existing prior. Smoke checks exercise
J10 for one seed and three decisions; production evaluates five complete paired
episodes per arm with the initial probe and all ten round endpoints.

## J12 extension

`ambi_closed_loop_critics_h1_j12_575k.json`,
`ambi_closed_loop_critics_h2_j12_575k.json`, and
`ambi_closed_loop_critics_h3_j12_575k.json` each add only the two J12 critic
arms at the named horizon. N128/B256/C16/A4 gives 192 critic updates and 48
actor/temperature updates per real decision, collecting 1536, 3072, and 4608
imagined transitions at H1, H2, and H3.

All J12 matrices use replay capacity 4608 so every round is retained. H3 fills
it exactly; the preceding J10 capacity of 3840 would fail the existing retention
guard at H3/J12. Earlier matrices and all other scientific controls remain
unchanged. Preparation creates two new performance identities per horizon and
reuses the prior. Both smoke arms use J12 for one seed and three decisions;
production retains five complete paired episodes, with the initial model probe
and all twelve round endpoints.
