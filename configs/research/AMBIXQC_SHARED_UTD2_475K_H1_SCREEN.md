# AMBI-XQC shared UTD2 475k: first inner-adaptation screen

This exploratory screen applies fresh H1 XQC adaptation at every real decision
of five full HumanoidWalk episodes. It is a checkpoint experiment, not backbone
training and not the SAC `closed-loop-refinement-v1` reference recipe.

The only supported checkpoint is shared auxiliary representation, XQC UTD2,
475,000 training decisions from the six-bank 500k campaign. Its SHA256 is
`00d9347f00eb7719f55d4e3ef7abff8c24d7f2e214c923cdeba7b8e34afa1368`.
The existing six-bank inventory supplies the checkpoint, metadata, training
validation and original source-run association; the driver verifies all of them.

## Four conditions

| Array index | Critic initialization / frozen horizon | Target | J | Actor and temperature LR |
|---|---|---|---:|---:|
| 0 | auxiliary return / auxiliary return | reward only | 1 | 5e-5 |
| 1 | main XQC / main XQC | entropy augmented | 1 | 5e-5 |
| 2 | auxiliary return / auxiliary return | reward only | 4 | 1.25e-5 |
| 3 | main XQC / main XQC | entropy augmented | 4 | 1.25e-5 |

All conditions use H1, 256 imagined branches per round, batch size 256, three
critic updates per round, policy delay 3, critic LR 5e-5, replay capacity 1024 with
replacement, and collection followed by updates each round. The saved critic
optimizer rate at this checkpoint is approximately 4.35e-5; 5e-5 is a nearby
constant inner rate. The inner optimizer starts fresh and does not inherit the
outer optimizer's moments or resume its schedule.

The checkpoint's inherited temperature is approximately 0.00047462 and its
saved real reward scale is approximately 38.012. These values are read from
the checkpoint, not independently initialized tuning parameters.

J1 uses 256 imagined transitions and C3/A1/T1 updates per decision; J4 uses 1024
transitions and C12/A4/T4. The sum of actor learning rates is held at 5e-5.
Adam trajectories, collected data and critic-update work still differ; this is
not an exact trust-region or actor-step-only comparison. Earlier shared-backbone
experiments showed excessive policy drift at J8 with actor LR 5e-5, motivating
this smaller initial screen and a lower per-update rate for longer adaptation.

The persistent XQC actor initializes the inner actor and supplies frozen outer
horizon actions. Temperature is inherited and adapted on the delayed actor
schedule. Return-only targets omit the explicit entropy subtraction, while
actor entropy regularization remains active. The two routes are pure pairings;
there is no mixed soft-value/return-only bootstrap in this screen.

Inner actor BN uses inherited running statistics without updating its buffers;
actor weights and BN affine parameters still learn. This matters at H1 because
every actor-training latent is the same root state. Critic BN retains XQC's
joined replay/policy training batches, target batch statistics without buffer
updates, and running statistics for actor-loss and frozen outer-tail queries.
Reward normalization uses the frozen real-data scale. The saved training replay
archive is retained but is not sampled by this experiment.

At every real decision the actor, selected online/target critics, temperature,
replay and optimizers reset from the frozen checkpoint. The executed action is
the adapted actor mean. Evaluator state digests cover the complete frozen outer
state, including auxiliary networks, optimizers, BN, reward state and RNG.
Per-decision diagnostics are recorded for every action. Actor-BN regression
tests additionally verify buffer ownership, gradients and repeated reset behavior.

## Protocol and reuse

Production uses environment seeds 101–105, controller seed 12345 and 500 real
decisions per episode. Each condition runs only its own inner controller.
Existing actor-mean episodes at this exact checkpoint are reused; MPPI and prior
episodes are not rerun. Reference acceptance checks its manifest hash,
checkpoint and sidecar hashes, clean source provenance, complete frozen prior
episodes and identical environment/action/seed protocol. The earlier evaluator
has a different source fingerprint; compatibility is asserted for the reference
episode contract, not by claiming identical source code or runtime timing.

Smoke mode runs the same four controllers for seeds 101–102 and three decisions
each, with no publication and no incompatible full-episode prior reference.
Production requires all four passing smoke bundles from the exact source and
matrix, plus a successful actor-BN regression log. Smokes are correctness checks,
not performance estimates. Review return gains and policy KL together with
critic loss/disagreement, gradient norms, clipping and timing. One training seed
and five evaluation episodes support exploratory conclusions only.

## Running and publication

The matrix is `ambixqc_humanoid_shared_utd2_475k_h1_screen.json`; the driver is
`run_ambixqc_inner_475k_screen.py`. Use `--eval-series-spec-dir` separately for
each `--index` before creating four explicit New identities with `eval_series.py`.
The run-map JSON has schema `ambixqc-inner-475k-run-map-v1`, the pinned
`checkpoint_sha256`, and `runs` mapping all four `controller/...` selectors to
distinct absolute registry directories. The prior-reference JSON has schema
`ambixqc-inner-475k-prior-reference-v1`, `checkpoint_manifest_sha256`,
`checkpoint_sha256`, absolute `bundle_path`, and `manifest_sha256`.

Submit `slurm/run_ambixqc_inner_475k_screen_oscar.sbatch` with the exact clean
source SHA, existing locked DMControl Python, immutable inventory and durable
result root. Override GPU type and array concurrency from live availability;
the default requests one GPU, six CPUs and 32 GiB for each condition. Use the
same GPU model for these four conditions. The launcher checks the runtime lock,
rejects output/cache reuse and keeps W&B disabled on GPU workers. Production
validates finite complete decision traces, exact C/A/T and model-step counts,
frozen normalization/state, selected critic semantics and paired gains before
staging an immutable result pointer. Partial outputs remain available on failure.

Run one `slurm/publish_ambixqc_inner_475k_screen_oscar.sbatch` CPU publisher per
registry, supplying `EVAL_CONDITION_INDEX` and the complete GPU array in
`EVAL_COMPUTE_JOBS`. The publisher retains the standard sole-owner lock and
uncertain-write reconciliation. After compute terminates it allows 180 seconds
for delayed pointer visibility, requires exactly the expected checkpoint and
selector, and verifies remote acknowledgement before success. An empty final
scan or missing acknowledgement fails explicitly. Repair publication from saved
bundles and existing registries; do not rerun completed GPU evaluations.
