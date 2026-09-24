# Reward-critic alpha, Retrace and horizon follow-ups at 575k

The matrix `ambi_closed_loop_alpha0_lambda1_h4_575k.json` contains 62 new
settings, each evaluated for five full 500-decision episodes. It uses the
575,000-step target −10.5/shared checkpoint from `ambi/aux6428346x0`, SHA256
`0c6955db7cb8555a67d7863344b70be68f4b3250814d131e647ee6f9ef01a042`.
Both the adapted inner critic and frozen terminal critic use `aux_return`.
Inner targets are reward-only; terminal entropy is disabled. The actor and
terminal actor use the SAC policy.

| Panel | Horizons | Inner rounds J | Estimator | Alpha | Executed action | Settings |
| --- | --- | --- | --- | --- | --- | --- |
| A | 1, 2, 3 | 1, 2, 4, 6, 8, 10 | One-step and Retrace λ=0.9 | Zero | Final squashed-Gaussian sample | 36 |
| B | 1, 2, 3 | 1, 2, 4, 6, 8, 10 | Retrace λ=1 | Adaptive | `tanh(mu)` | 18 |
| C | 4 | 1, 2, 4, 6, 8, 10, 12, 14 | One-step | Adaptive | `tanh(mu)` | 8 |

Panel A sets `inner_entropy_enabled=false` and
`inner_temperature_mode=inherit_outer`. The entropy gate makes alpha exactly
zero; it does not set an invalid numeric temperature of zero. Actor updates
maximize return Q without an entropy bonus. Critic targets remain reward-only.
No temperature optimizer is allocated or stepped. Sampling still uses the
adapted Gaussian at its full standard deviation: zero alpha does not make
execution deterministic. Return-critic initialization matches the return
objective; the pretrained stochastic actor remains the initialization.

Panels B and C retain the inherited adaptive temperature, initially
0.004603903274983168, and inherited target entropy −10.5. Retrace λ=1 still uses
clipped importance coefficients `min(1, pi/mu)` and a frozen return tail; it is
not an uncorrected Monte Carlo return. H1 Retrace has no interior multi-step
correction but its data sampling can differ from the one-step implementation.

Every decision starts fresh from the frozen checkpoint: actor, critic,
temperature, optimizers and replay have action scope. Every inner round collects
128 trajectories and performs 16 critic and 4 actor updates, with batch size
256. Panel A has zero temperature updates; B/C perform 4 per round. Retrace
samples `ceil(256/H)` whole trajectories (256, 128 or 86 for H1–3), so H3 uses
258 critic rows per update. Actor and temperature batches remain 256. C uses
one-step transition sampling. Replay is retained across rounds, sampled with
replacement and reset at the next real decision.

The replay capacity is `max(historical_capacity(J), 128*H*J)`. The historical
capacity is 3072 through J8, 3840 at J10, 4608 at J12 and 5376 at J14. H4 thus
uses 3072 through J6, then 4096/5120/6144/7168 at J8/10/12/14. No setting evicts
collected transitions. H4 extrapolates beyond this checkpoint's recurrent
training horizon of three; the resolver warning is expected. Changing H also
changes collected data, replay distribution and terminal-boundary frequency,
so an H4-versus-H3 difference cannot isolate model error.

Environment seeds are 101–105 with controller seed 55 and the existing
SHA256 episode-seed mapping. The outer model stays frozen. All settings retain
32 fixed-noise, entropy-free return-to-go probes initially and after every
round, full inner traces, strict compilation with zero fallback, and the
isolated reproducible execution RNG for sampled actions.

## Matched references and interpretation

The hash-pinned reference inventory contains 56 completed adapted-policy
settings plus the completed prior. References keep their original source
commits, manifests, episodes and publication identities; no reference is rerun.

- A pairs each zero-alpha result against adaptive-alpha sampled execution at
  the same H/J and estimator (λ=0.9 for Retrace).
- B pairs λ=1 mean execution against one-step mean execution at the same H/J.
  This tests the estimator change. Existing λ=0.9 Retrace references use sampled
  execution; any secondary comparison to those changes both λ and execution.
- C pairs H4 against H3 one-step mean execution at the same J, including the
  completed J12 and J14 points. Replay allocation changes only to retain all
  transitions. Its companion overview adds H4 to the existing mean-action
  horizon comparison without rerunning earlier points.

Pairing uses the exact environment and solver seed identities. Publishers
show partial progress only for validated complete five-episode settings and
keep missing results absent. The aggregate overview separates the three panels
and retains source run IDs; per-setting performance, diagnostic traces and
probe histories remain available.

## Preparation and execution

`slurm/ambi_closed_loop_alpha0_lambda1_h4.py prepare` accepts `--root`,
`--checkpoint`, `--inventory`, `--registry`, and optional `--matrix`,
`--references`, `--group`, `--label`. It verifies the completed references,
generates evaluation-series specifications separately under `specs/policy_sample`
and `specs/mean`, checks canonical planner identities
and creates exactly 62 new local publication registrations. Preparation does
not evaluate episodes or publish W&B runs. The matrix's unselected `prior`
variant satisfies the preset schema only; it never creates a run or a result.

`worker --root ROOT --index INDEX [--smoke]` owns one complete setting.
Indices 0–9 are maximum-budget smoke representatives: six A paths at J10,
three B paths at J10, and H4/J14. Smokes use seed 101 for three real decisions.
Production uses all five full episodes. Completion receipts pin manifest and
trace hashes after scientific validation and sealing.

`slurm/run_ambi_closed_loop_alpha0_lambda1_h4_oscar.sbatch` supports
`EVAL_MODE=prepare|worker|watch`, `CAMPAIGN_ROOT`,
`EXPECTED_ACTION_MODES_SHA`, `EVAL_SMOKE`, and the standard
`CHECKPOINT_PATH`, `CHECKPOINT_INVENTORY`, `EVAL_REFERENCES_PATH`,
`EVAL_MATRIX_PATH`, `EVAL_CAMPAIGN_GROUP`, `EVAL_CAMPAIGN_LABEL` preparation
variables. Resources and array concurrency are set by submission using live
Oscar allowance. Source must be a clean checkout at the pinned commit.
