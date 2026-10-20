# Automatic entropy plus a fixed prior-KL penalty

This follow-up adds a prior-KL penalty to the existing entropy-regularized inner
SAC objective. For each actor update it minimizes

```text
alpha_t * log pi(a | z) - Q_return(z, a)
    + beta * KL(pi(. | z) || frozen_prior(. | z)).
```

The prior is the inherited outer SAC actor, frozen throughout the solve.
The KL is the analytic diagonal-Gaussian KL in pre-tanh coordinates; the shared
invertible tanh transform preserves the distributional KL. The entropy term
uses the existing corrected squashed-action density. Automatic alpha tuning,
its inherited initialization, target entropy, fresh optimizer and learning rate
`0.0003` are retained. Beta is fixed throughout every solve at either one or
three times the **checkpoint's outer alpha**. It does not track the adapting
inner alpha. This is a soft penalty, not an adaptive coefficient or hard KL
budget. Actor and critic learning rates remain `0.0003`.

## Full sampled episodes

`ambi_entropy_kl_episodes.json` defines exactly four new cells:

| Checkpoint | Outer alpha | Beta multipliers | Fixed beta values |
| --- | ---: | --- | --- |
| SAC-aux 100k | 0.003365497337654233 | 1, 3 | 0.003365497337654233, 0.010096492012962699 |
| MPPI-aux 400k | 0.013081684708595276 | 1, 3 | 0.013081684708595276, 0.03924505412578583 |

The exact checkpoint and sidecar hashes remain in `ambi_entropy_probe.json`.
Each cell runs five paired environment seeds (101–105), controller seed 55,
and complete 500-decision episodes. Real actions are **sampled**, matching
training execution. This is the sampled extension of the original early-H3/J8
evaluation, distinct from the named mean-action `closed-loop-refinement-v1`.

Only `inner_outer_policy_kl_coef` changes from the existing sampled entropy
baseline. Each decision uses fresh cloned priors, H3/J8, 128 rollouts per round,
batch size 256, and C16/A4 per round: 128 critic, 32 actor and 32 temperature
updates. Reward-only auxiliary return Q initializes the inner critic and
supplies the horizon tail. No outer training, transfer, or weight writeback is
performed. The complete outer state remains frozen. The existing 32-rollout
model-return probe runs at decisions 0/100/200/300/400, before adaptation and
after each round; its cost is reported separately.

The campaign reuses compatible sampled prior and entropy baseline bundles from
the pinned early-H3/J8 campaign. These references are not rerun. Existing
KL-only results can be shown as a separate comparison; those disable entropy
and temperature updates and therefore remain a distinct objective.

## Paired saved-root diagnostics

The complementary `ambi_entropy_kl_probe.json` diagnostic uses the same 50 saved
roots: two checkpoints, five environment seeds and decisions 0/10/50/100/250.
Three full-solve arms use beta multipliers **0, 1 and 3**, each retaining
automatic entropy at alpha learning rate `0.0003`. Initialization, first replay,
first critic phase and diagnostic anchor noise are paired. Later policy and
imagined replay may diverge. No trajectories are recollected.

The root diagnostic records entropy, prior KL, mean/std changes, alpha dynamics,
Q/entropy/KL gradient balance and actual Adam-step projections. Those are
instrumented model diagnostics, not measured real return. Their timings
include observer overhead. Aggregate roots within each environment seed before
summarizing across seeds, and separate reset roots from later observations.

## Execution and publication

`slurm/ambi_entropy_kl_episodes.py prepare --root <new-campaign-root>` creates
the four local curve identities and pins source commit, matrix/profile,
checkpoint/sidecar, inventory and reused reference hashes. Selectors are
`entropy_kl/<checkpoint>_beta1` and `entropy_kl/<checkpoint>_beta3`; cell IDs use
`<checkpoint>__beta1` and `<checkpoint>__beta3`. The campaign ID is
`entropy-kl-episodes-20261020`.

Inside a scheduled GPU allocation, use
`slurm/run_ambi_entropy_kl_episodes_oscar.sbatch --campaign <campaign.json>
--index <0..3>`. The launcher requires the tested clean commit, locked Python
interpreter and explicit allocated CPU count. `--smoke-steps 8` or `32` runs
one seed with the full per-decision solve and writes a separate smoke bundle.
No resource request is hard-coded: choose GPU, CPU, memory, walltime and
parallelism after the measured GPU smoke and live account-capacity check.
Recent approximately 1.3-second decisions imply about 54 minutes for 2,500
controller decisions before initialization, model probes, validation and I/O;
use measured end-to-end throughput with a modest margin, not the previous
40-minute limit that left SAC workers incomplete.

One worker owns all five seeds and one result directory. There is no seed-shard
merge and no overwriting of existing outputs. Completed receipts require full
trace/seed coverage, 32 actual temperature steps per solve, sampled execution,
the exact beta and alpha learning rate, finite entropy/KL/alpha diagnostics,
strict CUDA compilation and unchanged outer state. Hash mismatches stop before
GPU evaluation. Failed workers preserve their saved partial bundles and write
a failure receipt when Python can handle the error; a scheduler kill may leave
only the started record and partial bundle.

GPU workers disable W&B. The separate `slurm/ambi_entropy_kl_publish.py`
publisher creates a dedicated native comparison with four new entropy-plus-KL
curves and the existing entropy/KL-only reference curves. Full traces stay on
Oscar; bounded result summaries and trace indexes are published. Finalize with
`--once --require-complete` so missing or failed cells cannot silently appear
complete. Diagnostic banks remain local reports and are never published as
episode-return curves.
