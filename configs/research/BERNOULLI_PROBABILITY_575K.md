# Bernoulli retention probability at the frozen 575k checkpoint

[The probability matrix](ambi_bernoulli_probability_575k.json) adds **p = 0.25
and 0.75** for actor-only, critic-only and joint weight transfer. It retains
H = 1, 2, 3; J = 1, 2, 4, 6; environment seeds 101–103; controller seed 55;
and full 500-decision episodes. This is **72 new settings and 216 episodes**.
The completed [p = 0.5 screen](BERNOULLI_TRANSFER_575K.md) and matching fresh
controls are reused without rerunning their episodes.

The mechanism and diagnostics are unchanged: p is the probability of copying
each previous adapted parameter, with all other parameters restored to the
frozen prior. First decisions are fresh. Actor and critic mask streams remain
private, and the same random uniforms are used across probabilities, so masks
are nested at corresponding decisions within each component. Learners reset
their optimizers, temperature, targets and replay as documented in the
original screen. All resulting parameters remain trainable during the solve.

Retaining less may suppress harmful inherited changes while retaining more may
save useful adaptation. The screen tests that tradeoff at each H/J setting; it
does not establish a general optimum from one checkpoint and three development
seeds. The [sampled diagnostics](BERNOULLI_TRANSFER_DIAGNOSTICS.md) compare the
prior, donor, masked initialization and final learner at fixed states within a
decision. Cross-arm full-episode diagnostic averages can also reflect the
different states their controllers visit.

Preparation binds both sets of historical evidence before launching:

```bash
python slurm/ambi_transfer_discovery_campaign.py prepare \
  --root /path/to/new/campaign --checkpoint /path/to/575k.pt \
  --matrix configs/research/ambi_bernoulli_probability_575k.json \
  --reference-root /path/to/historical/discovery/campaign \
  --bernoulli-reference-root /path/to/completed/p50/campaign
```

The sbatch equivalents are `CAMPAIGN_MATRIX`, `REFERENCE_CAMPAIGN_ROOT` and
`BERNOULLI_REFERENCE_CAMPAIGN_ROOT`. The prepared manifest pins the source,
matrix and each reference campaign, result, manifest and completion receipt by
hash. Reused p = 0.5 results must match the checkpoint, seeds, budgets and
diagnostic protocol; they retain their original source and timing.

Eight same-commit GPU smokes cover all six new arms at H3/J2, joint p = 0.25
at H1/J6, and joint p = 0.75 at H2/J6. Each smoke uses seeds 101 and 102 for
three real decisions and verifies exact diagnostic-on/off controller and RNG
isolation. Production requires every smoke receipt. Concurrency uses current
Oscar capacity and retains L40S hardware for comparable controller timing.

The new `ambi-inner-bench` workspace compares 25%, 50% and 75% retention with
the historical fresh baseline and includes diagnostic and timing panels.
Reused cells are labeled historical and excluded from the 72-setting progress
denominator. Existing campaign workspaces and result files remain intact.
