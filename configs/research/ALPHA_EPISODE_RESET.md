# Resetting temperature between matched episodes

This control isolates whether carrying temperature across episode boundaries
causes the MPPI500k policy-improvement regression seen in the completed
[temperature preservation experiment](ALPHA_PRESERVATION.md). It resets alpha
to the checkpoint value (approximately 0.0230667758733) and recreates its Adam
optimizer at each episode boundary. Alpha and Adam still carry across decisions
within an episode. Actor, critic, their optimizers and imagined replay remain
fresh at every decision. The outer checkpoint is frozen.

The new protocol is `episode-temperature-reset-v1`, campaign
`alpha-episode-reset-20261022`, selector `alpha_episode_reset/episode`. The
existing evaluator runs each three-episode stream in one process without
`--temperature-chain`; its ordinary episode reset also restores the paired
controller RNG for each environment seed. No engine or evaluator changes are
needed. Existing results retain their original protocol and publication identity.

The workload matches the completed MPPI500k cross-episode experiment:

- Five streams, with ordered environment seeds 101/102/103, 102/103/104,
  103/104/105, 104/105/101 and 105/101/102; controller seed 55 and SHA256-v1
  per-episode RNG seeding.
- Three full 500-decision episodes per stream, sampled real actions.
- H3/J8/C16/A4/N128/B256; reward-only auxiliary return critic for inner
  learning and terminal bootstrap; one temperature update per actor update.
- Actor, critic and temperature learning rates 0.0003, target entropy -10.5;
  no KL, Bernoulli copying, Retrace or extra temperature repeats.

Only the temperature and temperature-optimizer scopes change from `run` to
`episode`. Each episode therefore contains 16,000 temperature updates, starting
at Adam step and lifetime count zero. The driver checks checkpoint alpha at all
three initial decisions, 32-step Adam/lifetime continuity at subsequent
decisions, reset flags, full update budgets, exact seed order, reward totals,
finite diagnostics, unchanged outer state and trace hashes.

The primary endpoint remains episode three, now reporting the paired return
difference from across-episode carry, as well as gain over the sampled prior
and difference from fresh-alpha SAC. All three episode positions, individual
paired outcomes, sample standard deviations and entropy trajectories remain
visible. A positive difference from across-episode carry favors episode reset.
Use entropy and return together: entropy near target alone does not establish
successful policy improvement. Five streams are an exploratory comparison;
no equivalence or noninferiority claim is implied.

Because each episode resets the scientific state and reuses the same
per-seed RNG, the five balanced episode-reset curves should agree across
positions up to hardware/numerical effects. Repeating a seed at another
position is not an additional independent observation. Compare five paired
streams at each episode position rather than pooling 15 returns as independent
seeds. The first episode also provides a useful matching check because both
methods start from checkpoint alpha and fresh Adam there.

Preparation revalidates all five completed cross-episode histories from
`/oscar/scratch/rgao48/ambi/alpha-preservation/20261022-b315ca33/campaign.json`,
including frozen-state checks and full diagnostic traces. It also revalidates
and reuses that campaign's existing sampled prior and fresh-alpha controls.
Compact reference summaries and exact source hashes are pinned into the new
campaign; existing artifacts are never overwritten. No new every-decision-reset
or prior evaluation is submitted.

```bash
python slurm/ambi_alpha_episode_reset.py prepare --root "$EPISODE_RESET_ROOT"
python slurm/ambi_alpha_episode_reset.py run \
  --campaign "$EPISODE_RESET_ROOT/campaign.json" --index 0
```

Indices 0–4 own independent output directories. Before production, stream 0
runs two eight-decision episodes with `--smoke-steps 8` and a dedicated smoke
root for each selected GPU class. This checks the actual episode boundary and
measures runtime and memory. The scheduler reuses the measured preservation
resource procedure, requests one CPU and 4 GiB host memory per GPU worker,
and requires the separate comparison workspace before dispatch.

Each output contains `results.json`, a full trace bundle, `chain-summary.json`
(the common stream-summary filename, despite the reset protocol), and a
validated receipt. The publisher shows the two reused controls, existing
cross-episode carry, and new episode-reset results as separate native curves.
Incomplete or invalid streams remain explicit progress/failure states; they
cannot silently become partial five-stream performance estimates.
