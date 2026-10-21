# MPPI reference for entropy/KL checkpoint comparisons

`ambi_mppi_checkpoint_comparison.json` evaluates the frozen MPPI-trained
auxiliary-return backbone `ambi/mppi7530927x0` at 400k and 500k training
decisions. The companion profile pins checkpoint, metadata and existing
sampled-prior result hashes. No prior episodes or outer training are repeated.

The planner uses H3, eight iterations, 512 candidates, 64 elites, 24 actor
trajectories, temperature 0.5 and standard-deviation bounds 0.05–2. It shifts
the previous plan within each episode, resetting between episodes. Its terminal
bootstrap is the frozen online auxiliary-return critic with mean-pair reduction
and a sampled frozen SAC actor action. Work is 12,336 model steps per decision;
actor, critic and temperature optimizer updates are zero.

Five environment seeds 101–105 and controller seed 55 use the existing SHA256
episode seed scheme. Episodes last 500 real decisions. MPPI uses weighted elite
selection in evaluation mode, without the additional proposal noise used in
training. The protocol records `mppi_weighted_elite` and explicitly opts into
the `ambi_sampled_prior-v1` reference, rather than relabeling its action rule.
All gains subtract the same sampled-prior episodes used by the inner-SAC curves.

`slurm/ambi_mppi_checkpoint_comparison.py prepare --root ROOT` creates a new
campaign and raw evaluation registries, then adds the MPPI identity to a
campaign-local copy of the persistent curve profile. After reinstalling the
five-curve workspace layout, `publish --campaign ROOT/campaign.json` exposes
pending results. Each scheduled worker runs `run --campaign ... --index 0|1`;
32-decision smoke runs remain separate. `publish --require-complete` validates
and uploads complete summaries and updates the connected MPPI curve.

Exact episode records remain on Oscar. Uploaded source summaries are bounded
to 16 MiB per checkpoint, and the persistent curve keeps compact provenance.
Return, paired gain, sample standard deviations, controller time and actual
planner work are retained. SAC entropy/temperature diagnostics do not apply to
MPPI and are omitted; no SAC optimizer traces are manufactured for MPPI.
The four existing curve IDs and measurements are retained.

MPPI controller time is the summed wall time of `model.predict`, including
first-call startup. Model loading and environment stepping are excluded. This
path performs no unscored warmup, whereas the existing SAC evaluations exclude
one warmup action from their measured time. Thus startup accounting is
conservative for MPPI; the timing comparison is not an identical warmed-only
benchmark. Each MPPI result, bundle run and normalized record provenance stores
this explicit `timing_convention`. Return and paired-gain comparisons are
unaffected.
