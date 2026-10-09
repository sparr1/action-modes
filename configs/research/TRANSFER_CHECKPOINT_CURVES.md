# Transfer candidates across the auxiliary-return checkpoint bank

This exploratory evaluation measures policy improvement throughout the training
of `ambi/aux6428346x0`. It evaluates the 80 available periodic checkpoints,
25k through 2M training decisions at 25k intervals, using the shortlist selected
at 575k. The versioned selection is `ambi_transfer_checkpoint_curves.json` and
the exact transfer definitions are `ambi_transfer_checkpoint_shortlist.json`.

| Controller | H | J | Transfer scope |
| --- | --- | --- | --- |
| Bernoulli 75%, actor and critic | 1 | 2 | All learned parameters |
| Bernoulli 50%, critic | 1 | 4 | All learned critic parameters; actor fresh |
| Actor shrink-to-prior 50% | 1 | 4 | Actor weight matrices only; critic fresh |
| Fresh SAC | 1 | 2 | No transfer |
| Fresh SAC | 1 | 4 | No transfer |
| Frozen prior mean | — | — | No inner solve; validated existing results |

Bernoulli copying draws independent masks for each learned parameter entry:
selected entries take the previous solve's final value, and other entries take
the frozen prior value. Masks use private actor/critic random streams and are
redrawn at each decision. Dense shrinkage initializes actor weight matrices as
`prior + 0.5 * (previous_final - prior)`; biases, normalization parameters and
buffers reset. Both mechanisms use only the immediately preceding solve in the
same episode. The first solve starts from the prior, and episode reset clears
the donor.

All inner controllers reset Adam, temperature and imagined replay at each real
decision. The critic target starts from the initialized online critic. Every
round uses C16/A4/N128/B256, a reward-only critic target, and the frozen
auxiliary-return terminal critic. The actor retains its entropy term. Initial
alpha and target entropy inherit the selected checkpoint's outer values.
Adaptation executes the actor mean after every solve; the outer checkpoint
remains frozen. No training continuation or online speedup is measured here.

Each checkpoint/setting owns five complete 500-decision episodes, using
environment seeds 101–105 and controller seed 55. Existing private per-episode
solver streams are preserved. A complete new campaign has 400 evaluated cells
and 2,000 episodes; the 400 historical prior episodes are reused after strict
checkpoint, protocol, metadata and source verification. Historical transfer
screens had different seed counts or parameter scopes, so their results are
not substituted into the new five-seed evaluation.

Paired improvement over the prior subtracts the frozen SAC actor's return for
the same checkpoint and environment seed before computing mean and sample SD.
The matched fresh SAC controls distinguish transfer gains from the benefit of
inner optimization itself. Compare controller return, paired gain, episode SD,
and controller latency over training; a high return at the selection checkpoint
alone is insufficient evidence of a broadly useful mechanism.

All already-computed finite scalar inner metrics are retained, including Q,
TD error, entropy, losses, gradients, temperatures and realized update budgets.
Sampled observational diagnostics also remain enabled at decisions
0, 1, 25, 100, 250 and 499, with stationary fits at 25 and 250. They cover fixed
model targets, action gradients, policy changes, feature activity/rank and
stationary-fit plasticity. Probes use independent randomness and disposable
copies, are measured separately from controller time, and are model surrogates
rather than real-return ground truth. Diagnostic coverage is a completion gate;
GPU smoke additionally verifies exact action, reward, learner and RNG isolation.
See `TRANSFER_DIAGNOSTICS.md` for the measurement definitions.

Controller time includes initialization, prediction and donor export. The
steady timing summary excludes decisions 0–9; complete controller timing and
first-decision costs remain in raw records. Compute uses L40S GPUs to retain
the prior timing hardware class. Historical prior late timing remains missing.

The checkpoint inventory pins weights and metadata hashes. Preparation checks
the shared backbone identity, generates exact per-checkpoint contracts and
records the evaluator's full source fingerprint. The Oscar worker is
`slurm/ambi_transfer_curve_campaign.py`, launched through
`slurm/run_ambi_transfer_curves_oscar.sbatch` from a clean pinned Git commit.
Production requires smoke receipts at the earliest checkpoint, 575k and the
latest checkpoint, covering all five settings with two seeds and three
decisions. Independent checkpoint/setting cells are parallelized according to
live account and hardware availability.

Publication creates a new comparison attempt in `ambi-inner-bench`, with one
curve per setting and a black prior curve. The isolated workspace shows return,
paired improvement, episode variability, controller time and explicit
pending/running/completed progress. Error bands are sample SD across five
paired episodes, not confidence intervals. Checkpoints are repeated
measurements of one trained backbone, not independent training seeds; 575k
remains the exploratory selection point.

The six scientific runs retain immutable checkpoint-result histories. A
separate publication-only overview logs the combined points and progress
tables with `run.log`, which registers their table metric types for the W&B
workspace. Summary-only table updates were readable through the API but did
not render as panels. The saved workspace selects only the overview so the
six curves are not duplicated; their fixed colors remain consistent across panels.
The overview is presentation state and is not an additional evaluated method.

The publisher persists its overview run ID before contacting W&B and resumes
that ID after interruptions. Restarting the CPU publisher with `watch` safely
adds the overview to an existing prepared campaign. Workspace version 3 can
upgrade only an exact recognized version-2 layout, preserving its URL and all
other saved views; unknown user edits cause a visible setup failure instead
of an overwrite. API table/schema checks and authenticated browser rendering
are both required before calling the comparison visible.

Newly accepted W&B checkpoint artifacts store decision traces as lossless
`decisions-seed-<seed>.jsonl.gz` files. The publisher compresses them only after
validating the raw worker results; raw JSONL files on Oscar remain unchanged.
Every scalar and diagnostic row is retained. Gzip output has a zero timestamp
and no original-filename header, and decompression is checked against the raw
SHA256 and byte count before publication. The normalized record's
`provenance.decision_trace_storage.files` maps each compressed artifact name to
its original filename, raw/compressed hashes and sizes, and encoding.

Read a downloaded trace with `gzip.open(path, 'rt')` in Python, or recover the
raw bytes with `gzip -dc decisions-seed-101.jsonl.gz > decisions-seed-101.jsonl`.
The recorded raw SHA256 verifies the recovered file. Compression affects only
new records; previously accepted records and their publication identities are
never repacked or rewritten.

The existing H3 MPPI return-Q and soft-Q results can be displayed alongside
these curves without adding evaluation cells. `utils/transfer_mppi_comparison.py`
validates the two original 80-checkpoint registries against their published
record fingerprints, this campaign's checkpoint hashes and five paired seeds.
It recomputes return statistics and prior gains directly from the saved episodes.
The original sources remain `24f3f6be21b74961beeb7e846b880613` (return Q) and
`88e1989e3f31474c987798c1bd5a0370` (soft Q). Both use the same frozen SAC
proposal/terminal actor and MPPI H3/N512/E64/pi24/I8 with proposal-mean execution;
only the terminal critic route differs.

A separate publication-only run logs 160 MPPI table rows under the existing
overview selection. Its run ID, source pins and content hash are persisted in
`publication_root/mppi-overlay.json` before upload. Copy the verified published
receipt to the Oscar publication root before starting a publisher that should
retain this overlay. The publisher reads that receipt solely for the v5 layout;
it does not add scientific runs, rewrite the six existing histories, or change
the 400-cell completion count. The v5 upgrade preserves the saved view URL and
recognizes exact earlier layouts, rejecting unknown user edits.

MPPI has no matching H1 fresh-SAC control, so its matched-fresh gains stay null.
MPPI timing is also omitted: the first 40 checkpoints reuse a historical GPU
pool and include different diagnostic timing, while no matching decisions
10–499 timing is available. Return, paired-prior gain, sample SD and minimum
seed return use all five existing episodes at each checkpoint.

## J6 extension after 500k

`ambi_transfer_checkpoint_j6_after500k_curves.json` selects H1/J6 fresh SAC,
critic Bernoulli 50% copying of all learned critic parameters, and 50% actor
matrix shrinkage. It uses `ambi_transfer_checkpoint_j6_after500k_shortlist.json`
and restricts the existing backbone inventory to 525k–2M, inclusive, every 25k.
The 500k checkpoint itself is excluded. This is 60 checkpoints, 180 new cells,
and 900 full episodes on seeds 101–105. The joint Bernoulli 75% arm is omitted.
All C16/A4/N128/B256 settings, independent solver/transfer randomness, frozen
outer state, diagnostics and controller timing definitions above are retained.
J6 uses 768 imagined H1 transitions, 96 critic updates, and 24 actor updates per
real decision. Smoke checks cover all three settings at 525k, 575k and 2M.

Pass the new selection through `CAMPAIGN_CONFIG` to the existing Oscar CPU
preparation wrapper. The default remains the original J2/J4 campaign. Keep the
new campaign and publication directories separate from the completed original
campaign, and pin the new tested source commit before preparation.

The completed J2/J4, base actor and H3 MPPI evaluations remain reference data.
The J6 publication uses new scientific run identities and adds only its three
new curves to the existing comparison view. Its prior registry is a validated
read-only reference to the original published prior records. New transfer
artifacts link the existing prior artifact rather than uploading its traces
again. The new overview excludes prior rows, so the shared display contains
each checkpoint/setting once. Paired gains over fresh SAC use the new J6 fresh
control; they remain pending until both J6 settings complete at a checkpoint.
Use `prepare --comparison-host-publication-root ORIGINAL/publication` (or
`COMPARISON_HOST_PUBLICATION_ROOT` in the CPU wrapper) to bind the existing
comparison. The publisher pins the completed host and prior record hashes,
allocates three new scientific runs and one overview, and upgrades the exact
original v5 view to v6 at the same URL. It preserves unknown view edits by
refusing the upgrade. The combined tables contain 820 rows: 480 original,
160 MPPI, and 180 J6, including pending rows with null metrics.

The publisher retries transient shared-filesystem JSON read failures
(`ESTALE`, `EAGAIN`, `ETIMEDOUT`) up to four attempts, reopening the file each
time. Invalid JSON and scientific validation failures still stop immediately.
Collection failures now write a phase-specific `publisher-failure.json`, so
an interrupted upload can be diagnosed and resumed against the same registry
without repeating completed evaluations.

## J8 extension after 500k

`ambi_transfer_checkpoint_j8_after500k_curves.json` and its matching shortlist
repeat the three J6 mechanisms at H1/J8 over the same 60 checkpoints from 525k
through 2M. All five seeds, diagnostics, transfer definitions, inherited
temperature, controller timing and frozen-backbone checks are unchanged.
Each decision uses 1,024 imagined H1 transitions, 128 critic updates and
32 actor updates. The extension has 180 new cells and 900 full episodes, with
smoke checks for all three settings at 525k, 575k and 2M.

Use a separate campaign and publication directory with the new selection
through `CAMPAIGN_CONFIG`. Reuse the validated historical prior and retain the
completed J2/J4/J6 and MPPI comparisons. J8 transfer gains against fresh SAC
must use its new matched J8 fresh control. The scientific evaluator and learner
implementation remain unchanged.
