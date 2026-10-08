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
