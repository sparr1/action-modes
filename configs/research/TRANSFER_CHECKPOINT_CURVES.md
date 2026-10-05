# Transfer candidates across the auxiliary-return checkpoint bank

This exploratory evaluation extends six settings selected at checkpoint 575k
across the available checkpoints of `ambi/aux6428346x0`. The initial campaign
uses all 80 periodic checkpoints, from 25k through 2M decisions at 25k intervals.
The versioned selection is in `ambi_transfer_checkpoint_curves.json`.

| Inner horizon | Inner rounds | Actor retention | Critic retention |
| --- | --- | --- | --- |
| 2 | 10 | 0.5 | 0 |
| 1 | 4 | 0.5 | 0 |
| 2 | 6 | 0 | 0.5 |
| 3 | 6 | 0 | 0.5 |
| 2 | 2 | 1 | 0.5 |
| 1 | 1 | 1 | 0 |

At every real decision, retained parameters initialize as
`(1 - retention) * frozen_prior + retention * previous_solve_final`.
The first solve starts from the prior, and episode reset clears the donor.
Targets are copied from the selected online critic. Adam, temperature and
imagined replay reset at each solve. Every round uses C16/A4/N128/B256; the
critic target is reward-only, with the frozen auxiliary-return terminal critic.
The actor keeps its learned entropy term. The adapted actor's mean action is
executed after every solve, with no skipped decisions or outer learning.

Each checkpoint/setting owns three complete 500-decision episodes, environment
seeds 101–103 and controller seed 55, with the existing private per-episode
solver streams. The same seeds are selected explicitly from historical
five-episode prior bundles. Paired gain is transfer-controller return minus
the frozen SAC prior's mean-action return for the same checkpoint and seed.
This measures the combined benefit of inner improvement and transfer over the
prior; it does not isolate transfer against a fresh inner solve.

The checkpoint inventory pins weights and metadata hashes. Preparation checks
the common backbone configuration and generates a separate exact-checkpoint
contract for each checkpoint. Historical 575k results are imported only after
checking their original receipts, source compatibility, resolved controller
parameters, full decision traces, seeds and frozen-state verification. They
retain their original provenance and are excluded from production work.

New evaluations retain all already-computed finite scalar inner metrics,
including Q, TD error, actor entropy, losses, gradients, temperatures and
realized update budgets. This adds no diagnostic rollouts or independent
reference probes. Reused 575k results retain their original lean coverage;
unrecorded metrics remain unavailable. The separate transfer-diagnostic
protocol is documented in `TRANSFER_DIAGNOSTICS.md`.

Controller timing includes initialization, prediction and donor export. The
steady timing summary excludes decisions 0–9 in each episode; complete timing
and first-decision costs remain in the raw records. Compute uses L40S GPUs to
retain the original timing hardware class. Historical prior timing is not
presented as a matched controller benchmark.

The Oscar worker is `slurm/ambi_transfer_curve_campaign.py`, launched through
`slurm/run_ambi_transfer_curves_oscar.sbatch` from a clean pinned Git commit.
Production requires smoke receipts at the earliest checkpoint, 575k and the
latest checkpoint, covering every selected setting with two seeds and three
decisions. Concurrency is chosen from live account and hardware availability.

Publication creates an explicit new comparison attempt in `ambi-inner-bench`,
with one curve per setting and one prior curve. The saved workspace shows
return, paired gain, steady controller time and pending/running/completed
progress, using a distinct color for each setting and black for the prior.
Error bands are sample standard deviations across three episodes, not
confidence intervals. Checkpoints are repeated measurements of one trained
backbone and do not provide independent training seeds. The settings were
selected using 575k performance, so that checkpoint is a selection point.
