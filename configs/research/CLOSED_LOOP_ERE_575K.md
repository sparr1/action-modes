# ERE replay at the 575k shared-representation checkpoint

The matrix `ambi_closed_loop_ere_f025_575k.json` covers **H = 1, 2, 3** and
**J = 1, 2, 4, 6, 8, 10** with final replay fraction **f = 0.25**. It selects
the target-entropy −10.5, shared-representation backbone
`rwgao_b-brown-university/ambi/aux6428346x0` at 575,000 training decisions.
The checkpoint SHA256 is
`0c6955db7cb8555a67d7863344b70be68f4b3250814d131e647ee6f9ef01a042`.

Every real decision starts a fresh inherited actor, return critic, target
critic, replay and optimizer state. Both inner and terminal critics use the
return-only critic; critic targets are one-step reward-only. The final actor
mean action is executed throughout each complete 500-decision episode. The
outer model remains frozen. The paired environment seeds are 101–105 with
controller seed 55, and 32 independent model-return probes are retained at
initialization and after each round.

Each round collects N128 imagined rollouts, performs C16 critic updates followed
by A4 actor/temperature updates, and uses batch size 256 with replacement.
The temperature is inherited from the checkpoint (initial alpha
0.004603903274983168) and adapted with inherited target −10.5. ERE changes the
eligible replay window for both critic and actor; all other settings match
the corresponding historical uniform evaluation. Replay capacity retains all
collected transitions, matching each historical configuration.

With R available collection rounds, component update k among K updates uses
the latest `min(R, max(1, ceil(R * 0.25 ** (k / (K - 1)))))` rounds. Each entire
collection round remains intact across its trajectories and rollout depths.
Critic and actor schedules restart separately with K16 and K4. This is fixed
recency strength: there is no annealing across rounds or real decisions. It
is an AMBI adaptation of ERE, not the original paper's complete training recipe.

At J1, the eligible window always contains the sole collection round. Tests
verify identical replay sampling and RNG state for uniform and ERE in this
case. The three completed J1 uniform evaluations are therefore reused with
their original manifests, trace hashes, scientific identities, performance
and training run IDs. The campaign creates 15 new settings / 75 episodes.
All 18 historical uniform controls and the frozen-prior reference remain
available for paired comparison; no reference episodes are regenerated.

`ambi_closed_loop_ere_f025_575k_refs.json` pins immutable reference manifests,
checkpoint and metadata hashes, commits and original publication identities.
Preparation validates full coverage, seed pairing, runtime compatibility,
resolved configurations and publication receipts. Each new planner explicitly
creates a new evaluation-series registry entry. The CPU publisher owns W&B
publication, retains training traces, and presents return and paired differences
against both uniform replay and the frozen prior. Missing results remain
missing. Five environment seeds support exploratory paired comparisons, not
independent training-seed conclusions.

Prepare on a CPU allocation, then run the selected smoke indices and production
indices through `slurm/run_ambi_closed_loop_ere_oscar.sbatch`. This launcher
uses `CAMPAIGN_ROOT`, `EXPECTED_ACTION_MODES_SHA`, and `EVAL_MODE`
(`prepare`, `worker`, or `watch`). Preparation additionally requires
`CHECKPOINT_INVENTORY` and `EVAL_REFERENCES_PATH`; `EVAL_MATRIX_PATH` is an
optional matrix override. Registry ownership is pinned to
`/oscar/scratch/rgao48/ambi/evaluation-series/runs`. Worker
validation checks every replay window, sampling count, optimizer count,
finite metric, reference hash, and frozen outer-state invariant.
