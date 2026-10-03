# Training horizon versus frozen MPPI horizon

This campaign compares the shared auxiliary-return SAC backbones trained with
H=1, 3, 5, and 7. Their entropy target is -10.5, with automatic temperature,
directly clamped SAC actor log standard deviation, and the same training recipe.
The H=3 source is `ambi/aux6428346x0`; the other three sources are identified by
their original training launch records. Evaluation does not retrain anything.

Each backbone and explicitly selected checkpoint is evaluated with MPPI horizons
1, 3, 5, and 7. Every planner runs both terminal-critic choices: the online soft
SAC critic and the online auxiliary return-only critic. Both use the same frozen
SAC actor for policy proposals and terminal actions, raw predicted rewards,
mean-pair terminal Q, and no explicit entropy correction. The inactive
`inner_critic_source` remains `sac`; only `inner_horizon_critic_source` selects
the return critic. The policy prior is evaluated once per backbone/checkpoint.

The tested MPPI settings remain 512 candidates, 64 elites, 24 policy
trajectories, eight effective iterations, temperature 0.5, standard deviation
bounds [0.05, 2], weighted proposal-mean execution, and previous-mean warm start
within each episode. The exact model-step budget is
`8 * 512 * planning_H + 24 * (planning_H - 1)`:

| MPPI horizon | Model steps per real decision |
|---:|---:|
| 1 | 4,096 |
| 3 | 12,336 |
| 5 | 20,576 |
| 7 | 28,816 |

The saved training horizon is inherited from the checkpoint and is never
replaced with the planning horizon. Planning beyond the training horizon can
emit an expected model-bias warning. Five full episodes use environment seeds
101–105, controller seed 55, the existing SHA256 episode RNG scheme, and at most
500 decisions. Frozen model, optimizer, temperature and update counters are
checked; no outer or inner learning occurs.

## Preparation

`slurm/ambi_aux_horizon_mppi_campaign.py` is orchestration only. It calls the
existing evaluator and evaluation-series APIs without changing their scientific
implementation. Its versioned recipe is `ambi_aux_horizon_mppi_campaign.json`.

An inventory must select one common nonempty checkpoint grid for all four
backbones. No checkpoint range is assumed. Every row lies on the existing 25K
grid through two million training decisions. Inventory shape:

```json
{
  "schema_version": 1,
  "backbones": [
    {
      "train_horizon": 1,
      "source_run": "entity/project/run-id",
      "training_launch": "/absolute/training/launch.json",
      "checkpoints": [
        {"step": 575000, "path": "/absolute/model_575000"}
      ],
      "prior_bundles": {},
      "mppi_bundles": {}
    }
  ]
}
```

Supply all four backbone entries. Optional row fields are `sha256`,
`metadata_sha256`, `metadata_path`, and `bytes`. Preparation computes and pins
missing hashes, checks supplied hashes/sizes, verifies every sidecar's training
horizon and complete scientific recipe, and binds the original launch source
commit/run identity. Hashing a full checkpoint bank belongs on an allocated CPU
node, not a login node.

Reuse maps are explicit: `prior_bundles` maps checkpoint-step strings to
completed bundle paths; `mppi_bundles` maps step strings to objects whose keys
are MPPI-horizon strings and whose values are completed two-arm bundles. For
example, `"575000": {"3": "/absolute/existing/mppi/bundle"}` reuses both H=3
planner arms at 575K. Compatible historical H=3/H=3 and prior results are staged
into the new attempt after exact checkpoint, scientific source, planner,
protocol, episode completeness and frozen-state validation. Reused cells are
removed from compute tasks. Existing artifacts and evaluation curves remain
unchanged.

From the clean tested checkout, using the existing locked DMControl interpreter:

```bash
python slurm/ambi_aux_horizon_mppi_campaign.py prepare \
  --inventory /absolute/selected-inventory.json \
  --output-root /absolute/fresh-campaign \
  --registry-root /oscar/scratch/rgao48/ambi/evaluation-series/runs \
  --attempt-label aux-horizon-mppi
```

The output root must not already exist. This writes `preparation.json`, hashed
checkpoint inventories, generated matrices, metadata-only curve specifications,
and explicit new run assignments. There are 32 MPPI curves and four prior curves.
Preparation contacts no W&B service and executes no environment steps. Curve
creation is intentionally New; rerunning preparation creates a distinct attempt
and therefore requires a fresh output root.

## Workers and publication

`preparation.json` contains `prior_tasks` and `mppi_tasks`; indices are zero based.
One MPPI task owns one backbone/checkpoint/planning-H result with both critic
arms. Each task retains all five paired seeds, so no seed-shard merge is needed.
Run missing priors first, then MPPI workers after their references exist:

```bash
python slurm/ambi_aux_horizon_mppi_campaign.py worker \
  --campaign /absolute/fresh-campaign/preparation.json --kind prior --index 0
python slurm/ambi_aux_horizon_mppi_campaign.py worker \
  --campaign /absolute/fresh-campaign/preparation.json --kind mppi --index 0
```

Use the Oscar wrapper with live account/QOS limits and GPU availability. Multiple
adjacent task indices may run sequentially in one allocation to reduce scheduler
startup overhead or stay within submission limits. Keep useful independent
allocations concurrent. Temporary files and caches belong in scratch or node
local storage. Production output directories refuse replacement.

Before production, add `--smoke` to selected MPPI workers to execute two seeds
and three decisions with the actual production matrix and checkpoint. Smoke
bundles go under a separate `smoke/` directory and are never staged for
publication. Select at least H=1 planning on every training horizon; include
larger-H cases needed to validate the selected hardware/resource requests.
Production workers verify checkpoint hashes again, resolved horizons, both
critic routes, exact model-step budget, zero optimizer steps, complete paired
episodes and finite diagnostics before staging. An MPPI task also validates its
prior pairing before staging. GPU workers never initialize W&B.

Run the existing CPU evaluation-series publishers for all selected run
directories, then wait for base MPPI records to be acknowledged. Stage the
maintained paired-reference supplements with:

```bash
python slurm/ambi_aux_horizon_mppi_campaign.py pair \
  --campaign /absolute/fresh-campaign/preparation.json --train-horizon 1
```

Repeat for training horizons 3, 5 and 7, then run the publishers again for their
MPPI curves. This uses the existing strict pairing and immutable supplemental
publication journal. Gains are return minus the corresponding backbone's frozen
policy-mean return, paired by environment and solver seed. The evaluator's
`--reference-bundle` path is deliberately not used because its prior and MPPI
action-rule identities differ. Missing checkpoints, pending base publication,
incompatible references and altered artifacts fail explicitly. Pairing is
idempotent after acknowledgement and does not reevaluate any episodes.
