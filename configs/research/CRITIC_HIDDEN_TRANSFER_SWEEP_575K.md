# Matched 575K critic hidden-layer transfer sweep

This extends the [original H3 sweep](TRANSFER_SWEEP_575K.md) with eight new
conditions: soft/soft or return/return critics, J1 or J8, and solving every
real decision or every H=3 decisions. Each condition retains trainable critic
hidden layers and normalization and creates fresh Xavier-uniform weights and
zero biases for each final output layer at every solve, including the first.
The actor resets to its checkpoint prior. The target critic copies the online
critic after head initialization; replay, temperature, and all optimizer states
reset each solve. See [lifecycle semantics](CRITIC_HIDDEN_TRANSFER_575K.md).

The two launch matrices are:

- `ambi_critic_hidden_transfer_sweep_575k.json`: every-decision solving.
- `ambi_critic_hidden_transfer_hold_h_sweep_575k.json`: solve at decisions
  0, 3, 6, …, evaluating the held feedback actor on new observations in between.

Both select four `soft_soft_j1/critic_hidden_warm`,
`soft_soft_j8/critic_hidden_warm`, `return_return_j1/critic_hidden_warm`, and
`return_return_j8/critic_hidden_warm` conditions. The examples and their default
single-condition selection remain unchanged.

All requested algorithm settings match the old full-critic conditions exactly,
except `inner_critic_transfer_head="random"`: H3/C16/A4/N128/B256,
replay capacity 3072, uniform J from the first solve, controller seed 55,
environment seeds 101–105, and 500 mean-action decisions per episode. The
checkpoint is `aux6428346x0` at 575000 decisions, SHA256
`0c6955db7cb8555a67d7863344b70be68f4b3250814d131e647ee6f9ef01a042`.
This is 40 new full episodes. Fresh, actor-transfer, full-critic-transfer and
prior controls are historical comparison references; they are not rerun,
relabeled, or appended to a changed implementation identity.

## Planning and preparation

Run from the repository root with the locked DMControl Python. Planning is
read-only and checks the entire requested parameter dictionary against the
matched full-critic controls, in addition to the protocol and checkpoint pins:

```bash
python slurm/ambi_transfer_sweep_campaign.py plan --campaign-mode critic-hidden
python slurm/ambi_transfer_sweep_campaign.py prepare \
  --campaign-mode critic-hidden --root NEW_CAMPAIGN_ROOT \
  --inventory CHECKPOINT_INVENTORY
```

Preparation requires a clean committed checkout and creates a new directory;
it never replaces an existing campaign. The new manifest has schema version 2,
kind `ambi-critic-hidden-transfer-sweep-v1`, `campaign_mode="critic-hidden"`,
eight cells labeled `critic_hidden`, and explicit hidden/head initialization
metadata. The historical default `full` mode still prepares its original
24 conditions plus one prior reference with schema version 1.

## Oscar execution

Use `slurm/run_ambi_transfer_sweep_oscar.sbatch` from the clean synchronized
source checkout, pin `EXPECTED_ACTION_MODES_SHA`, `PYTHON_BIN`,
`CHECKPOINT_INVENTORY`, and a new `CAMPAIGN_ROOT`, and choose resources and array
concurrency from current account availability. Set `CAMPAIGN_MODE=critic-hidden`
for `EVAL_MODE=prepare`. Workers derive the mode from the immutable manifest.

First submit `EVAL_MODE=worker,EVAL_SMOKE=1` with array indices `0-7`, one L40S
per worker. Every new critic/J/cadence combination runs the actual checkpoint
for seven decisions on two seeds, covering first solves, later solves, held
feedback, episode reset, and an incomplete final hold block. Index 0 also runs
the four opt-in hidden-transfer CUDA lifecycle/strict-compilation cases from
`tests/test_critic_hidden_transfer_cuda_gate.py`; reports are checked and sealed
into its completion receipt.

After all eight smokes succeed, submit `EVAL_MODE=worker,EVAL_SMOKE=0` with
array indices `0-7`. Each worker owns all five paired full episodes for one
condition. Production workers refuse to start until every smoke receipt and
the CUDA gate are present and hash-verified. Worker directory ownership,
source/checkpoint/config identity, hidden-transfer metadata, trace chronology,
head/target reset flags, cumulative critic updates, frozen outer state and full
episode completion are validated before a completion receipt is written.
Do not retry an interrupted worker into an occupied output directory; use an
explicit new campaign attempt with its own publication identity.

```bash
python slurm/ambi_transfer_sweep_campaign.py status --root CAMPAIGN_ROOT --verify
```

GPU workers write immutable local bundles. A separate CPU publisher handles W&B
and comparisons against explicitly supplied historical campaign/publication
roots. Those references retain their original source identity and run URLs;
the new hidden-transfer results receive new identities and publication runs.
