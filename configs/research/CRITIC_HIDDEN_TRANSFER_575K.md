# Critic hidden-layer transfer with a fresh output head

This opt-in extension retains the inner critic's trainable hidden layers and
normalization parameters across solves while randomly reinitializing its final
output layer at **every solve, including the first solve of an episode**.
It supports both soft/soft and return/return critics and both every-decision
and hold-H feedback execution. Existing full-critic, actor-only, and fresh
configurations keep their behavior and result identities.

The motivation is the multi-head critic ablation in
[Disentangling Transfer in Continual Reinforcement Learning](https://papers.neurips.cc/paper_files/paper/2022/file/2938ad0434a6506b125d8adaff084a4a-Paper-Conference.pdf).
That paper transfers between distinct tasks and measures subsequent SAC learning
over long real-environment training runs. AMBI transfers between short imagined
solves within one task. This is a paper-inspired initialization ablation, not a
reproduction of its task sequence, head bank, architecture, or sample budget.
Its [official supplementary code](https://papers.neurips.cc/paper_files/paper/2022/file/2938ad0434a6506b125d8adaff084a4a-Supplemental-Conference.zip)
uses fresh Glorot-uniform (Xavier-uniform), zero-bias output layers in
`continualworld/sac/models.py`. Its transfer path in `continualworld/sac/sac.py`
retains the lagged target critic; this AMBI variant instead preserves our
per-solve hard-copy lifecycle, copying the online critic after the head reset.

## Configuration and lifecycle

Use the existing auxiliary dense critic-only transfer settings, plus:

```json
{
  "inner_actor_scope": "action",
  "inner_critic_scope": "episode",
  "inner_critic_transfer_head": "random",
  "inner_critic_target_initialization": "online",
  "inner_rebase_persistent": false
}
```

`inner_critic_transfer_head` defaults to `"retain"`, which preserves the existing
full-critic transfer. The `"random"` mode requires auxiliary SAC, dense actor and
critic clones, checkpoint-prior network initialization, action-local actor,
replay, temperature and all optimizer scopes, online target initialization,
no persistent rebasing, and no outer writeback. Simultaneous actor and critic
transfer remains unsupported.

At the first solve, the selected checkpoint critic supplies the hidden layers,
including normalization parameters. At subsequent solves, these retain the
previous solve's adapted values. Every final linear weight matrix in the critic
ensemble receives fresh **Xavier-uniform** initialization, and its output bias
is zeroed. This resets the complete scalar or distributional-logit output head,
not hidden-layer biases or normalization. It neither restores the checkpoint
head nor uses AMBI's usual zero-weight output initialization. All retained and
reset critic parameters remain trainable.

After the head reset, the target critic is hard-copied from the resulting online
critic. The actor resets to the checkpoint prior, replay is emptied, temperature
is restored from the checkpoint, and all optimizer moments, optimizer steps,
per-solve update counters and target-update cadence reset. The episode's
cumulative critic update count persists as a record of body adaptation; it
does not imply that the current head has received all those updates.

Held decisions evaluate the solved feedback actor on fresh observations without
head resets, learning, or random initialization. Episode boundaries, explicit
evaluation resets, and checkpoint loads clear scientific state. Allocation
pooling can retain tensor and optimizer objects but cannot retain expired
values or change parameter references. Head initialization uses the seeded
controller stream; diagnostic probes must not consume that stream.

The encoder, dynamics, reward model, outer actor and outer critics stay frozen.
The frozen terminal critic is never reset or carried. Soft/soft retains
entropy-augmented inner targets and the outer terminal entropy correction;
return/return retains reward-only auxiliary targets and no terminal entropy
correction.

## Evaluation examples

| Matrix | Protocol | Solve cadence |
|---|---|---|
| `ambi_critic_hidden_transfer_575k.json` | `critic-hidden-transfer-v1` | Every decision |
| `ambi_critic_hidden_transfer_hold_h_575k.json` | `critic-hidden-transfer-hold-h-v1` | Every H decisions |

Each matrix contains `soft_soft/critic_hidden_warm` and
`return_return/critic_hidden_warm`, defaulting to **only the return-only arm**.
The one-arm comparison groups satisfy the matrix schema; they do not claim a
within-matrix baseline contrast. Fresh and full-critic references remain in the
[existing critic-transfer matrices](CRITIC_TRANSFER_575K.md), under their
original protocols. Historical protocols reject the new random-head mode, and
the new protocols require it.

The four examples use checkpoint `aux6428346x0` at step 575000, SHA-256
`0c6955db7cb8555a67d7863344b70be68f4b3250814d131e647ee6f9ef01a042`.
They preserve H3/J1/C16/A4/N128/B256, uniform J from the first solve, replay
capacity 3072, controller seed 55, environment seeds 101–105, 500-decision
mean-action episodes, and 32 diagnostic to-go rollouts. Hold-H solves at
decisions 0, 3, 6, …; its final episode block has two decisions. The selected
checkpoint supplies actor and entropy settings, including target entropy −10.5.

List the examples without loading a checkpoint or running an episode:

```bash
python evaluate_ambi_checkpoint.py \
  --matrix configs/research/ambi_critic_hidden_transfer_575k.json --list-presets
```

On a separately authorized compute allocation, evaluate the default return-only
arm with a new output bundle:

```bash
python evaluate_ambi_checkpoint.py \
  --matrix configs/research/ambi_critic_hidden_transfer_575k.json \
  --checkpoint /path/to/step_575000.pt \
  --device cuda \
  --bundle-dir /path/to/new-critic-hidden-transfer-bundle \
  --output /path/to/new-critic-hidden-transfer-results.json
```

Select `--preset soft_soft/critic_hidden_warm` for soft critics, or substitute
the hold-H matrix for the other cadence. Transfer diagnostics require a bundle
and full episodes; these protocols reject model-only root-bank evaluation.
The generic evaluator writes local results. Existing campaign launchers remain
unchanged, and these examples do not launch or publish anything automatically.

## Diagnostics and identity

`inner_critic_head_reinitialized` is one at every actual solve, including the
first, and zero at held decisions. `inner_critic_transferred` is one only at
solves after the first and refers to carried hidden layers in this mode.
`inner_critic_target_reinitialized` is one at every solve and zero when held.
Existing cumulative and per-solve update counters keep their meanings.

Results explicitly label **critic hidden transfer + fresh head**, the value
semantics and any hold cadence. Scientific metadata pins the retained
component, Xavier-uniform/zero-bias initializer, first-solve reset, ensemble
output extent, retained normalization, and target-copy ordering. The default
`"retain"` value is omitted from historical planner identities. New-head
results have distinct identities; an implementation change does not authorize
automatic reuse of historical results.

Local fixture and CPU Inductor validation does not establish CUDA compilation
or performance. On an allocated GPU, run the opt-in gate in the locked runtime:

```bash
AMBI_RUN_CRITIC_HIDDEN_TRANSFER_CUDA_GATE=1 \
  environments/dmcontrol/.venv/bin/python -m pytest -q -s \
  tests/test_critic_hidden_transfer_cuda_gate.py
```

Its four cases cover both critic types and solve intervals 1 and 3, checking
deterministic parity and compiled dropout. A smoke evaluation using the actual
575k checkpoint also remains required before production experiments.
