# Temperature carry across complete episodes

This campaign asks whether the inner temperature can adapt across episode
boundaries when the actor, critics and replay are still initialized afresh at
every real decision. It uses the frozen SAC auxiliary-return checkpoint at
100k and MPPI auxiliary-return checkpoint at 400k. The protocol identifier is
`run-temperature-chain-v1`; it is distinct from independent-seed checkpoint
evaluation and from the previous within-episode temperature experiment.

Each checkpoint has five independent streams. A stream evaluates three full
500-decision episodes, with ordered environment seeds:

| Stream | Episode 1 | Episode 2 | Episode 3 |
| --- | --- | --- | --- |
| 0 | 101 | 102 | 103 |
| 1 | 102 | 103 | 104 |
| 2 | 103 | 104 | 105 |
| 3 | 104 | 105 | 101 |
| 4 | 105 | 101 | 102 |

The independent statistical unit is a stream. Episodes inside a stream depend
on earlier temperature updates. Each episode position contains all five seeds
across streams, so an apparent position effect is not caused by changing the
set of environment seeds. Controller seed 55 is hashed with each environment
seed as before. The seed-derived solver RNG is reset at episode boundaries;
only log-temperature, temperature Adam state and lifetime counters persist.

All cells execute sampled actions with H=3, J=8, 16 critic and four actor updates
per round, 128 rollouts per round, batch size 256, and learning rates 0.0003.
Each actor update has one temperature step: 32 per decision, 16,000 per episode,
48,000 per complete stream. Alpha starts from the frozen backbone's value once
per stream. There is no KL penalty, Bernoulli transfer, Retrace, actor/critic
carry, outer writeback or outer training. The entropy target remains inherited
from the backbone (-10.5 for these checkpoints).

The prior, fresh-alpha/action-reset reference and within-episode-alpha-carry
reference are reused from
`/oscar/scratch/rgao48/ambi/alpha-adaptation/20261021-9ad45fe9/campaign.json`.
Preparation checks completed paired-seed coverage, checkpoint identity, frozen
state, resolved protocol and saved hashes, then pins immutable reference
summaries. Reusing a reference at each episode position does not create new
independent reference observations. The primary paired comparison is against
within-episode carry; comparisons against fresh reset and the base policy are
also reported. Return and variance estimates must distinguish stream-level
variation from descriptive within-stream episode variation.

Prepare from a clean tested Git checkout in a short CPU allocation:

```bash
python slurm/ambi_alpha_carry_chain.py prepare --root "$CAMPAIGN_ROOT"
```

Each GPU worker owns one cell, indexed checkpoint-major (0–4 SAC, 5–9 MPPI):

```bash
python slurm/ambi_alpha_carry_chain.py run \
  --campaign "$CAMPAIGN_ROOT/campaign.json" --index 0
```

Use `slurm/run_ambi_alpha_carry_chain_oscar.sbatch` inside an explicit Slurm
allocation, with `EXPECTED_ACTION_MODES_SHA` and `PYTHON_BIN` set. It verifies
the clean source and sets thread counts from `SLURM_CPUS_PER_TASK`. Output is
offline and exclusive: an existing output directory cannot be overwritten.
Full production consists of ten independent GPU workers, 30 episodes and
15,000 decisions. Splitting an individual stream across workers is unsupported
because it would lose its temperature and Adam state.

Before production, run stream 0 for each checkpoint on each selected GPU class
with `--smoke-steps 8` (two eight-decision episodes). Four-decision smokes are
available for debugging, but the dispatch gate requires eight. Set
`--smoke-root "$CAMPAIGN_ROOT/hardware/GPU_TYPE/smoke"`; each adjacent
`CHECKPOINT__stream0-steps8.hardware.json` records the allocated GPU type, device
name/capability, node, job ID and `nvidia-smi` UUID/driver/memory CSV. The smoke
and hardware record must come from the same allocation. Every smoke checks
exact cross-episode alpha, Adam and lifetime continuity, one initial chain
reset, no episode reset, fresh actor/critic/replay scopes, full optimizer and
model-step budgets, frozen backbone and finite traces.

`slurm/submit_ambi_alpha_carry_chain.py benchmark` reads completed smoke receipts
without submitting. `dispatch` additionally requires the matching W&B workspace
to be installed and API-verified. Both accept `--campaign`, `--worker-repo`,
`--python-bin`, and optional `--gpu-types` with ten concrete GPU class names.
The default assigns all streams to Quadro RTX 6000. Each assigned
checkpoint/GPU-class combination requires its own measured smoke. The helper
does not infer live account capacity; check current allocations and QOS before
calling dispatch. Ten jobs request ten CPUs, 40 GiB host memory and ten GPUs
in aggregate; the cluster schedules them independently.

Production requests one CPU and 4 GiB per GPU, checked against measured smoke
RSS below 3 GiB. Runtime sizing uses 1,497 warm decisions, three separately
measured cold first decisions, 15 probed roots, and measured startup/I/O
overhead. It adds 25% plus two minutes and rounds up to five minutes. This avoids
extrapolating a short smoke's unusually high fraction of cold decisions to all
1,500 decisions. At the previous Quadro warm rate near 1.55 seconds/decision,
compute is roughly 40 minutes per stream before startup and margin; the new
hardware-specific smoke determines the actual request. The default walltime
guard is 90 minutes and can be tightened explicitly.

Submission uses a durable intent journal and refuses an ambiguous retry after
a lost scheduler acknowledgement. Production captures GPU provenance before
starting the worker. Two serialized CPU publishers merge each checkpoint only
after its five streams finish; the final publisher requires the complete
campaign. Missing, nonfinite, altered or incomplete streams remain progress or
failure states and are never included in a completed performance estimate.
The dedicated chain publisher uses new scientific identities and workspace,
rather than publishing dependent episodes into the old independent-seed series.

Per-stream outputs are `results.json`, the complete trace `bundle`,
`chain-summary.json` and a hash-pinned `receipt.json`. Compact rows retain
stream index, episode index, original seed, decision index, global decision
index, alpha start/end, entropy and target, optimizer/lifetime start/end,
reset flags, reward, per-episode cumulative reward and controller timing.
An incomplete worker preserves any completed trace shards but produces no
validated result receipt. It cannot be silently resumed or merged as complete.
