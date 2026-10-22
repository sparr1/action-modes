# Inner temperature adaptation sanity check

This frozen-checkpoint comparison tests repeated temperature updates and
temperature persistence before considering new backbone training. It uses the
existing SAC auxiliary-return 100k and MPPI auxiliary-return 400k checkpoints
pinned in `ambi_entropy_probe.json`. These represent the weak-checkpoint entropy
failure and a stronger checkpoint where temperature can support exploration.

Every condition uses sampled real actions, five paired seeds 101–105, controller
seed 55 and full 500-decision episodes. This is the sampled-action extension of
closed-loop refinement evaluation, rather than its mean-action reference recipe.
The encoder, dynamics, reward, outer actor and critics remain frozen. The
H3/J8/C16/A4/N128 recipe performs 128 critic updates and 32 actor updates per
decision with B256 and fresh actor, critic, optimizers and imagined replay.
No KL, Bernoulli, Retrace, real replay or weight writeback is used.

| Arm | Temperature steps per actor update | Alpha/Adam lifetime |
|---|---:|---|
| Reset baseline | 1 | One decision |
| Repeat 100 | 100 | One decision |
| Carry | 1 | One episode |

All learning rates remain 0.0003. Repeat 100 makes 100 Adam steps using the same
detached entropy estimate after each actor update, then proceeds to the next
actor update. It is a stronger dual update, not 100 new policy observations or
proof of entropy convergence. The actor/critic update budgets remain unchanged.
Carry preserves both log-alpha and its Adam moments/step count across real
decisions; it resets both to the checkpoint initialization at each new episode.
The saved outer temperature and target entropy are never overwritten.

Only four new production cells are run: repeat 100 and carry at each checkpoint.
Existing complete sampled-prior and reset-baseline results are reused after
checking checkpoint hashes, execution, seeds, frozen outer state, settings and
immutable result/manifest hashes. Fresh two-episode reset smokes compare the
first decisions against historical alpha, entropy and reward traces. Historical
traces lack the new optimizer-lifetime fields; they retain alpha/entropy/return
curves, while continuity assertions apply to the new runs.

The worker validates every optimizer event and decision. It checks 32 versus
3,200 actual temperature steps, unchanged actor/critic work, episode reset flags,
temperature and Adam continuity, fixed target entropy, finite diagnostics,
complete ordinary episodes, unchanged outer state and trace/result agreement.
Compact `decision-summary.json` files retain per-seed alpha initial/final,
training-batch entropy, target, temperature step/lifetime counters, reward,
cumulative reward and controller time. Complete optimizer traces stay on Oscar;
receipts hash all trace shards and compact summaries. Entropy is the update-time
batch statistic, not guaranteed final-root policy entropy.

```bash
python slurm/ambi_alpha_adaptation_episodes.py prepare --root ROOT
python slurm/ambi_alpha_adaptation_episodes.py run --campaign ROOT/campaign.json --index 0 --smoke-steps 8 --baseline --cpu-threads 1
python slurm/ambi_alpha_adaptation_episodes.py run --campaign ROOT/campaign.json --index 0 --smoke-steps 32 --cpu-threads 1
python slurm/ambi_alpha_adaptation_episodes.py run --campaign ROOT/campaign.json --index 0 --smoke-steps 32 --cpu-threads 2
python slurm/ambi_alpha_adaptation_episodes.py run --campaign ROOT/campaign.json --index 1 --smoke-steps 8 --cpu-threads 1
python slurm/ambi_alpha_adaptation_episodes.py run --campaign ROOT/campaign.json --index 0
```

Indices are SAC100k repeat, SAC100k carry, MPPI400k repeat, MPPI400k carry.
Repeat the reset and carry smokes on indices 2 and 3 for MPPI400k. Every smoke
runs seeds 101 and 102 to exercise episode boundaries. Resource requests are
chosen at submission from measured throughput and current limits, not embedded
in the launcher. CPU scaling smokes have independent output ownership, as do
the reset smokes and production. No output is overwritten on retry.
