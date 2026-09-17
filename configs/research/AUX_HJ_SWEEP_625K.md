# Horizon and round sweep at 625k

`ambi_aux_hj_sweep_625k.json` defines 36 closed-loop controllers on the
seed55 detached auxiliary-return backbone `ambi/aux6434715x3`, checkpoint625k.
H is 1/2/3 and J is 1/2/4. Each condition has paired environment seeds101–105,
controller55, and 500 real decisions. Every decision starts a fresh actor,
critic, target, temperature, optimizers and replay. All rounds retain their
data: N128 × H3 × J4 = 1536 transitions fits the fixed 2048-row replay.
Sampling is with replacement; B256/C32/A4 per round, learning rates3e-4,
target tau.01, dropout enabled, final mean action execution, strict compilation.

| Arm | Inner initialization | Interior critic target | Outer boundary | Actor entropy |
|---|---|---|---|---|
| soft_soft | Soft Q | Entropy-augmented | Soft Q plus frozen outer entropy | Inherited automatic alpha |
| soft_return | Soft Q | Entropy-augmented | Return Q only | Inherited automatic alpha |
| return_return_alpha | Return Q | Reward-only | Return Q only | Inherited automatic alpha |
| return_return_zero | Return Q | Reward-only | Return Q only | Off |

Boundary entropy uses the frozen outer coefficient, independently of adapting
inner alpha. There are no interior targets at H1. Alpha-on uses the saved
coefficient and inherited target entropy; its fresh optimizer updates with
every actor update. Inner training never updates the outer checkpoint.

The launcher runs a complete five-seed condition per GPU, amortizing startup
and compilation. Submit longer J4 conditions first, with up to the live account
allowance of12 concurrent GPUs, accepting L40S/A6000/A40/A5500. A CPU watcher
publishes complete panels as they become ready; GPU tasks never contact W&B.
Failed workers remain visible and are never silently converted to partial
results. All attempted scientific configurations and source hashes are saved.

Each condition gets an ordinary checkpoint-performance run and a grouped inner
training diagnostic run. Diagnostic history includes per-critic-update and
per-actor-update means/minima/maxima, per-real-decision metrics and per-seed
decision curves. Full compressed event traces, resolved configuration, metric
definitions, five paired episode returns, and 32 independent model probes at
initialization and after each round remain downloadable. Update metrics are
pre-update minibatch measurements; probes are post-round measurements. The
existing model probes use reward plus terminal Q only: they explicitly exclude
entropy and are not the entropy-inclusive training objective. Their absolute
values are not comparable across different H or terminal critics.

Reuse compatible prior and H1/J1 return/return results only after checking
checkpoint/sidecar identity, complete seeds, protocol, planner settings and
immutable bundle hashes. Reuse preserves original source provenance and marks
the campaign entry as reused. Commit02b8e97 adds a default-off boundary branch
and configuration/identity plumbing; it changes neither prior execution nor
return-tail execution. Corrected soft-boundary results must be evaluated anew.
