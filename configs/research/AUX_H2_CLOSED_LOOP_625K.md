# H2 closed-loop refinement at 625k

The initial H2 condition uses soft Q for both inner critic initialization and
the frozen outer bootstrap, with automatic policy alpha enabled. Select
`critic/soft_q` from `ambi_aux_closed_loop_625k_auto_alpha_h2.json`. Relative to
the completed H1 soft/soft auto-alpha condition, only `inner_rollout_horizon`
changes from 1 to 2. This matrix retains the existing critic selectors for
explicit later comparisons; selecting the matrix does not launch a sweep.

Use the seed55 backbone `rwgao_b-brown-university/ambi/aux6434715x3` at 625,000
training decisions. Retain J1/N128/C32/A4/B256: collect 128 two-step imagined
trajectories (256 transitions), then perform 32 critic and four actor updates.
The four automatic temperature updates start from saved outer alpha
0.004345251712948084 at every real decision, with a fresh optimizer, learning
rate 0.0003 and squashed target entropy −21. Actor, critic, replay and optimizers
reset each decision, and real execution uses the adapted actor's mean action.

Critic fitting remains reward-only. The first imagined transition uses the
inner target critic at a sampled inner-policy action; the final transition
uses the frozen outer soft critic at a sampled outer-policy action. No extra
entropy bonus is added to that boundary. Inner target initialization remains
the inherited online critic, with target update rate 0.01 after every critic
update. Policy entropy is enabled independently of critic-target entropy.

Evaluate five full 500-decision episodes with environment seeds 101–105 and
controller seed 55. Reuse verified prior episodes and the H1 reference after
checking checkpoint, scientific source, protocol and solver seeds. Model
probes use 32 independent two-step trajectories at initialization and after J1.
Their score is reward at depth zero plus discounted reward at depth one plus
the twice-discounted frozen soft-Q tail; it is not an observed reward return.
Compare H1 and H2 through paired full-episode returns rather than raw probe
scores, whose horizons differ.

For GPU workers and the CPU merger/publisher, set:

```bash
AMBI_ALPHA_MODE=auto
AMBI_CRITIC_MODE=soft_q
AMBI_HORIZON=2
```

The campaign helper also accepts `--horizon 2`. Horizon defaults to 1 for
existing launches. Each worker, merge receipt and publication binds the chosen
horizon, rejects H1/H2 mismatches, checks the realized transition/update counts,
and requires unchanged full outer state and zero strict-compilation fallbacks.
Create a new performance series and H2 diagnostic attempt; do not append H2
results to the H1 planner identity.
