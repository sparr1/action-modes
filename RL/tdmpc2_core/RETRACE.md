# Finite-trajectory Retrace in the inner SAC solve

`inner_sac_return_estimator="retrace"` opts into complete-trajectory critic
updates. The default remains `one_step`; inactive Retrace parameters do not
change planner/resume identity, checkpoint schemas, or optimizer/RNG behavior.
Outer training and frozen checkpoint weights are unchanged.

This bounded port of the canonical implementation supports ordinary latent
policies, action-local state, round collection, critic-first updates, retained
replay, and a frozen outer boundary without an extra entropy correction.
It supports scalar and distributional critics, auxiliary return-Q routing,
and sampled final execution through `inner_eval_execution_action=policy_sample`.
The separate one-hot horizon-conditioned architecture, interleaved component
updates, round replay reset, and outer terminal-entropy correction are rejected.

Collection saves each action's pre-tanh sample and its actual behavior
log-density before later updates. At a critic update, every valid suffix of a
sampled trajectory receives the detached target

```text
c[t+1] = lambda * min(1, pi(a[t+1]|z[t+1]) / mu(a[t+1]|z[t+1]))
G[t] = r[t] + gamma * (1 - terminated[t])
             * (V[z[t+1]] + c[t+1] * (G[t+1] - Q_target[z[t+1], a[t+1]]))
```

The horizon boundary uses the selected frozen outer actor/critic continuation
and omits the successor trace term. True termination removes both terms.
The reward-only experiment adds no entropy to critic targets; actor entropy
and temperature adaptation retain their existing objectives. Policy density
ratios use the unscaled current actor and the stored collection-time density.

Lambda is validated in [0,1], with implementation default 1.0; the selected
experiment explicitly uses 0.9. `H=1` has no trace correction and its target
equals the existing one-step target under matched batches/noise. The complete
solver uses distinct trajectory collection and replay paths, so this does not
imply bitwise equality of full episodes. Lambda zero gives one-step targets at
all sampled positions.

Critic batch size defaults to `ceil(inner_batch_size / H)` trajectories. With
B256, H1/H2/H3 draw 256/128/86 trajectories and train on 256/256/258 rows per
critic update when no modeled termination occurs. Actor and temperature
minibatches remain 256 transitions. Replay capacity counts nominal transition
slots; usable capacity is `floor(capacity/H)*H`. Complete trajectories are the
storage unit and padding never enters actor training. The experiment requires
enough capacity to retain all J*N*H transitions for one real action.

Diagnostics record trajectory draws, actual critic rows, retained trajectories,
effective capacity, trace coefficients, effective trace lengths and absolute
corrections. Active Retrace settings enter the planner and exact-resume
identities; schema 8 pins its protocol. Portable outer-weight loading can switch
estimators. Tests cover suffix indexing, masks, stored densities, H1/lambda-zero
equivalence, private gradients/RNG, exact resume, compilation, and sampled
reward-only execution at all three horizons.

The target recurrence and source provenance are documented in
[`common/retrace.py`](common/retrace.py); no external implementation dependency
is introduced.
