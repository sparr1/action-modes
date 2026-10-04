# Inner SAC transfer diagnostics

`evaluate_ambi_transfer_diagnostics.py` implements
`inner-sac-transfer-diagnostics-v1`: controlled interventions between successive
decisions of a frozen-checkpoint episode. This is a separate diagnostic protocol;
it does not change the fresh adaptation definition of `closed-loop-refinement-v1`
or the production transfer protocols.

## Horizons and budget

The default grid is **H = 1, 2, 3, 5, 7**, **J = 1, 8**, with four source
histories: fresh, actor carry, critic carry, and joint carry. Both lists accept
arbitrary positive integers. Every source history has its own root bank and its
own donor actor/critic from the immediately preceding solve. Each root is then
forked into all four initialization combinations.

The source checkpoint/preset supplies C, A, N, batch size, learning rates,
critic routing, entropy, and terminal-value semantics. J is uniform from the
first decision. All components use action-local storage: the diagnostic hook
explicitly injects selected weight donors after fresh initialization. Replay,
Adam, temperature and target state are not implicitly carried. Each target
normally copies the selected online critic. A separate target crossing selects
prior/carried online weights independently of the online learner.

Replay capacity is enlarged to at least `H * J * N` so the longer-horizon
settings retain their complete imagined data. **This is an update-budget
comparison, not an equal-compute comparison:** holding J/C/A/N fixed increases
model work with H. Source histories also visit different states across H.
Only the forks at an individual root are state-matched. The horizon portability
probe compares H−1 with H at an identical imagined successor, holding state,
action, model and continuation policy fixed. H1 reports this test as inapplicable.

## Entry point

Resolve the full prospective study without loading networks or starting episodes:

```bash
python evaluate_ambi_transfer_diagnostics.py \
  --checkpoint /path/to/575k.pt \
  --matrix configs/research/ambi_critic_transfer_575k.json \
  --preset return_return/fresh \
  --horizons 1 2 3 5 7 --rounds 1 8 \
  --output-dir /path/to/new-diagnostic-bundle --dry-run
```

The checkpoint contract in this example verifies the historical 575k identity.
A different checkpoint requires its corresponding compatible preset matrix.
For soft/soft use `--preset soft_soft/fresh`; reward and soft studies must have
separate output directories. Configuration validation rejects unsupported
operators instead of silently changing their objective.

A deliberately small functional run (not sufficient for a research conclusion):

```bash
python evaluate_ambi_transfer_diagnostics.py \
  --checkpoint /path/to/575k.pt \
  --preset return_return/fresh \
  --horizons 1 2 5 --rounds 1 --histories fresh joint \
  --seeds 101 --decisions 1 --max-steps 2 \
  --mc-rollouts 4 --action-count 4 --fit-states 4 --fit-steps 2 \
  --output-dir /path/to/new-smoke-bundle --device cpu
```

Use scheduler compute for checkpoint-scale studies on Oscar. This implementation
does not submit jobs or initialize online W&B. Existing result directories are
never overwritten. Decisions are zero-based and must be at least one: decision
zero has no preceding solve to transfer. Source episodes that terminate before
a requested decision record that root as unreached.

## Measurements

1. **Actor × critic comparison.** Fresh/fresh, carried/fresh, fresh/carried,
   and carried/carried use the same root, donor history, and private solver RNG.
   The `common` lane fixes collection behavior to the prior; complete replay
   hashes must match across branches. The `natural` lane lets each learner
   collect its own data. Pairing RNG alone does not make natural replay equal.
2. **Critic action guidance.** Fixed held-out actions include prior/carried means
   and samples. Independent frozen-model rollouts evaluate the configured
   finite-horizon return under prior, carried, and current continuation policies.
   Outputs include bias, centered and action-relative error, ranking, selection
   regret and paired uncertainty. Pre-tanh directional perturbations measure
   the return change along the critic's action gradient at two step sizes,
   separately at the prior and current actor means. Current-mean value error
   is also recorded outside the fixed common action bank.
   Online and target probes use their configured actor and target reductions.
3. **Bootstrap initialization.** A common-data 2×2 crossing holds the actor
   fresh and varies online and target initialization independently. This isolates
   the starting bootstrap function from online critic weights. It does not carry
   historical target lag. Stationary Monte Carlo fitting supplies a separate
   no-bootstrap learning check; it uses a different diagnostic dataset and is
   not itself a same-minibatch TD-versus-Monte-Carlo causal ablation.
4. **Portability.** Donor predictions are checked at the preceding root,
   imagined successor, and actual encoded successor with matched continuation
   policies. The additional H−1/H comparison measures horizon conflict rather
   than assuming an unconditioned critic must suffer from it.
5. **Stationary learning.** Prior and carried actor/critic clones fit identical
   fixed targets using fresh Adam, identical batches and paired dropout RNG.
   Critic labels are decoded model-return estimates; actor labels imitate the
   best candidate action in a fixed independently generated bank. Train and
   held-out states are disjoint by index. Report step-zero and subsequent losses,
   parameter movement, gradients, penultimate-feature rank and inactive units.
   Starting error and label noise matter: these curves support, but do not by
   themselves establish, loss of plasticity. Feature rank is bounded by probe
   sample count. Actor mean-action fitting does not test covariance fitting.
6. **Model/terminal error and real utility.** Optional real branches use the
   exact saved simulator state and explicit paired noise. Compare learned Q to
   independent model return A; then A to real H-prefix + the same frozen value B;
   then B to real H-prefix + a finite sampled prior tail C. C has an explicit
   cutoff and no appended value; it is not infinite-horizon ground truth.
   True termination masks tails, while incomplete time-limit branches are marked
   unavailable. Prefix calibration and actual replanning are separate outcomes.

Learner snapshots are taken at initialization, after the first critic block,
after the first actor block, selected completed rounds, and immediately before
the final action-local cleanup (`pre_reset`). The last snapshot is the next
decision's donor. It is not a snapshot before the next solve's initialization.
`--capture-rounds` limits completed-round captures; initial/block/final captures
remain. Every snapshot contains independent actor/online/target payloads,
temperature, policy bounds, and the actual actor Q divisor.

Critic references use the snapshot's own entropy coefficient and Q scale.
For causal comparisons of actor objectives across branches/stages, all forks
are scored with the **fresh branch's initial** coefficient in reward units.
This avoids changing the evaluation objective merely because alpha adapted.
The reported model-objective Monte Carlo SE is conditional on eight fixed,
paired first-action samples; it excludes their sampling uncertainty. Pair-head
selection is integrated analytically and probe dropout is disabled; these are
expected-function checks, not stochastic gradient-variance estimates.

## Optional real branches

Add, for example:

```text
--real-rollouts 32 --real-tail-steps 1000 --replan-steps 100 --replan-repeats 3
```

Real prefix evaluation supports the configured reward or soft objective,
including outer boundary entropy when enabled. Replanning separately reports:

- `first_action_only`: each branch's first action with the same fresh future solver;
- `memory_only`: the same fresh-branch first action, each branch's final memory,
  and a common joint-transfer future rule;
- `full`: each branch's first action and its own future transfer rule.

Future solver streams are paired across interventions, with independently
identified repeats. They are not independent environment episodes. Normal
episode truncation is preserved for replanning; fixed-prefix calibration may
explicitly enable continuing simulation. Exact built-in capture currently
supports **DMControl Humanoid Walk state observations**. Other environments
must supply the explicit calibration-state protocol. Model-only diagnostics do
not require simulator snapshot support.

`--save-snapshots` additionally stores the simulator state and donor modules/RNG
at selected roots. `donor.pt` is trusted local Python/PyTorch serialization and
must not be loaded from an untrusted source. Numerical root records are retained
without this option. Snapshots can be large; live module copies are temporary.

## Output and validation

Each root has `diagnostics.json`; every source episode has `episode.json`.
`summary.json` aggregates repetitions within roots, roots within episodes, then
episodes with equal weight. It keeps H/J/history/data-lane separate and reports
episode-level standard errors, unavailable with one episode. `report.md` has
three compact tables covering component effects, critic guidance and stationary
learning/horizon portability. MC replicates are never counted as independent
episodes. No real-return improvement is inferred from a model objective.

`manifest.started.json` records the exact checkpoint, source hashes, settings,
runtime, and semantics. `manifest.json` exists only after successful completion
and verifies that both outer learners stayed unchanged. An interrupted directory
is partial; use a new directory rather than overwriting it.

Focused validation:

```bash
python -m pytest -q \
  tests/test_transfer_diagnostic_engine.py \
  tests/test_transfer_diagnostic_metrics.py \
  tests/test_transfer_diagnostic_real.py \
  tests/test_transfer_diagnostic_evaluator.py
```

The exact Humanoid restoration test is opt-in with
`AMBI_RUN_REAL_DMCONTROL_TESTS=1`; state-only macOS checks can use
`MUJOCO_GL=disable`. The diagnostic solver intentionally runs eager, isolating
probe batch sizes and experimental hooks from compiled production kernels.
