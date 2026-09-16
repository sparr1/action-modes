# Auxiliary-return backbone: closed-loop refinement at 625k

The first selected condition uses learned SAC soft-Q for both inner critic
initialization and frozen horizon bootstrap. The actor and terminal actor are
the saved SAC actor. Inner alpha remains zero; reward-only finite-horizon
fitting adds no explicit entropy term, while the learned soft-Q tail retains
its entropy-trained semantics.

Use `ambi_aux_closed_loop_625k.json`, `critic/soft_q`, checkpoint 625000 from
`rwgao_b-brown-university/ambi/aux6434715x3`. All ordinary action-local settings
match `ambi_aux_return_j1.json`: J1/N128/H1/C32/A4/B256, learning rates 3e-4,
replay 2048, prior initialization, online target initialization, fresh learners
at every real decision, mean execution, frozen outer state, clipped log std
and unscaled Q inherited from this backbone.

This small closed-loop panel uses environment seeds 101–105, controller seed55,
and 500 decisions per episode. At each real decision, 32 paired model-return
probes run at the true initialization and after the round. Their RNG is separate
from learning; reward and frozen soft-Q bootstrap components are retained. These
are soft-Q-tailed model scores, not pure environment-return predictions.

The workers run one complete seed each without W&B. Reuse the verified prior
bundle at the same checkpoint. `slurm/ambi_aux_closed_loop.py` validates and
seals each shard, then merges all five, refusing missing/overlapping seeds,
changed identities, corrupted traces or incomplete probe coverage. Only the
complete panel is published: one new checkpoint-performance curve with paired
prior gains, plus a separate round/update-axis diagnostic run and portable HTML.
Existing checkpoint curves remain unchanged. Episode-cluster probe intervals
use 2000 bootstrap resamples, seed 20260912, and are exploratory with five seeds.

The episode-merging helper and diagnostic exporter are reused from the tested
historical entropy study; only the episode subset of the merger is integrated.
No learner or evaluator equations change. Tests compare serial and sharded
execution, verify probes preserve control behavior, and check publication data.
