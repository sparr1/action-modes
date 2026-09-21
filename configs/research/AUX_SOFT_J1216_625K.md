# Soft/soft scaling beyond J8

`ambi_aux_soft_j1216_625k.json` adds eight full-episode closed-loop settings:
H1/H2/H3 crossed with J12/J16 at C16, plus H3/J12 and H3/J16 at C32.
Every setting uses five paired environment seeds 101–105, controller seed 55,
and the frozen auxiliary-return SAC backbone at 625k. Existing J8 measurements
and frozen-prior episodes are reused without evaluation or republication.

Keep N128, B256, A4 per round, critic-first round updates, actor/critic learning
rates .0003, tau .01, inherited adaptive temperature, soft inner initialization,
entropy-augmented fitting and soft outer terminal bootstrap including entropy.
Uniform replay holds every round in the current decision; capacity 6144 covers
128*3*16 transitions. The learner, optimizer and replay reset at every real
decision. No ERE, interleaving, horizon conditioning, or execution override is
enabled. Strict compilation follows the original mixed-GPU evaluation protocol.

J12 performs 48 actor/temperature updates and 192 or 384 critic updates per
decision; J16 performs 64 and 256 or 512 respectively. J16/C16 versus J8/C32
matches critic updates only: actor work and imagined data differ.

Preparation verifies immutable J8 bundles and exact planner identities, allowing
only J, derived work totals, and full-retention capacity to change. C16 references
pin source fe87ae07b2a7ed751cd6865a3f60b2eae88e6abb; C32 references pin
28964eef209a2aa6deb73b898549ca405f38f293. Historical run IDs remain intact.

Use `run_ambi_aux_j_extension_prepare_oscar.sbatch`, then the existing H/J
worker with `--smoke` (H3/J16 at both C budgets), followed by array indices 0–7.
Schedule up to eight concurrent configurations according to live resources.
The CPU watcher uses four publishers; provision 8 CPUs and 64 GiB and retain
scratch W&B staging. Every production task evaluates all five seeds.

Performance runs retain paired prior gains. Training runs retain complete
per-update/decision traces, horizon diagnostics and frozen-model probes, and
log `comparison/j8_gain_*` against the matching H/C reference. Those probes
omit explicit entropy and are not observed Monte Carlo returns. Primary scaling
evidence comes from paired real full-episode returns; five evaluation seeds at
one trained checkpoint remain exploratory. J24/J32 are outside this matrix.
