# Current-round inner data at625k

`ambi_aux_round_replay_sweep_625k.json` compares soft/soft and soft/return
at H={1,2,3}, J={1,2,4} with the original tau0.01 H/J campaign. Its only
scientific setting change is `inner_replay_reset_each_round=true` (default false).
Allocated replay capacity stays2048.

At the start of every round the engine clears only replay, before collecting
new imagined transitions. It preserves monotonically increasing transition IDs
for unique-sample diagnostics. The learner, targets, temperature and optimizers
continue adapting. This option requires canonical SAC with action-local replay,
round update timing, no explorer population and no real replay mixing. It
also handles early termination without retaining rows from an older round.

Actor, critic, target, alpha and optimizer state continue across J rounds.
They still reset from the frozen625k prior at every real decision. C32/A4,
B256, replacement sampling and all entropy settings are unchanged. Thus data
are reused *within* each round; this is removal of cross-round replay, not
one-update-per-transition learning. Buffers contain128/256/384 rows atH1/H2/H3.
The J1 settings are fresh execution controls: their available data match the
original J1 runs, while J2/J4 isolate the loss of earlier-round data. The prior
episodes and original full-replay baselines are reused after identity checks.

There are18new five-episode panels, seeds101–105/controller55,500decisions per
episode. Pairing and all training diagnostics match `AUX_HJ_SWEEP_625K.md`.
Run the existing preparation/worker/publisher flow with the new matrix and
`--baseline-kind round_replay --baseline-campaign <original-HJ-root>`.
The baseline validator permits only the replay-reset setting change; checkpoint,
seeds, protocol and all other resolved planner settings must match. The new
opt-in implementation has a distinct scientific identity. Historical baseline
source is explicitly pinned to06d7077 (scientifically identical to the H/J
campaign), with both identities retained; no global compatibility check is relaxed.

Performance curves retain paired gain versus the frozen prior. Training runs
add `comparison/all_round_replay_gain_mean`, paired sample SD and exploratory
95% bootstrap intervals, with per-seed values in `replay-comparison.json`.
Five episodes and the previously observed execution variation do not establish
a robust ranking; inspect the J1 controls when interpreting this screen.

The focused tests verify every sampled transition ID belongs to the newest
round and compare parameters, optimizer/RNG states and executed actions against
an explicit-clear reference. Existing full-replay behavior and omitted/default planner identities remain
unchanged. Use a short GPU smoke before releasing the full panel.
