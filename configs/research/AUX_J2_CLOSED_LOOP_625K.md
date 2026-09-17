# Two-round closed-loop refinement at 625k

Use `ambi_aux_closed_loop_625k_auto_alpha_h1_j2.json`, selector `critic/soft_q`,
on seed55 auxiliary-return backbone `rwgao_b-brown-university/ambi/aux6434715x3`
at 625,000 training decisions. Relative to the completed H1/J1 soft/soft
automatic-alpha condition, only `inner_rounds` changes from one to two.

Each real decision starts with fresh inherited SAC actor and soft critic,
target initialized from online critic, replay and optimizers. Each round
collects N128 H1 trajectories, fits the critic for C32 steps, then performs
A4 actor and four automatic-alpha updates, with batch size 256. Round two
collects using the adapted actor; actor, critic, target, alpha, optimizers and
replay persist between rounds within this solve. They reset at the next real
decision. Total per solve: 256 transitions, 64 critic, eight actor and eight
temperature updates. Replay capacity remains 2048 with replacement sampling.

Saved initial alpha is 0.004345251712948084, with temperature learning rate
0.0003 and squashed target entropy -21. Actor/critic learning rates remain
0.0003, target tau 0.01, critic dropout enabled and real execution uses the
final actor mean. Critic fitting remains reward-only: because H1 puts every
transition at the boundary, every target uses the frozen outer soft Q at a
sampled frozen outer-policy action, with no added boundary entropy. Increasing
J does not activate an interior inner-target bootstrap at H1.

Evaluate five paired full episodes (500 real decisions, seeds 101-105,
controller 55). Reuse the matching frozen-prior and H1/J1 results. The completed
H2/J1/C64/A8 condition has matching total transition and optimizer counts, but
a different depth distribution and update order. These are scientific
comparisons, not controlled hardware timing comparisons.

Retain 32 independent H1 model probes at initialization and after each round:
actor axis 0/4/8, critic axis 0/32/64. Five full episodes yield 7,500 paired
diagnostic rows. Verify unchanged outer state and no strict-compile fallback.
At the two J values, H1 probes have the same horizon but average over different
controller-visited states; full-episode paired environment returns remain the
primary outcome.

Set `AMBI_HORIZON=1`, `AMBI_ROUNDS=2`, `AMBI_UPDATE_BUDGET=c32_a4`,
`AMBI_ALPHA_MODE=auto`, `AMBI_CRITIC_MODE=soft_q` for workers and publisher.
The helper accepts `--rounds 2`; its default remains one for existing runs.
Worker, merge and publication receipts bind the round count and reject mixing
J1 and J2. Create new performance and diagnostic series; do not append to J1.
