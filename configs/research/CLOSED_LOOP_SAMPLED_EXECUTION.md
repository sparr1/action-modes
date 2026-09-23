# Sampled execution after closed-loop refinement

This is an explicit execution variant of the full-episode closed-loop refinement
procedure. At every real decision, create fresh prior-initialized actor, critic,
optimizers and replay; run the requested inner rounds; execute a sample from the
final adapted policy, `tanh(mu + std * epsilon)`. The Gaussian standard deviation
is the actor's learned standard deviation with scale one. The frozen evaluation
API remains active: outer learning, writeback and training exploration are off.

The original mean-execution results retain their original identities and history.
The new evaluation-only setting is `inner_eval_execution_action=policy_sample`;
its action rule is `squashed_gaussian_sample`. The default setting remains `mean`.
The separate training-time `inner_execution_action` remains `mean` in this matrix.

## Selected comparison

`ambi_closed_loop_sampled_h{1,2,3}_575k.json` provides one separate campaign for
each rollout horizon. Each matrix selects exactly three settings: return-only
critic, J1/J2/J4, C16/A4, N128 and B256. The H3 matrix remains the default;
existing H3 runs continue under their original source and publication identities.
Adding H1 and H2 requires six new settings, with their own overviews and workers.
The 575,000-decision
target -10.5/shared backbone checkpoint has SHA256
`0c6955db7cb8555a67d7863344b70be68f4b3250814d131e647ee6f9ef01a042`.
Actor entropy adapts from the inherited checkpoint alpha; return-Q fitting
remains reward-only. Replay capacity is 3,072, retained across rounds within
each real decision, then reset. Environment seeds 101–105 and controller seed 55
match the completed historical mean-execution runs. Every episode has 500 real
decisions. Thirty-two independent model probes accompany initialization and
every completed round at every decision.

Historical counterparts are selected at the same H and J. Their source commits,
performance run IDs and manifest hashes are pinned separately by horizon in
`slurm/ambi_closed_loop_sampled.py`:

| Horizon | Historical mean source |
|---|---|
| H1 | `4f98943012ff3019fe0d556eef9654c481486fc6` |
| H2 | `589ed0c0ba469245085af4db0ae740c692786695` |
| H3 | `95dbae6769e976970e94a3987b070b31dbbd5b37` |

Preparation checks their pinned
manifest hashes, every trace hash, complete publication receipts, checkpoint,
protocol, planner and seed identities. Sampled workers check all resolved
settings against the counterpart, allowing only the new execution setting;
the action rule is the sole allowed environment-protocol difference. Scientific
source changes are explicit: this comparison changes final execution only;
the original source's identities are never rewritten.

The primary paired outcome is **sampled return minus historical mean return at
the same H, J, environment seed and controller seed**. It measures the effect of
executing the adapted actor stochastically. It does not measure improvement over
a sampled frozen prior. No new prior evaluations are included and no mean-prior
reference is attached to a sampled bundle. The unselected matrix `prior` entry
exists only to satisfy the preset schema; it is never scheduled by this campaign.

Pairing seeds does not keep subsequent states, imaginary rollouts or real actions
identical once the controllers diverge. Report all five paired values, their
sample standard deviation and exploratory percentile bootstrap intervals
(2,000 seed resamples, bootstrap seed 20260912).

## Preparation, execution and publication

Use `slurm/ambi_closed_loop_sampled.py prepare` with `--root`, `--checkpoint`,
`--inventory`, `--registry` and `--mean-reference-root`. Pass `--matrix` to select
H1 or H2; omit it for the original H3 default. The reference path is the
historical campaign for the selected horizon, containing its three
`return_return_alpha_h{H}_j{1,2,4}_c16/bundle` directories. Preparation creates
three new immutable performance identities and a new comparison overview per
horizon. Default group names and labels follow the selected horizon. A mixed-H
campaign or a mean counterpart from another horizon is rejected.

One short J4 smoke per new horizon checks three decisions, strict CUDA compilation, complete
traces, finite metrics, unchanged outer state, and actual sampled execution.
The measured sampled flag must be one throughout; average and maximum distance
from the corresponding same-forward mean action must be positive. A zero minimum
distance is allowed because float32 tanh saturation can make individual actions
coincide. After the smoke passes, each of three independent workers owns all five
episodes for its setting. The launcher supports CPU `prepare`/`watch` and GPU
`worker` modes, using isolated scratch directories for W&B and compiler caches.

`slurm/ambi_closed_loop_sampled_publish.py watch` owns publication. Its dedicated
overview shows historical mean and new sampled return curves versus J, paired
sample-minus-mean curves and intervals, all per-seed values, setting status and
links. Historical mean points appear immediately; sampled points appear only
after complete validation. Evaluated and fully published states are distinct.
Native scalars use `sampled_execution/*` against `axis/inner_rounds`; each
measurement is logged once. Actual sampling proof is visible under `execution/*`.

Per-setting performance runs preserve the normal checkpoint-series identity.
Separate diagnostic runs contain model-return probes, reward/bootstrap
components, critic losses and TD errors, actor losses/entropy/alpha, actual
decision rewards, complete trace artifacts, and the hash-pinned historical
comparison provenance. The old mean-comparison workspace is untouched. An
uncertain publication or existing overview-start journal requires inspection;
the watcher never blindly appends duplicate scientific rows on restart.
