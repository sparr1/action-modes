"""Fresh H1 SAC with a common replay bank and three execution selectors.

This experiment changes action selection only. The model score is the expected
H1 reward-only Bellman target, with frozen auxiliary return Q and frozen SAC
terminal policy; it is not a real-environment oracle. All selection randomness
is private, and optional held-out diagnostics never participate in selection.
"""
from __future__ import annotations

import hashlib
import math
from numbers import Integral
import time

import numpy as np
import torch

from utils.ambi_benchmark import solver_seed
from utils.matched_action_audit import _frozen, _reference
from utils.transfer_diagnostics import Reference, json_value, validate_controller


ARMS = ("actor_mean", "learned_q", "model_score")
SOLVE_CONTRACT = dict(inner_rollout_horizon=1, inner_rounds=6,
    inner_rollouts_per_round=128, inner_critic_updates_per_round=16,
    inner_actor_updates_per_round=4, inner_batch_size=256,
    inner_actor_initialization="prior", inner_critic_initialization="prior",
    inner_critic_target_initialization="online", inner_actor_source="sac",
    inner_horizon_actor_source="sac", inner_critic_source="aux_return",
    inner_horizon_critic_source="aux_return", inner_sac_critic_target="reward_only",
    inner_terminal_entropy="none", inner_q_actor_reduction="mean_pair",
    mppi_terminal_q_reduction="mean_pair", inner_q_target_reduction="min_pair",
    inner_outer_replay_fraction=0., inner_replay_reset_each_round=False,
    inner_first_action_rounds=None, inner_diagnostic_rollouts=0,
    inner_replay_sampling="with_replacement", inner_rebase_persistent=False)


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def validate_bypass_controller(wrapped):
    """Fail closed when a config would change the fixed fresh-solve treatment."""
    validate_controller(wrapped)
    differences = {key: (getattr(wrapped.cfg, key, None), value)
        for key, value in SOLVE_CONTRACT.items()
        if getattr(wrapped.cfg, key, None) != value}
    if differences:
        raise ValueError(f"Critic bypass requires the fixed fresh H1/J6 solve: {differences}")
    if int(wrapped.cfg.inner_replay_capacity) < 768:
        raise ValueError("Critic bypass needs capacity for all 768 replay transitions.")


def _synchronize(reference):
    if torch.device(reference.device).type == "cuda":
        torch.cuda.synchronize(reference.device)


def complete_replay_actions(reference, root, *, expected=768):
    """Return copied actions only after verifying all transitions are root-local."""
    expected = _integer(expected, "expected", 1)
    engine = reference.engine
    replay = engine.state.replay if engine.state.replay is not None else engine._action_pool.replay
    if replay is None or replay.size != expected or replay.next_sample_id != expected:
        raise RuntimeError("SAC replay is incomplete or overflowed.")
    if not bool((replay.horizon_end[:expected] == 1).all()):
        raise RuntimeError("H1 replay contains a non-boundary transition.")
    roots = replay.z[:expected]
    if (roots.shape[1:] != root.shape[1:]
            or not torch.equal(roots, roots[:1].expand_as(roots))
            or not torch.allclose(roots[:1], root, atol=1e-6, rtol=1e-6)):
        raise RuntimeError("H1 replay does not contain exactly the current root.")
    actions = replay.action[:expected].detach().clone()
    if (actions.shape != (expected, int(reference.cfg.action_dim))
            or not torch.isfinite(actions).all() or bool((actions.abs() > 1).any())):
        raise RuntimeError("Replay has invalid normalized actions.")
    return actions


def _check_bank(reference, root, actions):
    _reference(reference)
    if (root.ndim != 2 or root.shape[0] != 1 or not torch.isfinite(root).all()
            or actions.ndim != 2 or not len(actions)
            or actions.shape[1] != int(reference.cfg.action_dim)
            or not torch.isfinite(actions).all() or bool((actions.abs() > 1).any())):
        raise ValueError("One finite root and finite normalized [N,A] bank are required.")


@torch.no_grad()
def learned_q_scores(reference, root, actions, critic):
    """Exact expectation of the configured mean-pair, with dropout disabled."""
    _check_bank(reference, root, actions)
    if reference.cfg.inner_q_actor_reduction != "mean_pair":
        raise ValueError("Learned-Q selection requires mean_pair.")
    with _frozen(reference, critic):
        heads = reference.q_heads(critic, root.expand(len(actions), -1), actions)
    if (heads.ndim != 3 or heads.shape[1:] != (len(actions), 1)
            or not torch.isfinite(heads).all()):
        raise RuntimeError("Invalid learned critic values.")
    return heads.mean(0).flatten()


@torch.no_grad()
def model_return_samples(reference, root, actions, *, seed, mc_rollouts=32,
                         max_expanded_batch=4096):
    """H1 target draws [MC, action], with common terminal noise across actions.

    Deterministic dynamics/reward are evaluated once per action. Only terminal
    policy/Q evaluation expands to MC draws. This is algebraically identical
    to the H1 forced-action audit score (including its named terminal stream).
    """
    _check_bank(reference, root, actions)
    seed = _integer(seed, "seed")
    count = _integer(mc_rollouts, "mc_rollouts", 2)
    limit = _integer(max_expanded_batch, "max_expanded_batch", count)
    if reference.cfg.mppi_terminal_q_reduction != "mean_pair":
        raise ValueError("Direct model selection requires terminal mean_pair.")
    chunk = max(1, limit // count)
    draws = reference.noise((count, 1, int(reference.cfg.action_dim)),
                            solver_seed(seed, "tail"))
    values = []
    with _frozen(reference):
        for start in range(0, len(actions), chunk):
            candidate = actions[start:start + chunk]
            reward, successor, terminated = reference.transition(
                root.expand(len(candidate), -1), candidate)
            tail = reference.tail(successor[None].expand(count, -1, -1).reshape(
                count * len(candidate), -1), draws.expand(-1, len(candidate), -1).reshape(
                count * len(candidate), -1)).reshape(count, len(candidate))
            done = terminated.reshape(1, len(candidate)).bool()
            # Do not allow unused terminal NaNs to leak through multiplication.
            continuation = torch.where(done, 0., tail)
            result = reward.reshape(1, len(candidate)) + reference.discount * continuation
            if not torch.isfinite(result).all():
                raise RuntimeError("Nonfinite direct model return.")
            values.append(result)
    return torch.cat(values, dim=1)


def stable_argmax(values):
    """First exact maximum wins: bank order prefers actor, then prior, then replay."""
    if values.ndim != 1 or not len(values) or not torch.isfinite(values).all():
        raise ValueError("Selection requires finite one-dimensional scores.")
    return int(values.argmax().item())


def _stats(samples):
    samples = samples.detach().double().cpu()
    return dict(mean=float(samples.mean()),
                se=float(samples.std(unbiased=True) / math.sqrt(len(samples))),
                draws=samples.tolist())


def run_bypass_decision(wrapped, observation, *, arm, solve_seed, selection_seed,
                        validation_seed, shadow=False, mc_rollouts=32,
                        max_expanded_batch=4096):
    """Run exactly one fresh solve and select its executed environment action.

    Timings synchronize CUDA and include all required solve/bank/selection work
    and action conversion. Shadow selector comparisons and independent MC
    validation are reported separately and cannot affect the executed action.
    The caller must hash the frozen outer state before/after each episode.
    """
    if arm not in ARMS:
        raise ValueError(f"Unknown critic bypass arm: {arm}")
    if not isinstance(shadow, bool):
        raise TypeError("shadow must be bool.")
    seeds = [_integer(value, name) for value, name in (
        (solve_seed, "solve_seed"), (selection_seed, "selection_seed"),
        (validation_seed, "validation_seed"))]
    if len(set(seeds)) != 3:
        raise ValueError("Solve, selection, and validation require independent seeds.")
    count = _integer(mc_rollouts, "mc_rollouts", 2)
    limit = _integer(max_expanded_batch, "max_expanded_batch", count)
    if torch.device(wrapped.agent.device).type == "cuda":
        torch.cuda.synchronize(wrapped.agent.device)
    started = time.perf_counter()
    validate_bypass_controller(wrapped)
    reference = Reference(wrapped, rollouts=count)
    engine = reference.engine
    engine.reset_for_evaluation(int(solve_seed), reuse_action_pool=True)
    action, _ = wrapped.predict(observation, deterministic=True, episode_start=True,
                                collect_diagnostics=False)
    metrics = wrapped.agent.last_inner_metrics
    work = dict(actor_updates=int(metrics["inner_actor_optimizer_steps"]),
                critic_updates=int(metrics["inner_critic_optimizer_steps"]),
                model_steps=int(metrics["inner_model_steps"]))
    if work != dict(actor_updates=24, critic_updates=96, model_steps=768):
        raise RuntimeError(f"Realized SAC work differs from the fixed solve: {work}")
    _synchronize(reference)
    solve_done = time.perf_counter()
    pool = engine._action_pool
    if pool.actor is None or pool.critic is None:
        raise RuntimeError("Fresh SAC did not retain its final actor and critic allocations.")
    with torch.no_grad(), _frozen(reference, pool.actor, pool.critic):
        root = reference.encode(np.asarray(observation)[None]).detach()
        actor_mean = reference.model.policy_stats(root, policy=pool.actor,
                                                  **reference.bounds)["mean"].detach()
        prior_mean = reference.model.policy_stats(root, policy=engine._actor_base,
            **engine._actor_options)["mean"].detach()
        returned = torch.as_tensor(wrapped._scale_action(np.asarray(action)),
                                   device=reference.device).reshape_as(actor_mean)
        if not torch.allclose(returned, actor_mean, atol=2e-6, rtol=2e-6):
            raise RuntimeError("Returned SAC action disagrees with the final actor mean.")
        replay = complete_replay_actions(reference, root)
        bank = torch.cat((actor_mean, prior_mean, replay))
    labels = ("actor_mean", "prior_mean", *(f"replay/{i:04d}" for i in range(len(replay))))
    _synchronize(reference)
    bank_done = time.perf_counter()
    scores, model_samples = None, None
    choices = {"actor_mean": 0}
    if arm == "learned_q":
        scores = learned_q_scores(reference, root, bank, pool.critic)
        choices[arm] = stable_argmax(scores)
    elif arm == "model_score":
        model_samples = model_return_samples(reference, root, bank, seed=int(selection_seed),
            mc_rollouts=count, max_expanded_batch=limit)
        choices[arm] = stable_argmax(model_samples.mean(0))
    selected_index = choices[arm]
    normalized = bank[selected_index].detach().cpu().numpy().copy()
    # Preserve the ordinary actor's original environment action bit for bit.
    executed = np.asarray(action).copy() if arm == "actor_mean" else np.asarray(
        wrapped._unscale_action(normalized)).copy()
    _synchronize(reference)
    control_done = time.perf_counter()
    diagnostics = None
    if shadow:
        if scores is None:
            scores = learned_q_scores(reference, root, bank, pool.critic)
        if model_samples is None:
            model_samples = model_return_samples(reference, root, bank, seed=int(selection_seed),
                mc_rollouts=count, max_expanded_batch=limit)
        choices["learned_q"] = stable_argmax(scores)
        choices["model_score"] = stable_argmax(model_samples.mean(0))
        aliases = (*ARMS, "prior_mean")
        indices = [*(choices[name] for name in ARMS), 1]
        heldout = model_return_samples(reference, root, bank[indices], seed=int(validation_seed),
            mc_rollouts=count, max_expanded_batch=limit)
        diagnostics = dict(choices={name: dict(index=index, label=labels[index],
            normalized_action=bank[index].detach().cpu().tolist(),
            learned_q=float(scores[index]),
            selection_model=_stats(model_samples[:, index]),
            validation_model=_stats(heldout[:, column]),
            validation_gain_vs_prior=_stats(heldout[:, column] - heldout[:, -1]),
            validation_gain_vs_actor=_stats(heldout[:, column] - heldout[:, 0]))
            for column, (name, index) in enumerate(zip(aliases, indices))},
            bank_sha256=hashlib.sha256(bank.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
            bank_count=len(bank), replay_count=len(replay),
            selection_model_means=model_samples.mean(0).detach().cpu().tolist(),
            selection_q_scores=scores.detach().cpu().tolist(),
            selection_seed=int(selection_seed), validation_seed=int(validation_seed),
            mc_rollouts=count, root=root.detach().cpu().tolist(),
            semantics=dict(learned_q="expected mean-pair, dropout disabled",
                model="H1 reward + gamma frozen SAC-prior auxiliary return Q; no entropy",
                tie_break="actor mean, prior mean, replay insertion order",
                validation="independent terminal noise; fixed selected actions"))
    diagnostics = json_value(diagnostics)
    _synchronize(reference)
    finished = time.perf_counter()
    return dict(action=executed, normalized_action=normalized, arm=arm,
        selected_index=selected_index, selected_label=labels[selected_index],
        bank_count=len(bank), replay_count=len(replay), solve_seed=int(solve_seed),
        work=work, diagnostics=diagnostics,
        timing=dict(solve_seconds=solve_done-started, bank_seconds=bank_done-solve_done,
            selection_seconds=control_done-bank_done, control_seconds=control_done-started,
            diagnostics_seconds=finished-control_done),
        checks=dict(h1_replay_root_only=True, h1_replay_complete=True,
                    execution_mean_verified=True, independent_selection_validation=True))
