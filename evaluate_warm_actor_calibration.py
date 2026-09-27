"""Matched model/real calibration of policies captured on warm trajectories.

This is evaluation only. Each immutable root owns independent, paired noise;
the captured policy is held for H steps and the sampled frozen prior follows.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import torch

from evaluate_ambi_checkpoint import (
    _close_resources, _file_sha256, _initialize_frozen_model, _make_env,
    _outer_state_digest,
)
from utils.ambi_benchmark import atomic_json, solver_seed
from utils.ambi_research import load_preset_matrix, resolve_preset
from utils.checkpoint_context import load_checkpoint_context


INFERENCE_OVERRIDES = {"compile": False, "compile_strict": False}
TERMINAL_REDUCTIONS = ("expected_mean_pair", "expected_min_pair", "min_all")


def paired_noise(root_id, horizon, tail_steps, rollouts, action_dim):
    seed = solver_seed(55, "warm-real-calibration", root_id)
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((horizon, rollouts, action_dim)).astype(np.float32),
            rng.standard_normal((tail_steps, rollouts, action_dim)).astype(np.float32), seed)


class FrozenCallbacks:
    """Use routed prior/return critic and encode each new real observation."""
    def __init__(self, wrapped, policy, bounds, pair_indices, action_mode="sample"):
        self.wrapped = wrapped
        self.model = wrapped.agent.model
        self.engine = wrapped.agent.inner_engine
        self.device = wrapped.agent.device
        self.policy = policy
        self.bounds = bounds
        self.pair_indices = pair_indices
        if action_mode not in {"sample", "mean"}:
            raise ValueError("Action mode must be sample or mean.")
        self.action_mode = action_mode

    def encode(self, observations):
        return self.model.encode(torch.as_tensor(observations, device=self.device, dtype=torch.float32))

    @torch.no_grad()
    def actor(self, observations, noise):
        z = self.encode(observations)
        if self.action_mode == "mean":
            action = self.model.policy_stats(z, policy=self.policy, **self.bounds)["mean"]
        else:
            action, _ = self.model.pi(z, policy=self.policy,
                noise=torch.as_tensor(noise, device=self.device, dtype=z.dtype), **self.bounds)
        return action.cpu().numpy()

    @torch.no_grad()
    def q(self, observations, actions):
        from RL.tdmpc2_core.inner_trace import evaluate_frozen_outer_q
        z = self.encode(observations)
        value = evaluate_frozen_outer_q(self.model, z,
            torch.as_tensor(actions, device=self.device, dtype=z.dtype),
            reduction=self.wrapped.cfg.mppi_terminal_q_reduction,
            pair_indices=(self.pair_indices if self.wrapped.cfg.mppi_terminal_q_reduction.endswith("_pair") else None),
            critic=self.engine._horizon_critic)
        return value.reshape(-1).cpu().numpy()

    @torch.no_grad()
    def q_heads(self, observations, actions):
        from RL.tdmpc2_core.inner_trace import evaluate_frozen_outer_q
        z = self.encode(observations)
        value = evaluate_frozen_outer_q(self.model, z,
            torch.as_tensor(actions, device=self.device, dtype=z.dtype),
            reduction="all", critic=self.engine._horizon_critic)
        return value[..., 0].transpose(0, 1).cpu().numpy()


@torch.no_grad()
def model_branches(wrapped, observation, policy, bounds, prefix_noise, tail_noise,
                   pair_indices, mode):
    from RL.tdmpc2_core.inner_trace import evaluate_frozen_outer_q
    from utils.ambi_real_calibration import terminal_q_reductions
    if mode not in {"sample", "mean"}:
        raise ValueError("Action mode must be sample or mean.")
    model, engine, cfg = wrapped.agent.model, wrapped.agent.inner_engine, wrapped.cfg
    if (cfg.mppi_terminal_q_reduction.endswith("_pair")
            and model.q_backend.pair_size != model.q_backend.num_q and pair_indices is None):
        raise ValueError("A paired terminal-Q reduction requires explicit pair_indices.")
    if prefix_noise.ndim != 3 or prefix_noise.shape[0] < 1 or prefix_noise.shape[1] < 1:
        raise ValueError("Prefix noise must be [positive horizon, positive rollouts, action_dim].")
    if tail_noise.ndim != 3 or tail_noise.shape[0] < 1 or tail_noise.shape[1:] != prefix_noise.shape[1:]:
        raise ValueError("Tail noise must match prefix rollouts and actions.")
    if prefix_noise.shape[2] != int(cfg.action_dim):
        raise ValueError("Prefix action dimension differs from the checkpoint.")
    if not np.isfinite(prefix_noise).all() or not np.isfinite(tail_noise).all():
        raise ValueError("Policy noise must be finite.")
    # Zero prefix Gaussian noise selects tanh(mu), while the tail stays sampled.
    prefix = np.zeros_like(prefix_noise) if mode == "mean" else prefix_noise
    noise = torch.as_tensor(np.concatenate((prefix, tail_noise[:1])),
                            device=wrapped.agent.device, dtype=torch.float32)
    modes = {module: bool(module.training)
             for root in (model, policy) for module in root.modules()}
    try:
        model.eval()
        policy.eval()
        z = model.encode(torch.as_tensor(observation, device=noise.device,
                                        dtype=torch.float32)[None])
        z = z.expand(prefix_noise.shape[1], -1).clone()
        reward = z.new_zeros((prefix_noise.shape[1], 1))
        alive = torch.ones_like(reward, dtype=torch.bool)
        discount = 1.0
        for step in range(prefix_noise.shape[0]):
            action, _ = model.pi(z, policy=policy, noise=noise[step], **bounds)
            joint = model.joint_input(z, action)
            reward += torch.where(alive, discount * model.decode_reward(model.reward_from_joint(joint)), 0.0)
            z = model.next_from_joint(joint)
            if cfg.episodic:
                alive &= model.termination(z) <= float(cfg.inner_termination_threshold)
            discount *= float(wrapped.agent.discount)
        action, _ = model.pi(z, policy=engine._horizon_actor, noise=noise[-1],
                             **engine._horizon_actor_options)
        heads = evaluate_frozen_outer_q(model, z, action, reduction="all",
                                        critic=engine._horizon_critic)
        reduction = cfg.mppi_terminal_q_reduction
        q = model.q_backend.reduce(heads, reduction,
            pair_indices=pair_indices if reduction.endswith("_pair") else None,
            trusted_pair_indices=True)
        bootstrap = torch.where(alive, discount * q, 0.0)
        raw_heads = heads[..., 0].transpose(0, 1).cpu().numpy()
        expected = terminal_q_reductions(raw_heads)
        rewards, bootstraps = reward[:, 0].cpu().numpy(), bootstrap[:, 0].cpu().numpy()
        active = alive[:, 0].cpu().numpy()
        actions = action.cpu().numpy()
        rows = []
        for index, (r, b) in enumerate(zip(rewards, bootstraps)):
            row = dict(model_prefix_reward=float(r), model_bootstrap=float(b),
                       predicted_model_return=float(r + b),
                       model_endpoint_q_heads=raw_heads[index].tolist(),
                       model_endpoint_action=actions[index].tolist(),
                       model_prefix_terminated=not bool(active[index]))
            for name in TERMINAL_REDUCTIONS:
                endpoint_q = float(expected[name][index])
                contribution = discount * endpoint_q if active[index] else 0.0
                row.update({f"model_endpoint_q_{name}": endpoint_q,
                            f"model_bootstrap_{name}": contribution,
                            f"predicted_model_return_{name}": float(r) + contribution})
            rows.append(row)
        return rows
    finally:
        for module, was_training in modes.items():
            module.training = was_training


def boundary_diagnostics(root, family, round_index):
    """Join observations from the actual actor boundary, without pooling roots."""
    events = root.get("diagnostic_events", {})
    if isinstance(events, dict):
        events = events.get(family, [])
    matches = [event for event in events
               if event.get("phase") == "transfer_probe"
               and event.get("round_index", event.get("inner_round", event.get("round", 0))) == round_index
               and event.get("stage") == ("initial" if round_index == 0 else "post_round")]
    if not matches:
        return {}
    metrics = matches[-1].get("metrics", {})
    inner = metrics.get("transfer_root_q_inner_advantage_mean_all")
    frozen = metrics.get("transfer_root_q_frozen_advantage_mean_all")
    # Full diagnostic events remain in the immutable root artifact. Repeating
    # them for every MC replicate would multiply the result size needlessly.
    result = {}
    if inner is not None and frozen is not None:
        result.update(critic_preference_gap=inner-frozen,
                      adapted_actor_advantage=inner, frozen_actor_advantage=frozen)
    heads = [value for key, value in metrics.items()
             if key.startswith("transfer_root_q_inner_actor_head_")]
    if heads:
        result["critic_head_sd"] = float(np.std(heads))
    return result


def join_rows(predicted, real, *, metadata):
    if len(predicted) != len(real):
        raise ValueError("Model and simulator branch coverage differs.")
    rows = []
    for replicate, (estimate, observed) in enumerate(zip(predicted, real)):
        if not observed["mc_complete"] or not observed["episode_cutoff_complete"]:
            raise ValueError("A real branch did not complete the requested return.")
        a, b, c = (estimate["predicted_model_return"],
                   observed["real_bootstrapped_return"], observed["real_mc_return"])
        if not np.isfinite([a, b, c]).all():
            raise ValueError("Nonfinite calibrated return.")
        rows.append(dict(metadata, **estimate, **observed, replicate=replicate,
                         model_prefix_error=a-b, terminal_value_error=b-c,
                         total_prediction_error=a-c))
        for name in TERMINAL_REDUCTIONS:
            model_key, real_key = f"predicted_model_return_{name}", f"real_bootstrapped_return_{name}"
            if model_key in estimate and real_key in observed:
                model_return, real_return = estimate[model_key], observed[real_key]
                rows[-1].update({f"model_prefix_error_{name}": model_return-real_return,
                                 f"terminal_value_error_{name}": real_return-c,
                                 f"total_prediction_error_{name}": model_return-c})
    return rows


def evaluate_root(root_path, output, *, checkpoint=None, matrix=None, device="cuda",
                  rollouts=32, tail_steps=1000):
    from utils.ambi_real_calibration import SimulatorSnapshot, evaluate_real_branches
    if rollouts < 1 or tail_steps < 1:
        raise ValueError("Positive branch and tail counts are required.")
    root_path, output = Path(root_path).resolve(), Path(output).resolve()
    root = json.loads(root_path.read_text())
    checkpoint = Path(checkpoint or root["checkpoint"]).resolve()
    matrix = Path(matrix or root["matrix"]).resolve()
    checkpoint_hash = _file_sha256(checkpoint)
    matrix_hash = _file_sha256(matrix)
    if root.get("checkpoint_sha256") and root["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("Captured actor checkpoint mismatch.")
    if root.get("matrix_sha256") and root["matrix_sha256"] != matrix_hash:
        raise ValueError("Captured actor preset matrix mismatch.")
    identity = dict(root_sha256=_file_sha256(root_path), checkpoint_sha256=checkpoint_hash,
                    matrix_sha256=matrix_hash, inference_overrides=INFERENCE_OVERRIDES,
                    rollouts=rollouts, tail_steps=tail_steps, modes=["sample", "mean"],
                    protocol="warm-actor-branch-calibration-v1")
    if output.exists():
        previous = json.loads(output.read_text())
        if previous.get("status") == "complete" and previous.get("identity") == identity:
            return previous
        raise FileExistsError("Existing result is incomplete or has a different identity.")
    preset_matrix = load_preset_matrix(matrix)
    resolved = resolve_preset(matrix, root["selector"], preset_matrix,
                              checkpoint_context=load_checkpoint_context(checkpoint))
    resolved = deepcopy(resolved)
    # This worker never optimizes. Eager inference avoids needless compile startup.
    resolved["algorithm_config"]["alg_params"].update(INFERENCE_OVERRIDES)
    env, wrapped, branch_envs = None, None, []
    started = time.perf_counter()
    try:
        env = _make_env(resolved)
        wrapped, _ = _initialize_frozen_model(resolved, env, checkpoint, 55, device=device)
        cfg, engine = wrapped.cfg, wrapped.agent.inner_engine
        if cfg.inner_horizon_critic_source != "aux_return" or cfg.inner_horizon_actor_source != "sac":
            raise ValueError("This campaign requires the auxiliary return critic and frozen SAC tail.")
        if getattr(cfg, "aux_return_mode", None) != "sac":
            raise ValueError("The auxiliary critic must have been trained to evaluate the SAC prior.")
        if getattr(cfg, "inner_horizon_conditioning", "none") != "none":
            raise ValueError("Captured policies must be unconditioned for this campaign.")
        if cfg.critic_value_mode != "single":
            raise ValueError("This campaign requires the single reward-only auxiliary critic route.")
        horizon, discount = int(root["H"]), float(wrapped.agent.discount)
        if horizon != int(cfg.inner_rollout_horizon):
            raise ValueError("Root and model horizons differ.")
        remaining = int(root.get("episode_max_steps", 500)) - int(root["decision_index"])
        if horizon + tail_steps < remaining:
            raise ValueError("Tail does not reach the original episode cutoff.")
        frozen_before = _outer_state_digest(wrapped)
        snapshot = SimulatorSnapshot.from_dict(root["snapshot"])
        if not np.array_equal(np.asarray(root["observation"], dtype=np.float32),
                              snapshot.state()["environment"]["observation"]):
            raise ValueError("Captured root observation differs from its simulator snapshot.")
        prefix_noise, tail_noise, noise_seed = paired_noise(root["root_id"], horizon,
            tail_steps, rollouts, int(cfg.action_dim))
        pair_seed = solver_seed(55, "warm-calibration-q-pair", root["root_id"])
        generator = torch.Generator(device=wrapped.agent.device).manual_seed(pair_seed)
        indices = wrapped.agent.model.q_backend.sample_pair_indices(wrapped.agent.device, generator=generator)
        prior = FrozenCallbacks(wrapped, engine._horizon_actor, engine._horizon_actor_options, indices)
        for _ in range(rollouts):
            branch = _make_env(resolved)
            branch.reset(seed=int(root["seed"]))
            branch_envs.append(branch)
        common = dict(source_cell=root.get("source_cell", root.get("cell")), H=horizon,
            J=int(root["J"]), seed=int(root["seed"]), decision=int(root["decision_index"]),
            root_id=root["root_id"], branch_kind="prefix",
            return_semantics="discounted_H_prefix_then_sampled_frozen_prior", discount=discount,
            auxiliary_training_mode=cfg.aux_return_mode, tail_actor_source="sac",
            terminal_critic_source="aux_return", terminal_reduction=cfg.mppi_terminal_q_reduction)
        records, timings = [], []
        cache = output.with_suffix(".parts")
        cache.mkdir(parents=True, exist_ok=True)
        for actor in root["actors"]:
            path = root_path.parent / actor["path"]
            if _file_sha256(path) != actor["sha256"]:
                raise ValueError(f"Actor payload hash mismatch: {path}")
            policy = torch.load(path, map_location=wrapped.agent.device, weights_only=False)
            policy.eval().requires_grad_(False)
            family, stage = actor["family"], int(actor["round"])
            diag = boundary_diagnostics(root, family, stage)
            for mode in ("sample", "mean"):
                part_path = cache / f"{family}-r{stage}-{mode}.json"
                part_identity = dict(identity, actor_sha256=actor["sha256"], mode=mode)
                if part_path.exists():
                    part = json.loads(part_path.read_text())
                    if part["identity"] != part_identity or len(part["records"]) != rollouts:
                        raise ValueError("Cached branch identity or coverage differs.")
                else:
                    tick = time.perf_counter()
                    predicted = model_branches(wrapped, root["observation"], policy,
                        actor["bounds"], prefix_noise, tail_noise, indices, mode)
                    callbacks = FrozenCallbacks(wrapped, policy, actor["bounds"], indices, mode)
                    observed = evaluate_real_branches(branch_envs, snapshot, callbacks.actor,
                        prior.actor, prior.q, prefix_noise, tail_noise, discount=discount,
                        horizon=horizon, original_remaining_steps=remaining, outer_q_heads=prior.q_heads)
                    rows = join_rows(predicted, observed["rows"], metadata=dict(common, **diag,
                        actor_family=family, round=stage, action_mode=mode,
                        actor_sha256=actor["sha256"], actor_updates=actor.get("actor_updates", 0),
                        critic_updates=actor.get("critic_updates", 0)))
                    part = dict(identity=part_identity, records=rows,
                                seconds=time.perf_counter()-tick, timing=observed["timing"], work=observed["work"])
                    atomic_json(part_path, part)
                records.extend(part["records"])
                timings.append({"family":family,"round":stage,"mode":mode,"seconds":part["seconds"]})
            del policy
        if _outer_state_digest(wrapped) != frozen_before:
            raise RuntimeError("Calibration changed the frozen backbone.")
        result = dict(schema_version=1, status="complete", identity=identity, **common,
            records=records, noise_seed=noise_seed, q_pair_seed=pair_seed,
            q_pair_indices=None if indices is None else indices.cpu().tolist(), root_file=str(root_path),
            source_selector=root["selector"], terminal_q_reduction=cfg.mppi_terminal_q_reduction,
            horizon_actor_source=cfg.inner_horizon_actor_source,
            horizon_critic_source=cfg.inner_horizon_critic_source,
            total_seconds=time.perf_counter()-started, timings=timings, frozen_state_unchanged=True)
        atomic_json(output, result)
        return result
    finally:
        for branch in branch_envs:
            branch.close()
        if wrapped is not None:
            _close_resources(wrapped)
        if env is not None:
            env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--matrix")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rollouts", type=int, default=32)
    parser.add_argument("--tail-steps", type=int, default=1000)
    args = parser.parse_args(argv)
    result = evaluate_root(args.root, args.output, checkpoint=args.checkpoint, matrix=args.matrix,
                           device=args.device, rollouts=args.rollouts, tail_steps=args.tail_steps)
    print(json.dumps({"status":result["status"],"records":len(result["records"]),"output":args.output}))


if __name__ == "__main__":
    main()
