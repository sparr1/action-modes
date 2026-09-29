"""Paired fixed-root critic-BN probe; model consistency, not environment return."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import re

import numpy as np
import torch

import run_ambixqc_inner_475k_screen as source
from RL.tdmpc2_core.xqc_controller import LatentXQCBatch
from RL.tdmpc2_core.common import math as td_math
from RL.tdmpc2_core.inner_xqc import InnerXQCEngine
from RL.xqc_core import categorical_td_projection, select_lower_distribution

MODES = ("batch_update", "batch_no_update", "running")
SOLVER_SEEDS = (12345, 23456, 34567)
INSPECTION_STEPS = (0, 1, 3, 12)
SCHEMA = "ambixqc-bn-probe-v1"
SELECTION_SCHEMA = "ambixqc-bn-probe-selection-v1"
N = 256
CHECKPOINT_METADATA_SHA = "d02f3ea3eedfe706474803312e11d335fc0794d8ac59174f59b253072127090f"
TARGET_KIND = "expectation_of_categorical_projection(model_reward/scale + discount * frozen_auxiliary_min_distribution)"


def tree_hash(value):
    digest = hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            x = x.detach().cpu().contiguous()
            digest.update(str((str(x.dtype), tuple(x.shape))).encode())
            digest.update(x.numpy().tobytes())
        elif isinstance(x, np.ndarray):
            digest.update(str((str(x.dtype), x.shape)).encode()); digest.update(x.tobytes())
        elif isinstance(x, dict):
            for key in sorted(x): digest.update(str(key).encode()); visit(x[key])
        elif isinstance(x, (list, tuple)):
            for element in x: visit(element)
        elif isinstance(x, float):
            # Persistent reward trackers may intentionally contain -inf before
            # their first update. Hash its representation without treating it
            # as a finite diagnostic measurement.
            digest.update(x.hex().encode())
        else:
            digest.update(json.dumps(x, sort_keys=True, allow_nan=False).encode())
    visit(value)
    return digest.hexdigest()


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


@contextmanager
def preserve_rng():
    before = rng_state()
    try:
        yield
    finally:
        random.setstate(before["python"]); np.random.set_state(before["numpy"])
        torch.set_rng_state(before["torch"])
        if before["cuda"]: torch.cuda.set_rng_state_all(before["cuda"])
        if tree_hash(rng_state()) != tree_hash(before):
            raise RuntimeError("Global RNG restoration failed.")


def root_positions(count=32, episode_steps=500):
    if not 1 <= count <= 5 * episode_steps:
        raise ValueError("Root count must fit five prior trajectories.")
    allocations = [count // 5 + int(i < count % 5) for i in range(5)]
    return [(seed, int(step)) for seed, n in zip(source.SEEDS, allocations)
            for step in np.linspace(0, episode_steps - 1, n, dtype=int)]


@torch.no_grad()
def capture_roots(model, count=32, episode_steps=500):
    selected = set(root_positions(count, episode_steps))
    roots, episode_returns = [], {}
    agent = model.agent
    for seed in source.SEEDS:
        if not any(s == seed for s, _ in selected): continue
        observation, _ = model.env.reset(seed=seed)
        total = 0.0
        for decision in range(episode_steps):
            if (seed, decision) in selected:
                roots.append({"root_id": f"seed-{seed}-decision-{decision}", "seed": seed,
                              "decision": decision, "observation": np.asarray(observation, dtype=np.float32).tolist()})
            z = agent.model.encode(model._obs_to_tensor(observation).to(agent.device).unsqueeze(0))
            action, _ = agent.xqc_controller.sample_action(z, deterministic=True)
            action = model._unscale_action(action[0].cpu().numpy())
            observation, reward, terminated, truncated, _ = model.env.step(action)
            total += float(reward)
            if terminated or truncated:
                if decision + 1 != episode_steps:
                    raise ValueError("Prior trajectory ended before the requested root grid.")
                break
        episode_returns[str(seed)] = total
    if [(r["seed"], r["decision"]) for r in roots] != root_positions(count, episode_steps):
        raise RuntimeError("Fixed-root capture is incomplete.")
    return roots, episode_returns


def _cpu_tree(value):
    if torch.is_tensor(value): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: _cpu_tree(v) for k, v in value.items()}
    return value


def heldout_targets(rewards, tail_log_probs, support, reward_scale, discount):
    """Match the reward-only categorical target, including atomwise clipping."""
    selected, tail_values, _ = select_lower_distribution(tail_log_probs, support)
    probabilities, clip_fraction = categorical_td_projection(
        selected, rewards / reward_scale, torch.ones_like(rewards),
        torch.full_like(rewards, discount), torch.zeros_like(rewards), support,
    )
    return {
        "heldout_target": (probabilities * support).sum(dim=-1),
        "heldout_unprojected_target": rewards / reward_scale + discount * tail_values,
        "heldout_target_probabilities": probabilities,
        "target_clip_fraction": clip_fraction,
    }


@torch.no_grad()
def paired_inputs(agent, root_z, solver_seed, root_id, *, count=N, max_updates=12):
    seed = int.from_bytes(hashlib.sha256(f"bn-probe-v1:{root_id}:{solver_seed}".encode()).digest()[:8], "big") % (2**63-1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    device, dtype = root_z.device, root_z.dtype
    def noise(shape): return torch.randn(shape, generator=generator, dtype=dtype).to(device)
    actor = agent.xqc_controller.actor
    scale = float(agent.reward_normalizer.scale)
    def transitions():
        z = root_z.expand(count, -1).clone()
        action, _ = actor.sample(z, bn_mode="running", noise=noise((count, agent.cfg.action_dim)))
        joint = agent.model.joint_input(z, action)
        reward = td_math.two_hot_inv(agent.model.reward_from_joint(joint), agent.cfg).reshape(-1)
        next_z = agent.model.next_from_joint(joint)
        return {"latents": z, "actions": action, "rewards": reward,
                "next_latents": next_z, "bootstrap_mask": torch.ones_like(reward)}
    train, heldout = transitions(), transitions()
    heldout_noise = noise((count, agent.cfg.action_dim))
    heldout_next_actions, _ = actor.sample(heldout["next_latents"], bn_mode="running", noise=heldout_noise)
    tail_log = agent.aux_return.critic.log_probs(heldout["next_latents"], heldout_next_actions, bn_mode="running")
    support = agent.aux_return.critic.support
    return {"train": train, "heldout": heldout, "heldout_next_noise": heldout_noise,
            "heldout_next_actions": heldout_next_actions,
            **heldout_targets(heldout["rewards"], tail_log, support, scale, float(agent.discount)),
            "reward_scale": scale, "discount": float(agent.discount),
            "sample_indices": torch.randint(count, (max_updates, count), generator=generator).to(device),
            "bootstrap_noises": noise((max_updates, count, agent.cfg.action_dim)),
            "actor_noise": noise((count, agent.cfg.action_dim)), "derived_seed": seed}


def clone_workspace(outer, critic, *, actor_lr=5e-5, critic_lr=5e-5):
    # Reuse inherited tensors without random network initialization. Construction
    # via clone_for_inner otherwise consumes global RNG and repeats expensive QR.
    controller = deepcopy(outer)
    controller.configure_compile(enabled=False, strict=False)
    controller.reset_prior_from_(outer, critic_source=critic)
    return controller.make_workspace(actor_lr=actor_lr, critic_lr=critic_lr,
                                     actor_lr_end=actor_lr, critic_lr_end=critic_lr, transition_steps=12)


def critic_batch(data, index=None):
    values = data["train"]
    if index is not None: values = {key: tensor[index] for key, tensor in values.items()}
    return LatentXQCBatch(**values, discount=data["discount"])


def critic_only_update(workspace, outer, outer_critic, data, step, mode):
    batch = critic_batch(data, data["sample_indices"][step])
    objective = workspace.controller.critic_objective(
        batch, next_noise=data["bootstrap_noises"][step], reward_scale=data["reward_scale"],
        outer_terminal_mask=torch.ones(batch.latents.shape[0], dtype=torch.bool, device=batch.latents.device),
        outer_controller=outer, outer_critic=outer_critic, outer_critic_is_return=True,
        critic_target_kind="reward_only", critic_bn_mode=mode)
    workspace.zero_critic_grad(); objective.loss.backward(); workspace.step_critic()
    # Actor/temperature remain fixed; their usual helper owns this increment.
    workspace.update_step += 1
    return float(objective.loss.detach()), float(objective.clip_fraction.detach())


def gradient_comparison(left, right):
    left, right = left.double().flatten(), right.double().flatten()
    if not bool(torch.isfinite(left).all() & torch.isfinite(right).all()):
        raise ValueError("Action-gradient comparisons require finite gradients.")
    a, b = left.norm(), right.norm()
    cosine_defined, ratio_defined = bool((a > 0) & (b > 0)), bool(b > 0)
    return {"cosine": float(((left @ right) / (a*b)).clamp(-1, 1)) if cosine_defined else None,
            "norm_ratio": float(a / b) if ratio_defined else None,
            "cosine_defined": cosine_defined, "norm_ratio_defined": ratio_defined,
            "left_norm": float(a), "right_norm": float(b)}


def q_views(critic, data):
    held = data["heldout"]
    values, grads = {}, {}
    for name, bn_mode in (("running", "running"), ("joined", "batch_no_update")):
        action = held["actions"].detach().clone().requires_grad_(True)
        z = torch.cat((held["latents"], held["next_latents"]), dim=0)
        joined_action = torch.cat((action, data["heldout_next_actions"]), dim=0)
        q = critic.values_from_log_probs(critic.log_probs(z, joined_action, bn_mode=bn_mode))[:, :action.shape[0]]
        gradient, = torch.autograd.grad(q.min(dim=0).values.sum(), action)
        values[name], grads[name] = q.detach(), gradient.detach()
    return values, grads


def bn_buffers(critic):
    return {name: value.detach().clone() for name, value in critic.named_buffers()
            if name.endswith(("running_mean", "running_var"))}


def bn_drift(critic, initial):
    return {name: float((value - initial[name]).double().square().mean().sqrt())
            for name, value in bn_buffers(critic).items()}


def inspect(workspace, outer, data, initial_bn, *, actor_lr=5e-5):
    controller = workspace.controller
    before = tree_hash({"controller": controller.state_dict(), "workspace": workspace.state_dict(), "rng": rng_state()})
    values, gradients = q_views(controller.critic, data)
    target = data["heldout_target"]
    running, joined = values["running"].mean(dim=0), values["joined"].mean(dim=0)
    metrics = {
        "heldout_running_q_mse": float((running-target).square().mean()),
        "heldout_joined_q_mse": float((joined-target).square().mean()),
        "heldout_running_min_q_mse": float((values["running"].min(dim=0).values-target).square().mean()),
        "heldout_running_unprojected_q_mse": float((running-data["heldout_unprojected_target"]).square().mean()),
        "running_joined_q_rmse": float((running-joined).square().mean().sqrt()),
        "running_joined_q_mean_difference": float((running-joined).mean()),
        "running_vs_joined_action_gradient": gradient_comparison(gradients["running"], gradients["joined"]),
        "target_clip_fraction": float(data["target_clip_fraction"]),
        "bn_drift_rms": bn_drift(controller.critic, initial_bn),
    }
    diagnostic = clone_workspace(outer, controller.critic, actor_lr=actor_lr)
    actor_before = tree_hash(bn_buffers(diagnostic.controller.actor))
    objective = diagnostic.controller.actor_objective(data["train"]["latents"],
        actor_noise=data["actor_noise"], actor_bn_mode="running")
    diagnostic.step_actor_and_temperature(objective.loss, objective.entropy.mean())
    root = data["train"]["latents"][:1]
    with torch.no_grad():
        old_mean, old_logstd = outer.actor.distribution(root, bn_mode="running")
        new_mean, new_logstd = diagnostic.controller.actor.distribution(root, bn_mode="running")
        metrics["one_actor_step_deployed_kl"] = float(InnerXQCEngine._gaussian_kl(new_mean, new_logstd, old_mean, old_logstd).mean())
        metrics["one_actor_step_mean_action_l2"] = float((new_mean.tanh()-old_mean.tanh()).norm())
    if tree_hash(bn_buffers(diagnostic.controller.actor)) != actor_before:
        raise RuntimeError("Diagnostic actor step mutated inherited running BN buffers.")
    after = tree_hash({"controller": controller.state_dict(), "workspace": workspace.state_dict(), "rng": rng_state()})
    if before != after: raise RuntimeError("Diagnostic queries mutated critic, optimizer or global RNG.")
    metrics["diagnostics_state_sha256"] = before
    metrics["diagnostics_state_unchanged"] = True
    return metrics, gradients["running"]


def probe_pair(agent, data, *, inspection_steps=INSPECTION_STEPS, actor_lr=5e-5, critic_lr=5e-5):
    if tuple(inspection_steps) not in (INSPECTION_STEPS, (0, 1, 3)):
        raise ValueError("Inspection steps must be production 0/1/3/12 or smoke 0/1/3.")
    if bool(agent.cfg.episodic): raise ValueError("This probe supports the selected non-episodic H1 checkpoint only.")
    outer, selected = agent.xqc_controller, agent.aux_return.critic
    input_hash = tree_hash(data)
    original_hash = tree_hash({"actor": outer.actor.state_dict(), "critic": selected.state_dict(), "rng": rng_state()})
    initial_bn = bn_buffers(selected)
    controllers = {mode: clone_workspace(outer, selected, actor_lr=actor_lr, critic_lr=critic_lr) for mode in MODES}
    fixed_actor_hash = tree_hash(outer.actor.state_dict())
    initial_target_bn_hash = tree_hash(initial_bn)
    records = []
    for step in range(max(inspection_steps)+1):
        if step in inspection_steps:
            snapshot, gradients = {}, {}
            for mode, workspace in controllers.items():
                metrics, gradient = inspect(workspace, outer, data, initial_bn, actor_lr=actor_lr)
                snapshot[mode], gradients[mode] = metrics, gradient
            for mode in MODES:
                snapshot[mode]["running_gradient_vs_batch_update"] = gradient_comparison(gradients[mode], gradients["batch_update"])
                records.append({"mode": mode, "critic_updates": step, **snapshot[mode]})
        if step == max(inspection_steps): break
        for mode, workspace in controllers.items():
            critic_only_update(workspace, outer, selected, data, step, mode)
            if (tree_hash(workspace.controller.actor.state_dict()) != fixed_actor_hash
                    or workspace.actor_optimizer_steps != 0 or workspace.temperature_optimizer_steps != 0
                    or tree_hash(bn_buffers(workspace.controller.critic_target)) != initial_target_bn_hash):
                raise RuntimeError("Critic-only update changed actor, temperature counters or target BN buffers.")
    if original_hash != tree_hash({"actor": outer.actor.state_dict(), "critic": selected.state_dict(), "rng": rng_state()}):
        raise RuntimeError("Probe changed outer networks or global RNG.")
    if tree_hash(data) != input_hash:
        raise RuntimeError("Probe mutated shared paired inputs.")
    for record in records:
        record["paired_inputs_unchanged"] = True
    return records


def choose_mode(records, root_ids, solver_seeds):
    expected = {(root, int(seed), mode) for root in root_ids for seed in solver_seeds for mode in MODES}
    selected_rows = [row for row in records if row["critic_updates"] == 3]
    actual = [(row["root_id"], int(row["solver_seed"]), row["mode"]) for row in selected_rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Selection requires complete paired root/seed/mode results at three critic updates.")
    scores = {}
    for mode in MODES:
        values = [float(row["heldout_running_q_mse"]) for row in selected_rows if row["mode"] == mode]
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError("Selection MSE must be finite and nonnegative.")
        scores[mode] = float(np.mean(values))
    # Only alternatives B/C are eligible; A is the retained control.
    selected = min(("running", "batch_no_update"), key=lambda mode: scores[mode])
    return selected, scores


def validate_selection(path, source_sha=None):
    path = Path(path).resolve()
    saved = source.banks.read_json(path)
    if (saved.get("schema") != SELECTION_SCHEMA or saved.get("mode") != "production"
            or saved.get("checkpoint_sha256") != source.CHECKPOINT_SHA
            or saved.get("root_count") != 32 or saved.get("solver_seeds") != list(SOLVER_SEEDS)
            or saved.get("inspection_steps") != list(INSPECTION_STEPS)
            or saved.get("outer_state_unchanged") is not True or saved.get("global_rng_unchanged") is not True
            or saved.get("rule") != "heldout_running_q_mse_at_3_updates"
            or re.fullmatch(r"[0-9a-f]{40}", str(saved.get("source_sha"))) is None
            or (source_sha is not None and saved.get("source_sha") != source_sha)):
        raise ValueError("Selection does not identify the complete production probe.")
    artifacts = {}
    for name in ("roots", "results", "paired_inputs"):
        artifact = (path.parent / saved[f"{name}_file"]).resolve()
        if path.parent not in artifact.parents or source.file_sha256(artifact) != saved[f"{name}_sha256"]:
            raise ValueError(f"Selection {name} artifact hash differs.")
        artifacts[name] = source.banks.read_json(artifact)
    entries = artifacts["paired_inputs"]["files"]
    for entry in entries:
        artifact = (path.parent / entry["path"]).resolve()
        if path.parent not in artifact.parents or source.file_sha256(artifact) != entry["sha256"]:
            raise ValueError("Paired input artifact is missing or changed.")
    roots = artifacts["roots"]["roots"]
    results = artifacts["results"]
    root_ids = [root["root_id"] for root in roots]
    expected_pairs = {(root, seed) for root in root_ids for seed in SOLVER_SEEDS}
    actual_pairs = [(entry["root_id"], entry["solver_seed"]) for entry in entries]
    if (len(roots) != 32 or len(set(root_ids)) != 32
            or [(root["seed"], root["decision"]) for root in roots] != root_positions()
            or artifacts["roots"].get("checkpoint_sha256") != source.CHECKPOINT_SHA
            or results.get("schema") != SCHEMA
            or results.get("mode") != "production" or results.get("root_count") != 32
            or results.get("inspection_steps") != list(INSPECTION_STEPS)
            or results.get("solver_seeds") != list(SOLVER_SEEDS)
            or results.get("source_sha") != saved["source_sha"]
            or results.get("checkpoint_sha256") != source.CHECKPOINT_SHA
            or results.get("checkpoint_metadata_sha256") != CHECKPOINT_METADATA_SHA
            or results.get("target") != TARGET_KIND
            or results.get("outer_state_unchanged") is not True
            or results.get("global_rng_unchanged") is not True
            or re.fullmatch(r"[0-9a-f]{64}", str(results.get("outer_state_before_sha256"))) is None
            or re.fullmatch(r"[0-9a-f]{64}", str(results.get("global_rng_sha256"))) is None
            or results.get("outer_state_before_sha256") != results.get("outer_state_after_sha256")
            or len(actual_pairs) != len(set(actual_pairs)) or set(actual_pairs) != expected_pairs):
        raise ValueError("Probe results or roots have incompatible provenance.")
    expected_records = {(root, seed, mode, step) for root, seed in expected_pairs
                        for mode in MODES for step in INSPECTION_STEPS}
    actual_records = [(r["root_id"], r["solver_seed"], r["mode"], r["critic_updates"]) for r in results["records"]]
    paired_hashes = {(entry["root_id"], entry["solver_seed"]): entry["tensor_sha256"] for entry in entries}
    if (len(actual_records) != len(set(actual_records)) or set(actual_records) != expected_records
            or any(r.get("diagnostics_state_unchanged") is not True
                   or r.get("paired_inputs_unchanged") is not True
                   or r.get("paired_inputs_tensor_sha256") != paired_hashes[(r["root_id"], r["solver_seed"])]
                   for r in results["records"])):
        raise ValueError("Probe records are incomplete, unpaired or have mutable diagnostics.")
    def finite(value):
        if isinstance(value, dict): return all(finite(v) for v in value.values())
        if isinstance(value, list): return all(finite(v) for v in value)
        return not isinstance(value, float) or math.isfinite(value)
    if not finite(results): raise ValueError("Probe results contain nonfinite diagnostics.")
    selected, scores = choose_mode(results["records"], root_ids, SOLVER_SEEDS)
    if selected != saved.get("selected_critic_bn_mode") or scores != saved.get("aggregate_scores"):
        raise ValueError("Selection does not match independently recomputed scores.")
    return saved


def run(checkpoint, output, *, device="cuda", mode="production", checkpoint_sha=source.CHECKPOINT_SHA):
    from evaluate_ambi_checkpoint import _initialize_frozen_model, _make_env, _outer_state_digest
    from utils.ambi_research import resolve_preset
    from utils.checkpoint_context import load_checkpoint_context
    from utils.ambi_benchmark import atomic_json, code_identity
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    if mode not in ("smoke", "production"): raise ValueError("Mode must be smoke or production.")
    if checkpoint_sha != source.CHECKPOINT_SHA or source.file_sha256(checkpoint) != checkpoint_sha:
        raise ValueError("Probe requires the pinned shared UTD2 475k checkpoint.")
    if source.file_sha256(str(checkpoint)+".metadata.json") != CHECKPOINT_METADATA_SHA:
        raise ValueError("Probe checkpoint metadata differs from the pinned sidecar.")
    code = code_identity()
    if not code.get("commit") or code.get("dirty") is not False: raise ValueError("Probe requires a clean committed checkout.")
    if output == source.ROOT or source.ROOT in output.parents: raise ValueError("Probe outputs must be outside source.")
    output.mkdir(parents=True, exist_ok=False)
    (output / "paired-inputs").mkdir()
    steps = INSPECTION_STEPS if mode == "production" else (0, 1, 3)
    seeds = SOLVER_SEEDS if mode == "production" else SOLVER_SEEDS[:1]
    count = 32 if mode == "production" else 1
    context = load_checkpoint_context(checkpoint)
    resolved = resolve_preset(source.MATRIX, "controller/return_return_j1", checkpoint_context=context)
    params = resolved["algorithm_config"]["alg_params"]
    params.update(compile=False, compile_strict=False, inner_actor_bn_mode="running")
    records, files = [], []
    before_rng = tree_hash(rng_state())
    with preserve_rng():
        env = _make_env(resolved)
        try:
            model, _ = _initialize_frozen_model(resolved, env, checkpoint, 12345, device=device)
            agent = model.agent
            before_outer = _outer_state_digest(model)
            roots, returns = capture_roots(model, count=count, episode_steps=500)
            atomic_json(output / "roots.json", {"checkpoint_sha256": checkpoint_sha,
                "roots": roots, "prior_episode_returns": returns, "capture": "deterministic_persistent_actor_mean"})
            for root in roots:
                with torch.no_grad():
                    obs = np.asarray(root["observation"], dtype=np.float32)
                    z = agent.model.encode(model._obs_to_tensor(obs).to(agent.device).unsqueeze(0))
                for seed in seeds:
                    data = paired_inputs(agent, z, seed, root["root_id"], max_updates=max(steps))
                    relative = f"paired-inputs/{root['root_id']}-solver-{seed}.pt"
                    torch.save(_cpu_tree(data), output / relative)
                    files.append({"path": relative, "sha256": source.file_sha256(output / relative),
                                  "root_id": root["root_id"], "solver_seed": seed, "tensor_sha256": tree_hash(data)})
                    pair = probe_pair(agent, data, inspection_steps=steps)
                    records.extend({"root_id": root["root_id"], "solver_seed": seed,
                                    "paired_inputs_tensor_sha256": files[-1]["tensor_sha256"], **record} for record in pair)
                    print(json.dumps({"root": root["root_id"], "solver_seed": seed, "records_completed": len(records)}), flush=True)
            after_outer = _outer_state_digest(model)
            if before_outer != after_outer: raise RuntimeError("Probe changed complete frozen outer state.")
        finally:
            env.close()
    if tree_hash(rng_state()) != before_rng: raise RuntimeError("Probe advanced global RNG.")
    selected, scores = choose_mode(records, [root["root_id"] for root in roots], seeds)
    result = {"schema": SCHEMA, "mode": mode, "source_sha": code["commit"], "checkpoint_sha256": checkpoint_sha,
              "checkpoint_metadata_sha256": source.file_sha256(str(checkpoint)+".metadata.json"),
              "inspection_steps": list(steps), "solver_seeds": list(seeds), "root_count": count,
              "outer_state_before_sha256": before_outer, "outer_state_after_sha256": after_outer,
              "global_rng_sha256": before_rng, "outer_state_unchanged": True, "global_rng_unchanged": True,
              "metrics_units": "reward_scale_normalized",
              "target": TARGET_KIND,
              "records": records, "aggregate_scores": scores}
    atomic_json(output / "results.json", result)
    atomic_json(output / "paired-inputs.json", {"files": files})
    selection = {"schema": SELECTION_SCHEMA, "mode": mode, "source_sha": code["commit"],
                 "checkpoint_sha256": checkpoint_sha, "root_count": count, "solver_seeds": list(seeds),
                 "inspection_steps": list(steps), "selected_critic_bn_mode": selected,
                 "rule": "heldout_running_q_mse_at_3_updates", "aggregate_scores": scores,
                 "outer_state_unchanged": True, "global_rng_unchanged": True}
    for name in ("roots", "results", "paired_inputs"):
        filename = name.replace("_", "-")+".json"
        selection.update({f"{name}_file": filename, f"{name}_sha256": source.file_sha256(output/filename)})
    atomic_json(output / "selection.json", selection)
    if mode == "production": validate_selection(output / "selection.json", source_sha=code["commit"])
    (output / "PASS").write_text("PASS\n")
    return selection


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=("smoke", "production"), default="production")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.checkpoint, args.output, device=args.device, mode=args.mode), indent=2))


if __name__ == "__main__":
    main()
