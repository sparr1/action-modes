"""Observe the pinned, released BOOM trainer without changing its updates."""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import time
from unittest.mock import patch

UPSTREAM = "b3b2f09c03533c15b804ced9abc6767e73ff8e1e"


def jsonable(value):
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


class Recorder:
    def __init__(self, output, cfg, online, method="BOOM", upstream_commit=UPSTREAM):
        self.output = output
        self.start = time.perf_counter()
        self.updates = 0
        self.window = defaultdict(list)
        self.crossed = False
        self.agent = None
        self.upstream_commit = upstream_commit
        self.wandb = None
        self.video = None
        self.last_checkpoint = -1
        self.last_step = 0
        self.events = (output / "events.jsonl").open("x", buffering=1)
        self.raw = (output / "updates.jsonl").open("x", buffering=1024 * 1024)
        from omegaconf import OmegaConf
        config = OmegaConf.to_container(cfg, resolve=True)
        config.update(upstream_commit=upstream_commit, threshold=2.0,
                      reference_method=f"released {method}",
                      action_repeat=2, reward_aggregation="sum",
                      observer_commit=subprocess.check_output(
                          ["git", "rev-parse", "HEAD"], text=True).strip())
        (output / "resolved_config.json").write_text(json.dumps(jsonable(config), indent=2))
        if online:
            import wandb
            self.wandb = wandb.init(
                entity=cfg.wandb_entity, project=cfg.wandb_project,
                id=os.environ["WANDB_RUN_ID"], resume="never",
                name=f"{method} released | humanoid-walk | S>2 | {cfg.steps // 1000}k | {cfg.acceleration} | seed{cfg.seed}",
                group="boom-upstream-threshold-reference-20261104",
                tags=["upstream-reference", method, "threshold-diagnostics"],
                config=jsonable(config), dir=str(output))
            self.wandb.define_metric("env_step")
            self.wandb.define_metric("*", step_metric="env_step")
            self.wandb.log({"env_step": 0, "progress/started": 1})

    def publish(self, row):
        row = jsonable(row)
        self.events.write(json.dumps(row) + "\n")
        if self.wandb is not None:
            self.wandb.log(row)

    def checkpoint(self, name, step):
        import torch
        path = self.output / "checkpoints" / f"{name}.pt"
        path.parent.mkdir(exist_ok=True)
        torch.save({"model": self.agent.model.state_dict(),
                    "scale": self.agent.scale.state_dict(),
                    "env_step": step, "upstream_commit": self.upstream_commit}, path)

    def update(self, step, metrics):
        self.updates += 1
        self.last_step = step
        row = {"env_step": step, "update": self.updates, **metrics}
        self.raw.write(json.dumps(jsonable(row)) + "\n")
        for key, value in metrics.items():
            self.window[key].append(float(value))
        if self.updates % 100 == 0:
            self.flush_window(step)
        if step > 0 and step % 25000 == 0 and step != self.last_checkpoint:
            self.checkpoint(f"step-{step:07d}", step)
            self.last_checkpoint = step

    def flush_window(self, step):
        if not self.window:
            return
        row = {"env_step": step, "progress/updates": self.updates,
               "progress/elapsed_seconds": time.perf_counter() - self.start}
        for key, values in self.window.items():
            row[f"diagnostic/{key}"] = sum(values) / len(values)
            row[f"diagnostic/{key}_min"] = min(values)
            row[f"diagnostic/{key}_max"] = max(values)
        self.publish(row)
        self.window.clear()
        self.raw.flush()

    def log(self, metrics, category="train"):
        step = int(metrics.get("step", 0))
        self.publish({"env_step": step,
                      **{f"{category}/{k}": v for k, v in metrics.items()}})
        print(json.dumps(jsonable({"category": category, **metrics})), flush=True)

    def finish(self, agent=None):
        self.flush_window(self.last_step)
        if agent is not None:
            self.checkpoint("final", self.last_step)
        self.raw.close()
        self.events.close()
        if self.wandb is not None:
            self.wandb.finish()


def observed_class(base):
    import torch

    class ObservedBOOM(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.recorder = None
            self.observe = True
            self.in_actor = False
            self.actor_metrics = {}
            original_pi = self.model.pi

            def pi(*a, **kw):
                out = original_pi(*a, **kw)
                if self.observe and self.in_actor:
                    with torch.no_grad():
                        mean, sample, legacy_log_pi, log_std = (x.detach() for x in out)
                        jac = torch.log(torch.relu(1 - sample.square()) + 1e-6).sum(-1, keepdim=True)
                        d = self.cfg.action_dim
                        # Undo upstream's extra Gaussian factor, retaining its floored Jacobian.
                        joint_log_pi = (legacy_log_pi + jac) / d - (d - 1) * 0.5 * math.log(2 * math.pi) - jac
                        self.actor_metrics.update(
                            post_tanh_entropy_floored=float(-joint_log_pi.mean()),
                            upstream_entropy_statistic=float(-legacy_log_pi.mean()),
                            pre_tanh_entropy=float((log_std + 0.5 * math.log(2 * math.pi * math.e)).sum(-1).mean()),
                            action_saturation=float((sample.abs() >= .99).float().mean()),
                            mean_saturation=float((mean.abs() >= .99).float().mean()),
                            actor_std_mean=float(log_std.exp().mean()),
                            actor_std_min=float(log_std.exp().min()))
                return out

            self.model.pi = pi

        def update_pi(self, zs, action, mu, std, task, step=None):
            if step is None:
                step = getattr(self, "reference_step", 0)
            if not self.observe:
                return super().update_pi(zs, action, mu, std, task, step)
            self.actor_metrics = {}
            self.in_actor = True
            clip = torch.nn.utils.clip_grad_norm_
            first_crossing = False

            def observed_clip(*args, **kwargs):
                nonlocal first_crossing
                active = self.scale.value > 2.0
                if active and self.recorder is not None and not self.recorder.crossed:
                    first_crossing = True
                    self.recorder.crossed = True
                    self.recorder.checkpoint("first-active-before-actor", step)
                norm = clip(*args, **kwargs)
                self.actor_metrics["actor_grad_norm"] = float(norm)
                self.actor_metrics["actor_clip_active"] = float(norm > self.cfg.grad_clip_norm)
                return norm

            try:
                with patch.object(torch.nn.utils, "clip_grad_norm_", observed_clip):
                    result = super().update_pi(zs, action, mu, std, task, step)
            finally:
                self.in_actor = False
            if first_crossing:
                self.recorder.checkpoint("first-active-after-actor", step)
                self.recorder.publish({"env_step": step, "event/first_gate_crossing": 1,
                                       "event/crossing_scale": self.scale.value,
                                       "event/crossing_actor_grad_norm": self.actor_metrics["actor_grad_norm"]})
                print(f"FIRST_GATE_CROSSING step={step} scale={self.scale.value}", flush=True)
            return result

        def update(self, replay_sample, step):
            start = time.perf_counter()
            metrics = super().update(replay_sample, step)
            if self.observe and self.recorder is not None:
                self.recorder.update(step, {
                    **metrics, **self.actor_metrics,
                    "gate_active": float(self.scale.value > 2.0),
                    "weighted_alignment_loss": self.cfg.action_dim * self.lamda * metrics["pi_fkl_loss"],
                    "update_seconds": time.perf_counter() - start})
            return metrics

    return ObservedBOOM


def validate_observer(agent, output):
    """Numerical update parity and exact RNG, with a native/native repeat control."""
    import numpy as np
    import torch
    cfg = agent.cfg
    batch = (torch.randn(4, 256, 67, device="cuda"),
             torch.rand(3, 256, 21, device="cuda") * 2 - 1,
             torch.rand(3, 256, 21, device="cuda") * 2 - 1,
             torch.ones(3, 256, 21, device="cuda"),
             torch.rand(3, 256, 1, device="cuda"), None)
    agent.observe = False
    agent.update(batch, 0)  # Compile both paths before comparing.
    rows = []
    for scale in (1., 3.):
        model = deepcopy(agent.model.state_dict())
        grads = {name: None if p.grad is None else p.grad.detach().clone()
                 for name, p in agent.model.named_parameters()}
        optim = deepcopy(agent.optim.state_dict())
        pi_optim = deepcopy(agent.pi_optim.state_dict())
        rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all(), np.random.get_state(), random.getstate())
        results = []
        for enabled in (False, False, True):
            agent.model.load_state_dict(model)
            # Upstream clips all model parameters before clearing the actor's
            # previous gradients. They are therefore part of its update state.
            for name, p in agent.model.named_parameters():
                p.grad = None if grads[name] is None else grads[name].clone()
            agent.optim.load_state_dict(deepcopy(optim))
            agent.pi_optim.load_state_dict(deepcopy(pi_optim))
            agent.scale._value.fill_(scale)
            torch.set_rng_state(rng[0]); torch.cuda.set_rng_state_all(rng[1])
            np.random.set_state(rng[2]); random.setstate(rng[3])
            agent.observe = enabled
            metrics = agent.update(batch, 0)
            results.append((deepcopy(agent.model.state_dict()), deepcopy(agent.optim.state_dict()),
                            deepcopy(agent.pi_optim.state_dict()), metrics,
                            torch.get_rng_state(), torch.cuda.get_rng_state_all(),
                            np.random.get_state(), random.getstate()))

        differences = {}
        def equal(a, b, path="root"):
            if isinstance(a, torch.Tensor):
                if a.is_floating_point():
                    torch.testing.assert_close(a, b, rtol=2e-5, atol=1e-7, msg=path)
                    differences[path] = float((a - b).abs().max())
                else:
                    assert torch.equal(a, b), path
            elif isinstance(a, np.ndarray):
                assert np.array_equal(a, b)
            elif isinstance(a, dict):
                assert a.keys() == b.keys()
                for k in a: equal(a[k], b[k], f"{path}.{k}")
            elif isinstance(a, (list, tuple)):
                assert len(a) == len(b)
                for i, (x, y) in enumerate(zip(a, b)): equal(x, y, f"{path}[{i}]")
            elif isinstance(a, float):
                assert math.isclose(a, b, rel_tol=2e-5, abs_tol=1e-7), (path, a, b)
                differences[path] = abs(a - b)
            else:
                assert a == b, (a, b)
        equal(results[0], results[1])
        native_repeat = max(differences.values(), default=0)
        differences.clear()
        equal(results[1], results[2])
        rows.append({"initial_scale": scale, "numerical_parity": True, "rng_exact": True,
                     "native_repeat_max_absolute_difference": native_repeat,
                     "observed_max_absolute_difference": max(differences.values(), default=0),
                     "rtol": 2e-5, "atol": 1e-7})
    (output / "parity.json").write_text(json.dumps({"upstream": UPSTREAM, "checks": rows}, indent=2))
    print("OBSERVER_PARITY_PASSED", rows, flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mode", choices=("validate", "benchmark", "train"))
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument("--steps", type=int, default=400000)
    parser.add_argument("--acceleration", choices=("native", "planner-graph", "compiled-planner", "model-planner-graph"), default="native")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    assert subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip() == UPSTREAM
    assert not subprocess.check_output(["git", "-C", str(args.upstream), "status", "--porcelain"], text=True).strip()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.update(LAZY_LEGACY_OP="0", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      WANDB_MODE="online" if args.mode == "train" else "disabled")
    sys.path.insert(0, str(args.upstream))
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    assert torch.cuda.is_available()
    from omegaconf import OmegaConf
    from boom.common.parser import parse_cfg
    from boom.common.seed import set_seed
    from boom.common.buffer import Buffer
    from boom.envs import make_env
    from boom.boom_alg import BOOM
    from boom.trainer.online_trainer import OnlineTrainer
    cfg = OmegaConf.load(args.upstream / "boom/config.yaml")
    del cfg["defaults"]
    cfg.task = "humanoid-walk"
    cfg.seed = args.seed
    cfg.exp_name = "released-threshold-reference"
    cfg.disable_wandb = args.mode != "train"
    cfg.wandb_entity = "rwgao_b-brown-university"
    cfg.wandb_project = "ambi"
    with patch("hydra.utils.get_original_cwd", return_value=str(Path.cwd())):
        cfg = parse_cfg(cfg)
    cfg.work_dir = args.output
    cfg.steps = args.steps
    cfg.acceleration = args.acceleration
    if args.mode == "benchmark":
        cfg.steps = 4100
        cfg.eval_episodes = 1
    set_seed(cfg.seed)
    env = make_env(cfg)
    agent = observed_class(BOOM)(cfg)
    if args.mode == "validate":
        validate_observer(agent, args.output)
        if args.acceleration != "native":
            from upstream_reference_acceleration import validate_planner
            validate_planner(agent, "boom", args.output, args.acceleration)
        if args.acceleration == "model-planner-graph":
            from upstream_reference_acceleration import validate_model_update
            validate_model_update(agent, "boom", args.output)
        return
    from upstream_reference_acceleration import install_planner
    install_planner(agent, "boom", args.acceleration)
    recorder = Recorder(args.output, cfg, args.mode == "train")
    agent.recorder = recorder
    recorder.agent = agent
    trainer = OnlineTrainer(cfg=cfg, env=env, agent=agent, buffer=Buffer(cfg), logger=recorder)
    benchmark = []
    affinity = sorted(os.sched_getaffinity(0))
    original_update = agent.update
    wall = cpu = None
    window_start = None
    if args.mode == "benchmark":
        assert len(affinity) >= 2
        def measured_update(batch, step):
            nonlocal wall, cpu, window_start
            if step in (2700, 3000, 3300, 3600, 3900):
                now, usage = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
                consumed = usage.ru_utime + usage.ru_stime
                if wall is not None:
                    benchmark.append({"cpus": len(os.sched_getaffinity(0)), "decisions": step - window_start,
                                      "seconds": now - wall, "cpu_seconds": consumed - cpu})
                    (args.output / "benchmark.json").write_text(json.dumps(benchmark, indent=2))
                n = 1 if step in (2700, 3300, 3900) else 2
                os.sched_setaffinity(0, affinity[:n])
                wall, cpu, window_start = now, consumed, step
            return original_update(batch, step)
        agent.update = measured_update
    trainer.train()
    if args.mode == "benchmark":
        os.sched_setaffinity(0, affinity)
        print("BENCHMARK", json.dumps(benchmark), flush=True)


if __name__ == "__main__":
    main()
