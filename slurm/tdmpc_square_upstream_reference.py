"""Released TD-M(PC)^2 with observational threshold diagnostics."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from unittest.mock import patch

from boom_upstream_reference import Recorder, observed_class, validate_observer

UPSTREAM = "d1c2632c36effd2f7b661bfe5f822a3db8054d40"


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mode", choices=("validate", "benchmark", "train"))
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--acceleration", choices=("native", "planner-graph", "compiled-planner"), default="native")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    assert subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip() == UPSTREAM
    assert not subprocess.check_output(["git", "-C", str(args.upstream), "status", "--porcelain"], text=True).strip()
    os.environ.update(LAZY_LEGACY_OP="0", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      WANDB_MODE="online" if args.mode == "train" else "disabled")
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, str(args.upstream / "tdmpc_square"))
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    # The released TD-M(PC)^2 entrypoint leaves matmul TF32 at Torch's default.
    torch.backends.cudnn.benchmark = True
    assert torch.cuda.is_available()
    from omegaconf import OmegaConf
    from tdmpc_square.common.parser import parse_cfg
    from tdmpc_square.common.seed import set_seed
    from tdmpc_square.common.buffer import Buffer
    from tdmpc_square.envs import make_env
    from tdmpc_square.tdmpc_square import TDMPC2
    from tdmpc_square.trainer.online_trainer import OnlineTrainer
    cfg = OmegaConf.load(args.upstream / "tdmpc_square/tdmpc_square/config.yaml")
    del cfg["defaults"]
    cfg.task = "humanoid-walk"
    cfg.seed = args.seed
    cfg.exp_name = "released-threshold-reference"
    cfg.disable_wandb = args.mode != "train"
    cfg.save_video = False
    cfg.wandb_entity = "rwgao_b-brown-university"
    cfg.wandb_project = "ambi"
    with patch("hydra.utils.get_original_cwd", return_value=str(Path.cwd())):
        cfg = parse_cfg(cfg)
    cfg.work_dir = args.output
    cfg.steps = args.steps
    cfg.acceleration = args.acceleration
    assert cfg.actor_mode == "residual" and cfg.scale_threshold == 2.0 and cfg.prior_coef == 1.0
    if args.mode == "benchmark":
        cfg.steps = 4100
        cfg.eval_episodes = 1
    set_seed(cfg.seed)
    env = make_env(cfg)

    class InterfaceAdapter(TDMPC2):
        """Only adapt call signatures/metric names; invoke native implementations."""
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.reference_step = 0
            self.lamda = self.cfg.prior_coef / 61

        def update_pi(self, zs, action, mu, std, task, step):
            return super().update_pi(zs, action, mu, std, task)

        def update(self, buffer, step):
            self.reference_step = step
            if isinstance(buffer, tuple):
                batch = buffer
                class FixedBuffer:
                    def sample(self): return batch
                buffer = FixedBuffer()
            metrics = super().update(buffer)
            return {**metrics, "pi_q_loss": metrics["pi_loss_q"],
                    "pi_fkl_loss": metrics["pi_loss_prior"]}

    agent = observed_class(InterfaceAdapter)(cfg)
    if args.mode == "validate":
        validate_observer(agent, args.output)
        if args.acceleration != "native":
            from upstream_reference_acceleration import validate_planner
            validate_planner(agent, "tdmpc_square", args.output, args.acceleration)
        p = args.output / "parity.json"
        receipt = json.loads(p.read_text()); receipt["upstream"] = UPSTREAM
        p.write_text(json.dumps(receipt, indent=2))
        return
    from upstream_reference_acceleration import install_planner
    install_planner(agent, "tdmpc_square", args.acceleration)
    recorder = Recorder(args.output, cfg, args.mode == "train",
                        method="TD-M(PC)^2", upstream_commit=UPSTREAM)
    recorder.agent = agent
    agent.recorder = recorder
    trainer = OnlineTrainer(cfg=cfg, env=env, agent=agent, buffer=Buffer(cfg), logger=recorder)
    native_update = agent.update
    affinity = sorted(os.sched_getaffinity(0))
    benchmark = []
    wall = cpu = window_start = None

    def update(buffer):
        nonlocal wall, cpu, window_start
        step = trainer._step
        if args.mode == "benchmark" and step in (2700, 3000, 3300, 3600, 3900):
            assert len(affinity) >= 2
            now, usage = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
            consumed = usage.ru_utime + usage.ru_stime
            if wall is not None:
                benchmark.append({"cpus": len(os.sched_getaffinity(0)), "decisions": step - window_start,
                                  "seconds": now - wall, "cpu_seconds": consumed - cpu})
                (args.output / "benchmark.json").write_text(json.dumps(benchmark, indent=2))
            n = 1 if step in (2700, 3300, 3900) else 2
            os.sched_setaffinity(0, affinity[:n])
            wall, cpu, window_start = now, consumed, step
        return native_update(buffer, step)

    agent.update = update
    trainer.train()
    if args.mode == "benchmark":
        os.sched_setaffinity(0, affinity)
        print("BENCHMARK", json.dumps(benchmark), flush=True)


if __name__ == "__main__":
    main()
