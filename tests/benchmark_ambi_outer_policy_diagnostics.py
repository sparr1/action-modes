"""Bounded CPU microbenchmark; synthetic updates only, no environment rollouts.

Run from the repository root with the locked interpreter. This intentionally
uses a reduced network and does not establish GPU or production throughput.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import platform
import statistics
import tempfile
import time

import torch

from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
from tests.test_ambi_config_decoupling import _build_cfg
from utils.outer_policy_diagnostics import OuterPolicyDiagnostics


def make_agent(enabled):
    cfg = _build_cfg(
        compile=False, wandb=False, discount=0.99, ent_coef="auto_1.0",
        target_entropy=-21., train_unroll_horizon=3,
        outer_critic_target="entropy_augmented", outer_q_actor_reduction="mean_pair",
        outer_q_target_reduction="min_pair", outer_actor_entropy_mode="squashed",
        log_std_mapping="direct_clamp", log_std_min=-10, log_std_max=2,
        inner_operator="none", dropout=0.01,
    )
    cfg.enc_dim = cfg.mlp_dim = cfg.latent_dim = 64
    cfg.obs_shape = {"state": (67,)}
    cfg.action_dim = 21
    cfg.batch_size = 256
    cfg.device = "cpu"
    cfg.seed_steps = 2500
    cfg.outer_policy_diagnostics = enabled
    cfg.outer_policy_diagnostics_early_every = 100
    cfg.outer_policy_diagnostics_early_until = 10000
    cfg.outer_policy_diagnostics_every = 1000
    cfg.outer_policy_diagnostics_states = 32
    cfg.outer_policy_diagnostics_samples = 32
    cfg.outer_policy_diagnostics_seed = 12345
    return AMBITDMPC2Agent(cfg)


def benchmark(updates=300, repeats=5, warmup=20):
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(811)
    data = (
        torch.randn(4, 256, 67, generator=generator),
        torch.randn(3, 256, 21, generator=generator).tanh(),
        torch.randn(3, 256, 1, generator=generator),
        torch.zeros(3, 256, 1),
    )
    bank = torch.randn(32, 67, generator=generator)
    torch.manual_seed(55)
    reference = make_agent(False)
    for _ in range(warmup):
        reference._update(*data)
    initial = deepcopy(reference.state_dict())
    optimizers = {name: deepcopy(getattr(reference, name).state_dict()) for name in ("optim", "pi_optim", "ent_coef_optim")}
    conditions = ("off", "learner", "learner_and_fixed_bank")
    rows = []
    expected_final = None
    with tempfile.TemporaryDirectory(prefix="ambi-outer-diag-benchmark-") as directory:
        for repetition in range(repeats):
            # Rotate execution order to reduce systematic warmup/thermal bias.
            order = conditions[repetition % 3:] + conditions[:repetition % 3]
            for condition in order:
                agent = make_agent(condition != "off")
                agent.load_state_dict(initial)
                for name, state in optimizers.items():
                    getattr(agent, name).load_state_dict(deepcopy(state))
                recorder = None
                if condition != "off":
                    recorder = OuterPolicyDiagnostics(agent.cfg, Path(directory) / f"{repetition}-{condition}")
                    recorder.bank = {index: bank[slot].clone() for slot, index in enumerate(recorder.indices)}
                torch.manual_seed(991)
                before_timing = dict(recorder.timing) if recorder else {}
                started = time.perf_counter()
                for update in range(1, updates + 1):
                    agent._update(*data)
                    packet = agent.drain_outer_policy_diagnostics()
                    if packet is not None:
                        recorder.learner(packet, env_step=2501, updates=update, phase="microbenchmark", run=None)
                        if condition == "learner_and_fixed_bank":
                            recorder.probe(agent, env_step=2501, updates=update, phase="microbenchmark", run=None)
                seconds = time.perf_counter() - started
                # Probe observability remains checked outside the timed region.
                if expected_final is None:
                    expected_final = deepcopy(agent.state_dict())
                else:
                    for key, expected in expected_final.items():
                        torch.testing.assert_close(agent.state_dict()[key], expected, rtol=0, atol=0, equal_nan=True)
                row = dict(repetition=repetition, condition=condition, seconds=seconds, updates=updates,
                           events=len(recorder.rows) if recorder else 0,
                           diagnostic_timing={key: value - before_timing[key] for key, value in recorder.timing.items()} if recorder else {})
                rows.append(row)
                print(json.dumps(row), flush=True)
    summaries = {}
    for condition in conditions:
        values = [row["seconds"] for row in rows if row["condition"] == condition]
        summaries[condition] = dict(mean_seconds=statistics.mean(values), sample_std_seconds=statistics.stdev(values) if len(values) > 1 else 0., min_seconds=min(values), max_seconds=max(values))
    comparisons = {}
    for condition in conditions[1:]:
        paired = []
        for repetition in range(repeats):
            baseline = next(row["seconds"] for row in rows if row["condition"] == "off" and row["repetition"] == repetition)
            measured = next(row["seconds"] for row in rows if row["condition"] == condition and row["repetition"] == repetition)
            paired.append(100 * (measured / baseline - 1))
        comparisons[condition] = dict(paired_overhead_percent=paired, mean_percent=statistics.mean(paired), sample_std_percent=statistics.stdev(paired) if len(paired) > 1 else 0.)
    return dict(platform=platform.platform(), torch_version=torch.__version__, cpu_threads=torch.get_num_threads(),
                cuda_available=torch.cuda.is_available(), fixture=dict(obs_dim=67, action_dim=21, latent_dim=64, mlp_dim=64, enc_dim=64, num_q=5, q_bins=101, batch_size=256, horizon=3, dropout=.01, updates=updates, repeats=repeats, warmup_updates=warmup, cadence=100, fixed_bank_states=32, fixed_bank_samples=32, compile=False),
                rows=rows, summaries=summaries, comparisons=comparisons,
                limitations="Synthetic repeated batch, reduced network, one CPU thread; includes local event serialization but no environment interaction, replay sampling, compilation, W&B or production GPU execution. Initialization and setup excluded. Final parameters verified exactly identical across conditions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.updates < 100 or args.repeats < 1 or args.warmup < 0:
        parser.error("Require updates >= 100, repeats >= 1, and warmup >= 0.")
    result = benchmark(args.updates, args.repeats, args.warmup)
    output = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output)
