"""Mean execution preserves the adapted BOOM policy and stochastic SAC solve."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
from RL.tdmpc2_core.whole_inner_graph import WholeInnerGraph, validate_training_graph
from tests.test_ambi_root_local_sac import _build_cfg
from tests.test_inner_entropy_graph import (
    CUDA, assert_graph_resume_continuation, assert_tree_close, inner_snapshot,
    make_agent, outer_snapshot,
)

ROOT = Path(__file__).resolve().parents[1]


def mean_agent(*, device="cpu", compiled=False):
    cfg = deepcopy(make_agent(update="per_update").cfg)
    cfg.device = device
    cfg.compile = cfg.compile_strict = compiled
    cfg.inner_execution_action = "mean"
    cfg.outer_actor_objective = "boom"
    cfg.outer_boom_coef = .0042
    cfg.store_behavior_policy = True
    agent = AMBITDMPC2Agent(cfg)
    agent.model.eval()
    agent.actor_loss_scale.fill_(4.)
    return agent


def test_recipe_changes_only_execution_and_run_metadata():
    folder = ROOT / "configs/dmcontrol"
    source = json.loads((folder / "algs/ambi_posttanh_prior_outer_adapt_boom_x0p01.json").read_text())
    target = json.loads((folder / "algs/ambi_boom_mean_execution.json").read_text())
    for config in (source, target):
        config["alg_params"] = {k: v for k, v in config["alg_params"].items() if not k.startswith("wandb")}
    assert source["alg_params"]["inner_execution_action"] == "policy_sample"
    source["alg_params"]["inner_execution_action"] = "mean"
    assert target == source
    manifest = json.loads((folder / "experiments/ambi_boom_mean_execution.json").read_text())
    reference = json.loads((folder / "experiments/ambi_posttanh_prior_strength.json").read_text())
    for key in ("overrides_alg", "env_params", "trials", "logs", "save_trials", "checkpoint_every", "save_strat", "log_info", "log_type"):
        assert manifest[key] == reference[key]
    assert manifest["configs"] == ["ambi_boom_mean_execution"]
    cfg = _build_cfg(**target["alg_params"])
    assert cfg.inner_behavior_action == "policy_sample"
    assert cfg.inner_execution_action == "mean" and cfg.store_behavior_policy
    validate_training_graph(SimpleNamespace(cfg=cfg, device=torch.device("cuda"), _active_trace=None, _retrace_enabled=False))


@pytest.mark.parametrize("execution", ["policy_sample", "mean"])
def test_training_graph_accepts_validated_execution_modes(execution):
    agent = mean_agent()
    agent.cfg.inner_execution_action = execution
    agent.cfg.compile = agent.cfg.compile_strict = True
    agent.inner_engine.device = torch.device("cuda")
    validate_training_graph(agent.inner_engine)
    agent.cfg.inner_execution_action = "mean_plus_gaussian"
    with pytest.raises(ValueError, match="sampled or mean"):
        validate_training_graph(agent.inner_engine)


def test_native_mean_action_matches_adapted_policy_and_preserves_labels():
    agent = mean_agent()
    engine = agent.inner_engine
    root = torch.zeros(1, agent.cfg.latent_dim)
    before = outer_snapshot(agent)
    action, metrics, _, labels = engine.act(root, collect_diagnostics=False, return_behavior_policy=True)
    torch.testing.assert_close(action, labels["pre_tanh_mean"].tanh(), rtol=0, atol=0)
    with torch.no_grad():
        expected, info = agent.model.pi(root, policy=engine._action_pool.actor, deterministic=True, **engine._inner_policy_kwargs())
    torch.testing.assert_close(action, expected[0], rtol=0, atol=0)
    torch.testing.assert_close(labels["log_std"], info["log_std"][0], rtol=0, atol=0)
    assert torch.isfinite(labels["log_std"]).all()
    assert metrics["inner_actor_optimizer_steps"] == agent.cfg.inner_actor_updates_per_action
    assert metrics["inner_critic_optimizer_steps"] == agent.cfg.inner_critic_updates_per_action
    assert_tree_close(outer_snapshot(agent), before)


@CUDA
def test_cuda_mean_capture_preserves_labels_and_inner_state():
    native, captured = (mean_agent(device="cuda", compiled=True) for _ in range(2))
    captured.load(deepcopy(native.checkpoint_state()))
    root = torch.zeros(1, native.cfg.latent_dim, device="cuda")
    control = WholeInnerGraph(native.inner_engine, root, capture=False, return_behavior_policy=True)
    graph = WholeInnerGraph(captured.inner_engine, root, return_behavior_policy=True)
    for index in range(3):
        initial = torch.full_like(native.actor_loss_scale, 4. + index)
        before = [outer_snapshot(agent) for agent in (native, captured)]
        expected = control(root + .1 * index, initial_actor_loss_scale=initial)
        actual = graph(root + .1 * index, initial_actor_loss_scale=initial)
        assert_tree_close(actual, expected, rtol=5e-5, atol=5e-7)
        assert_tree_close(inner_snapshot(captured.inner_engine), inner_snapshot(native.inner_engine), rtol=5e-5, atol=5e-7)
        torch.testing.assert_close(actual[0], actual[3]["pre_tanh_mean"].tanh(), rtol=5e-5, atol=5e-7)
        with torch.no_grad():
            action, info = captured.model.pi(root + .1 * index, policy=captured.inner_engine._action_pool.actor,
                                             deterministic=True, **captured.inner_engine._inner_policy_kwargs())
        torch.testing.assert_close(actual[0], action[0], rtol=5e-5, atol=5e-7)
        torch.testing.assert_close(actual[3]["log_std"], info["log_std"][0], rtol=5e-5, atol=5e-7)
        torch.testing.assert_close(graph.rng_state, control.rng_state, rtol=0, atol=0)
        for agent, state in zip((native, captured), before):
            assert_tree_close(outer_snapshot(agent), state)
        with torch.no_grad():
            for left, right in zip(native.model.parameters(), captured.model.parameters()):
                left.add_(.0001)
                right.add_(.0001)


@CUDA
def test_cuda_mean_graph_native_graph_and_checkpoint_continuation():
    agent = mean_agent(device="cuda", compiled=True)
    agent.cfg.inner_cuda_graph = True
    agent.act(torch.zeros(agent.cfg.obs_shape["state"]), collect_diagnostics=False)
    assert_graph_resume_continuation(agent)
