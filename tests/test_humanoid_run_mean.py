"""Mean execution preserves stochastic adaptation and the sampled counterpart."""
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
SOURCE_STEM = "ambi_boom_x0p01_humanoid_run"
BOOM = True


def mean_agent(*, device="cpu", compiled=False):
    cfg = deepcopy(make_agent(update="per_update").cfg)
    cfg.device = device
    cfg.compile = cfg.compile_strict = compiled
    cfg.inner_execution_action = "mean"
    if BOOM:
        cfg.outer_actor_objective = "boom"
        cfg.outer_boom_coef = .0042
        cfg.store_behavior_policy = True
    agent = AMBITDMPC2Agent(cfg)
    agent.model.eval()
    agent.actor_loss_scale.fill_(4.)
    return agent


def test_recipe_changes_only_execution_and_publication_metadata():
    folder = ROOT / "configs/dmcontrol"
    source = json.loads((folder / "algs" / (SOURCE_STEM + ".json")).read_text())
    target = json.loads((folder / "algs" / (SOURCE_STEM + "_mean.json")).read_text())
    for config in (source, target):
        config["alg_params"] = {k: v for k, v in config["alg_params"].items() if not k.startswith("wandb")}
    assert source["alg_params"]["inner_execution_action"] == "policy_sample"
    source["alg_params"]["inner_execution_action"] = "mean"
    assert target == source
    manifest = json.loads((folder / "experiments" / (SOURCE_STEM + "_mean.json")).read_text())
    reference = json.loads((folder / "experiments" / (SOURCE_STEM + ".json")).read_text())
    for key in ("overrides_alg", "env_params", "trials", "logs", "save_trials",
                "checkpoint_every", "save_strat", "log_info", "log_type"):
        assert manifest[key] == reference[key]
    assert manifest["configs"] == [SOURCE_STEM + "_mean"]
    assert manifest["env_params"] == {"task": "humanoid-run", "obs": "state", "render_mode": None}
    cfg = _build_cfg(**target["alg_params"])
    assert cfg.inner_behavior_action == "policy_sample" and cfg.inner_execution_action == "mean"
    assert cfg.critic_geometry_diagnostics and cfg.critic_geometry_every == 1000
    assert (cfg.inner_rounds, cfg.inner_rollout_horizon, cfg.inner_solve_interval) == (8, 3, 1)
    assert cfg.inner_actor_loss_scale_scope == "action" and cfg.inner_actor_loss_scale_update == "per_update"
    if BOOM:
        assert cfg.outer_boom_coef == .0042 and cfg.store_behavior_policy
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


def test_native_mean_action_matches_adapted_policy():
    agent = mean_agent()
    engine = agent.inner_engine
    root = torch.zeros(1, agent.cfg.latent_dim)
    before = outer_snapshot(agent)
    result = engine.act(root, collect_diagnostics=False,
                        **({"return_behavior_policy": True} if BOOM else {}))
    action, metrics = result[:2]
    with torch.no_grad():
        expected, info = agent.model.pi(root, policy=engine._action_pool.actor,
                                        deterministic=True, **engine._inner_policy_kwargs())
    torch.testing.assert_close(action, expected[0], rtol=0, atol=0)
    if BOOM:
        torch.testing.assert_close(action, result[3]["pre_tanh_mean"].tanh(), rtol=0, atol=0)
        torch.testing.assert_close(result[3]["log_std"], info["log_std"][0], rtol=0, atol=0)
    assert metrics["inner_actor_optimizer_steps"] == agent.cfg.inner_actor_updates_per_action
    assert metrics["inner_critic_optimizer_steps"] == agent.cfg.inner_critic_updates_per_action
    assert_tree_close(outer_snapshot(agent), before)


@CUDA
def test_cuda_mean_capture_preserves_inner_state():
    native, captured = (mean_agent(device="cuda", compiled=True) for _ in range(2))
    captured.load(deepcopy(native.checkpoint_state()))
    root = torch.zeros(1, native.cfg.latent_dim, device="cuda")
    kwargs = {"return_behavior_policy": True} if BOOM else {}
    control = WholeInnerGraph(native.inner_engine, root, capture=False, **kwargs)
    graph = WholeInnerGraph(captured.inner_engine, root, **kwargs)
    for index in range(3):
        initial = torch.full_like(native.actor_loss_scale, 4. + index)
        before = [outer_snapshot(agent) for agent in (native, captured)]
        expected = control(root + .1 * index, initial_actor_loss_scale=initial)
        actual = graph(root + .1 * index, initial_actor_loss_scale=initial)
        assert_tree_close(actual, expected, rtol=5e-5, atol=5e-7)
        assert_tree_close(inner_snapshot(captured.inner_engine), inner_snapshot(native.inner_engine), rtol=5e-5, atol=5e-7)
        with torch.no_grad():
            action, info = captured.model.pi(root + .1 * index, policy=captured.inner_engine._action_pool.actor,
                                             deterministic=True, **captured.inner_engine._inner_policy_kwargs())
        torch.testing.assert_close(actual[0], action[0], rtol=5e-5, atol=5e-7)
        if BOOM:
            torch.testing.assert_close(actual[0], actual[3]["pre_tanh_mean"].tanh(), rtol=5e-5, atol=5e-7)
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
