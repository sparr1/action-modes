"""Production graph identity, protocol guards, and scheduled resume validation."""

from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from RL.tdmpc2_core.whole_inner_graph import (
    preflight_training_graph_state, training_graph_state, validate_training_graph,
    training_graph_act,
)
from tests.benchmark_whole_inner_graph import assert_tree_close, make_agent
from utils.resume_identity import scientific_trial_parameters

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "configs/dmcontrol/algs/ambi_full_h3_j10_k1_reward_only_graph.json"


def test_k1_recipe_preserves_linked_run_learning_settings_and_user_budget():
    new = json.loads(RECIPE.read_text())
    old = json.loads((RECIPE.parent / "ambi_full_h3_j10_k3_reward_only.json").read_text())
    for key, value in old["alg_params"].items():
        if key.startswith("wandb") or key in {"inner_solve_interval", "buffer_size", "inner_log_std_mapping"}:
            continue
        assert new["alg_params"][key] == value, key
    assert (new["seed"], new["total_steps"], new["checkpoint_every"], new["save_strat"]) == (55, 1000000, 25000, "all")
    assert new["alg_params"]["buffer_size"] == 500000
    assert new["alg_params"]["inner_cuda_graph"] is True
    assert new["alg_params"]["inner_solve_interval"] == 1
    from slurm.ambi_reward_training import GRAPH_MANIFEST, validate_campaign, smoke_recipe
    validate_campaign(GRAPH_MANIFEST)
    _, _, smoke, _ = smoke_recipe(12, 120, GRAPH_MANIFEST)
    assert smoke["alg_params"]["wandb"] is False
    assert smoke["alg_params"]["wandb_mode"] == "disabled"


def test_graph_mode_is_scientific_identity_but_disabled_default_is_compatible():
    native = {"alg": "AMBITDMPC2/AMBITDMPC2", "alg_params": {}}
    disabled = {**native, "alg_params": {"inner_cuda_graph": False}}
    graph = {**native, "alg_params": {"inner_cuda_graph": True}}
    assert scientific_trial_parameters(native) == scientific_trial_parameters(disabled)
    assert scientific_trial_parameters(native) != scientific_trial_parameters(graph)


@pytest.fixture
def graph_engine():
    agent = make_agent(device="cpu", tiny=True, compile_enabled=False, config=RECIPE)
    engine = agent.inner_engine
    engine.cfg.inner_cuda_graph = True
    engine.cfg.compile = engine.cfg.compile_strict = True
    engine.device = torch.device("cuda")
    return engine


@pytest.mark.parametrize("key,value", [
    ("inner_solve_interval", 2), ("inner_hold_encoder", "current"),
    ("obs", "rgb"), ("aux_return_mode", "sac"),
    ("outer_behavior_policy_kl_schedule", "fixed"), ("compile_strict", False),
    ("inner_execution_action", "policy_mean"),
])
def test_training_protocol_rejects_unvalidated_modes(graph_engine, key, value):
    setattr(graph_engine.cfg, key, value)
    with pytest.raises(ValueError, match=key):
        validate_training_graph(graph_engine)


def test_graph_checkpoint_requires_complete_compatible_rng_state(graph_engine):
    saved = training_graph_state(graph_engine)
    assert saved["calls"] == 0 and saved["rng_state"] is None
    assert preflight_training_graph_state(graph_engine, saved) == saved
    for bad in (None, {}, {**saved, "calls": 2}, {**saved, "seed": 1},
                {**saved, "protocol": "native"}, {**saved, "rng_state": torch.zeros(2)}):
        with pytest.raises(ValueError):
            preflight_training_graph_state(graph_engine, bad)
    graph_engine.cfg.inner_cuda_graph = False
    with pytest.raises(ValueError, match="native"):
        preflight_training_graph_state(graph_engine, saved)
    assert preflight_training_graph_state(graph_engine, None) is None


def test_graph_action_records_prior_version_for_resume_boundary(graph_engine):
    graph_engine._whole_graph_runner = lambda root: (root, {}, [])
    graph_engine._timer_start = lambda: None
    graph_engine._timer_stop = lambda *args: None
    training_graph_act(graph_engine, torch.zeros(1, 8))
    assert graph_engine.action_index == 1
    assert graph_engine.state.outer_version == graph_engine.agent.outer_version
    graph_engine.prepare_training_resume_boundary()


def assert_graph_resume_continuation(agent):
    """GPU smoke: a recaptured process must reproduce graph/native/graph actions.

    Called after real training has populated the graph, before portable model
    loading intentionally clears run state. Also exercises the original sparse
    diagnostics, an episode reset, a changed prior, and exact RNG restoration.
    """
    engine = agent.inner_engine
    assert engine._whole_graph_runner is not None and engine._whole_graph_runner.calls > 0
    agent.prepare_training_resume_boundary()
    saved = deepcopy(agent.training_state_dict())
    original_runner = engine._whole_graph_runner
    outputs = []
    for attempt in range(2):
        if attempt:
            agent.load_training_state_dict(saved)
            assert agent.inner_engine._whole_graph_runner is None
        agent.reset()
        outer_cpu = torch.random.get_rng_state().clone()
        outer_cuda = torch.cuda.get_rng_state().clone()
        sequence = []
        # Scheduled native diagnostics must not invalidate graph storage.
        interval = agent.cfg.inner_solve_interval
        diagnostics = (False, True, False) if interval == 1 else (False, False, False, True, False, False, False)
        for index, diagnostic in enumerate(diagnostics):
            obs = torch.full(agent.cfg.obs_shape["state"], 0.1 * index)
            action = agent.act(obs, t0=index == 0, collect_diagnostics=diagnostic)
            metrics = deepcopy(agent.last_inner_metrics)
            assert all(math.isfinite(float(value)) for value in metrics.values())
            solve = index % interval == 0
            assert metrics["inner_cuda_graph_used"] == float(solve and not diagnostic)
            assert metrics["inner_critic_optimizer_steps"] == (160 if solve else 0)
            assert metrics["inner_actor_optimizer_steps"] == (40 if solve else 0)
            if interval > 1:
                assert metrics["inner_action_age"] == index % interval
                assert metrics["inner_solve_performed"] == float(solve)
            if diagnostic:
                assert metrics["inner_diagnostics_sampled"] == 1
            sequence.append((action.clone(), {key: value for key, value in metrics.items()
                                             if not key.endswith("seconds")}))
        torch.testing.assert_close(torch.random.get_rng_state(), outer_cpu, rtol=0, atol=0)
        torch.testing.assert_close(torch.cuda.get_rng_state(), outer_cuda, rtol=0, atol=0)
        agent.prepare_training_resume_boundary()
        outputs.append((sequence, deepcopy(agent.training_state_dict())))
    assert agent.inner_engine._whole_graph_runner is not original_runner
    assert_tree_close(outputs[0], outputs[1], rtol=0, atol=0)
    # Return exactly to the pre-test boundary; this helper adds no training.
    agent.load_training_state_dict(saved)
