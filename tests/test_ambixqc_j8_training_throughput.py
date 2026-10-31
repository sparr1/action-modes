"""The J8 timing must execute the requested budget and include outer learning."""

import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).parent / "benchmarks/ambixqc_j8_training_throughput.py"
SPEC = importlib.util.spec_from_file_location("j8_timing", PATH)
timing = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(timing)


def test_recipe_resolves_requested_j8_counts():
    from RL.AMBIXQC import AMBIXQC
    import gymnasium as gym

    run, manifest = timing.recipe("j8", True, 20, 96)
    wrapper = object.__new__(AMBIXQC)
    wrapper.env = gym.make("Pendulum-v1")
    wrapper.run_params = {**run, "device": "cpu"}
    wrapper.experiment_params = manifest
    wrapper.custom_params = run["alg_params"]
    try:
        cfg = wrapper._build_cfg({**run["alg_params"], "device": "cpu"})
    finally:
        wrapper.env.close()
    assert (cfg.inner_critic_updates_per_action, cfg.inner_actor_updates_per_action,
            cfg.inner_temperature_updates_per_action) == (128, 32, 32)
    assert cfg.inner_model_step_budget == 3072
    assert cfg.inner_replay_capacity == 3072
    assert cfg.utd == cfg.xqc_utd == 1
    assert cfg.aux_return_mode == "off"
    assert not cfg.wandb and run["checkpoint_every"] is None


def test_window_times_through_final_outer_update():
    clock = iter([10., 12., 15., 19.])
    cpu = iter([1., 7.])
    syncs = []
    window = timing.TrainingWindow(502, 3, lambda: syncs.append(True),
                                  wall=lambda: next(clock), cpu=lambda: next(cpu))
    window.before_action(502, 2)
    window.after_decision(502, 3, .2)
    window.after_decision(503, 4, .3)
    window.after_decision(504, 5, .4)
    assert window.result["wall_seconds"] == 9.
    assert window.result["outer_updates"] == 3
    assert window.result["decisions_per_second"] == pytest.approx(1 / 3)
    assert len(syncs) == 2


def test_window_rejects_missing_outer_update():
    clock = iter([0., 1.])
    window = timing.TrainingWindow(1, 1, lambda: None, wall=lambda: next(clock), cpu=lambda: 0)
    window.before_action(1, 0)
    with pytest.raises(RuntimeError, match="outer update"):
        window.after_decision(1, 0, .1)


@pytest.mark.parametrize("warmup,measured", [(20, 108), (21, 107)])
def test_k3_rejects_partial_timing_blocks(warmup, measured):
    with pytest.raises(ValueError, match="complete solve blocks"):
        timing.recipe("j8", True, warmup, measured, 3)


@pytest.fixture
def held_model():
    import torch
    from test_ambixqc_prior_checkpoint import _wrapper

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    model = _wrapper(inner_operator="xqc")
    adapter = timing.HeldPolicyBenchmark(model.agent, 3)
    yield model, adapter
    model.flush_checkpoints()
    model.env.close()
    torch.set_num_threads(threads)


def test_held_policy_uses_new_observation_and_frozen_encoder_actor(held_model):
    from copy import deepcopy
    import torch
    from test_ambixqc_core import _batch, _tree_equal

    model, held = held_model
    agent, engine = model.agent, model.agent.inner_engine
    first = agent.act(torch.zeros(3), eval_mode=True, collect_diagnostics=False)
    encoder_state = deepcopy(held.encoder.state_dict())
    actor_state = deepcopy(held.actor.state_dict())
    workspace_state = deepcopy(engine._workspace_pool.state_dict())
    agent._update(*_batch(agent))
    assert not _tree_equal(encoder_state, agent.model._encoder["state"].state_dict())
    outer = agent.frozen_outer_state()
    random_state = torch.get_rng_state().clone()
    inner_rng = engine.rng.training_state_dict()
    observation = torch.tensor([2., -3., 5.])
    with torch.no_grad():
        expected, _ = held.actor.sample(held.encoder(observation.unsqueeze(0)),
                                        deterministic=True, bn_mode="running")
    action = agent.act(observation, eval_mode=True, collect_diagnostics=True)
    torch.testing.assert_close(action, expected[0], rtol=0, atol=0)
    assert not torch.equal(action, first)
    assert _tree_equal(outer, agent.frozen_outer_state())
    assert _tree_equal(encoder_state, held.encoder.state_dict())
    assert _tree_equal(actor_state, held.actor.state_dict())
    assert _tree_equal(workspace_state, engine._workspace_pool.state_dict())
    assert torch.equal(random_state, torch.get_rng_state())
    after_rng = engine.rng.training_state_dict()
    for key, before in inner_rng["streams"].items():
        assert torch.equal(before, after_rng["streams"][key]) == (key != "execution")
    assert _tree_equal(inner_rng["phase_streams"], after_rng["phase_streams"])
    assert agent.last_inner_metrics["inner_policy_held"] == 1
    assert all(agent.last_inner_metrics[k] == 0 for k in timing.WORK_KEYS)
    assert agent.last_inner_rollout_lengths == []
    assert engine.action_index == 1


def test_k3_real_training_keeps_outer_utd_and_solves_once_per_block(held_model):
    model, held = held_model
    rows = []
    original = model.agent.act

    def observe(*args, **kwargs):
        action = original(*args, **kwargs)
        rows.append(dict(model.agent.last_inner_metrics))
        return action

    model.agent.act = observe
    model.learn(10)
    assert model._global_step == model.agent.reward_normalizer.count == 10
    assert model.agent.num_updates == model._num_updates == model.agent.xqc_workspace.update_step == 6
    assert model.agent.inner_engine.action_index == 2
    assert [r["inner_action_age"] for r in rows] == [0, 1, 2, 0, 1]
    assert [r["inner_solve_performed"] for r in rows] == [1, 0, 0, 1, 0]
    assert [r["inner_model_steps"] for r in rows] == [4, 0, 0, 4, 0]


def test_held_reset_mode_change_and_failure_restart_solve(held_model):
    import torch

    model, held = held_model
    agent = model.agent
    obs = torch.zeros(3)
    agent.act(obs, collect_diagnostics=False)
    agent.act(obs, collect_diagnostics=False)
    agent.reset()
    agent.act(obs, collect_diagnostics=False)
    assert agent.last_inner_metrics["inner_action_age"] == 0
    agent.act(obs, eval_mode=True, collect_diagnostics=False)
    assert agent.last_inner_metrics["inner_action_age"] == 0
    agent.act(obs, t0=True, eval_mode=True, collect_diagnostics=False)
    assert agent.last_inner_metrics["inner_action_age"] == 0
    with pytest.raises(RuntimeError):
        agent.act(torch.zeros(99), eval_mode=True)
    assert not held.ready and held.index == 0
    agent.act(obs, eval_mode=True, collect_diagnostics=False)
    assert agent.last_inner_metrics["inner_action_age"] == 0
