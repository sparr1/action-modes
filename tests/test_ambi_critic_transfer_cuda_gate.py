"""Opt-in CUDA lifecycle gate, complementary to real-checkpoint campaign smokes.

The fixture uses five distributional Q heads, both selected critic semantics,
H3/J1/C2/A2 and two seven-decision episodes. It checks exact solve boundaries
on CUDA while comparing eager and strict Inductor execution. Production model
shapes, the 575K weights, and J1/J8 campaign cells are checked by campaign smokes.
"""

import gc
import json
import math
import os
from pathlib import Path
import time

import pytest
import torch

from tests.test_actor_transfer_solve_cadence import learner_snapshot
from tests.test_aux_actor_transfer import _trace
from tests.test_aux_critic_transfer import (
    assert_optimizer_reset,
    component_ids,
    critic_params,
)
from tests.test_ambi_inner_decoupling import _assert_tree_equal, _clone_tree
from tests.test_ambi_root_local_sac import _model_from_params


def _global_rng(device):
    state = {"cpu": torch.random.get_rng_state().clone()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device).clone()
    return state


def _instrument_boundaries(model):
    engine = model.agent.inner_engine
    prepare = engine._prepare_workspace
    observed = {"previous": None, "solves": 0, "allocations": []}

    def checked_prepare(*, t0):
        prepare(t0=t0)
        state = engine.state
        if t0:
            observed["solves"] = 0
        expected = engine._critic_base.state_dict() if t0 else observed["previous"]
        assert expected is not None
        _assert_tree_equal(state.critic.state_dict(), expected)
        _assert_tree_equal(state.critic_target.state_dict(), expected)
        _assert_tree_equal(state.actor.state_dict(), engine._actor_base.state_dict())
        assert state.critic_lifetime_steps == observed["solves"] * 2
        assert state.actor_lifetime_steps == state.temperature_lifetime_steps == 0
        assert state.critic_steps == state.actor_steps == state.temperature_steps == 0
        assert state.target_steps == state.critic_target_steps == 0
        assert state.replay.size == 0
        torch.testing.assert_close(engine.alpha, engine._initial_inner_alpha(), rtol=0, atol=0)
        assert_optimizer_reset(state.actor_optim, state.actor.parameters())
        assert_optimizer_reset(state.critic_optim, state.critic.parameters())
        assert_optimizer_reset(state.temperature_optim, [state.log_alpha])
        observed["allocations"].append(component_ids(state))
        observed["solves"] += 1

    engine._prepare_workspace = checked_prepare
    return observed


def _assert_close_modules(left, right):
    for name in ("actor", "critic", "critic_target"):
        a = getattr(left.state, name) or getattr(left._action_pool, name)
        b = getattr(right.state, name) or getattr(right._action_pool, name)
        for parameter_a, parameter_b in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(parameter_a, parameter_b, atol=2e-5, rtol=2e-4)


def _run_fixture_gate(critic, interval, *, device="cuda", compile_enabled=True):
    started = time.perf_counter()
    params = critic_params(
        critic, device=device, inner_rollout_horizon=3, inner_solve_interval=interval,
        train_unroll_horizon=3, q_representation="distributional", num_q=5,
        dropout=.01, inner_critic_target_update_interval=3,
    )
    eager = compiled = None
    try:
        eager = _model_from_params(params)
        compiled = _model_from_params(dict(params, compile=compile_enabled, compile_strict=True))
        assert eager.agent.device.type == compiled.agent.device.type == torch.device(device).type
        checkpoint = _clone_tree(eager.agent.checkpoint_state())
        compiled.agent.load(checkpoint)
        models = (eager, compiled)
        observers = [_instrument_boundaries(model) for model in models]
        outer = [_clone_tree(model.agent.checkpoint_state()) for model in models]
        solve_count = 0
        max_action_difference = 0.
        global_rng = _global_rng(eager.agent.device)
        for seed in (101, 102):
            for model in models:
                model.agent.inner_engine.reset_for_evaluation(seed, reuse_action_pool=True)
            for decision in range(7):
                observation = torch.tensor([1. + .07 * decision, -.1 * decision, .3])
                held = decision % interval != 0
                actions = []
                for model, observer, frozen in zip(models, observers, outer):
                    engine = model.agent.inner_engine
                    if held:
                        before = learner_snapshot(engine)
                        private_rng = _clone_tree(engine.rng.training_state_dict())
                        lifetime_before = engine.state.critic_lifetime_steps
                        with torch.no_grad():
                            root = engine.model.encode(observation.to(engine.device).unsqueeze(0))
                            expected_action = engine.model.policy_stats(
                                root, policy=engine._held_actor,
                                log_std_mapping=model.cfg.inner_log_std_mapping,
                                log_std_min=model.cfg.inner_log_std_min,
                                log_std_max=model.cfg.inner_log_std_max,
                            )["mean"][0]
                    trace = _trace(3)
                    action = model.agent.act(observation, t0=decision == 0, eval_mode=True, trace=trace)
                    actions.append(action)
                    metrics = model.agent.last_inner_metrics
                    assert torch.isfinite(action).all() and action.abs().max() <= 1
                    for value in metrics.values():
                        if torch.is_tensor(value):
                            assert torch.isfinite(value).all()
                        elif isinstance(value, (int, float)):
                            assert math.isfinite(value)
                    assert metrics["inner_compile_fallback"] == 0
                    assert metrics["inner_solve_performed"] == (not held)
                    assert metrics["inner_actor_transferred"] == 0
                    assert metrics["inner_critic_transferred"] == (decision > 0 and not held)
                    assert metrics["inner_critic_target_reinitialized"] == (not held)
                    assert metrics["inner_solve_index"] == decision // interval
                    if held:
                        torch.testing.assert_close(action, expected_action.cpu(), rtol=0, atol=0)
                        _assert_tree_equal(learner_snapshot(engine), before)
                        _assert_tree_equal(engine.rng.training_state_dict(), private_rng)
                        assert engine.state.critic_lifetime_steps == lifetime_before
                        assert trace.events == []
                    else:
                        assert metrics["inner_critic_updates_initial"] == 2 * decision // interval
                        assert metrics["inner_critic_optimizer_steps"] == metrics["inner_actor_optimizer_steps"] == 2
                        assert metrics["inner_target_updates"] == 0
                        assert trace.events[0]["metrics"]["inner_critic_target_reinitialized"] == 1
                        observer["previous"] = _clone_tree(engine.state.critic.state_dict())
                        # A stale target must be overwritten at the next solve.
                        # Held feedback actions never consult this target.
                        with torch.no_grad():
                            for parameter in engine.state.critic_target.parameters():
                                parameter.add_(3.)
                    _assert_tree_equal(model.agent.checkpoint_state(), frozen)
                    _assert_tree_equal(_global_rng(engine.device), global_rng)
                torch.testing.assert_close(actions[0], actions[1], atol=2e-5, rtol=2e-4)
                max_action_difference = max(max_action_difference, (actions[0] - actions[1]).abs().max().item())
                _assert_close_modules(eager.agent.inner_engine, compiled.agent.inner_engine)
                _assert_tree_equal(eager.agent.inner_engine.rng.training_state_dict(),
                                   compiled.agent.inner_engine.rng.training_state_dict())
                solve_count += not held
        for model, observer in zip(models, observers):
            assert all(ids == observer["allocations"][0] for ids in observer["allocations"])
            model.agent.load(checkpoint)
            engine = model.agent.inner_engine
            assert engine.state.critic is engine._action_pool.critic is engine._held_actor is None
            model.agent.act(torch.ones(3), t0=True, eval_mode=True)
            assert model.agent.last_inner_metrics["inner_critic_transferred"] == 0
            assert model.agent.last_inner_metrics["inner_critic_updates_initial"] == 0
            assert model.agent.last_inner_metrics["inner_compile_fallback"] == 0
            _assert_tree_equal(model.agent.checkpoint_state(), checkpoint)
        if eager.agent.device.type == "cuda":
            torch.cuda.synchronize(eager.agent.device)
        return {
            "passed": True, "fixture": "tiny-five-head-distributional-critic-transfer",
            "production_checkpoint_tested": False, "critic": critic, "solve_interval": interval,
            "H": 3, "J": 1, "C": 2, "A": 2, "seeds": [101, 102], "decisions_per_episode": 7,
            "solves_per_controller": solve_count, "checkpoint_load_followup_solves_per_controller": 1,
            "compile_strict": compile_enabled, "compile_fallback": False,
            "exact_lifecycle_boundaries": True, "target_reinitialized": True,
            "target_clock_restarted": True, "optimizer_references_and_reset": True,
            "allocation_reuse": True, "held_state_and_rng_unchanged": True,
            "outer_state_unchanged": True, "checkpoint_load_clears_transfer": True,
            "eager_compiled_action_max_absolute_difference": max_action_difference,
            "device": (torch.cuda.get_device_name(eager.agent.device)
                       if eager.agent.device.type == "cuda" else "cpu"),
            "torch": torch.__version__, "elapsed_seconds": time.perf_counter() - started,
            "source_commit": os.environ.get("EXPECTED_ACTION_MODES_SHA"),
        }
    finally:
        for model in (eager, compiled):
            if model is not None:
                model.close()
        del eager, compiled
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@pytest.mark.skipif(os.environ.get("AMBI_RUN_CRITIC_TRANSFER_CUDA_GATE") != "1",
                    reason="set AMBI_RUN_CRITIC_TRANSFER_CUDA_GATE=1 on an allocated GPU")
@pytest.mark.parametrize("critic", ["soft", "return"])
@pytest.mark.parametrize("interval", [1, 3])
def test_cuda_critic_transfer_lifecycle_and_strict_compile_parity(critic, interval):
    assert torch.cuda.is_available(), "Requested critic-transfer CUDA gate requires a GPU."
    output = Path(os.environ["AMBI_CRITIC_TRANSFER_GATE_OUTPUT_ROOT"])
    assert output.is_dir(), "Create a fresh CUDA gate output directory before running."
    result = _run_fixture_gate(critic, interval)
    with (output / f"critic-{critic}-interval{interval}.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("CRITIC_TRANSFER_CUDA_GATE_REPORT " + json.dumps(result, sort_keys=True))
