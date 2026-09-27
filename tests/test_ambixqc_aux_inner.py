"""Auxiliary return critics supply values while the primary policy stays canonical."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from RL.tdmpc2_core.inner_xqc import InnerXQCEngine
from RL.tdmpc2_core.xqc_controller import (
    LatentXQCConfig, LatentXQCController, LatentXQCWorkspace,
)
from test_ambixqc_core import _batch, _tree_equal
from test_ambixqc_inner import _agent
from test_ambixqc_prior_checkpoint import _wrapper


def _engine(*, source="xqc", horizon_source="xqc", target="entropy_augmented", terminal="outer"):
    torch.manual_seed(41)
    agent, _ = _agent(oracle_rollout=True)
    agent.cfg.inner_critic_source = source
    agent.cfg.inner_horizon_critic_source = horizon_source
    agent.cfg.inner_critic_target = target
    agent.cfg.inner_terminal_bootstrap = terminal
    agent.xqc_controller = LatentXQCController(2, 1, LatentXQCConfig(
        actor_net_arch=(8,), critic_net_arch=(8,), num_atoms=7,
        vmin=-3, vmax=3, init_temperature=0.3, target_entropy=-0.5,
        optimizer_backend="single_tensor",
    ))
    auxiliary = deepcopy(agent.xqc_controller.critic)
    with torch.no_grad():
        for name, parameter in auxiliary.named_parameters():
            if name.endswith("bias"):
                parameter.add_(0.2)
        for name, buffer in auxiliary.named_buffers():
            if name.endswith("running_mean"):
                buffer.add_(0.7)
            elif name.endswith("running_var"):
                buffer.add_(0.3)
    agent.aux_return = SimpleNamespace(
        critic=auxiliary, critic_target=deepcopy(auxiliary),
    )
    return InnerXQCEngine(agent)


@pytest.mark.parametrize("source", ["xqc", "aux_return"])
def test_selected_online_critic_and_bn_seed_both_inner_critics_and_reset(source):
    engine = _engine(source=source)
    outer = engine.outer_controller
    selected = outer.critic if source == "xqc" else engine.agent.aux_return.critic
    engine._prepare_action()
    workspace = engine.state.workspace
    local = workspace.controller
    assert _tree_equal(local.actor.state_dict(), outer.actor.state_dict())
    assert torch.equal(local.log_temperature, outer.log_temperature)
    assert _tree_equal(local.critic.state_dict(), selected.state_dict())
    assert _tree_equal(local.critic_target.state_dict(), selected.state_dict())
    assert all(not optimizer.state for optimizer in (
        workspace.actor_optimizer, workspace.critic_optimizer, workspace.temperature_optimizer,
    ))

    engine._collect_round(torch.zeros(1, 2))
    engine._update_slot()
    assert all(optimizer.state for optimizer in (
        workspace.actor_optimizer, workspace.critic_optimizer, workspace.temperature_optimizer,
    ))
    assert not _tree_equal(local.critic.state_dict(), selected.state_dict())
    engine._release_action()
    with torch.no_grad():
        for name, buffer in selected.named_buffers():
            if name.endswith("running_mean"):
                buffer.add_(0.1)
    engine._prepare_action()
    assert engine.state.workspace is workspace
    assert _tree_equal(local.actor.state_dict(), outer.actor.state_dict())
    assert torch.equal(local.log_temperature, outer.log_temperature)
    assert _tree_equal(local.critic.state_dict(), selected.state_dict())
    assert _tree_equal(local.critic_target.state_dict(), selected.state_dict())
    assert workspace.update_step == workspace.actor_optimizer_steps == workspace.temperature_optimizer_steps == 0
    assert all(not optimizer.state for optimizer in (
        workspace.actor_optimizer, workspace.critic_optimizer, workspace.temperature_optimizer,
    ))
    engine._release_action()


@pytest.mark.parametrize("source", ["xqc", "aux_return"])
@pytest.mark.parametrize("horizon_source", ["xqc", "aux_return"])
@pytest.mark.parametrize("target", ["entropy_augmented", "reward_only"])
@pytest.mark.parametrize("terminal", ["inner", "outer"])
def test_inner_source_target_combinations_keep_primary_actor_and_frozen_outer(
    monkeypatch, source, horizon_source, target, terminal,
):
    engine = _engine(source=source, horizon_source=horizon_source, target=target, terminal=terminal)
    primary = engine.outer_controller
    auxiliary = engine.agent.aux_return
    primary_before = deepcopy(primary.state_dict())
    aux_before = deepcopy(auxiliary.critic.state_dict())
    aux_target_before = deepcopy(auxiliary.critic_target.state_dict())
    model_before = deepcopy(engine.model.state_dict())
    selected_module = primary.critic if source == "xqc" else auxiliary.critic
    selected_bn = dict(selected_module.named_buffers())
    calls = []
    update = LatentXQCWorkspace.update

    def checked_update(workspace, batch, **kwargs):
        assert kwargs.get("critic_target_kind", "entropy_augmented") == target
        assert kwargs["reward_scale"] == 2.5
        if terminal == "outer":
            assert kwargs["outer_controller"] is primary
            assert kwargs["outer_terminal_mask"].dtype == torch.bool
            if horizon_source == "aux_return":
                assert kwargs["outer_critic"] is auxiliary.critic
                assert kwargs["outer_critic_is_return"] is True
            else:
                assert "outer_critic" not in kwargs
                assert not kwargs.get("outer_critic_is_return", False)
        else:
            assert "outer_controller" not in kwargs
            assert "outer_critic" not in kwargs
        calls.append(kwargs)
        return update(workspace, batch, **kwargs)

    monkeypatch.setattr(LatentXQCWorkspace, "update", checked_update)
    global_rng = torch.get_rng_state().clone()
    action, metrics, _ = engine.act(torch.zeros(1, 2), eval_mode=True)
    local = engine._workspace_pool.controller
    assert len(calls) == 4
    assert torch.isfinite(action).all()
    assert all(torch.isfinite(torch.as_tensor(value)).all() for value in metrics.values())
    assert metrics["inner_critic_optimizer_steps"] == 4
    assert metrics["inner_actor_optimizer_steps"] == metrics["inner_temperature_optimizer_steps"] == 2
    assert not _tree_equal(local.actor.state_dict(), primary.actor.state_dict())
    assert not torch.equal(local.log_temperature, primary.log_temperature)
    assert not _tree_equal(local.critic.state_dict(), selected_module.state_dict())
    # Target updates preserve inherited BN, while the local online BN trains.
    for name, value in local.critic_target.named_buffers():
        assert torch.equal(value, selected_bn[name])
    assert any(
        not torch.equal(value, selected_bn[name])
        for name, value in local.critic.named_buffers()
        if name.endswith(("running_mean", "running_var"))
    )
    assert _tree_equal(primary_before, primary.state_dict())
    assert _tree_equal(aux_before, auxiliary.critic.state_dict())
    assert _tree_equal(aux_target_before, auxiliary.critic_target.state_dict())
    assert _tree_equal(model_before, engine.model.state_dict())
    assert all(parameter.grad is None for module in (primary, auxiliary.critic, auxiliary.critic_target, engine.model)
               for parameter in module.parameters())
    assert torch.equal(global_rng, torch.get_rng_state())
    assert engine.state.workspace is None


def test_auxiliary_initialization_requires_an_available_head_before_allocating():
    engine = _engine(source="aux_return")
    engine.agent.aux_return = None
    with pytest.raises(ValueError, match="auxiliary return critic is required"):
        engine._prepare_action()
    assert engine.state.workspace is None


def test_explicit_default_selections_preserve_legacy_actions_rng_and_state():
    explicit = _engine(terminal="inner")
    legacy = _engine(terminal="inner")
    for key in ("inner_critic_source", "inner_horizon_critic_source", "inner_critic_target"):
        delattr(legacy.cfg, key)
    action, _, _ = explicit.act(torch.zeros(1, 2), eval_mode=True)
    legacy_action, _, _ = legacy.act(torch.zeros(1, 2), eval_mode=True)
    assert torch.equal(action, legacy_action)
    assert _tree_equal(explicit.training_state_dict(), legacy.training_state_dict())
    assert _tree_equal(explicit._workspace_pool.controller.state_dict(), legacy._workspace_pool.controller.state_dict())


@pytest.mark.parametrize("source", ["xqc", "aux_return"])
def test_trained_auxiliary_head_remains_frozen_across_repeated_real_wrapper_actions(source):
    wrapper = _wrapper(
        aux_return_mode="xqc", inner_critic_source=source,
        inner_horizon_critic_source="aux_return", inner_terminal_bootstrap="outer",
    )
    try:
        agent = wrapper.agent
        agent._update(*_batch(agent))
        agent.observe_reward(2.0, False, False)
        agent.load(deepcopy(agent.checkpoint_state()), frozen_evaluation=True)
        before = agent.frozen_outer_state()
        observation, _ = wrapper.env.reset(seed=101)
        global_rng = torch.get_rng_state().clone()
        actions = []
        for _ in range(2):
            wrapper.reset_for_evaluation(77, reuse_action_pool=True)
            actions.append(wrapper.predict(observation, deterministic=True)[0])
            assert _tree_equal(before, agent.frozen_outer_state())
            assert agent.last_inner_metrics["inner_horizon_critic_source_aux_return"] == 1
            assert agent.last_inner_metrics["inner_critic_source_aux_return"] == float(source == "aux_return")
            assert agent.last_inner_metrics["inner_critic_target_reward_only"] == float(source == "aux_return")
        assert (actions[0] == actions[1]).all()
        assert torch.equal(global_rng, torch.get_rng_state())
    finally:
        wrapper.env.close()
