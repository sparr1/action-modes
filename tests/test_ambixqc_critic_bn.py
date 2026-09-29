"""Online critic BN choices and independent local alpha rates preserve ownership."""
from copy import deepcopy

import pytest
import torch

from test_ambixqc_compile import (
    _batch, _build_cfg, _controller, _noise, _assert_module_close,
    _assert_nested_close, _assert_objective_close,
)
from test_ambixqc_core import _tiny_model, _tree_equal


MODES = ("batch_update", "batch_no_update", "running")


def buffers(module):
    return deepcopy(dict(module.named_buffers()))


@pytest.mark.parametrize("mode", MODES)
def test_critic_bn_config_and_temperature_override(mode):
    cfg = _build_cfg(inner_critic_bn_mode=mode.upper(), inner_actor_lr=1.25e-5,
                     inner_temperature_lr=5e-5)
    assert cfg.inner_critic_bn_mode == mode
    assert cfg.inner_temperature_lr == 5e-5
    assert _build_cfg(inner_actor_lr=1.25e-5).inner_temperature_lr == 1.25e-5
    assert _build_cfg(inner_actor_lr=1.25e-5, inner_temperature_lr=None).inner_temperature_lr == 1.25e-5


@pytest.mark.parametrize("value", [True, 1, None, [], "eval", ""])
def test_invalid_critic_bn_config(value):
    with pytest.raises(ValueError, match="inner_critic_bn_mode"):
        _build_cfg(inner_critic_bn_mode=value)


@pytest.mark.parametrize("value", [True, 0, -1, float("nan"), float("inf")])
def test_invalid_temperature_override(value):
    with pytest.raises(ValueError, match="inner_temperature_lr"):
        _build_cfg(inner_temperature_lr=value)


@pytest.mark.parametrize("entry", ["critic_objective", "workspace"])
@pytest.mark.parametrize("mode", [True, None, [], "eval"])
def test_invalid_mode_fails_before_forward_or_optimizer_mutation(entry, mode):
    controller = _controller()
    workspace = controller.make_workspace(actor_lr=5e-5, critic_lr=5e-5)
    before = deepcopy(controller.state_dict()), deepcopy(workspace.state_dict())
    with pytest.raises(ValueError, match="critic_bn_mode"):
        if entry == "critic_objective":
            controller.critic_objective(_batch(), next_noise=_noise(), critic_bn_mode=mode)
        else:
            workspace.update(_batch(), next_noise=_noise(), actor_noise=_noise(), critic_bn_mode=mode)
    assert _tree_equal(before, (controller.state_dict(), workspace.state_dict()))
    assert all(p.grad is None for p in controller.parameters())


def test_default_update_is_exact_and_temperature_default_stays_tied():
    left = _controller().make_workspace(actor_lr=5e-5, critic_lr=5e-5)
    right = _controller().make_workspace(actor_lr=5e-5, critic_lr=5e-5, temperature_lr=5e-5)
    for slot in range(4):
        kwargs = dict(next_noise=_noise(seed=40+slot), actor_noise=_noise(seed=50+slot))
        a = left.update(_batch(), **kwargs)
        b = right.update(_batch(), critic_bn_mode="batch_update", **kwargs)
        assert _tree_equal(a, b)
        assert _tree_equal(left.controller.state_dict(), right.controller.state_dict())
        assert _tree_equal(left.state_dict(), right.state_dict())


def test_batch_commit_changes_only_buffers_not_critic_loss_or_gradient():
    controllers = [_controller() for _ in MODES]
    objectives, gradients = [], []
    batch = _batch()
    for mode, controller in zip(MODES, controllers):
        initial = buffers(controller.critic)
        target_initial = buffers(controller.critic_target)
        actor_initial = deepcopy(controller.actor.state_dict())
        expected_running = controller.critic.log_probs(batch.latents, batch.actions, bn_mode="running")
        objective = controller.critic_objective(batch, next_noise=_noise(), critic_bn_mode=mode)
        objective.loss.backward()
        objectives.append(objective)
        gradients.append([p.grad.clone() for p in controller.critic.parameters()])
        assert _tree_equal(initial, buffers(controller.critic)) == (mode != "batch_update")
        assert _tree_equal(target_initial, buffers(controller.critic_target))
        assert _tree_equal(actor_initial, controller.actor.state_dict())
        assert all(torch.isfinite(g).all() for g in gradients[-1])
        assert sum(g.abs().sum() for g in gradients[-1]) > 0
        if mode == "running":
            torch.testing.assert_close(objective.current_log_probs, expected_running, atol=0, rtol=0)
            for name, p in controller.critic.named_parameters():
                if "batch_norm" in name:
                    assert p.grad is not None and torch.isfinite(p.grad).all()
    _assert_objective_close(objectives[0], objectives[1], atol=0, rtol=0)
    _assert_nested_close(gradients[0], gradients[1], atol=0, rtol=0)
    assert not torch.equal(objectives[0].current_log_probs, objectives[2].current_log_probs)
    # Normalization choice changes only predictions, not the target equation.
    torch.testing.assert_close(objectives[0].target_probabilities, objectives[2].target_probabilities, atol=0, rtol=0)


@pytest.mark.parametrize("source", ["xqc", "aux_return"])
def test_inner_pool_resets_critic_buffers_and_independent_temperature_rate(source):
    model = _tiny_model(
        aux_return_mode="xqc", inner_actor_bn_mode="running",
        inner_critic_source=source, inner_horizon_critic_source=source,
        inner_terminal_bootstrap="outer", inner_rollout_horizon=1,
        inner_rounds=2, inner_updates_per_round=3, inner_rollouts_per_round=4,
        inner_batch_size=4, inner_replay_capacity=8, inner_actor_lr=1.25e-5,
        inner_temperature_lr=5e-5,
    )
    try:
        agent, outputs = model.agent, []
        outer = agent.frozen_outer_state()
        selected = agent.aux_return.critic if source == "aux_return" else agent.xqc_controller.critic
        initial = buffers(selected)
        observation, _ = model.env.reset(seed=101)
        for mode, rate in [("running", 5e-5), ("batch_update", 1.25e-5),
                           ("batch_no_update", 5e-5), ("running", 5e-5)]:
            agent.cfg.inner_critic_bn_mode = mode
            agent.cfg.inner_temperature_lr = rate
            model.reset_for_evaluation(77, reuse_action_pool=True)
            action = model.predict(observation, deterministic=True)[0]
            local = agent.inner_engine._workspace_pool
            assert local.update_step == 6 and local.actor_optimizer_steps == 2
            assert local.temperature_optimizer_steps == 2
            assert local.temperature_optimizer.param_groups[0]["lr"] == rate
            assert local.actor_optimizer.param_groups[0]["lr"] == 1.25e-5
            assert _tree_equal(initial, buffers(local.controller.critic)) == (mode != "batch_update")
            assert _tree_equal(initial, buffers(local.controller.critic_target))
            assert _tree_equal(buffers(agent.xqc_controller.actor), buffers(local.controller.actor))
            assert _tree_equal(outer, agent.frozen_outer_state())
            outputs.append((action.copy(), deepcopy(local.controller.state_dict()), deepcopy(local.state_dict())))
        assert (outputs[0][0] == outputs[-1][0]).all()
        assert _tree_equal(outputs[0][1:], outputs[-1][1:])
    finally:
        model.env.close()


@pytest.mark.parametrize("mode", MODES)
def test_inner_controls_do_not_change_utd_two_shared_outer_learning(mode):
    from test_ambixqc_update_ratio import _Replay
    base = dict(xqc_utd=2, aux_return_mode="xqc", aux_return_detach_representation=False)
    a, b = _tiny_model(**base), _tiny_model(**base, inner_critic_bn_mode=mode, inner_temperature_lr=2e-4)
    try:
        for _ in range(2):
            a.agent.update(_Replay(a.agent))
            b.agent.update(_Replay(b.agent))
            assert _tree_equal(a.agent.frozen_outer_state(), b.agent.frozen_outer_state())
    finally:
        a.env.close()
        b.env.close()


def test_temperature_override_changes_only_first_alpha_step():
    base = _controller().make_workspace(actor_lr=1.25e-5, critic_lr=5e-5)
    fixed = _controller().make_workspace(actor_lr=1.25e-5, critic_lr=5e-5, temperature_lr=5e-5)
    for workspace in (base, fixed):
        workspace.update(_batch(), next_noise=_noise(), actor_noise=_noise(),
                         actor_bn_mode="running", critic_bn_mode="running")
    assert _tree_equal(base.controller.actor.state_dict(), fixed.controller.actor.state_dict())
    assert _tree_equal(base.controller.critic.state_dict(), fixed.controller.critic.state_dict())
    assert not torch.equal(base.controller.log_temperature, fixed.controller.log_temperature)


@pytest.mark.parametrize("mode", MODES)
def test_compile_fallback_restores_buffers_and_preserves_mode(mode, monkeypatch):
    reference, compiled = _controller(), _controller()
    compiled.configure_compile(enabled=True, strict=False)
    compiled._critic_loss_region.enabled = True
    mutable = compiled._critic_loss_region.mutable_buffers[0]
    def fake_compile(function, **kwargs):
        def broken(*args):
            with torch.no_grad():
                mutable.add_(100)
            raise RuntimeError("injected first-call failure")
        return broken
    monkeypatch.setattr(torch, "compile", fake_compile)
    expected = reference.critic_objective(_batch(), next_noise=_noise(), critic_bn_mode=mode)
    with pytest.warns(RuntimeWarning, match="Falling back"):
        actual = compiled.critic_objective(_batch(), next_noise=_noise(), critic_bn_mode=mode)
    _assert_objective_close(expected, actual, atol=0, rtol=0)
    _assert_module_close(reference, compiled, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware is unavailable")
def test_cuda_compiled_critic_modes_and_independent_alpha_match_eager():
    source = _controller(device="cuda")
    eager, compiled = _controller(device="cuda"), _controller(device="cuda")
    compiled.configure_compile(enabled=True, strict=True)
    workspaces = [controller.make_workspace(actor_lr=1.25e-5, critic_lr=5e-5, temperature_lr=5e-5)
                  for controller in (eager, compiled)]
    for mode in (*MODES, "running"):
        for workspace in workspaces:
            workspace.reset_from_(source)
        for slot in range(4):
            metrics = [workspace.update(_batch(device="cuda"), next_noise=_noise(device="cuda", seed=30+slot),
                        actor_noise=_noise(device="cuda", seed=40+slot), actor_bn_mode="running", critic_bn_mode=mode)
                       for workspace in workspaces]
            _assert_nested_close(*metrics, atol=2e-4, rtol=1e-3)
        torch.cuda.synchronize()
        _assert_module_close(eager, compiled, atol=2e-4, rtol=1e-3)
        _assert_nested_close(*(w.state_dict() for w in workspaces), atol=1e-3, rtol=1e-3)
        assert compiled.compile_status["critic_compiled"] and not compiled.compile_status["fallback"]
