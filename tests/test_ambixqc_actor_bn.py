"""Actor-only running BatchNorm preserves learning and all other XQC semantics."""

from copy import deepcopy

import pytest
import torch

from test_ambixqc_compile import (
    _batch as _latent_batch, _build_cfg, _controller, _noise,
    _assert_module_close, _assert_nested_close, _assert_parameter_grads_close,
)
from test_ambixqc_core import _batch, _tiny_model, _tree_equal


def _workspace(controller):
    return controller.make_workspace(actor_lr=5e-5, critic_lr=5e-5)


def _buffers(module):
    return deepcopy(dict(module.named_buffers()))


def test_actor_bn_config_defaults_and_normalizes_explicit_modes():
    assert _build_cfg().inner_actor_bn_mode == "batch_update"
    assert _build_cfg(inner_actor_bn_mode="running").inner_actor_bn_mode == "running"
    assert _build_cfg(inner_actor_bn_mode="RUNNING").inner_actor_bn_mode == "running"


@pytest.mark.parametrize("mode", [None, True, 1, [], "", "batch_no_update", "eval"])
def test_actor_bn_config_rejects_invalid_modes(mode):
    with pytest.raises(ValueError, match="inner_actor_bn_mode"):
        _build_cfg(inner_actor_bn_mode=mode)


@pytest.mark.parametrize("mode", [None, True, [], "batch_no_update", "eval"])
@pytest.mark.parametrize("entry", ["workspace", "actor_objective"])
def test_invalid_actor_bn_mode_rejected_before_any_state_mutation(mode, entry):
    controller = _controller()
    workspace = _workspace(controller)
    before = deepcopy(controller.state_dict()), deepcopy(workspace.state_dict())
    rng = torch.get_rng_state().clone()
    batch, noise = _latent_batch(), _noise()
    with pytest.raises(ValueError, match="actor_bn_mode"):
        if entry == "workspace":
            workspace.update(batch, next_noise=noise, actor_noise=noise, actor_bn_mode=mode)
        else:
            controller.actor_objective(batch.latents, actor_noise=noise, actor_bn_mode=mode)
    assert _tree_equal(before[0], controller.state_dict())
    assert _tree_equal(before[1], workspace.state_dict())
    assert all(parameter.grad is None for parameter in controller.parameters())
    assert torch.equal(rng, torch.get_rng_state())


def test_default_and_explicit_batch_update_preserve_values_gradients_optimizers_and_rng():
    default, explicit = _workspace(_controller()), _workspace(_controller())
    batch = _latent_batch()
    rng = torch.get_rng_state().clone()
    for slot in range(4):
        kwargs = {"next_noise": _noise(seed=30 + slot), "actor_noise": _noise(seed=50 + slot)}
        default_metrics = default.update(batch, **kwargs)
        explicit_metrics = explicit.update(batch, **kwargs, actor_bn_mode="batch_update")
        assert _tree_equal(default_metrics, explicit_metrics)
        assert _tree_equal(default.controller.state_dict(), explicit.controller.state_dict())
        assert _tree_equal(default.state_dict(), explicit.state_dict())
        _assert_parameter_grads_close(default.controller.actor, explicit.controller.actor, atol=0, rtol=0)
    actions = [workspace.controller.sample_action(batch.latents, deterministic=True)[0]
               for workspace in (default, explicit)]
    torch.testing.assert_close(*actions, atol=0, rtol=0)
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("mode", ["batch_update", "running"])
def test_actor_bn_mode_only_changes_actor_training_forward_and_preserves_critic_rules(mode, monkeypatch):
    controller = _controller()
    workspace = _workspace(controller)
    actor_before, critic_before, target_before = map(
        _buffers, (controller.actor, controller.critic, controller.critic_target)
    )
    actor_weights = deepcopy(controller.actor.state_dict())
    alpha_before = controller.log_temperature.detach().clone()
    calls = {"actor": [], "critic": [], "target": []}
    for name, module, method in (
        ("actor", controller.actor, "sample"),
        ("critic", controller.critic, "log_probs"),
        ("target", controller.critic_target, "log_probs"),
    ):
        original = getattr(module, method)

        def capture(*args, _name=name, _original=original, **kwargs):
            calls[_name].append(kwargs["bn_mode"])
            return _original(*args, **kwargs)

        monkeypatch.setattr(module, method, capture)
    # H1 has identical current states: freezing statistics must preserve
    # useful actor gradients rather than disabling autograd or alpha updates.
    batch = _latent_batch()
    batch.latents[:] = batch.latents[0].clone()
    metrics = workspace.update(
        batch, next_noise=_noise(seed=31), actor_noise=_noise(seed=37), actor_bn_mode=mode,
    )
    assert calls == {"actor": ["running", mode], "critic": ["batch_update", "running"],
                     "target": ["batch_no_update"]}
    assert _tree_equal(actor_before, _buffers(controller.actor)) == (mode == "running")
    assert not _tree_equal(critic_before, _buffers(controller.critic))
    assert _tree_equal(target_before, _buffers(controller.critic_target))
    assert not _tree_equal(actor_weights, controller.actor.state_dict())
    assert not torch.equal(alpha_before, controller.log_temperature)
    assert metrics["actor_update_accepted"] == 1
    assert workspace.actor_optimizer_steps == workspace.temperature_optimizer_steps == 1
    if mode == "running":
        assert controller.actor.mean.weight.grad.abs().sum() > 0
        assert controller.actor.blocks[0].linear.weight.grad.abs().sum() > 0
        affine_names = [name for name, _ in controller.actor.named_parameters()
                        if "batch_norm" in name]
        assert "input_batch_norm.weight" in affine_names
        assert "input_batch_norm.bias" in affine_names
        assert len(affine_names) == 2 * (1 + len(controller.actor.blocks))
        for name, parameter in controller.actor.named_parameters():
            if name in affine_names:
                assert parameter.grad is not None
                assert torch.isfinite(parameter.grad).all(), name
                assert parameter.grad.abs().sum() > 0, name
                assert not torch.equal(actor_weights[name], parameter), name


@pytest.mark.parametrize("source", ["xqc", "aux_return"])
def test_h1_running_actor_bn_preserves_outer_and_repeats_across_pool_mode_changes(source, monkeypatch):
    model = _tiny_model(
        aux_return_mode="xqc", inner_actor_bn_mode="running",
        inner_critic_source=source, inner_horizon_critic_source=source,
        inner_terminal_bootstrap="outer", inner_rounds=2,
        inner_rollouts_per_round=4, inner_rollout_horizon=1,
        inner_updates_per_round=3, inner_batch_size=4, inner_replay_capacity=8,
    )
    try:
        agent = model.agent
        agent.observe_reward(2.0, False, False)
        agent._update(*_batch(agent))
        model.load(deepcopy(agent.checkpoint_state()), frozen_evaluation=True)
        before = agent.frozen_outer_state()
        outer_actor_bn = _buffers(agent.xqc_controller.actor)
        actor_weights = deepcopy(dict(agent.xqc_controller.actor.named_parameters()))
        selected = agent.xqc_controller.critic if source == "xqc" else agent.aux_return.critic
        engine = agent.inner_engine
        prepare = engine._prepare_action
        prepared_ids = []

        def check_prepare():
            prepare()
            workspace = engine.state.workspace
            prepared_ids.append(id(workspace))
            assert _tree_equal(outer_actor_bn, _buffers(workspace.controller.actor))
            assert _tree_equal(selected.state_dict(), workspace.controller.critic.state_dict())
            assert _tree_equal(selected.state_dict(), workspace.controller.critic_target.state_dict())
            assert workspace.update_step == 0
            assert all(not optimizer.state for optimizer in (
                workspace.actor_optimizer, workspace.critic_optimizer, workspace.temperature_optimizer,
            ))

        monkeypatch.setattr(engine, "_prepare_action", check_prepare)
        observation, _ = model.env.reset(seed=101)
        rng = torch.get_rng_state().clone()

        def solve(mode):
            agent.cfg.inner_actor_bn_mode = mode
            model.reset_for_evaluation(77, reuse_action_pool=True)
            action = model.predict(observation, deterministic=True)[0]
            workspace = engine._workspace_pool
            local = workspace.controller
            assert workspace.update_step == 6
            assert workspace.actor_optimizer_steps == workspace.temperature_optimizer_steps == 2
            assert _tree_equal(outer_actor_bn, _buffers(local.actor)) == (mode == "running")
            assert not _tree_equal(actor_weights, dict(local.actor.named_parameters()))
            assert not torch.equal(local.log_temperature, agent.xqc_controller.log_temperature)
            assert _tree_equal(_buffers(selected), _buffers(local.critic_target))
            assert not _tree_equal(_buffers(selected), _buffers(local.critic))
            assert _tree_equal(before, agent.frozen_outer_state())
            assert torch.equal(rng, torch.get_rng_state())
            return action.copy(), deepcopy(local.state_dict()), deepcopy(workspace.state_dict())

        first = solve("running")
        solve("batch_update")
        repeated = solve("running")
        assert len(set(prepared_ids)) == 1  # Both modes use the same allocation.
        assert (first[0] == repeated[0]).all()
        assert _tree_equal(first[1], repeated[1])
        assert _tree_equal(first[2], repeated[2])
    finally:
        model.env.close()


def test_inner_running_mode_does_not_change_outer_training():
    default = _tiny_model()
    running = _tiny_model(inner_actor_bn_mode="running")
    try:
        assert _tree_equal(default.agent.frozen_outer_state(), running.agent.frozen_outer_state())
        for _ in range(2):
            default.agent._update(*_batch(default.agent))
            running.agent._update(*_batch(running.agent))
            assert _tree_equal(default.agent.frozen_outer_state(), running.agent.frozen_outer_state())
    finally:
        default.env.close()
        running.env.close()


def test_compile_wrapper_receives_explicit_mode_and_reuses_controller_without_stale_mode(monkeypatch):
    reference, candidate = _controller(), _controller()
    candidate.configure_compile(enabled=True, strict=True)
    candidate._actor_loss_region.enabled = True  # Exercise the boundary on CPU.
    modes = []

    def fake_compile(function, **kwargs):
        def compiled(*args):
            modes.append(args[-1])
            return function(*args)
        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    for mode in ("running", "batch_update", "running"):
        batch, noise = _latent_batch(), _noise()
        expected = reference.actor_objective(batch.latents, actor_noise=noise, actor_bn_mode=mode)
        actual = candidate.actor_objective(batch.latents, actor_noise=noise, actor_bn_mode=mode)
        assert _tree_equal(vars(expected), vars(actual))
        assert _tree_equal(reference.actor.state_dict(), candidate.actor.state_dict())
    assert modes == ["running", "batch_update", "running"]
    assert candidate.compile_status["actor_compiled"]
    assert not candidate.compile_status["fallback"]


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA hardware is unavailable",
)
def test_strict_cuda_compiled_actor_bn_modes_match_eager_with_reset_and_alternation():
    eager, compiled = _controller(device="cuda"), _controller(device="cuda")
    compiled.load_state_dict(deepcopy(eager.state_dict()))
    compiled.configure_compile(enabled=True, strict=True)
    eager_ws, compiled_ws = _workspace(eager), _workspace(compiled)
    source = _controller(device="cuda")
    source.load_state_dict(deepcopy(eager.state_dict()))
    batch = _latent_batch(device="cuda")
    batch.latents[:] = batch.latents[0].clone()
    for mode in ("running", "batch_update", "running"):
        eager_ws.reset_from_(source)
        compiled_ws.reset_from_(source)
        initial = _buffers(source.actor)
        for slot in range(4):
            noise = _noise(device="cuda", seed=53 + slot)
            eager_metrics = eager_ws.update(
                batch, next_noise=noise, actor_noise=noise, actor_bn_mode=mode,
            )
            compiled_metrics = compiled_ws.update(
                batch, next_noise=noise, actor_noise=noise, actor_bn_mode=mode,
            )
            _assert_nested_close(eager_metrics, compiled_metrics, atol=2e-4, rtol=1e-3)
        torch.cuda.synchronize()
        _assert_module_close(eager, compiled, atol=2e-4, rtol=1e-3)
        _assert_nested_close(eager_ws.state_dict(), compiled_ws.state_dict(), atol=1e-3, rtol=1e-3)
        if mode == "running":
            assert _tree_equal(initial, _buffers(compiled.actor))
        assert eager_ws.update_step == compiled_ws.update_step == 4
        assert compiled_ws.actor_optimizer_steps == compiled_ws.temperature_optimizer_steps == 2
        assert compiled.compile_status["actor_compiled"]
        assert not compiled.compile_status["fallback"]
