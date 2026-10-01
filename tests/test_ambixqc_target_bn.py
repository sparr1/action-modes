"""Target BN ablations preserve the default learner and state ownership."""
from copy import deepcopy

import pytest
import torch

from test_ambixqc_compile import (
    _batch, _build_cfg, _controller, _noise, _assert_module_close,
    _assert_nested_close, _assert_objective_close,
)
from test_ambixqc_core import _tiny_model, _tree_equal


MODES = ("batch_no_update", "running")


def test_target_bn_config_defaults_and_normalizes_explicit_modes():
    assert _build_cfg().inner_critic_target_bn_mode == "batch_no_update"
    for mode in MODES:
        assert _build_cfg(inner_critic_target_bn_mode=mode.upper()).inner_critic_target_bn_mode == mode


@pytest.mark.parametrize("mode", [True, 1, None, [], "batch_update", "eval", ""])
def test_invalid_target_bn_configuration(mode):
    with pytest.raises(ValueError, match="inner_critic_target_bn_mode"):
        _build_cfg(inner_critic_target_bn_mode=mode)


@pytest.mark.parametrize("entry", ["critic_objective", "workspace"])
@pytest.mark.parametrize("mode", [True, None, [], "batch_update"])
def test_invalid_target_mode_fails_before_forward_or_optimizer_mutation(entry, mode):
    controller = _controller()
    workspace = controller.make_workspace(actor_lr=5e-5, critic_lr=5e-5)
    before = deepcopy((controller.state_dict(), workspace.state_dict()))
    with pytest.raises(ValueError, match="critic_target_bn_mode"):
        if entry == "critic_objective":
            controller.critic_objective(_batch(), next_noise=_noise(), critic_target_bn_mode=mode)
        else:
            workspace.update(_batch(), next_noise=_noise(), actor_noise=_noise(), critic_target_bn_mode=mode)
    assert _tree_equal(before, (controller.state_dict(), workspace.state_dict()))
    assert all(p.grad is None for p in controller.parameters())


def test_explicit_default_update_and_compiled_call_shape_are_unchanged(monkeypatch):
    workspaces = [_controller().make_workspace(actor_lr=5e-5, critic_lr=5e-5) for _ in MODES]
    calls = []
    for workspace in workspaces:
        original = workspace.controller._critic_loss_components
        def record(*args, _original=original):
            calls.append(len(args))
            return _original(*args)
        monkeypatch.setattr(workspace.controller, "_critic_loss_components", record)
    for slot in range(4):
        kwargs = dict(next_noise=_noise(seed=40+slot), actor_noise=_noise(seed=50+slot))
        left = workspaces[0].update(_batch(), **kwargs)
        right = workspaces[1].update(_batch(), critic_target_bn_mode="batch_no_update", **kwargs)
        assert _tree_equal(left, right)
        assert _tree_equal(workspaces[0].controller.state_dict(), workspaces[1].controller.state_dict())
        assert _tree_equal(workspaces[0].state_dict(), workspaces[1].state_dict())
    assert calls == [10] * 8  # Historical loss call: no added optional operands.


@pytest.mark.parametrize("mode", MODES)
def test_target_modes_use_expected_moments_and_parameter_only_polyak(mode):
    controller = _controller()
    with torch.no_grad():
        for name, buffer in controller.critic_target.named_buffers():
            if name.endswith("input_batch_norm.running_mean"):
                buffer.add_(.2)
            elif name.endswith("running_var"):
                buffer.mul_(1.8)
    batch = _batch()
    initial = deepcopy(controller.state_dict())
    rng = torch.get_rng_state().clone()
    base = controller.critic_objective(batch, next_noise=_noise(), critic_bn_mode="running",
                                       critic_target_bn_mode=mode)
    with torch.no_grad():
        for name, buffer in controller.critic_target.named_buffers():
            if name.endswith("input_batch_norm.running_mean"):
                buffer.sub_(.9)
    perturbed = deepcopy(controller.state_dict())
    changed = controller.critic_objective(batch, next_noise=_noise(), critic_bn_mode="running",
                                          critic_target_bn_mode=mode)
    assert torch.equal(base.target_probabilities, changed.target_probabilities) == (mode == "batch_no_update")
    torch.testing.assert_close(base.current_log_probs, changed.current_log_probs, rtol=0, atol=0)
    assert _tree_equal(perturbed, controller.state_dict())
    assert torch.equal(rng, torch.get_rng_state())
    workspace = controller.make_workspace(actor_lr=5e-5, critic_lr=5e-5)
    before_target_params = {n: p.clone() for n, p in controller.critic_target.named_parameters()}
    before_target_buffers = deepcopy(dict(controller.critic_target.named_buffers()))
    result = workspace.update(batch, next_noise=_noise(), actor_noise=_noise(),
                              actor_bn_mode="running", critic_bn_mode="batch_update",
                              critic_target_bn_mode=mode)
    assert result["target_updated"] == 1
    assert _tree_equal(before_target_buffers, dict(controller.critic_target.named_buffers()))
    assert not _tree_equal(initial, controller.state_dict())
    tau = controller.config.tau
    online = dict(controller.critic.named_parameters())
    for name, parameter in controller.critic_target.named_parameters():
        expected = before_target_params[name].lerp(online[name], tau)
        torch.testing.assert_close(parameter, expected, atol=1e-7, rtol=1e-6)
    assert all(p.grad is None for p in controller.critic_target.parameters())


def test_target_setting_does_not_change_shared_utd_two_backbone_learning():
    from test_ambixqc_update_ratio import _Replay
    base = dict(xqc_utd=2, aux_return_mode="xqc", aux_return_detach_representation=False)
    models = [_tiny_model(**base, inner_critic_target_bn_mode=mode) for mode in MODES]
    try:
        for _ in range(2):
            for model in models:
                model.agent.update(_Replay(model.agent))
            assert _tree_equal(*(model.agent.frozen_outer_state() for model in models))
    finally:
        for model in models:
            model.env.close()


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("outer", [False, True])
def test_compiled_dispatch_and_atomic_fallback_preserve_target_mode(mode, outer, monkeypatch):
    reference, compiled = _controller(), _controller()
    compiled.configure_compile(enabled=True, strict=False)
    compiled._critic_loss_region.enabled = True
    mutable = compiled._critic_loss_region.mutable_buffers[0]
    seen = []
    def fake_compile(function, **kwargs):
        def broken(*args):
            seen.append(args)
            with torch.no_grad():
                mutable.add_(100)
            raise RuntimeError("injected first-call failure")
        return broken
    monkeypatch.setattr(torch, "compile", fake_compile)
    kwargs = dict(next_noise=_noise(), critic_target_bn_mode=mode)
    if outer:
        kwargs.update(outer_controller=_controller(), outer_terminal_mask=torch.tensor([False, True, False, True]))
    expected = reference.critic_objective(_batch(), **kwargs)
    with pytest.warns(RuntimeWarning, match="Falling back"):
        actual = compiled.critic_objective(_batch(), **kwargs)
    _assert_objective_close(expected, actual, atol=0, rtol=0)
    _assert_module_close(reference, compiled, atol=0, rtol=0)
    assert len(seen[0]) == (16 if mode == "running" else 15 if outer else 10)
    assert compiled.compile_status["fallback"]


def test_strict_compile_failure_is_never_retried_as_eager(monkeypatch):
    controller = _controller()
    controller.configure_compile(enabled=True, strict=True)
    controller._critic_loss_region.enabled = True
    calls = []
    def fake_compile(function, **kwargs):
        def broken(*args):
            calls.append(args[-1])
            raise RuntimeError("injected strict failure")
        return broken
    monkeypatch.setattr(torch, "compile", fake_compile)
    with pytest.raises(RuntimeError, match="Compiled AMBI-XQC critic loss failed"):
        controller.critic_objective(_batch(), next_noise=_noise(), critic_target_bn_mode="running")
    assert calls == ["running"]
    assert not controller.compile_status["fallback"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware is unavailable")
def test_cuda_compiled_target_modes_match_eager_with_inner_and_outer_rows():
    source, outer = _controller(device="cuda"), _controller(device="cuda")
    eager, compiled = _controller(device="cuda"), _controller(device="cuda")
    compiled.configure_compile(enabled=True, strict=True)
    workspaces = [c.make_workspace(actor_lr=5e-5, critic_lr=5e-5) for c in (eager, compiled)]
    for mode in (*MODES, "running"):
        for workspace in workspaces:
            workspace.reset_from_(source)
        for slot in range(4):
            metrics = [w.update(_batch(device="cuda"), next_noise=_noise(device="cuda", seed=30+slot),
                        actor_noise=_noise(device="cuda", seed=40+slot), actor_bn_mode="running",
                        critic_bn_mode="running", critic_target_bn_mode=mode, outer_controller=outer,
                        outer_terminal_mask=torch.tensor([False, True, False, True], device="cuda"))
                       for w in workspaces]
            _assert_nested_close(*metrics, atol=2e-4, rtol=1e-3)
        torch.cuda.synchronize()
        _assert_module_close(eager, compiled, atol=2e-4, rtol=1e-3)
        _assert_nested_close(*(w.state_dict() for w in workspaces), atol=1e-3, rtol=1e-3)
        assert compiled.compile_status["critic_compiled"] and not compiled.compile_status["fallback"]
