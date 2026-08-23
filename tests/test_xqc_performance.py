import copy

import numpy as np
import pytest
import torch

import RL.xqc_core as xqc_core
from RL.sac_core import ReplayBuffer
from RL.xqc_core import XQCAgent, XQCConfig


def _config(**overrides):
    values = {
        "actor_net_arch": (8,),
        "critic_net_arch": (8,),
        "num_atoms": 5,
        "vmin": -2.0,
        "vmax": 2.0,
        "num_interactions": 8,
        "updates_per_step": 2,
        "gradient_steps": 2,
        "batch_size": 4,
        "policy_delay": 3,
        "reward_normalization": False,
        "seed": 37,
        "device": "cpu",
    }
    values.update(overrides)
    return XQCConfig(**values)


def _batch(offset=0.0):
    return {
        "obs": torch.tensor(
            [
                [0.2 + offset, -0.5],
                [1.1, 0.3 + offset],
                [-0.7, 0.8],
                [0.4, -0.9 + offset],
            ]
        ),
        "actions": torch.tensor([[0.4], [-0.2], [0.6], [-0.5]]),
        "rewards": torch.tensor([[0.3], [-0.4], [1.2], [0.1]]),
        "next_obs": torch.tensor(
            [
                [0.3 + offset, -0.4],
                [1.0, 0.2 + offset],
                [-0.5, 0.9],
                [0.1, -0.6 + offset],
            ]
        ),
        "dones": torch.tensor([[0.0], [0.0], [1.0], [0.0]]),
    }


def _assert_nested_close(actual, expected):
    if torch.is_tensor(actual):
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_nested_close(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for actual_value, expected_value in zip(actual, expected):
            _assert_nested_close(actual_value, expected_value)
    elif isinstance(actual, float):
        assert actual == pytest.approx(expected, abs=1e-9, rel=1e-9)
    else:
        assert actual == expected


class _JoinedReplay:
    def __init__(self, batches):
        self.batches = batches
        self.calls = []

    def sample(self, batch_size, device):
        self.calls.append((batch_size, torch.device(device)))
        expected = sum(batch["obs"].shape[0] for batch in self.batches)
        assert batch_size == expected
        return {
            key: torch.cat([batch[key] for batch in self.batches], dim=0).to(device)
            for key in self.batches[0]
        }


def test_one_large_replay_draw_preserves_the_sequential_utd_sample_stream():
    replay = ReplayBuffer(obs_dim=2, action_dim=1, capacity=16)
    for index in range(16):
        replay.add(
            [index, -index],
            [index / 16.0],
            index + 0.5,
            [index + 1, -index - 1],
            index % 3 == 0,
            False,
        )

    np.random.seed(913)
    rng_state = np.random.get_state()
    sequential = [replay.sample(4, torch.device("cpu")) for _ in range(2)]
    np.random.set_state(rng_state)
    joined = replay.sample(8, torch.device("cpu"))

    for key in joined:
        torch.testing.assert_close(
            joined[key], torch.cat([batch[key] for batch in sequential], dim=0)
        )


def test_batched_utd_sampling_matches_sequential_updates_and_returns_last_metrics():
    batches = (_batch(), _batch(offset=0.125))
    reference = XQCAgent(2, 1, _config())
    optimized = XQCAgent(2, 1, _config())

    reference._update_once(batches[0], collect_metrics=False)
    expected_metrics = reference._update_once(batches[1])

    replay = _JoinedReplay(batches)
    actual_metrics = optimized.update(replay, gradient_steps=2, batch_size=4)

    assert replay.calls == [(8, torch.device("cpu"))]
    assert actual_metrics.keys() == expected_metrics.keys()
    for key in actual_metrics:
        assert actual_metrics[key] == pytest.approx(
            expected_metrics[key], abs=1e-6, rel=1e-5
        )
    _assert_nested_close(optimized.state_dict(), reference.state_dict())


def test_nonfinal_utd_update_skips_diagnostic_gradient_reductions(monkeypatch):
    agent = XQCAgent(2, 1, _config())

    def fail_if_called(_parameters):
        raise AssertionError("non-final UTD updates must not compute grad norms")

    monkeypatch.setattr(xqc_core, "_global_grad_norm", fail_if_called)
    assert agent._update_once(
        _batch(),
        next_noise=torch.zeros((4, 1)),
        actor_noise=torch.zeros((4, 1)),
        collect_metrics=False,
    ) == {}


def test_debug_batch_validation_is_strict_and_production_path_has_no_finite_fence():
    agent = XQCAgent(2, 1, _config())
    batch = _batch()
    batch["rewards"][0] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        agent._prepared_batch(batch, validate_finite=True)
    prepared = agent._prepared_batch(batch, validate_finite=False)
    assert torch.isnan(prepared["rewards"][0])

    checked = XQCAgent(2, 1, _config(debug_checks=True))
    before = copy.deepcopy(checked.state_dict())
    replay = _JoinedReplay((batch, batch))
    with pytest.raises(ValueError, match="finite"):
        checked.update(replay, gradient_steps=2, batch_size=4)
    _assert_nested_close(checked.state_dict(), before)


def test_optimizer_backend_contract_is_explicit_and_cpu_reference_stays_single_tensor():
    automatic = XQCAgent(2, 1, _config(optimizer_backend="auto"))
    explicit = XQCAgent(2, 1, _config(optimizer_backend="single_tensor"))
    assert automatic.actor_optimizer.defaults["foreach"] is False
    assert explicit.actor_optimizer.defaults["foreach"] is False
    assert automatic.actor_optimizer.defaults["fused"] is None

    with pytest.raises(ValueError, match="requires CUDA"):
        XQCAgent(2, 1, _config(optimizer_backend="fused"))
    with pytest.raises(ValueError, match="optimizer_backend"):
        _config(optimizer_backend="not-a-backend")


def test_compile_region_is_lazy_and_reuses_one_fixed_shape_graph(monkeypatch):
    buffer = torch.zeros(())
    compile_calls = []

    def eager(value):
        buffer.add_(value)
        return value * 2.0

    def fake_compile(function, **kwargs):
        compile_calls.append(kwargs)
        return function

    monkeypatch.setattr(torch, "compile", fake_compile)
    region = xqc_core._MutableCompileRegion(
        "test region", eager, (buffer,), enabled=True, strict=False
    )
    assert region(torch.tensor(1.0)) == 2.0
    assert region(torch.tensor(2.0)) == 4.0
    assert buffer == 3.0
    assert compile_calls == [{"fullgraph": False, "dynamic": False}]


def test_compile_strict_changes_failure_policy_not_graph_mode(monkeypatch):
    compile_calls = []

    def fake_compile(function, **kwargs):
        compile_calls.append(kwargs)
        return function

    monkeypatch.setattr(torch, "compile", fake_compile)
    region = xqc_core._MutableCompileRegion(
        "strict region", lambda value: value + 1, (), enabled=True, strict=True
    )
    assert region(torch.tensor(2.0)) == 3.0
    assert compile_calls == [{"fullgraph": False, "dynamic": False}]


def test_compile_first_call_failure_restores_bn_state_before_eager_fallback(monkeypatch):
    buffer = torch.zeros(())

    def eager(value):
        buffer.add_(value)
        return value * 2.0

    def fake_compile(_function, **_kwargs):
        def broken(value):
            buffer.add_(100.0 * value)
            raise RuntimeError("injected compiler failure")

        return broken

    monkeypatch.setattr(torch, "compile", fake_compile)
    region = xqc_core._MutableCompileRegion(
        "test region", eager, (buffer,), enabled=True, strict=False
    )
    with pytest.warns(RuntimeWarning, match="Falling back"):
        assert region(torch.tensor(1.0)) == 2.0
    assert buffer == 1.0
    assert region.failed is True
    assert region.enabled is False


@pytest.mark.parametrize("field", ("debug_checks", "compile", "compile_strict"))
def test_performance_boolean_options_are_strict(field):
    with pytest.raises(ValueError, match=field):
        _config(**{field: 1})


def test_update_noise_batch_preserves_cpu_rng_sequence():
    agent = XQCAgent(2, 1, _config())
    reference = torch.Generator(device="cpu")
    reference.set_state(agent.generator.get_state())

    expected = torch.stack(
        [
            torch.stack(
                [
                    torch.randn((4, 1), generator=reference),
                    torch.randn((4, 1), generator=reference),
                ]
            )
            for _ in range(2)
        ]
    )
    actual = agent._sample_update_noises(2, 4)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(agent.generator.get_state(), reference.get_state())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_compiled_regions_match_eager_update_state():
    eager = XQCAgent(
        2,
        1,
        _config(device="cuda", compile=False, optimizer_backend="single_tensor"),
    )
    compiled = XQCAgent(
        2,
        1,
        _config(device="cuda", compile=True, optimizer_backend="single_tensor"),
    )
    batch = {key: value.cuda() for key, value in _batch().items()}
    noise = torch.zeros((4, 1), device="cuda")

    for _ in range(4):
        eager_metrics = eager._update_once(
            batch, next_noise=noise, actor_noise=noise
        )
        compiled_metrics = compiled._update_once(
            batch, next_noise=noise, actor_noise=noise
        )

    assert compiled._critic_loss_region._compiled is not None
    assert compiled._actor_loss_region._compiled is not None
    assert compiled._critic_loss_region.failed is False
    assert compiled._actor_loss_region.failed is False
    for key in eager_metrics:
        assert compiled_metrics[key] == pytest.approx(
            eager_metrics[key], abs=1e-5, rel=1e-4
        )
    for eager_module, compiled_module in (
        (eager.actor, compiled.actor),
        (eager.critic, compiled.critic),
        (eager.critic_target, compiled.critic_target),
    ):
        for key, value in eager_module.state_dict().items():
            torch.testing.assert_close(
                compiled_module.state_dict()[key], value, atol=1e-5, rtol=1e-4
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_fused_adam_matches_single_tensor_within_optimizer_tolerance():
    reference = XQCAgent(
        2,
        1,
        _config(device="cuda", compile=False, optimizer_backend="single_tensor"),
    )
    fused = XQCAgent(
        2,
        1,
        _config(device="cuda", compile=False, optimizer_backend="fused"),
    )
    batch = {key: value.cuda() for key, value in _batch().items()}
    noise = torch.zeros((4, 1), device="cuda")

    for _ in range(4):
        reference._update_once(batch, next_noise=noise, actor_noise=noise)
        fused._update_once(batch, next_noise=noise, actor_noise=noise)

    assert fused.critic_optimizer.defaults["fused"] is True
    for reference_module, fused_module in (
        (reference.actor, fused.actor),
        (reference.critic, fused.critic),
        (reference.critic_target, fused.critic_target),
    ):
        for key, value in reference_module.state_dict().items():
            torch.testing.assert_close(
                fused_module.state_dict()[key], value, atol=1e-4, rtol=1e-4
            )
