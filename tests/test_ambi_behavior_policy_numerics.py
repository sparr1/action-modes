"""Regression coverage for finite KL gradients beyond float32 norm/moment range."""

from copy import deepcopy
import json
import math
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from RL.AMBITDMPC2 import AMBITDMPC2
from tests.test_ambi_behavior_policy_kl import _kl_model
from tests.test_ambi_inner_decoupling import _assert_tree_equal


ROOT = Path(__file__).resolve().parents[1]
DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
    ),
]


class _HumanoidSpaces(gym.Env):
    observation_space = gym.spaces.Box(
        -np.inf, np.inf, shape=(67,), dtype=np.float32
    )
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(21,), dtype=np.float32)
    spec = gym.envs.registration.EnvSpec("Numerics-v0", max_episode_steps=500)


def _full_size_model(schedule, *, objective="reverse_kl", device="cpu"):
    config = json.loads(
        (ROOT / f"configs/dmcontrol/algs/ambi_anchor_kl_{schedule}.json").read_text()
    )
    params = {
        **config["alg_params"],
        "device": device,
        "compile": False,
        "wandb": False,
        "eval_freq": None,
        "eval_value": False,
        "eval_inner_comparison": False,
        "outer_behavior_policy_objective": objective,
    }
    return AMBITDMPC2(
        "AMBITDMPC2",
        _HumanoidSpaces(),
        params,
        {**config, "device": device},
        {},
    )


def _actor_batch(agent, *, mean, log_std):
    horizon, batch = agent.cfg.train_unroll_horizon, agent.cfg.batch_size
    action_dim = agent.cfg.action_dim
    with torch.no_grad():
        agent.model._pi[-1].weight.zero_()
        agent.model._pi[-1].bias[:action_dim].fill_(mean)
        agent.model._pi[-1].bias[action_dim:].fill_(log_std)
        zs = agent.model.encode(
            torch.randn(horizon + 1, batch, 67, device=agent.device)
        ).detach()
    behavior_mean = torch.zeros(horizon, batch, action_dim, device=agent.device)
    behavior_log_std = torch.full_like(behavior_mean, -20.0)
    valid = torch.ones(horizon, batch, 1, dtype=torch.bool, device=agent.device)
    return zs, behavior_mean, behavior_log_std, valid


@pytest.mark.parametrize("objective", ["reverse_kl", "action_space_cross_entropy"])
@pytest.mark.parametrize("device", DEVICES)
def test_large_finite_behavior_gradient_matches_double_precision_clipping(
    objective, device
):
    torch.manual_seed(55)
    wrapper = _full_size_model("smooth", objective=objective, device=device)
    try:
        agent = wrapper.agent
        agent.behavior_policy_kl_eligible_updates = (
            agent.cfg.outer_behavior_policy_kl_ramp_updates
        )
        # Strictly inside the configured log-std bounds, not at a clamp kink.
        args = _actor_batch(agent, mean=0.0, log_std=1.9)
        parameters = list(agent.model._pi.parameters())
        before = [parameter.detach().clone() for parameter in parameters]
        raw_gradients = {}

        def capture(index):
            def hook(gradient):
                raw_gradients[index] = gradient.detach().clone()

            return hook

        handles = [p.register_hook(capture(i)) for i, p in enumerate(parameters)]
        try:
            metrics = agent._update_actor(*args)
        finally:
            for handle in handles:
                handle.remove()

        assert torch.isfinite(metrics["actor_loss"])
        assert all(torch.isfinite(g).all() for g in raw_gradients.values())
        reference = [
            torch.nn.Parameter(torch.zeros_like(g, dtype=torch.float64))
            for g in raw_gradients.values()
        ]
        for parameter, gradient in zip(reference, raw_gradients.values()):
            parameter.grad = gradient.double()
        expected_norm = torch.nn.utils.clip_grad_norm_(
            reference, agent.cfg.grad_clip_norm
        )
        assert expected_norm > torch.finfo(torch.float32).max ** 0.5
        assert torch.isfinite(metrics["actor_grad_norm"])
        assert float(metrics["actor_grad_norm"]) == pytest.approx(float(expected_norm))
        for index, expected in zip(raw_gradients, reference):
            torch.testing.assert_close(
                parameters[index].grad, expected.grad.float(), rtol=2e-6, atol=1e-7
            )
        assert any(not torch.equal(old, p) for old, p in zip(before, parameters))
        assert all(p.dtype == torch.float32 for p in agent.model.parameters())
        assert metrics["actor_loss"].dtype == torch.float32
    finally:
        wrapper.close()


@pytest.mark.parametrize("device", DEVICES)
def test_large_dual_violation_keeps_adam_finite_and_matches_float64_reference(device):
    torch.manual_seed(55)
    wrapper = _full_size_model("dual", device=device)
    try:
        agent = wrapper.agent
        args = _actor_batch(agent, mean=100.0, log_std=-2.0)
        coefficient = agent.log_behavior_policy_kl_coef
        reference = torch.nn.Parameter(coefficient.detach().double().clone())
        reference_optim = torch.optim.Adam(
            [reference],
            lr=agent.cfg.outer_behavior_policy_kl_dual_lr,
            eps=agent.cfg.adam_eps,
            capturable=agent.device.type == "cuda",
            foreach=agent.device.type == "cuda",
        )
        for step in range(3):
            if step:
                # Resume ordinary violations after the outlier. Infinite Adam
                # moments in the former implementation never recovered here.
                with torch.no_grad():
                    stats = agent.model.policy_stats(args[0][:-1])
                args = (args[0], stats["pre_tanh_mean"], stats["log_std"], args[3])
            before = coefficient.detach().clone()
            metrics = agent._update_actor(*args)
            violation = metrics["behavior_policy_kl_dual_violation"].double()
            reference_optim.zero_grad(set_to_none=True)
            (-reference * violation).mean().backward()
            reference_optim.step()
            state = agent.behavior_policy_kl_optim.state[coefficient]
            assert all(torch.isfinite(value).all() for value in state.values())
            assert not torch.equal(before, coefficient)
            torch.testing.assert_close(coefficient, reference, rtol=1e-12, atol=1e-12)
            assert metrics["actor_loss"].dtype == torch.float32
    finally:
        wrapper.close()


def _real_update(agent):
    horizon, batch = agent.cfg.train_unroll_horizon, agent.cfg.batch_size
    actions = torch.zeros(horizon, batch, agent.cfg.action_dim)
    agent._update(
        torch.randn(horizon + 1, batch, 3), actions,
        torch.zeros(horizon, batch, 1), torch.zeros(horizon, batch, 1),
        torch.zeros_like(actions), torch.zeros_like(actions),
        torch.ones(horizon, batch, 1, dtype=torch.bool),
    )


def test_double_precision_dual_roundtrips_exact_state_and_next_update():
    source = _kl_model("dual").agent
    _real_update(source)
    saved = deepcopy(source.training_state_dict())
    restored = _kl_model("dual").agent
    restored.load_training_state_dict(saved)
    _assert_tree_equal(restored.training_state_dict(), saved)
    rng = torch.random.get_rng_state()
    _real_update(source)
    torch.random.set_rng_state(rng)
    _real_update(restored)
    _assert_tree_equal(source.training_state_dict(), restored.training_state_dict())


def _legacy_dual_state(agent):
    saved = deepcopy(agent.checkpoint_state())
    dual = saved["behavior_policy_kl_state"]
    dual["log_coef"] = dual["log_coef"].float()
    for state in dual["optim"]["state"].values():
        for field in ("exp_avg", "exp_avg_sq"):
            state[field] = state[field].float()
    return saved


@pytest.mark.parametrize("endpoint", [None, "minimum", "maximum"])
def test_portable_legacy_dual_promotes_finite_state_without_mutating_checkpoint(
    endpoint,
):
    source = _kl_model("dual").agent
    _real_update(source)
    legacy = _legacy_dual_state(source)
    if endpoint is not None:
        bound = 1e-8 if endpoint == "minimum" else source.cfg.outer_behavior_policy_kl_dual_max
        legacy["behavior_policy_kl_state"]["log_coef"].fill_(math.log(bound))
    before = deepcopy(legacy)
    restored = _kl_model("dual").agent
    restored.load(legacy)
    coefficient = restored.log_behavior_policy_kl_coef
    assert coefficient.dtype == torch.float64
    torch.testing.assert_close(
        coefficient,
        legacy["behavior_policy_kl_state"]["log_coef"].double().clamp(
            min=math.log(1e-8), max=math.log(source.cfg.outer_behavior_policy_kl_dual_max)
        ),
    )
    optim_state = restored.behavior_policy_kl_optim.state[coefficient]
    for field in ("exp_avg", "exp_avg_sq"):
        assert optim_state[field].dtype == torch.float64
    _assert_tree_equal(legacy, before)
    _real_update(restored)


def test_exact_resume_rejects_legacy_dual_precision_before_mutation():
    source = _kl_model("dual").agent
    _real_update(source)
    legacy = deepcopy(source.training_state_dict())
    legacy["outer"] = _legacy_dual_state(source)
    target = _kl_model("dual").agent
    before = deepcopy(target.training_state_dict())
    with pytest.raises(ValueError, match="dtype"):
        target.load_training_state_dict(legacy)
    _assert_tree_equal(target.training_state_dict(), before)


@pytest.mark.parametrize("legacy", [False, True])
def test_nonfinite_dual_moments_are_rejected_before_any_load_mutation(legacy):
    source = _kl_model("dual").agent
    _real_update(source)
    saved = _legacy_dual_state(source) if legacy else deepcopy(source.checkpoint_state())
    for state in saved["behavior_policy_kl_state"]["optim"]["state"].values():
        state["exp_avg_sq"].fill_(float("inf"))
    target = _kl_model("dual").agent
    before = deepcopy(target.checkpoint_state())
    with pytest.raises(ValueError, match="finite"):
        target.load(saved)
    _assert_tree_equal(target.checkpoint_state(), before)
