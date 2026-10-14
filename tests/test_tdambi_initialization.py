"""Native TDAMBI preserves its update equations with scratch and critic LoRA."""

from copy import deepcopy
import random
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import pytest
import torch

from RL.TDAMBI import TDAMBI
from RL.tdmpc2_core.common import math as td_math
from RL.tdmpc2_core.common.inner_utils import InnerRNG
from RL.tdmpc2_core.common.lora import (
    LoRARLLinear, dense_lora_rl_critic, lora_rl_parameter_groups,
)
from RL.tdmpc2_core.common.scale import RunningScale
from RL.tdmpc2_core.common.world_model import WorldModel
from tests.test_ambi_latency_contract import _assert_tree_equal, _optimizer_tensor_pointers
from tests.test_tdambi_checkpoint import make_tdambi, pair, tiny_native_params
from tests.test_tdambi_inner import _assert_close_tree, _same_random_normal, _snapshot


INITIALIZATIONS = [(actor, critic) for actor in ("prior", "random")
                   for critic in ("prior", "random")]


def _native_cfg(**overrides):
    instance = object.__new__(TDAMBI)
    instance.env = gym.make("Pendulum-v1", max_episode_steps=5)
    instance.run_params = {"seed": 3, "device": "cpu"}
    try:
        return instance._build_cfg(tiny_native_params(**{"inner_rollout_horizon": 2, **overrides}))
    finally:
        instance.env.close()


@pytest.fixture
def loaded_native(pair):
    native, _ = pair
    native.agent.scale.value.fill_(37.5)
    checkpoint = deepcopy(native.agent.checkpoint_state())
    models = []

    def make(*, checkpoint_scale=True, **overrides):
        params = {"inner_updates_per_round": 3, **overrides}
        if str(params.get("inner_critic_adaptation", "clone")).lower() == "lora_rl":
            params.setdefault("inner_critic_lora_rank", 4)
        model = make_tdambi(**params)
        models.append(model)
        payload = deepcopy(checkpoint)
        if not checkpoint_scale:
            del payload["scale"]
        model.agent.load(payload)
        return model

    yield make
    for model in models:
        model.close()
        model.env.close()


def _prepare(engine, *, t0=True):
    with engine.rng.fork("initialization"):
        engine._prepare_workspace(t0=t0)
    return engine.state


def _dense(critic):
    return (dense_lora_rl_critic(critic)
            if any(isinstance(layer, LoRARLLinear) for layer in critic.modules())
            else critic)


@pytest.mark.parametrize("actor,critic", INITIALIZATIONS)
@pytest.mark.parametrize("adaptation", ["clone", "lora_rl"])
def test_native_initialization_choices_and_target_defaults(actor, critic, adaptation):
    params = dict(inner_actor_initialization=actor.upper(), inner_critic_initialization=critic.upper(),
                  inner_critic_adaptation=adaptation.upper())
    if adaptation == "lora_rl":
        params["inner_critic_lora_rank"] = 4
    cfg = _native_cfg(**params)
    assert cfg.inner_operator == "tdambi"
    assert cfg.inner_actor_initialization == actor
    assert cfg.inner_critic_initialization == critic
    assert cfg.inner_actor_adaptation == "clone"
    assert cfg.inner_critic_adaptation == adaptation
    expected_target = "online" if critic == "random" or adaptation == "lora_rl" else "outer_target"
    assert cfg.inner_critic_target_initialization == expected_target
    assert cfg.inner_temperature_updates_per_action == 0
    assert cfg.inner_critic_dropout_enabled
    assert cfg.inner_actor_adam_eps == 1e-5
    assert cfg.inner_adam_eps == 1e-8
    assert cfg.tdambi_value_coef == cfg.value_coef
    assert cfg.tdambi_entropy_coef == cfg.entropy_coef
    assert cfg.tdambi_scale_tau == cfg.tau
    if expected_target == "online":
        with pytest.raises(ValueError, match="TDAMBI.*target|target.*initialization"):
            _native_cfg(**params, inner_critic_target_initialization="outer_target")
    else:
        # Historical resolved native configs said online although the native
        # engine always used the saved target. Preserve their effective behavior.
        legacy = _native_cfg(**params, inner_critic_target_initialization="online")
        assert legacy.inner_critic_target_initialization == "outer_target"


@pytest.mark.parametrize("actor,critic", INITIALIZATIONS)
@pytest.mark.parametrize("adaptation", ["clone", "lora_rl"])
def test_loaded_native_backbone_is_frozen_and_inner_initialization_is_independent(
    loaded_native, actor, critic, adaptation,
):
    model = loaded_native(inner_actor_initialization=actor, inner_critic_initialization=critic,
                          inner_critic_adaptation=adaptation)
    outer, engine = model.agent.model, model.agent.inner_engine
    before = deepcopy(outer.state_dict())
    global_rng = torch.random.get_rng_state().clone()
    state = _prepare(engine)
    for local, prior, mode in ((state.actor, outer._pi, actor), (_dense(state.critic), outer._Qs, critic)):
        assert local is not prior
        if mode == "prior":
            _assert_tree_equal(local.state_dict(), prior.state_dict())
        else:
            assert any(not torch.equal(value, prior.state_dict()[name])
                       for name, value in local.state_dict().items())
            for layer in local.modules():
                if isinstance(layer, torch.nn.LayerNorm):
                    assert torch.all(layer.weight == 1)
                    assert not torch.count_nonzero(layer.bias)
    expected = (outer._target_Qs if critic == "prior" and adaptation == "clone"
                else _dense(state.critic))
    _assert_tree_equal(state.critic_target.state_dict(), expected.state_dict())
    if critic == "prior" and adaptation == "clone":
        assert not torch.equal(state.critic_target[0][-1].bias, state.critic[0][-1].bias)
    if critic == "random":
        assert all(not torch.count_nonzero(head[-1].weight) for head in state.critic)
    assert all(parameter.requires_grad for parameter in state.actor.parameters())
    assert all(not parameter.requires_grad for parameter in outer.parameters())
    assert not state.critic_target.training
    assert not any(isinstance(layer, LoRARLLinear) for layer in state.critic_target.modules())
    assert state.log_alpha is state.alpha_fixed is state.temperature_optim is None
    torch.testing.assert_close(state.tdambi_scale, torch.tensor([37.5]), rtol=0, atol=0)
    assert state.tdambi_scale.data_ptr() != model.agent.tdambi_checkpoint_scale.data_ptr()
    _assert_tree_equal(outer.state_dict(), before)
    torch.testing.assert_close(torch.random.get_rng_state(), global_rng, rtol=0, atol=0)


@pytest.mark.parametrize("adaptation,placement", [("clone", None), ("lora_rl", "input_hidden"),
                                                   ("lora_rl", "hidden")])
def test_native_scratch_learning_updates_only_allowed_parameters(loaded_native, monkeypatch, adaptation, placement):
    params = dict(inner_actor_initialization="random", inner_critic_initialization="random",
                  inner_critic_adaptation=adaptation, inner_critic_target_tau=1.0)
    if placement is not None:
        params["inner_critic_lora_layers"] = placement
    model = loaded_native(**params)
    outer, engine = model.agent.model, model.agent.inner_engine
    outer_before = deepcopy(outer.state_dict())
    initial = {}
    original_prepare = engine._prepare_workspace

    def capture(**kwargs):
        original_prepare(**kwargs)
        initial.update({name: deepcopy(getattr(engine.state, name).state_dict()) for name in ("actor", "critic")})

    monkeypatch.setattr(engine, "_prepare_workspace", capture)
    global_rng = torch.random.get_rng_state().clone()
    python_rng, numpy_rng = random.getstate(), repr(np.random.get_state())
    model.predict([0.2, -0.3, 0.7], deterministic=True, episode_start=True, collect_diagnostics=False)
    pool = engine._action_pool
    assert engine.state.actor_steps == engine.state.critic_steps == engine.state.critic_target_steps == 3
    assert pool.temperature_optim is pool.log_alpha is pool.alpha_fixed is None
    assert type(pool.actor_optim) is torch.optim.Adam
    assert pool.actor_optim.defaults["eps"] == 1e-5
    assert pool.critic_optim.defaults["eps"] == 1e-8
    assert not any(isinstance(layer, LoRARLLinear) for layer in pool.actor.modules())
    assert any(not torch.equal(value, initial["actor"][name]) for name, value in pool.actor.state_dict().items())
    assert any(not torch.equal(value, initial["critic"][name]) for name, value in pool.critic.state_dict().items())
    if adaptation == "lora_rl":
        assert type(pool.critic_optim) is torch.optim.AdamW
        assert [group["weight_decay"] for group in pool.critic_optim.param_groups] == [2e-4, 0.0]
        for path, layer in pool.critic.named_modules():
            if isinstance(layer, LoRARLLinear):
                torch.testing.assert_close(layer.base.weight, initial["critic"][f"{path}.base.weight"], rtol=0, atol=0)
                assert not layer.base.weight.requires_grad
                assert layer.base.weight.grad is None
                assert layer.base.bias.requires_grad
                assert torch.count_nonzero(layer.lora_B)
                assert all(parameter.requires_grad for parameter in layer.base.ln.parameters())
        assert all(head[-1].weight.requires_grad and head[-1].bias.requires_grad for head in pool.critic)
    else:
        assert type(pool.critic_optim) is torch.optim.Adam
        assert all(parameter.requires_grad for parameter in pool.critic.parameters())
    _assert_tree_equal(pool.critic_target.state_dict(), _dense(pool.critic).state_dict())
    _assert_tree_equal(outer.state_dict(), outer_before)
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in outer.parameters())
    torch.testing.assert_close(torch.random.get_rng_state(), global_rng, rtol=0, atol=0)
    assert random.getstate() == python_rng
    assert repr(np.random.get_state()) == numpy_rng
    assert model.agent.last_inner_metrics["inner_tdambi_scale_from_checkpoint"] == 1
    assert model.agent.last_inner_metrics["inner_tdambi_calibration_samples"] == 0
    torch.testing.assert_close(model.agent.tdambi_checkpoint_scale, torch.tensor([37.5]), rtol=0, atol=0)


@pytest.mark.parametrize("actor,critic", INITIALIZATIONS)
@pytest.mark.parametrize("adaptation", ["clone", "lora_rl"])
def test_native_cold_and_reused_workspaces_match_with_isolated_random_draws(loaded_native, actor, critic, adaptation):
    models = [loaded_native(inner_actor_initialization=actor, inner_critic_initialization=critic,
                             inner_critic_adaptation=adaptation) for _ in range(2)]
    discarded, reused = models
    for model in models:
        model.predict([0.0, 0.0, 0.0], deterministic=True, collect_diagnostics=False)
    pool = reused.agent.inner_engine._action_pool
    objects = {name: getattr(pool, name) for name in ("actor", "critic", "critic_target", "actor_optim", "critic_optim", "replay")}
    moments = tuple(_optimizer_tensor_pointers(getattr(pool, name)) for name in ("actor_optim", "critic_optim"))
    scale_pointer, replay_pointer = pool.tdambi_scale.data_ptr(), pool.replay._storage.data_ptr()
    previous = None
    for seed in (901, 902):
        discarded.agent.inner_engine.reset_for_evaluation(seed)
        reused.agent.inner_engine.reset_for_evaluation(seed, reuse_action_pool=True)
        for decision in range(2):
            expected, _ = discarded.predict([0.2, -0.3, 0.7], episode_start=decision == 0, collect_diagnostics=False)
            actual, _ = reused.predict([0.2, -0.3, 0.7], episode_start=decision == 0, collect_diagnostics=False)
            np.testing.assert_array_equal(actual, expected)
            _assert_tree_equal(_snapshot(reused), _snapshot(discarded))
            for name, value in objects.items():
                assert getattr(pool, name) is value
            assert tuple(_optimizer_tensor_pointers(getattr(pool, name)) for name in ("actor_optim", "critic_optim")) == moments
            assert pool.tdambi_scale.data_ptr() == scale_pointer
            assert pool.replay._storage.data_ptr() == replay_pointer
            assert reused.agent.last_inner_metrics["inner_tdambi_q_scale_initial"] == pytest.approx(37.5)
            current = {name: deepcopy(getattr(pool, name).state_dict()) for name in ("actor", "critic")}
            if previous is not None:
                for name, mode in (("actor", actor), ("critic", critic)):
                    if mode == "random":
                        assert any(not torch.equal(value, previous[name][key]) for key, value in current[name].items())
            previous = current


@pytest.mark.parametrize("adaptation", ["clone", "lora_rl"])
def test_native_scratch_updates_preserve_native_losses_scale_and_target_equations(loaded_native, adaptation):
    model = loaded_native(inner_actor_initialization="random", inner_critic_initialization="random",
                          inner_critic_adaptation=adaptation)
    engine, cfg = model.agent.inner_engine, model.cfg
    state = _prepare(engine)
    with engine.rng.action_fork():
        root = model.agent.model.encode(torch.tensor([[0.2, -0.3, 0.7]])).detach()
        engine._collect_round(root)
        batch = engine._sample_batch()
    native = WorldModel(deepcopy(cfg))
    native.load_state_dict(model.agent.model.state_dict(), strict=True)
    native._pi = deepcopy(state.actor)
    native._Qs = deepcopy(state.critic)
    native._target_Qs = deepcopy(state.critic_target)
    native.train()
    scale_cfg = deepcopy(cfg)
    scale_cfg.tau = cfg.tdambi_scale_tau
    scale = RunningScale(scale_cfg)
    scale.value.copy_(state.tdambi_scale)
    if adaptation == "lora_rl":
        critic_optim = torch.optim.AdamW(lora_rl_parameter_groups(native._Qs, cfg.inner_critic_lora_weight_decay),
                                        lr=cfg.inner_critic_lr, eps=1e-8, foreach=False)
    else:
        critic_optim = torch.optim.Adam(native._Qs.parameters(), lr=cfg.inner_critic_lr, eps=1e-8, foreach=False)
    actor_optim = torch.optim.Adam(native._pi.parameters(), lr=cfg.inner_actor_lr, eps=1e-5, foreach=False)
    oracle_rng = InnerRNG(cfg.seed, "cpu", extra_streams=("tdambi_calibration",))
    oracle_rng.load_training_state_dict(engine.rng.training_state_dict())
    for _ in range(3):
        with oracle_rng.action_fork():
            with oracle_rng.fork("bootstrap") as generator:
                with torch.no_grad(), patch("torch.randn_like", _same_random_normal(generator)):
                    next_action, _ = native.pi(batch["next_z"], None)
                    target = batch["reward"] + model.agent.discount * (1 - batch["terminated"]) * native.Q(
                        batch["next_z"], next_action, None, return_type="min", target=True,
                    )
                predictions = native.Q(batch["z"], batch["action"], None, return_type="all")
                critic_loss = cfg.tdambi_value_coef * torch.stack([
                    td_math.soft_ce(head, target, cfg).mean() for head in predictions
                ]).mean()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in native._Qs.parameters() if p.requires_grad], cfg.inner_critic_grad_clip_norm)
                critic_optim.step()
                critic_optim.zero_grad(set_to_none=True)
            with oracle_rng.fork("gradient_policy") as generator:
                with patch("torch.randn_like", _same_random_normal(generator)):
                    action, info = native.pi(batch["z"], None)
                q = native.Q(batch["z"], action, None, return_type="avg", detach=True)
                scale.update(q)
                actor_loss = -(scale(q) + cfg.tdambi_entropy_coef * info["scaled_entropy"]).mean()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(native._pi.parameters(), cfg.inner_actor_grad_clip_norm)
                actor_optim.step()
                actor_optim.zero_grad(set_to_none=True)
            with torch.no_grad():
                effective = _dense(native._Qs).state_dict()
                for name, value in native._target_Qs.state_dict().items():
                    value.lerp_(effective[name], cfg.inner_critic_target_tau)
        with engine.rng.action_fork():
            with engine.rng.fork("bootstrap"):
                critic_metrics = engine._tdambi_critic_step(batch)
            with engine.rng.fork("gradient_policy"):
                actor_metrics = engine._tdambi_actor_step(batch)
            engine._maybe_update_targets(critic_updated=True, actor_updated=True)
        torch.testing.assert_close(critic_metrics["critic_loss"], critic_loss)
        torch.testing.assert_close(actor_metrics["actor_loss"], actor_loss)
        torch.testing.assert_close(state.tdambi_scale, scale.value)
        for actual, expected in ((state.actor, native._pi), (state.critic, native._Qs),
                                 (state.critic_target, native._target_Qs),
                                 (state.actor_optim, actor_optim), (state.critic_optim, critic_optim)):
            _assert_close_tree(actual.state_dict(), expected.state_dict())
        _assert_tree_equal(engine.rng.training_state_dict(), oracle_rng.training_state_dict())
    assert state.actor_steps == state.critic_steps == state.critic_target_steps == 3
    assert state.temperature_optim is None


@pytest.mark.parametrize("adaptation", ["clone", "lora_rl"])
@pytest.mark.parametrize("mode,saved", [("checkpoint_or_calibrate", True), ("checkpoint", True),
                                       ("calibrate", True), ("checkpoint_or_calibrate", False),
                                       ("calibrate", False)])
def test_native_scratch_preserves_saved_and_frozen_prior_scale_initialization(
    loaded_native, monkeypatch, adaptation, mode, saved,
):
    model = loaded_native(inner_actor_initialization="random", inner_critic_initialization="random",
                          inner_critic_adaptation=adaptation, tdambi_scale_initialization=mode,
                          checkpoint_scale=saved)
    engine, outer = model.agent.inner_engine, model.agent.model
    calibrating = False
    calls = []
    original_calibrate = engine._calibrate_tdambi_scale
    original_policy = outer.pi_action
    original_values = outer.q_values

    def calibrate():
        nonlocal calibrating
        calibrating = True
        try:
            return original_calibrate()
        finally:
            calibrating = False

    def prior_policy(*args, **kwargs):
        if calibrating:
            assert "policy" not in kwargs
            assert not torch.is_grad_enabled()
            assert not outer._pi.training
            calls.append("frozen_prior_actor")
        return original_policy(*args, **kwargs)

    def prior_values(*args, **kwargs):
        if calibrating:
            assert not kwargs
            assert not torch.is_grad_enabled()
            assert not outer._Qs.training
            calls.append("frozen_online_prior_critic")
        return original_values(*args, **kwargs)

    monkeypatch.setattr(engine, "_calibrate_tdambi_scale", calibrate)
    monkeypatch.setattr(outer, "pi_action", prior_policy)
    monkeypatch.setattr(outer, "q_values", prior_values)
    model.predict([0.2, -0.3, 0.7], deterministic=True, collect_diagnostics=False)
    metrics = model.agent.last_inner_metrics
    should_calibrate = mode == "calibrate" or not saved
    assert model.cfg.tdambi_scale_initialization == mode
    if should_calibrate:
        assert calls == ["frozen_prior_actor", "frozen_online_prior_critic"]
        assert metrics["inner_tdambi_calibration_samples"] == model.cfg.inner_batch_size
        # The fixture's frozen online value heads have zero kernels, so their
        # prior Q range is zero and native calibration applies its unit floor.
        assert metrics["inner_tdambi_q_scale_initial"] == pytest.approx(1.0)
        assert model.agent.tdambi_scale_source == "first_collection_calibration"
    else:
        assert not calls
        assert metrics["inner_tdambi_calibration_samples"] == 0
        assert metrics["inner_tdambi_q_scale_initial"] == pytest.approx(37.5)
        assert model.agent.tdambi_scale_source == "checkpoint"
    assert metrics["inner_tdambi_scale_from_checkpoint"] == int(not should_calibrate)
    if saved:
        torch.testing.assert_close(model.agent.tdambi_checkpoint_scale, torch.tensor([37.5]), rtol=0, atol=0)
    else:
        assert model.agent.tdambi_checkpoint_scale is None


@pytest.mark.parametrize("adaptation", ["clone", "lora_rl"])
def test_native_scratch_checkpoint_scale_requirement_still_rejects_legacy_snapshots(loaded_native, adaptation):
    with pytest.raises(ValueError, match="requires a saved Q scale"):
        loaded_native(inner_actor_initialization="random", inner_critic_initialization="random",
                      inner_critic_adaptation=adaptation, tdambi_scale_initialization="checkpoint",
                      checkpoint_scale=False)
