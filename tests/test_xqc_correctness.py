import copy
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from RL.xqc_core import (
    BN_EPSILON,
    BN_MOMENTUM,
    OFFICIAL_XQC_COMMIT,
    DiscountedReturnNormalizer,
    FlaxBatchNorm1d,
    XQCActor,
    XQCAgent,
    XQCConfig,
    XQCTwinCritic,
    categorical_cross_entropy,
    categorical_td_projection,
    linear_learning_rate,
    polyak_update_parameters,
    project_unit_rows_,
    select_lower_distribution,
)


FIXTURE_PATH = (
    Path(__file__).parent / "fixtures/xqc_official_9a6832b.json"
)
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _test_device():
    requested = os.environ.get("AMBI_XQC_TEST_DEVICE", "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        if os.environ.get("AMBI_XQC_REQUIRE_CUDA") == "1":
            pytest.fail("AMBI_XQC_REQUIRE_CUDA=1 but CUDA is unavailable")
        pytest.skip("CUDA is unavailable")
    return torch.device(requested)


def _tensor(value, *, device=None, requires_grad=False):
    return torch.tensor(
        value,
        dtype=torch.float32,
        device=_test_device() if device is None else device,
        requires_grad=requires_grad,
    )


def _copy(parameter, value, *, transpose=False):
    tensor = _tensor(value, device=parameter.device)
    if transpose:
        tensor = tensor.T
    with torch.no_grad():
        parameter.copy_(tensor)


def _set_bn(batch_norm, *, scale, bias, mean, var):
    _copy(batch_norm.weight, scale)
    _copy(batch_norm.bias, bias)
    _copy(batch_norm.running_mean, mean)
    _copy(batch_norm.running_var, var)


def _load_fixture_actor():
    data = FIXTURE["actor"]
    actor = XQCActor(2, 1, hidden_dims=(3,)).to(_test_device())
    _set_bn(
        actor.input_batch_norm,
        scale=data["input_bn_scale"],
        bias=data["input_bn_bias"],
        mean=data["input_bn_mean"],
        var=data["input_bn_var"],
    )
    _copy(actor.blocks[0].linear.weight, data["hidden_kernel"], transpose=True)
    _set_bn(
        actor.blocks[0].batch_norm,
        scale=data["hidden_bn_scale"],
        bias=data["hidden_bn_bias"],
        mean=data["hidden_bn_mean"],
        var=data["hidden_bn_var"],
    )
    _copy(actor.mean.weight, data["mean_kernel"], transpose=True)
    _copy(actor.mean.bias, data["mean_bias"])
    _copy(actor.log_std.weight, data["log_std_kernel"], transpose=True)
    _copy(actor.log_std.bias, data["log_std_bias"])
    return actor


def _load_fixture_critic():
    data = FIXTURE["critic"]
    critic = XQCTwinCritic(
        2, 1, hidden_dims=(3,), num_atoms=5, vmin=-2.0, vmax=2.0
    ).to(_test_device())
    for head_index, head in enumerate(critic.q_networks):
        _set_bn(
            head.input_batch_norm,
            scale=data["input_bn_scale"][head_index],
            bias=data["input_bn_bias"][head_index],
            mean=data["input_bn_mean"][head_index],
            var=data["input_bn_var"][head_index],
        )
        _copy(
            head.blocks[0].linear.weight,
            data["hidden_kernel"][head_index],
            transpose=True,
        )
        _set_bn(
            head.blocks[0].batch_norm,
            scale=data["hidden_bn_scale"][head_index],
            bias=data["hidden_bn_bias"][head_index],
            mean=data["hidden_bn_mean"][head_index],
            var=data["hidden_bn_var"][head_index],
        )
        _copy(
            head.value.weight,
            data["output_kernel"][head_index],
            transpose=True,
        )
        _copy(head.value.bias, data["output_bias"][head_index])
    return critic


def _bn_buffers(module):
    return {
        name: value.detach().clone()
        for name, value in module.named_buffers()
        if name.endswith(("running_mean", "running_var"))
    }


def _linear_row_norms(module):
    return [
        torch.linalg.vector_norm(layer.weight.detach(), dim=1)
        for layer in module.modules()
        if isinstance(layer, nn.Linear)
    ]


def _tiny_config(**overrides):
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
        "seed": 7,
        "device": str(_test_device()),
    }
    values.update(overrides)
    return XQCConfig(**values)


def _fixed_batch(device=None):
    device = _test_device() if device is None else device
    return {
        "obs": torch.tensor(
            [[0.2, -0.5], [1.1, 0.3], [-0.7, 0.8], [0.4, -0.9]],
            device=device,
        ),
        "actions": torch.tensor([[0.4], [-0.2], [0.6], [-0.5]], device=device),
        "rewards": torch.tensor([[0.3], [-0.4], [1.2], [0.1]], device=device),
        "next_obs": torch.tensor(
            [[0.3, -0.4], [1.0, 0.2], [-0.5, 0.9], [0.1, -0.6]],
            device=device,
        ),
        "masks": torch.tensor([[1.0], [1.0], [0.0], [1.0]], device=device),
        "discount": torch.tensor([[0.99], [0.99], [0.99], [0.99]], device=device),
    }


def test_fixture_records_the_exact_official_environment():
    assert FIXTURE["metadata"] == {
        "official_commit": OFFICIAL_XQC_COMMIT,
        "jax": "0.4.30",
        "flax": "0.8.4",
        "optax": "0.2.3",
        "dtype": "float32",
    }


def test_flax_batch_norm_uses_population_variance_and_three_state_modes():
    bn = FlaxBatchNorm1d(2).to(_test_device())
    _set_bn(
        bn,
        scale=[1.5, 0.5],
        bias=[-0.25, 0.75],
        mean=[0.4, -0.2],
        var=[1.2, 0.8],
    )
    x = _tensor([[1.0, -1.0], [3.0, 2.0], [-2.0, 0.5]])
    initial_mean = bn.running_mean.clone()
    initial_var = bn.running_var.clone()

    running = bn(x, mode="running")
    expected_running = (x - initial_mean) / torch.sqrt(initial_var + BN_EPSILON)
    expected_running = expected_running * bn.weight + bn.bias
    torch.testing.assert_close(running, expected_running, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(bn.running_mean, initial_mean)

    batch_mean = x.mean(0)
    batch_var = (x.square().mean(0) - batch_mean.square()).clamp_min(0)
    no_update = bn(x, mode="batch_no_update")
    expected_batch = (x - batch_mean) / torch.sqrt(batch_var + BN_EPSILON)
    expected_batch = expected_batch * bn.weight + bn.bias
    torch.testing.assert_close(no_update, expected_batch, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(bn.running_mean, initial_mean)
    torch.testing.assert_close(bn.running_var, initial_var)

    updated = bn(x, mode="batch_update")
    torch.testing.assert_close(updated, expected_batch, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(
        bn.running_mean,
        BN_MOMENTUM * initial_mean + (1.0 - BN_MOMENTUM) * batch_mean,
    )
    torch.testing.assert_close(
        bn.running_var,
        BN_MOMENTUM * initial_var + (1.0 - BN_MOMENTUM) * batch_var,
    )
    with pytest.raises(ValueError, match="mode"):
        bn(x, mode="training")


def test_actor_matches_official_jax_forward_and_fixed_noise_log_probability():
    actor = _load_fixture_actor()
    data = FIXTURE["actor"]
    observations = _tensor(data["observations"])
    noise = _tensor(data["noise"])
    before = _bn_buffers(actor)

    mean, log_std = actor.distribution(observations, bn_mode="running")
    actions, log_probs = actor.sample(
        observations, bn_mode="running", noise=noise
    )
    torch.testing.assert_close(mean, _tensor(data["mean"]), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(
        log_std, _tensor(data["log_std"]), atol=1e-6, rtol=1e-5
    )
    torch.testing.assert_close(
        actions, _tensor(data["actions"]), atol=1e-6, rtol=1e-5
    )
    torch.testing.assert_close(
        log_probs, _tensor(data["log_probs"]), atol=1e-6, rtol=1e-5
    )
    for key, value in _bn_buffers(actor).items():
        torch.testing.assert_close(value, before[key])


def test_twin_critic_matches_official_running_and_batch_stat_forwards():
    critic = _load_fixture_critic()
    data = FIXTURE["critic"]
    observations = _tensor(data["observations"])
    actions = _tensor(data["actions"])

    running_log_probs = critic.log_probs(observations, actions, bn_mode="running")
    running_values = critic.values_from_log_probs(running_log_probs)
    torch.testing.assert_close(
        running_log_probs,
        _tensor(data["running_log_probs"]),
        atol=1e-6,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        running_values, _tensor(data["running_values"]), atol=1e-6, rtol=1e-5
    )

    no_update_before = _bn_buffers(critic)
    no_update = critic.log_probs(observations, actions, bn_mode="batch_no_update")
    torch.testing.assert_close(
        no_update, _tensor(data["batch_log_probs"]), atol=1e-6, rtol=1e-5
    )
    for key, value in _bn_buffers(critic).items():
        torch.testing.assert_close(value, no_update_before[key])

    batch_log_probs = critic.log_probs(observations, actions, bn_mode="batch_update")
    torch.testing.assert_close(
        batch_log_probs,
        _tensor(data["batch_log_probs"]),
        atol=1e-6,
        rtol=1e-5,
    )
    for index, head in enumerate(critic.q_networks):
        torch.testing.assert_close(
            head.input_batch_norm.running_mean,
            _tensor(data["updated_input_mean"][index]),
            atol=1e-6,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            head.input_batch_norm.running_var,
            _tensor(data["updated_input_var"][index]),
            atol=1e-6,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            head.blocks[0].batch_norm.running_mean,
            _tensor(data["updated_hidden_mean"][index]),
            atol=1e-6,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            head.blocks[0].batch_norm.running_var,
            _tensor(data["updated_hidden_var"][index]),
            atol=1e-6,
            rtol=1e-5,
        )


def test_default_architecture_shapes_initialization_and_independent_heads():
    actor = XQCActor(6, 2)
    critic = XQCTwinCritic(6, 2)
    actor_linears = [module for module in actor.modules() if isinstance(module, nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in actor_linears] == [
        (6, 256),
        (256, 256),
        (256, 256),
        (256, 256),
        (256, 2),
        (256, 2),
    ]
    for block in actor.blocks:
        assert block.linear.bias is None
    for head in critic.q_networks:
        linears = [module for module in head.modules() if isinstance(module, nn.Linear)]
        assert [(layer.in_features, layer.out_features) for layer in linears] == [
            (8, 512),
            (512, 512),
            (512, 512),
            (512, 512),
            (512, 101),
        ]
    assert critic.q1.value.weight.data_ptr() != critic.q2.value.weight.data_ptr()
    assert not torch.equal(critic.q1.value.weight, critic.q2.value.weight)


def test_categorical_projection_and_gradient_match_official_jax_fixture():
    data = FIXTURE["categorical"]
    target_log_probs = _tensor(data["target_log_probs"])
    projected, clip_fraction = categorical_td_projection(
        target_log_probs,
        _tensor(data["rewards"]),
        _tensor(data["masks"]),
        data["discount"],
        _tensor(data["entropy_shift"]),
        _tensor(data["support"]),
    )
    torch.testing.assert_close(
        projected, _tensor(data["projected_probs"]), atol=1e-6, rtol=1e-5
    )
    torch.testing.assert_close(
        clip_fraction, _tensor(data["clip_fraction"]), atol=1e-6, rtol=1e-5
    )
    pred_logits = _tensor(data["pred_logits"], requires_grad=True)
    loss = categorical_cross_entropy(F.log_softmax(pred_logits, dim=-1), projected)
    torch.testing.assert_close(
        loss, _tensor(data["cross_entropy"]), atol=1e-6, rtol=1e-5
    )
    loss.backward()
    torch.testing.assert_close(
        pred_logits.grad,
        _tensor(data["pred_logits_gradient"]),
        atol=1e-4,
        rtol=1e-4,
    )


def test_categorical_projection_covers_bins_clipping_terminals_and_entropy_shift():
    support = _tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    source = _tensor([[-math.inf, -math.inf, 0.0, -math.inf, -math.inf]])

    exact, _ = categorical_td_projection(
        source, _tensor([-1.0]), _tensor([0.0]), 0.9, _tensor([0.0]), support
    )
    torch.testing.assert_close(exact, _tensor([[0.0, 1.0, 0.0, 0.0, 0.0]]))

    halfway, _ = categorical_td_projection(
        source, _tensor([0.5]), _tensor([0.0]), 0.9, _tensor([0.0]), support
    )
    torch.testing.assert_close(halfway, _tensor([[0.0, 0.0, 0.5, 0.5, 0.0]]))

    shifted, _ = categorical_td_projection(
        source, _tensor([0.0]), _tensor([1.0]), 1.0, _tensor([-0.5]), support
    )
    torch.testing.assert_close(shifted, halfway)

    clipped, clip_fraction = categorical_td_projection(
        source, _tensor([100.0]), _tensor([1.0]), 0.9, _tensor([0.0]), support
    )
    torch.testing.assert_close(clipped, _tensor([[0.0, 0.0, 0.0, 0.0, 1.0]]))
    torch.testing.assert_close(clip_fraction, _tensor(1.0))
    torch.testing.assert_close(clipped.sum(-1), _tensor([1.0]))


def test_lower_head_selection_keeps_one_complete_distribution_and_ties_to_zero():
    support = _tensor([-1.0, 0.0, 1.0])
    head_zero = _tensor([[0.8, 0.1, 0.1], [0.2, 0.6, 0.2]]).log()
    head_one = _tensor([[0.1, 0.1, 0.8], [0.2, 0.6, 0.2]]).log()
    selected, values, indices = select_lower_distribution(
        torch.stack((head_zero, head_one)), support
    )
    torch.testing.assert_close(selected[0], head_zero[0])
    torch.testing.assert_close(selected[1], head_zero[1])
    assert indices.tolist() == [0, 0]
    torch.testing.assert_close(values, _tensor([-0.7, 0.0]))

    twin_loss = categorical_cross_entropy(
        torch.stack((head_zero, head_one)),
        _tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    )
    expected = -(head_zero[[0, 1], [0, 1]].mean() + head_one[[0, 1], [0, 1]].mean())
    torch.testing.assert_close(twin_loss, expected)


def test_projection_polyak_and_adam_match_official_jax_fixture():
    data = FIXTURE["projection_and_polyak"]
    hidden = nn.Linear(2, 2, bias=False).to(_test_device())
    final = nn.Linear(2, 2, bias=True).to(_test_device())
    _copy(hidden.weight, data["hidden_kernel"], transpose=True)
    _copy(final.weight, data["final_kernel"], transpose=True)
    _copy(final.bias, data["final_bias"])
    module = nn.Sequential(hidden, final)
    project_unit_rows_(module)
    torch.testing.assert_close(
        hidden.weight, _tensor(data["projected_hidden_kernel"]).T
    )
    torch.testing.assert_close(
        final.weight, _tensor(data["projected_final_kernel"]).T
    )
    torch.testing.assert_close(final.bias, _tensor(data["projected_final_bias"]))

    source = nn.Linear(2, 2, bias=False).to(_test_device())
    target = nn.Linear(2, 2, bias=False).to(_test_device())
    _copy(source.weight, data["polyak_source"], transpose=True)
    _copy(target.weight, data["polyak_target"], transpose=True)
    polyak_update_parameters(source, target, data["polyak_tau"])
    torch.testing.assert_close(target.weight, _tensor(data["polyak_result"]).T)

    optimizer_data = FIXTURE["optimizer"]
    parameter = nn.Parameter(_tensor(optimizer_data["initial"]))
    optimizer = torch.optim.AdamW(
        [parameter], lr=3e-4, eps=1e-8, weight_decay=0.0, foreach=False
    )
    for step, gradient in enumerate(optimizer_data["gradients"]):
        optimizer.param_groups[0]["lr"] = optimizer_data["learning_rates"][step]
        parameter.grad = _tensor(gradient)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.testing.assert_close(
            parameter,
            _tensor(optimizer_data["parameters"][step]),
            atol=1e-6,
            rtol=1e-5,
        )

    moments_before = {
        key: value.clone()
        for key, value in optimizer.state[parameter].items()
        if torch.is_tensor(value)
    }
    holder = nn.Module()
    holder.linear = nn.Linear(2, 2, bias=False).to(_test_device())
    holder.linear.weight = parameter
    project_unit_rows_(holder)
    for key, value in moments_before.items():
        torch.testing.assert_close(optimizer.state[parameter][key], value)


def test_reward_normalizer_matches_release_recurrence_boundaries_and_round_trip():
    normalizer = DiscountedReturnNormalizer(gamma=0.9)
    expected_returns = [1.0, 2.9, -1.0, 3.1, 5.0]
    for reward, done, expected in zip(
        [1.0, 2.0, -1.0, 4.0, 5.0],
        [False, False, True, False, True],
        expected_returns,
    ):
        normalizer.update(reward, done)
        assert normalizer.return_accumulator == pytest.approx(expected)
    assert normalizer.count == 5
    rewards = _tensor([1.0, -2.0])
    torch.testing.assert_close(
        normalizer.normalize(rewards), rewards / normalizer.scale
    )

    restored = DiscountedReturnNormalizer(gamma=0.9)
    restored.load_state_dict(normalizer.state_dict())
    assert restored.state_dict() == normalizer.state_dict()
    broken = copy.deepcopy(normalizer.state_dict())
    broken["var"] = -1.0
    before = restored.state_dict()
    with pytest.raises(ValueError, match="variance"):
        restored.load_state_dict(broken)
    assert restored.state_dict() == before


def test_agent_update_order_delay_bn_target_buffers_and_learning_rate_phases():
    agent = XQCAgent(
        2,
        1,
        _tiny_config(
            actor_lr=1e-2,
            critic_lr=1e-2,
            lr_end=1e-3,
            tau=0.5,
            target_update_interval=1,
        ),
    )
    batch = _fixed_batch(agent.device)
    noises = torch.zeros((4, 1), device=agent.device)
    target_buffers = _bn_buffers(agent.critic_target)
    old_target_parameters = {
        name: value.detach().clone()
        for name, value in agent.critic_target.named_parameters()
    }

    # Record the released dataflow: both critic losses see the joined 2B batch,
    # while the actor sees B only after the critic optimizer has stepped.
    online_batch_sizes = []
    target_batch_sizes = []
    actor_logits = []
    policy_log_probs = []
    critic_step_completed = False
    original_critic_step = agent.critic_optimizer.step
    original_actor_sample = agent.actor.sample

    def tracked_critic_step(*args, **kwargs):
        nonlocal critic_step_completed
        result = original_critic_step(*args, **kwargs)
        critic_step_completed = True
        return result

    def track_online(_module, inputs, output):
        online_batch_sizes.append(int(inputs[0].shape[0]))
        if inputs[0].shape[0] == 4:
            assert critic_step_completed
            actor_logits.append(output.detach().clone())

    def track_target(_module, inputs):
        target_batch_sizes.append(int(inputs[0].shape[0]))

    def tracked_actor_sample(*args, **kwargs):
        result = original_actor_sample(*args, **kwargs)
        if kwargs.get("bn_mode") == "batch_update":
            policy_log_probs.append(result[1].detach().clone())
        return result

    agent.critic_optimizer.step = tracked_critic_step
    online_hook = agent.critic.register_forward_hook(track_online)
    target_hook = agent.critic_target.register_forward_pre_hook(track_target)
    agent.actor.sample = tracked_actor_sample
    old_alpha = agent.temperature.detach().clone()

    first = agent._update_once(batch, next_noise=noises, actor_noise=noises)
    online_hook.remove()
    target_hook.remove()
    agent.critic_optimizer.step = original_critic_step
    agent.actor.sample = original_actor_sample
    assert online_batch_sizes == [8, 4]
    assert target_batch_sizes == [8]
    expected_policy_values = agent.critic.values_from_log_probs(
        F.log_softmax(actor_logits[0], dim=-1)
    )
    expected_actor_loss = (
        old_alpha * policy_log_probs[0]
        - expected_policy_values.min(dim=0).values
    ).mean()
    assert first["actor_loss"] == pytest.approx(
        float(expected_actor_loss.cpu()), rel=1e-5, abs=1e-6
    )
    assert first["actor_update_accepted"] == 1.0
    assert agent.update_step == agent.actor_optimizer_steps == 1
    assert agent.temperature_optimizer_steps == 1
    for name, value in agent.critic_target.named_parameters():
        expected = 0.5 * old_target_parameters[name] + 0.5 * dict(
            agent.critic.named_parameters()
        )[name]
        torch.testing.assert_close(value, expected, atol=1e-6, rtol=1e-5)
    for name, value in _bn_buffers(agent.critic_target).items():
        torch.testing.assert_close(value, target_buffers[name])

    actor_parameters = {
        name: value.detach().clone() for name, value in agent.actor.named_parameters()
    }
    actor_buffers = _bn_buffers(agent.actor)
    temperature = agent.temperature.detach().clone()
    second = agent._update_once(batch, next_noise=noises, actor_noise=noises)
    assert second["actor_update_accepted"] == 0.0
    assert agent.update_step == 2
    assert agent.actor_optimizer_steps == agent.temperature_optimizer_steps == 1
    for name, value in agent.actor.named_parameters():
        torch.testing.assert_close(value, actor_parameters[name], atol=1e-6, rtol=1e-5)
    assert any(
        not torch.equal(value, actor_buffers[name])
        for name, value in _bn_buffers(agent.actor).items()
    )
    torch.testing.assert_close(agent.temperature, temperature)

    third = agent._update_once(batch, next_noise=noises, actor_noise=noises)
    fourth = agent._update_once(batch, next_noise=noises, actor_noise=noises)
    assert [third["actor_update_accepted"], fourth["actor_update_accepted"]] == [
        0.0,
        1.0,
    ]
    assert agent.update_step == 4
    assert agent.actor_optimizer_steps == agent.temperature_optimizer_steps == 2
    assert fourth["actor_learning_rate"] == pytest.approx(
        linear_learning_rate(1e-2, 1e-3, 1, 16)
    )
    assert fourth["temperature_learning_rate"] == pytest.approx(
        fourth["actor_learning_rate"]
    )
    assert fourth["critic_learning_rate"] == pytest.approx(
        linear_learning_rate(1e-2, 1e-3, 3, 16)
    )
    assert all(math.isfinite(value) for value in fourth.values())
    for module in (agent.actor, agent.critic):
        for norms in _linear_row_norms(module):
            torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-6, rtol=0)
    assert agent.temperature < _tensor(0.01)


def test_agent_deterministic_inference_is_repeatable_and_advances_release_rng_once():
    agent = XQCAgent(2, 1, _tiny_config())
    before = _bn_buffers(agent.actor)
    initial_rng = agent.generator.get_state().clone()
    reference = torch.Generator(device="cpu")
    reference.set_state(initial_rng)
    torch.randn(
        (1, agent.action_dim),
        dtype=torch.float32,
        device="cpu",
        generator=reference,
    )
    after_one_draw = reference.get_state().clone()
    first = agent.act(np.array([0.25, -0.5], dtype=np.float32), deterministic=True)
    assert torch.equal(agent.generator.get_state(), after_one_draw)
    torch.randn(
        (1, agent.action_dim),
        dtype=torch.float32,
        device="cpu",
        generator=reference,
    )
    after_two_draws = reference.get_state().clone()
    second = agent.act(np.array([0.25, -0.5], dtype=np.float32), deterministic=True)
    assert torch.equal(agent.generator.get_state(), after_two_draws)
    np.testing.assert_array_equal(first, second)
    for name, value in _bn_buffers(agent.actor).items():
        torch.testing.assert_close(value, before[name])


def test_agent_uses_raw_replay_rewards_and_timeout_bootstraps():
    agent = XQCAgent(2, 1, _tiny_config(reward_normalization=True))
    agent.observe_reward(1.0, False, False)
    agent.observe_reward(2.0, False, True)
    raw = _fixed_batch(agent.device)
    raw["rewards"] = torch.full((4, 1), 3.0, device=agent.device)
    raw["dones"] = torch.zeros((4, 1), device=agent.device)
    raw.pop("masks")
    prepared = agent._prepared_batch(raw)
    torch.testing.assert_close(
        prepared["rewards"], raw["rewards"] / agent.reward_normalizer.scale
    )
    torch.testing.assert_close(prepared["masks"], torch.ones_like(prepared["masks"]))
    torch.testing.assert_close(raw["rewards"], torch.full_like(raw["rewards"], 3.0))
    assert agent.reward_normalizer.return_accumulator == 2.0


def test_agent_state_round_trip_and_validation_are_atomic():
    agent = XQCAgent(2, 1, _tiny_config())
    batch = _fixed_batch(agent.device)
    noise = torch.zeros((4, 1), device=agent.device)
    agent._update_once(batch, next_noise=noise, actor_noise=noise)
    state = copy.deepcopy(agent.state_dict())

    restored = XQCAgent(2, 1, _tiny_config(seed=19))
    restored.load_state_dict(state)
    assert restored.update_step == 1
    assert restored.actor_optimizer_steps == restored.temperature_optimizer_steps == 1
    for key, value in agent.actor.state_dict().items():
        torch.testing.assert_close(restored.actor.state_dict()[key], value)
    assert restored.reward_normalizer.state_dict() == agent.reward_normalizer.state_dict()

    invalid = copy.deepcopy(state)
    invalid["actor"].pop(next(iter(invalid["actor"])))
    before = {
        key: value.detach().clone() for key, value in restored.actor.state_dict().items()
    }
    with pytest.raises(ValueError):
        restored.load_state_dict(invalid)
    for key, value in restored.actor.state_dict().items():
        torch.testing.assert_close(value, before[key])


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"tau": 0.0}, "tau"),
        ({"gamma": 1.1}, "gamma"),
        ({"policy_delay": 0}, "policy_delay"),
        ({"num_atoms": 1}, "num_atoms"),
        ({"vmin": 2.0, "vmax": 2.0}, "vmin"),
        ({"actor_net_arch": ()}, "actor_net_arch"),
        ({"target_entropy": float("nan")}, "target_entropy"),
        ({"reward_normalization": 1}, "reward_normalization"),
    ],
)
def test_xqc_config_rejects_invalid_semantics(overrides, message):
    with pytest.raises(ValueError, match=message):
        XQCConfig(**overrides)
