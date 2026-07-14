import copy

import gymnasium as gym
import numpy as np
import pytest
import torch

from RL.SAC import SAC
from RL.sac_core import SACAgent, SACConfig


class StaticBatch:
    def __init__(self, *, reward_scale=1.0, batch_size=8):
        generator = torch.Generator().manual_seed(419)
        self.batch = {
            "obs": torch.randn(batch_size, 3, generator=generator),
            "actions": torch.tanh(
                torch.randn(batch_size, 2, generator=generator)
            ),
            "rewards": reward_scale
            * torch.randn(batch_size, 1, generator=generator),
            "next_obs": torch.randn(batch_size, 3, generator=generator),
            "dones": (
                torch.rand(batch_size, 1, generator=generator) < 0.25
            ).float(),
        }

    def sample(self, *_args):
        return self.batch


def distributional_config(**overrides):
    values = {
        "net_arch": (16,),
        "q_representation": "distributional",
        "q_num_bins": 11,
        "q_vmin": -5.0,
        "q_vmax": 5.0,
        "seed": 7,
        "device": "cpu",
    }
    values.update(overrides)
    return SACConfig(**values)


def test_distributional_critics_expose_logits_but_public_q_api_decodes_values():
    agent = SACAgent(3, 2, distributional_config())
    obs = torch.zeros(4, 3)
    actions = torch.zeros(4, 2)

    q1_logits, q2_logits = agent.q_predictions(obs, actions)
    q1, q2 = agent.q_values(obs, actions)

    assert q1_logits.shape == q2_logits.shape == (4, 11)
    assert q1.shape == q2.shape == (4, 1)
    assert agent.critic.qf1[-1].out_features == 11
    assert agent.critic_signature == {
        "q_representation": "distributional",
        "num_q": 2,
        "q_num_bins": 11,
        "q_vmin": -5.0,
        "q_vmax": 5.0,
    }
    assert torch.isfinite(q1).all()
    assert torch.isfinite(q2).all()


@pytest.mark.parametrize("ent_coef", ["auto", 0.2])
def test_distributional_sac_update_trains_both_critics_and_actor(ent_coef):
    agent = SACAgent(3, 2, distributional_config(ent_coef=ent_coef))
    q1_before = agent.critic.qf1[-1].weight.detach().clone()
    q2_before = agent.critic.qf2[-1].weight.detach().clone()
    actor_before = agent.actor.mu.weight.detach().clone()

    metrics = agent.update(StaticBatch(), gradient_steps=1, batch_size=8)

    assert not torch.equal(q1_before, agent.critic.qf1[-1].weight)
    assert not torch.equal(q2_before, agent.critic.qf2[-1].weight)
    assert not torch.equal(actor_before, agent.actor.mu.weight)
    assert metrics["critic_grad_norm"] > 0.0
    assert metrics["actor_grad_norm"] > 0.0
    assert 0.0 <= metrics["q_target_clip_fraction"] <= 1.0
    assert metrics["q_distribution_entropy"] > 0.0
    assert 0.0 < metrics["q_distribution_max_probability"] <= 1.0
    assert all(np.isfinite(value) for value in metrics.values())
    assert all(not parameter.requires_grad for parameter in agent.critic_target.parameters())

    if ent_coef == "auto":
        assert "ent_coef_loss" in metrics
        assert "ent_coef_grad_norm" in metrics
    else:
        assert metrics["ent_coef"] == pytest.approx(ent_coef)
        assert "ent_coef_loss" not in metrics
        assert "ent_coef_grad_norm" not in metrics


def test_distributional_targets_clip_to_support_without_nonfinite_losses():
    agent = SACAgent(
        3,
        2,
        distributional_config(q_vmin=-1.0, q_vmax=1.0, ent_coef=0.2),
    )
    replay = StaticBatch(reward_scale=1_000_000.0)

    metrics = agent.update(replay, gradient_steps=1, batch_size=8)

    assert metrics["q_target_clip_fraction"] > 0.0
    assert np.isfinite(metrics["critic_loss"])
    assert np.isfinite(metrics["actor_loss"])


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"q_representation": "quantile"}, "q_representation"),
        ({"q_num_bins": 1}, "at least 2"),
        ({"q_vmin": 2.0, "q_vmax": 2.0}, "smaller than"),
        ({"q_vmin": float("nan")}, "smaller than"),
    ],
)
def test_invalid_distributional_q_configuration_fails_before_network_construction(
    overrides, message
):
    with pytest.raises(ValueError, match=message):
        SACAgent(3, 2, distributional_config(**overrides))


def test_distributional_checkpoint_roundtrip_and_semantic_preflight():
    source = SACAgent(3, 2, distributional_config())
    source.update(StaticBatch(), gradient_steps=1, batch_size=8)
    state = source.state_dict()

    restored = SACAgent(3, 2, distributional_config(seed=99))
    restored.load_state_dict(state)
    for key, value in source.critic.state_dict().items():
        torch.testing.assert_close(
            value, restored.critic.state_dict()[key], rtol=0, atol=0
        )

    for incompatible in (
        SACAgent(3, 2, SACConfig(net_arch=(16,), device="cpu")),
        SACAgent(3, 2, distributional_config(q_vmin=-4.0)),
        SACAgent(3, 2, distributional_config(q_num_bins=13)),
    ):
        actor_before = copy.deepcopy(incompatible.actor.state_dict())
        with pytest.raises(ValueError, match="critic specification"):
            incompatible.load_state_dict(state)
        for key, value in actor_before.items():
            torch.testing.assert_close(
                value, incompatible.actor.state_dict()[key], rtol=0, atol=0
            )


def test_legacy_checkpoint_without_critic_spec_loads_only_into_scalar_sac():
    scalar = SACAgent(3, 2, SACConfig(net_arch=(16,), seed=3, device="cpu"))
    legacy_state = scalar.state_dict()
    legacy_state.pop("critic_spec")

    restored = SACAgent(3, 2, SACConfig(net_arch=(16,), seed=9, device="cpu"))
    restored.load_state_dict(legacy_state)

    distributional = SACAgent(3, 2, distributional_config())
    with pytest.raises(ValueError, match="critic specification"):
        distributional.load_state_dict(legacy_state)


def test_native_sac_wrapper_plumbs_distributional_config_and_rejects_ensembles():
    env = gym.make("Pendulum-v1")
    params = {
        "device": "cpu",
        "net_arch": [8],
        "q_representation": "distributional",
        "q_num_bins": 17,
        "q_vmin": -6,
        "q_vmax": 7,
        "wandb": False,
    }
    model = SAC("SAC", env, params, {"device": "cpu"}, {})

    assert model.cfg.q_representation == "distributional"
    assert model.cfg.q_num_bins == 17
    assert model.agent.critic_signature["num_q"] == 2

    with pytest.raises(ValueError, match="exactly two critics"):
        SAC("SAC", env, {**params, "num_q": 5}, {"device": "cpu"}, {})
    env.close()


def test_native_distributional_sac_runs_end_to_end_training_loop():
    env = gym.make("Pendulum-v1")
    model = SAC(
        "SAC",
        env,
        {
            "device": "cpu",
            "seed": 13,
            "learning_starts": 0,
            "train_freq": 1,
            "gradient_steps": 1,
            "batch_size": 4,
            "buffer_size": 32,
            "net_arch": [8],
            "q_representation": "distributional",
            "q_num_bins": 11,
            "q_vmin": -5,
            "q_vmax": 5,
            "wandb": False,
            "verbose": 0,
        },
        {"seed": 13, "device": "cpu", "env": "Pendulum-v1"},
        {},
    )

    model.learn(total_timesteps=4)

    assert model.num_timesteps == 4
    assert model.agent.num_updates == 4
    assert model.replay_buffer.size == 4
    assert model._last_metrics["critic_loss"] > 0.0
    assert all(np.isfinite(value) for value in model._last_metrics.values())
    env.close()
