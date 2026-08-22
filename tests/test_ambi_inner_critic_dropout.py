import gymnasium as gym
import pytest
import torch
import torch.nn as nn

from RL.AMBITDMPC2 import AMBITDMPC2


def _build_cfg(**params):
    """Resolve AMBI configuration without allocating model parameters."""

    algorithm = object.__new__(AMBITDMPC2)
    algorithm.env = gym.make("Pendulum-v1", max_episode_steps=5)
    algorithm.run_params = {
        "seed": 3,
        "device": "cpu",
        "env": "test-env",
        "total_steps": 12,
    }
    algorithm.custom_params = params
    try:
        return algorithm._build_cfg({"device": "cpu", **params})
    finally:
        algorithm.env.close()


def _tiny_params(**overrides):
    params = {
        "device": "cpu",
        "model_size": None,
        "enc_dim": 16,
        "mlp_dim": 16,
        "latent_dim": 8,
        "num_enc_layers": 2,
        "simnorm_dim": 4,
        "num_bins": 5,
        "vmin": -5,
        "vmax": 5,
        "batch_size": 2,
        "train_unroll_horizon": 2,
        "outer_planning_horizon": 2,
        "buffer_size": 32,
        "seed_steps": 4,
        "pretrain_steps": 1,
        "utd": 1,
        "compile": False,
        "episodic": False,
        "discount": 0.99,
        "wandb": False,
        "dropout": 0.75,
        "q_representation": "scalar",
        "num_q": 2,
        "q_num_bins": 5,
        "q_vmin": -5,
        "q_vmax": 5,
        "inner_operator": "sac",
        "inner_rounds": 1,
        "inner_rollouts_per_round": 4,
        "inner_rollout_horizon": 1,
        "inner_updates_per_round": 1,
        "inner_batch_size": 4,
        "inner_replay_capacity": 4,
        "inner_actor_adaptation": "clone",
        "inner_critic_adaptation": "clone",
        "inner_critic_lora_dropout": 0.65,
        "inner_temperature_mode": "inherit_outer",
        "inner_critic_target_tau": 1.0,
        "inner_critic_target_update_interval": 1,
        "inner_diagnostic_rollouts": 0,
        "inner_mppi_num_elites": 2,
        "inner_mppi_num_pi_trajs": 0,
    }
    params.update(overrides)
    return params


def _tiny_model(**overrides):
    env = gym.make("Pendulum-v1", max_episode_steps=5)
    return AMBITDMPC2(
        "AMBITDMPC2",
        env,
        _tiny_params(**overrides),
        {"seed": 13, "device": "cpu", "env": "test", "total_steps": 10},
        {},
    )


def _prepare_inner_critic(model):
    # AMBI zero-initializes Q output weights. Make them nonzero so stochastic
    # hidden activations remain observable at the output in this focused test.
    with torch.no_grad():
        for head in model.agent.model._Qs:
            weight = head[-1].weight
            weight.copy_(
                torch.linspace(-0.5, 0.5, weight.numel()).reshape_as(weight)
            )
    engine = model.agent.inner_engine
    with engine.rng.fork("initialization"):
        engine._prepare_workspace(t0=True)
    return engine


def _two_q_forwards(engine, z, action, *, detach):
    model = engine.model
    with engine.rng.fork("bootstrap"):
        first = model.q_predictions(
            z,
            action,
            qs=engine.state.critic,
            detach=detach,
        ).detach()
        second = model.q_predictions(
            z,
            action,
            qs=engine.state.critic,
            detach=detach,
        ).detach()
    return first, second


def test_inner_critic_dropout_flag_is_strict_and_defaults_enabled():
    assert _build_cfg().inner_critic_dropout_enabled is True
    assert (
        _build_cfg(inner_critic_dropout_enabled=False).inner_critic_dropout_enabled
        is False
    )

    for invalid in (None, 0, 1, "false"):
        with pytest.raises(ValueError, match="inner_critic_dropout_enabled"):
            _build_cfg(inner_critic_dropout_enabled=invalid)


@pytest.mark.parametrize("adaptation", ["clone", "lora"])
@pytest.mark.parametrize("enabled", [True, False])
def test_inner_critic_dropout_flag_controls_q_forwards_and_keeps_autograd(
    adaptation,
    enabled,
):
    model = _tiny_model(
        inner_critic_adaptation=adaptation,
        inner_critic_dropout_enabled=enabled,
    )
    try:
        engine = _prepare_inner_critic(model)
        critic = engine.state.critic
        target = engine.state.critic_target
        dropout_modules = [
            module for module in critic.modules() if isinstance(module, nn.Dropout)
        ]

        assert critic.training is enabled
        assert target.training is False
        assert dropout_modules
        assert all(module.training is enabled for module in dropout_modules)
        assert model.agent.model._Qs.training is False
        assert all(parameter.requires_grad for parameter in engine.state.critic_params)

        z = torch.randn(5, model.cfg.latent_dim)
        action = torch.randn(5, model.cfg.action_dim).tanh()
        raw = _two_q_forwards(engine, z, action, detach=False)
        detached = _two_q_forwards(engine, z, action, detach=True)
        if enabled:
            assert not torch.equal(*raw)
            assert not torch.equal(*detached)
        else:
            torch.testing.assert_close(*raw, rtol=0, atol=0)
            torch.testing.assert_close(*detached, rtol=0, atol=0)

            predictions = model.agent.model.q_predictions(
                z,
                action,
                qs=critic,
            )
            predictions.sum().backward()
            gradients = [
                parameter.grad for parameter in engine.state.critic_params
            ]
            assert any(
                gradient is not None
                and torch.isfinite(gradient).all()
                and torch.count_nonzero(gradient).item()
                for gradient in gradients
            )

            for parameter in engine.state.critic_params:
                parameter.grad = None
            differentiable_action = action.detach().requires_grad_(True)
            model.agent.model.q_predictions(
                z,
                differentiable_action,
                qs=critic,
                detach=True,
            ).sum().backward()
            assert differentiable_action.grad is not None
            assert torch.isfinite(differentiable_action.grad).all()
            assert torch.count_nonzero(differentiable_action.grad).item()
            assert all(
                parameter.grad is None for parameter in engine.state.critic_params
            )
    finally:
        model.env.close()


@pytest.mark.parametrize("adaptation", ["clone", "lora"])
def test_dropout_free_inner_critic_updates_and_stays_eval_across_actions(adaptation):
    model = _tiny_model(
        inner_critic_adaptation=adaptation,
        inner_critic_dropout_enabled=False,
        inner_critic_lr=1e-2,
        inner_critic_scope="run",
        inner_critic_optimizer_scope="run",
        inner_replay_scope="run",
    )
    try:
        engine = _prepare_inner_critic(model)
        critic = engine.state.critic
        before = [parameter.detach().clone() for parameter in engine.state.critic_params]

        model.agent.model.train()
        model.agent.act(torch.zeros(3), t0=True, eval_mode=False)

        assert engine.state.critic is critic
        assert critic.training is False
        assert engine.state.critic_target.training is False
        assert model.agent.model.training is True
        assert model.agent.model._target_Qs.training is False
        assert any(
            not torch.equal(parameter, initial)
            for parameter, initial in zip(engine.state.critic_params, before)
        )
        assert any(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and torch.count_nonzero(parameter.grad).item()
            for parameter in engine.state.critic_params
        )

        model.agent.act(torch.zeros(3), t0=False, eval_mode=False)
        assert engine.state.critic is critic
        assert critic.training is False
        assert all(
            not module.training
            for module in critic.modules()
            if isinstance(module, nn.Dropout)
        )
    finally:
        model.env.close()
