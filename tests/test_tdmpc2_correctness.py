from collections import OrderedDict

import gymnasium as gym
import numpy as np
import pytest
import torch

from RL.AMBITDMPC2 import AMBITDMPC2
from RL.TDMPC2 import TDMPC2Baseline
from RL.tdmpc2_core.ambi_agent import _polyak_update
from RL.tdmpc2_core.common.layers import api_model_conversion


def tiny_params(**overrides):
    params = {
        "device": "cpu",
        "model_size": None,
        "enc_dim": 32,
        "mlp_dim": 32,
        "latent_dim": 16,
        "num_enc_layers": 2,
        "num_q": 2,
        "simnorm_dim": 8,
        "num_bins": 11,
        "vmin": -5,
        "vmax": 5,
        "batch_size": 4,
        "train_unroll_horizon": 2,
        "outer_planning_horizon": 2,
        "buffer_size": 100,
        "seed_steps": 4,
        "pretrain_steps": 2,
        "utd": 1,
        "compile": False,
        "episodic": False,
        "discount": 0.99,
        "iterations": 2,
        "num_samples": 16,
        "num_elites": 4,
        "num_pi_trajs": 2,
        "inner_adaptation": "clone",
        "inner_iterations": 2,
        "inner_rollouts": 4,
        "inner_horizon": 2,
        "inner_updates_per_iteration": 2,
        "inner_batch_size": 8,
        "inner_tau": 1.0,
        "wandb": False,
        "dropout": 0.0,
    }
    params.update(overrides)
    return params


def make_model(cls, env, params=None, total_steps=12):
    return cls(
        cls.__name__,
        env,
        params or tiny_params(),
        {"seed": 3, "device": "cpu", "env": "test-env", "total_steps": total_steps},
        {},
    )


class OneStepTruncationEnv(gym.Env):
    metadata = {}

    def __init__(self, events):
        self.events = events
        self.observation_space = gym.spaces.Box(
            -1.0, 1.0, shape=(3,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            -1.0, 1.0, shape=(1,), dtype=np.float32
        )
        self.spec = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        self.events.append("env_step")
        return np.zeros(3, dtype=np.float32), 0.0, False, True, {}


def test_ambi_preserves_the_exact_tdmpc2_training_loop_and_ordering():
    assert AMBITDMPC2.learn is TDMPC2Baseline.learn
    events = []
    env = OneStepTruncationEnv(events)
    params = tiny_params(
        seed_steps=0,
        pretrain_steps=1,
        utd=1,
        train_unroll_horizon=1,
        outer_planning_horizon=1,
        batch_size=1,
        episode_length=1,
        inner_iterations=0,
        inner_horizon=1,
        inner_updates_per_iteration=0,
    )
    model = make_model(AMBITDMPC2, env, params, total_steps=2)
    original_add = model.buffer.add

    def record_add(episode):
        events.append("replay_add")
        return original_add(episode)

    def record_update(buffer):
        events.append("update")
        return {"num_updates": torch.tensor(1.0)}

    def record_act(
        obs,
        *,
        t0=False,
        eval_mode=False,
        task=None,
        collect_diagnostics=True,
    ):
        del collect_diagnostics
        events.append("act")
        return torch.zeros(model.cfg.action_dim)

    model.buffer.add = record_add
    model.agent.update = record_update
    model.agent.act = record_act
    model.learn(total_timesteps=2)
    assert events == [
        "env_step",
        "replay_add",
        "update",
        "act",
        "env_step",
        "replay_add",
        "update",
    ]
    assert model._num_updates == 2


@pytest.mark.parametrize("algorithm", [TDMPC2Baseline, AMBITDMPC2])
def test_small_end_to_end_training_and_truncation_bootstrap(algorithm):
    env = gym.make("Pendulum-v1", max_episode_steps=5)
    model = make_model(algorithm, env)
    model.learn(total_timesteps=12)
    assert model._global_step == 12
    assert model.buffer.num_eps >= 2
    assert model._last_train_metrics is not None
    for value in model._last_train_metrics.values():
        assert torch.isfinite(torch.as_tensor(value)).all()


def test_inner_clone_and_lora_do_not_mutate_outer_heads():
    for adaptation in ("clone", "lora"):
        env = gym.make("Pendulum-v1", max_episode_steps=5)
        model = make_model(AMBITDMPC2, env, tiny_params(inner_adaptation=adaptation))
        obs, _ = env.reset(seed=3)
        before = {key: value.detach().clone() for key, value in model.agent.model.state_dict().items()}
        action, _ = model.predict(obs, deterministic=False, episode_start=True)
        assert action.shape == env.action_space.shape
        for key, value in model.agent.model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_ephemeral_inner_target_hard_syncs_after_each_update():
    env = gym.make("Pendulum-v1", max_episode_steps=5)
    model = make_model(AMBITDMPC2, env)
    _, critic, target, *_ = model.agent._make_inner_modules()
    with torch.no_grad():
        for parameter in critic.parameters():
            parameter.add_(0.25)
    _polyak_update(critic, target, model.cfg.inner_tau)
    for online, target_parameter in zip(critic.parameters(), target.parameters()):
        torch.testing.assert_close(online, target_parameter, rtol=0, atol=0)


def test_soft_sac_actor_uses_direct_log_std_clamping():
    env = gym.make("Pendulum-v1", max_episode_steps=5)
    model = make_model(AMBITDMPC2, env)
    with torch.no_grad():
        for parameter in model.agent.model._pi.parameters():
            parameter.zero_()
    latent = torch.zeros(3, model.cfg.latent_dim)
    _, info = model.agent.model.pi(latent, deterministic=True)
    torch.testing.assert_close(info["log_std"], torch.zeros_like(info["log_std"]), rtol=0, atol=0)


def test_unsafe_or_unsupported_configs_fail_early():
    env = gym.make("Pendulum-v1", max_episode_steps=5)
    with pytest.warns(UserWarning, match="model-bias risk"):
        long_horizon = make_model(
            AMBITDMPC2,
            env,
            tiny_params(
                inner_horizon=3,
                train_unroll_horizon=2,
                outer_planning_horizon=2,
            ),
        )
    assert long_horizon.cfg.inner_horizon_ratio == pytest.approx(1.5)
    with pytest.raises(ValueError, match="compile=True is not supported"):
        make_model(TDMPC2Baseline, env, tiny_params(compile=True))

    image_env = gym.make("Pendulum-v1")
    image_env.observation_space = gym.spaces.Box(0, 255, shape=(3, 64, 64), dtype=np.uint8)
    with pytest.raises(NotImplementedError, match="vector state observations only"):
        make_model(TDMPC2Baseline, image_env)


def test_public_reset_clears_mppi_warm_start():
    env = gym.make("Pendulum-v1", max_episode_steps=5)
    model = make_model(TDMPC2Baseline, env)
    model.agent._prev_mean.fill_(0.75)
    model.reset()
    assert torch.count_nonzero(model.agent._prev_mean) == 0
    assert model._predict_t0 is True


def test_official_vectorized_checkpoint_keys_convert_exactly():
    env = gym.make("Pendulum-v1", max_episode_steps=5)
    model = make_model(TDMPC2Baseline, env)
    world_model = model.agent.model
    port_state = world_model.state_dict()
    official = OrderedDict()

    for key, value in port_state.items():
        if not key.startswith(("_Qs.modules_list.", "_target_Qs.modules_list.")):
            official[key] = value.clone() if torch.is_tensor(value) else value

    def pack(port_prefix, official_prefix):
        grouped = {}
        for key, value in port_state.items():
            if not key.startswith(port_prefix):
                continue
            remainder = key[len(port_prefix):]
            critic_index, layer_and_field = remainder.split(".", 1)
            grouped.setdefault(layer_and_field, {})[int(critic_index)] = value
        for layer_and_field, values in grouped.items():
            official[official_prefix + layer_and_field] = torch.stack(
                [values[index] for index in sorted(values)], dim=0
            )

    pack("_Qs.modules_list.", "_Qs.params.")
    pack("_Qs.modules_list.", "_detach_Qs_params.")
    pack("_target_Qs.modules_list.", "_target_Qs_params.")

    converted = api_model_conversion(port_state, official)
    assert set(converted) == set(port_state)
    for key, value in port_state.items():
        torch.testing.assert_close(converted[key], value, rtol=0, atol=0)


class TrueTerminationEnv(gym.Env):
    metadata = {}

    def __init__(self):
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.spec = None
        self._step = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        self._step += 1
        return np.zeros(3, dtype=np.float32), 1.0, self._step == 3, False, {}


def test_true_termination_requires_episdodic_mode():
    env = TrueTerminationEnv()
    params = tiny_params(
        seed_steps=2,
        episode_length=3,
        train_unroll_horizon=2,
        outer_planning_horizon=2,
        batch_size=2,
    )
    model = make_model(TDMPC2Baseline, env, params, total_steps=3)
    with pytest.raises(ValueError, match="episodic=true"):
        model.learn(total_timesteps=3)


def test_true_termination_trains_when_episodic_enabled():
    env = TrueTerminationEnv()
    params = tiny_params(
        seed_steps=3,
        pretrain_steps=1,
        episode_length=3,
        train_unroll_horizon=2,
        outer_planning_horizon=2,
        batch_size=2,
        episodic=True,
    )
    model = make_model(AMBITDMPC2, env, params, total_steps=7)
    model.learn(total_timesteps=7)
    assert model.agent.num_updates > 0
    assert torch.isfinite(torch.as_tensor(model._last_train_metrics["termination_loss"]))
