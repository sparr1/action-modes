from collections import OrderedDict
from contextlib import contextmanager

import gymnasium as gym
import numpy as np
import pytest
import torch

from RL.AMBITDMPC2 import AMBITDMPC2
from RL.TDMPC2 import TDMPC2Baseline


def _tiny_params(**overrides):
    params = {
        "device": "cpu",
        "model_size": None,
        "enc_dim": 32,
        "mlp_dim": 32,
        "latent_dim": 16,
        "num_channels": 1,
        "num_enc_layers": 2,
        "num_q": 2,
        "simnorm_dim": 8,
        "num_bins": 11,
        "vmin": -5,
        "vmax": 5,
        "batch_size": 2,
        "train_unroll_horizon": 2,
        "outer_planning_horizon": 2,
        "buffer_size": 100,
        "seed_steps": 2,
        "pretrain_steps": 1,
        "episode_length": 5,
        "compile": False,
        "iterations": 2,
        "num_samples": 16,
        "num_elites": 4,
        "num_pi_trajs": 2,
        "inner_operator": "none",
        "inner_rollout_horizon": 2,
        "dropout": 0.0,
    }
    params.update(overrides)
    return params


class PixelEnv(gym.Env):
    metadata = {}

    def __init__(self, *, observation_space=None, declared="rgb"):
        if declared is not None:
            self.observation_type = declared
        self.observation_space = observation_space or gym.spaces.Box(
            0, 255, shape=(9, 64, 64), dtype=np.uint8
        )
        self.action_space = gym.spaces.Box(
            -1.0, 1.0, shape=(2,), dtype=np.float32
        )
        self.spec = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype), {}

    def step(self, action):
        del action
        observation = np.zeros(
            self.observation_space.shape, dtype=self.observation_space.dtype
        )
        return observation, 0.0, False, False, {}


def _make_model(cls, env, params=None, *, total_steps=10):
    return cls(
        cls.__name__,
        env,
        params or _tiny_params(),
        {
            "seed": 3,
            "device": "cpu",
            "env": "pixel-test",
            "total_steps": total_steps,
        },
        {},
    )


def _populate_pixel_replay(model):
    rng = np.random.default_rng(17)
    initial = rng.integers(0, 256, (9, 64, 64), dtype=np.uint8)
    rows = model._start_episode_staging(model._obs_to_tensor(initial))
    for transition in range(3):
        observation = rng.integers(
            0, 256, (9, 64, 64), dtype=np.uint8
        )
        model._stage_transition(
            rows,
            model._obs_to_tensor(observation),
            np.zeros(model.cfg.action_dim, dtype=np.float32),
            float(transition) / 10.0,
            False,
        )
        rows += 1
    model.buffer.add(model._episode_staging[:rows])


def test_environment_declared_rgb_preserves_uint8_staging_and_replay_rows():
    model = _make_model(TDMPC2Baseline, PixelEnv())
    assert model.cfg.obs == "rgb"
    assert model.cfg.obs_shape == {"rgb": (9, 64, 64)}
    assert model.cfg.obs_dtype == "uint8"

    observation = np.arange(9 * 64 * 64, dtype=np.uint8).reshape(9, 64, 64)
    observation_tensor = model._obs_to_tensor(observation)
    assert observation_tensor.dtype == torch.uint8
    assert tuple(observation_tensor.shape) == (9, 64, 64)
    assert model._observation_staging.dtype == torch.uint8
    assert model._episode_staging["obs"].dtype == torch.uint8
    assert tuple(model._episode_staging["obs"].shape[1:]) == (9, 64, 64)
    transition_dict = model._to_td(observation)
    assert transition_dict["obs"].dtype == torch.uint8
    assert tuple(transition_dict["obs"].shape) == (1, 9, 64, 64)

    row = model._start_episode_staging(observation_tensor)
    model._stage_transition(
        row,
        observation_tensor,
        np.zeros(2, dtype=np.float32),
        0.0,
        False,
    )
    torch.testing.assert_close(model._episode_staging["obs"][1], observation_tensor)
    with pytest.raises(ValueError, match="dtype changed at runtime"):
        model._obs_to_tensor(observation.astype(np.float32))
    with pytest.raises(ValueError, match="shape changed at runtime"):
        model._obs_to_tensor(observation[:, :-1, :])


def test_rgb_mode_is_explicit_and_environment_declaration_is_authoritative():
    generic_pixels = PixelEnv(declared=None)
    with pytest.raises(NotImplementedError, match="vector state observations only"):
        _make_model(TDMPC2Baseline, generic_pixels)

    explicit = _make_model(
        TDMPC2Baseline,
        PixelEnv(declared=None),
        _tiny_params(obs="rgb"),
    )
    assert explicit.cfg.obs == "rgb"

    with pytest.raises(ValueError, match="does not match.*observation_type"):
        _make_model(
            TDMPC2Baseline,
            PixelEnv(declared="rgb"),
            _tiny_params(obs="state"),
        )


@pytest.mark.parametrize(
    ("space", "message"),
    [
        (
            gym.spaces.Box(0, 255, shape=(3, 64, 64), dtype=np.uint8),
            "shape.*9, 64, 64",
        ),
        (
            gym.spaces.Box(0, 255, shape=(9, 64, 64), dtype=np.float32),
            "dtype uint8",
        ),
        (
            gym.spaces.Box(1, 255, shape=(9, 64, 64), dtype=np.uint8),
            "bounds.*exactly",
        ),
    ],
)
def test_rgb_space_contract_is_strict(space, message):
    with pytest.raises(ValueError, match=message):
        _make_model(TDMPC2Baseline, PixelEnv(observation_space=space))


def test_rgb_encoder_contract_and_strict_compilation_fail_early():
    with pytest.raises(ValueError, match=r"latent_dim == 16 \* num_channels"):
        _make_model(
            TDMPC2Baseline,
            PixelEnv(),
            _tiny_params(latent_dim=32, num_channels=1),
        )

    with pytest.raises(ValueError, match="compile_strict=True"):
        _make_model(
            AMBITDMPC2,
            PixelEnv(),
            _tiny_params(compile_strict=True),
        )


def test_large_rgb_replay_footprint_warns_without_changing_capacity():
    with pytest.warns(UserWarning, match="36.9 GB.*capacity is unchanged"):
        model = _make_model(
            TDMPC2Baseline,
            PixelEnv(),
            _tiny_params(buffer_size=1_000_000),
            total_steps=1_000_000,
        )
    assert model.cfg.buffer_size == 1_000_000
    assert model.buffer.capacity == 1_000_000


def test_soft_world_model_encodes_time_major_rgb_like_upstream():
    model = _make_model(AMBITDMPC2, PixelEnv())
    world_model = model.agent.model
    observations = torch.randint(
        0, 256, (3, 2, 9, 64, 64), dtype=torch.uint8
    )

    torch.manual_seed(19)
    actual = world_model.encode(observations)
    torch.manual_seed(19)
    expected = torch.stack(
        [world_model._encoder["rgb"](time_obs) for time_obs in observations]
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert tuple(actual.shape) == (3, 2, model.cfg.latent_dim)


@pytest.mark.parametrize("algorithm", [TDMPC2Baseline, AMBITDMPC2])
def test_pixel_predict_returns_a_finite_environment_action(algorithm):
    model = _make_model(algorithm, PixelEnv())
    observation = np.zeros((9, 64, 64), dtype=np.uint8)
    action, state = model.predict(
        observation, deterministic=True, episode_start=True
    )
    assert state is None
    assert action.shape == model.env.action_space.shape
    assert action.dtype == np.float32
    assert np.isfinite(action).all()


@pytest.mark.parametrize("algorithm", [TDMPC2Baseline, AMBITDMPC2])
def test_pixel_replay_sample_and_optimizer_update_are_finite(algorithm):
    model = _make_model(algorithm, PixelEnv())
    _populate_pixel_replay(model)

    observations, actions, rewards, terminated, task = model.buffer.sample()
    assert observations.dtype == torch.uint8
    assert tuple(observations.shape) == (3, 2, 9, 64, 64)
    assert tuple(actions.shape[:2]) == (2, 2)
    assert tuple(rewards.shape[:2]) == (2, 2)
    assert tuple(terminated.shape[:2]) == (2, 2)
    assert task is None

    metrics = model.agent.update(model.buffer)
    assert list(metrics.keys())
    for value in metrics.values():
        assert torch.isfinite(torch.as_tensor(value)).all()


@pytest.mark.parametrize("algorithm", [TDMPC2Baseline, AMBITDMPC2])
def test_pixel_checkpoint_roundtrip_records_the_exact_observation_contract(
    algorithm,
):
    source = _make_model(algorithm, PixelEnv())
    with torch.no_grad():
        next(source.agent.model.parameters()).add_(0.125)
    checkpoint = source.agent.checkpoint_state()
    assert checkpoint["observation_spec"] == {
        "mode": "rgb",
        "shape": [9, 64, 64],
        "dtype": "uint8",
    }

    restored = _make_model(algorithm, PixelEnv())
    restored.agent.load(checkpoint)
    source_state = source.agent.model.state_dict()
    restored_state = restored.agent.model.state_dict()
    assert source_state.keys() == restored_state.keys()
    for key in source_state:
        torch.testing.assert_close(restored_state[key], source_state[key], rtol=0, atol=0)


def test_official_style_pixel_checkpoint_preserves_convolutional_encoder_keys():
    source = _make_model(TDMPC2Baseline, PixelEnv())
    port_state = source.agent.model.state_dict()
    official = OrderedDict()
    critic_prefixes = ("_Qs.modules_list.", "_target_Qs.modules_list.")
    for key, value in port_state.items():
        if not key.startswith(critic_prefixes):
            official[key] = value.detach().clone()

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
    assert any(key.startswith("_encoder.rgb.") for key in official)

    restored = _make_model(TDMPC2Baseline, PixelEnv())
    restored.agent.load(official)
    restored_state = restored.agent.model.state_dict()
    for key in port_state:
        torch.testing.assert_close(restored_state[key], port_state[key], rtol=0, atol=0)


def test_ambi_pixel_root_encoding_uses_private_observation_rng():
    model = _make_model(AMBITDMPC2, PixelEnv())
    observation = torch.randint(0, 256, (9, 64, 64), dtype=torch.uint8)
    observation_stream = model.agent.inner_engine.rng.phase_generators[
        "observation"
    ]
    private_before = observation_stream.get_state().clone()
    global_before = torch.random.get_rng_state().clone()

    captured_roots = []

    def capture_root(root_z, **kwargs):
        del kwargs
        captured_roots.append(root_z.detach().clone())
        return torch.zeros(model.cfg.action_dim), {}, []

    model.agent.inner_engine.act = capture_root
    model.agent.act(observation, t0=True, eval_mode=True)

    assert torch.equal(torch.random.get_rng_state(), global_before)
    assert not torch.equal(observation_stream.get_state(), private_before)
    assert len(captured_roots) == 1
    assert tuple(captured_roots[0].shape) == (1, model.cfg.latent_dim)
    assert model.agent.inner_engine.rng.STREAMS[:-1] == (
        "initialization",
        "collection",
        "replay",
        "bootstrap",
        "gradient_policy",
        "execution",
        "mppi",
        "diagnostics",
    )


def test_ambi_pixel_action_saves_default_rng_state_only_once(monkeypatch):
    model = _make_model(AMBITDMPC2, PixelEnv())
    observation = torch.zeros((9, 64, 64), dtype=torch.uint8)
    original_fork_rng = torch.random.fork_rng
    calls = 0

    @contextmanager
    def counted_fork_rng(*args, **kwargs):
        nonlocal calls
        calls += 1
        with original_fork_rng(*args, **kwargs):
            yield

    monkeypatch.setattr(torch.random, "fork_rng", counted_fork_rng)
    model.agent.act(observation, t0=True, eval_mode=True)
    assert calls == 1
