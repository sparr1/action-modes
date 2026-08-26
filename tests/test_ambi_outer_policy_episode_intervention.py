import random
from types import MethodType, SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

import RL.TDMPC2 as tdmpc2_module
from RL.AMBITDMPC2 import AMBITDMPC2
from RL.TDMPC2 import TDMPC2Baseline, WandbAccumulator, _DeviceMeanAccumulator
from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent


def _params(**overrides):
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
        "dropout": 0.0,
        "q_representation": "scalar",
        "num_q": 2,
        "q_num_bins": 5,
        "q_vmin": -5,
        "q_vmax": 5,
        "inner_operator": "sac",
        "inner_model_step_budget": 4,
        "inner_rounds": 1,
        "inner_rollout_horizon": 2,
        "inner_critic_updates_per_action": 1,
        "inner_actor_updates_per_action": 1,
        "inner_temperature_updates_per_action": 0,
        "inner_batch_size": 4,
        "inner_replay_capacity": 8,
        "inner_actor_adaptation": "clone",
        "inner_critic_adaptation": "clone",
        "inner_temperature_mode": "inherit_outer",
        "inner_critic_target_tau": 1.0,
        "inner_critic_target_update_interval": 1,
    }
    params.update(overrides)
    return params


def _build_cfg(**overrides):
    algorithm = object.__new__(AMBITDMPC2)
    algorithm.env = gym.make("Pendulum-v1", max_episode_steps=5)
    algorithm.run_params = {
        "seed": 17,
        "device": "cpu",
        "env": "test-env",
        "total_steps": 10,
    }
    algorithm.custom_params = _params(**overrides)
    try:
        return algorithm._build_cfg(algorithm.custom_params)
    finally:
        algorithm.env.close()


def _make_model(probability):
    env = gym.make("Pendulum-v1", max_episode_steps=1)
    return AMBITDMPC2(
        "AMBITDMPC2",
        env,
        _params(
            seed_steps=0,
            outer_policy_episode_probability=probability,
        ),
        {"seed": 17, "device": "cpu", "env": "test", "total_steps": 2},
        {},
    )


class _CaptureBuffer:
    def __init__(self):
        self.num_eps = 1
        self.episodes = []

    def add(self, episode):
        self.episodes.append(episode.clone())
        self.num_eps += 1


def _assert_numpy_rng_equal(actual, expected):
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2:] == expected[2:]


def test_outer_policy_episode_probability_defaults_and_validates_strictly():
    assert _build_cfg().outer_policy_episode_probability == 0.0
    assert _build_cfg(outer_policy_episode_probability=0.5).outer_policy_episode_probability == 0.5
    assert _build_cfg(outer_policy_episode_probability=1).outer_policy_episode_probability == 1.0

    for value in (True, "0.5", None, -0.01, 1.01, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="outer_policy_episode_probability"):
            _build_cfg(outer_policy_episode_probability=value)


def test_positive_outer_policy_probability_is_state_sac_only():
    with pytest.raises(ValueError, match="state observations"):
        _build_cfg(obs="rgb", outer_policy_episode_probability=0.5)
    with pytest.raises(ValueError, match="inner_operator='sac'"):
        _build_cfg(
            inner_operator="none",
            inner_model_step_budget=0,
            inner_rounds=0,
            inner_critic_updates_per_action=0,
            inner_actor_updates_per_action=0,
            inner_temperature_updates_per_action=0,
            outer_policy_episode_probability=0.5,
        )


class _BehaviorAgent:
    def __init__(self):
        self.device = torch.device("cpu")
        self.inner_calls = 0
        self.outer_calls = 0
        self.generator_ids = []
        self.last_inner_metrics = {}
        self.last_inner_rollout_lengths = []

    def act(self, obs, **kwargs):
        del obs, kwargs
        self.inner_calls += 1
        self.last_inner_metrics = {}
        return torch.tensor([float(self.inner_calls)])

    def act_outer_policy(self, obs, *, generator):
        del obs
        self.outer_calls += 1
        self.generator_ids.append(id(generator))
        return torch.rand(1, generator=generator)


def _episode_wrapper(start_step, *, probability=1.0, buffer_episodes=1):
    algorithm = object.__new__(AMBITDMPC2)
    algorithm.cfg = SimpleNamespace(
        seed=23,
        seed_steps=4,
        outer_policy_episode_probability=probability,
        inner_diagnostics_every=1,
    )
    algorithm.agent = _BehaviorAgent()
    algorithm.buffer = SimpleNamespace(num_eps=buffer_episodes)
    algorithm._episode_idx = 99
    algorithm._global_step = start_step
    algorithm._outer_policy_episode_eligible = False
    algorithm._outer_policy_episode_selected = False
    algorithm._outer_policy_action_generator = None
    return algorithm


@pytest.mark.parametrize(
    ("start_step", "eligible", "outer_calls", "inner_calls"),
    [(4, False, 0, 2), (5, True, 2, 0)],
)
def test_episode_start_boundary_fixes_one_behavior_for_whole_episode(
    monkeypatch,
    start_step,
    eligible,
    outer_calls,
    inner_calls,
):
    observed = []

    def fake_base(self, obs, total_timesteps, *, eval_pending):
        del obs, total_timesteps
        for _ in range(2):
            observed.append(
                (
                    self._outer_policy_episode_eligible,
                    self._outer_policy_episode_selected,
                    self._episode_payload_extras(),
                )
            )
            self._act_agent(torch.zeros(3), t0=False, eval_mode=False)
            # Crossing seed_steps within an episode must not change behavior.
            self._global_step += 1
        return True, eval_pending

    monkeypatch.setattr(TDMPC2Baseline, "_run_training_episode", fake_base)
    algorithm = _episode_wrapper(start_step)
    assert algorithm._run_training_episode(None, 100, eval_pending=False) == (
        True,
        False,
    )

    assert algorithm.agent.outer_calls == outer_calls
    assert algorithm.agent.inner_calls == inner_calls
    assert all(item[0] is eligible for item in observed)
    assert all(
        item[2]
        == {
            "rollout/outer_policy_episode": int(eligible),
            "rollout/outer_policy_episode_eligible": int(eligible),
        }
        for item in observed
    )
    if eligible:
        assert len(set(algorithm.agent.generator_ids)) == 1
    assert algorithm._outer_policy_episode_eligible is False
    assert algorithm._outer_policy_episode_selected is False
    assert algorithm._outer_policy_action_generator is None


def test_episode_flags_are_cleared_when_private_generator_construction_fails(
    monkeypatch,
):
    algorithm = _episode_wrapper(5, probability=1.0)
    monkeypatch.setattr(
        algorithm,
        "_make_outer_policy_action_generator",
        lambda episode_start_step: (_ for _ in ()).throw(
            RuntimeError(f"generator failed at {episode_start_step}")
        ),
    )
    monkeypatch.setattr(
        TDMPC2Baseline,
        "_run_training_episode",
        lambda *args, **kwargs: pytest.fail("base loop ran without a generator"),
    )

    with pytest.raises(RuntimeError, match="generator failed at 5"):
        algorithm._run_training_episode(None, 100, eval_pending=False)

    assert algorithm._outer_policy_episode_eligible is False
    assert algorithm._outer_policy_episode_selected is False
    assert algorithm._outer_policy_action_generator is None


def test_selection_uses_seed_and_episode_start_step_without_global_rng():
    algorithm = _episode_wrapper(10, probability=0.5)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()

    first = algorithm._select_outer_policy_episode(10)
    algorithm._episode_idx = 123456
    assert algorithm._select_outer_policy_episode(10) is first
    selections = [algorithm._select_outer_policy_episode(step) for step in range(10, 40)]
    assert selections == [algorithm._select_outer_policy_episode(step) for step in range(10, 40)]
    assert random.getstate() == python_state
    _assert_numpy_rng_equal(np.random.get_state(), numpy_state)
    torch.testing.assert_close(torch.random.get_rng_state(), torch_state, rtol=0, atol=0)


def test_selected_episode_uses_ordinary_replay_and_outer_utd(monkeypatch):
    algorithm = _make_model(1.0)
    buffer = _CaptureBuffer()
    algorithm.buffer = buffer
    algorithm._global_step = 1
    algorithm._pretrained = True
    update_calls = []

    def forbidden_inner_act(*args, **kwargs):
        raise AssertionError("selected outer episode entered the inner engine")

    def fake_update(replay):
        update_calls.append(replay)
        return {}

    monkeypatch.setattr(algorithm.agent.inner_engine, "act", forbidden_inner_act)
    monkeypatch.setattr(algorithm.agent, "update", fake_update)
    obs, _ = algorithm._reset_env(seed=31)
    torch_state = torch.random.get_rng_state().clone()
    try:
        completed, eval_pending = algorithm._run_training_episode(
            obs,
            2,
            eval_pending=False,
        )
    finally:
        algorithm.env.close()

    assert completed is True
    assert eval_pending is False
    assert len(buffer.episodes) == 1
    assert buffer.num_eps == 2
    assert len(update_calls) == int(algorithm.cfg.utd) == 1
    assert update_calls == [buffer]
    assert algorithm._num_updates == 1
    assert torch.isfinite(buffer.episodes[0]["action"][-1]).all()
    torch.testing.assert_close(torch.random.get_rng_state(), torch_state, rtol=0, atol=0)
    assert algorithm._outer_policy_action_generator is None


def _install_global_rng_behavior(monkeypatch, algorithm):
    def fake_act(obs, **kwargs):
        del obs, kwargs
        value = (
            random.random()
            + float(np.random.random())
            + float(torch.rand(()))
        ) / 3.0
        algorithm.agent.last_inner_metrics = {
            "inner_active": 1.0,
            "inner_rollouts": 0.0,
            "inner_steps": 0.0,
            "inner_updates": 0.0,
        }
        algorithm.agent.last_inner_rollout_lengths = []
        return torch.tensor([2.0 * value - 1.0], dtype=torch.float32)

    def fake_update(replay):
        del replay
        random.random()
        np.random.random()
        torch.rand(())
        return {}

    monkeypatch.setattr(algorithm.agent, "act", fake_act)
    monkeypatch.setattr(algorithm.agent, "update", fake_update)


def test_probability_zero_matches_direct_legacy_base_loop(monkeypatch):
    wrapped = _make_model(0.0)
    direct = _make_model(0.0)
    wrapped.buffer = _CaptureBuffer()
    direct.buffer = _CaptureBuffer()
    for algorithm in (wrapped, direct):
        algorithm._global_step = 1
        algorithm._pretrained = True
        _install_global_rng_behavior(monkeypatch, algorithm)

    original_python = random.getstate()
    original_numpy = np.random.get_state()
    original_torch = torch.random.get_rng_state().clone()
    random.seed(991)
    np.random.seed(991)
    torch.manual_seed(991)
    initial_python = random.getstate()
    initial_numpy = np.random.get_state()
    initial_torch = torch.random.get_rng_state().clone()
    try:
        wrapped_obs, _ = wrapped._reset_env(seed=41)
        wrapped_result = wrapped._run_training_episode(
            wrapped_obs,
            2,
            eval_pending=False,
        )
        wrapped_rng = (
            random.getstate(),
            np.random.get_state(),
            torch.random.get_rng_state().clone(),
        )

        random.setstate(initial_python)
        np.random.set_state(initial_numpy)
        torch.random.set_rng_state(initial_torch)
        direct_obs, _ = direct._reset_env(seed=41)
        direct_result = TDMPC2Baseline._run_training_episode(
            direct,
            direct_obs,
            2,
            eval_pending=False,
        )
        direct_rng = (
            random.getstate(),
            np.random.get_state(),
            torch.random.get_rng_state().clone(),
        )
    finally:
        wrapped.env.close()
        direct.env.close()
        random.setstate(original_python)
        np.random.set_state(original_numpy)
        torch.random.set_rng_state(original_torch)

    assert wrapped_result == direct_result == (True, False)
    assert wrapped._num_updates == direct._num_updates == 1
    assert len(wrapped.buffer.episodes) == len(direct.buffer.episodes) == 1
    for key in ("obs", "action", "reward", "terminated"):
        torch.testing.assert_close(
            wrapped.buffer.episodes[0][key],
            direct.buffer.episodes[0][key],
            rtol=0,
            atol=0,
            equal_nan=True,
        )
    assert wrapped_rng[0] == direct_rng[0]
    _assert_numpy_rng_equal(wrapped_rng[1], direct_rng[1])
    torch.testing.assert_close(wrapped_rng[2], direct_rng[2], rtol=0, atol=0)


class _OuterModel(torch.nn.Module):
    def __init__(self, *, fail=False, nonfinite=False):
        super().__init__()
        self.child = torch.nn.Dropout(p=0.5)
        self.fail = fail
        self.nonfinite = nonfinite
        self.encoded_shapes = []
        self.policy_calls = []

    def encode(self, obs):
        self.encoded_shapes.append(tuple(obs.shape))
        # Exercise isolation for an implicit default-generator draw.
        torch.rand(())
        return obs.float()

    def pi_action(self, z, *, deterministic, generator):
        self.policy_calls.append((deterministic, generator))
        if self.fail:
            raise RuntimeError("outer actor failed")
        action = torch.rand(
            (z.shape[0], 2),
            dtype=z.dtype,
            device=z.device,
            generator=generator,
        )
        if self.nonfinite:
            action[0, 0] = float("nan")
        return action


def _outer_agent(model):
    agent = object.__new__(AMBITDMPC2Agent)
    torch.nn.Module.__init__(agent)
    agent.device = torch.device("cpu")
    agent.model = model
    agent.last_inner_metrics = {"stale": 1.0}
    agent.last_inner_rollout_lengths = [7]
    return agent


def test_direct_outer_actor_is_stochastic_online_and_rng_mode_isolated():
    model = _OuterModel()
    model.train()
    model.child.eval()
    modes = tuple(module.training for module in model.modules())
    agent = _outer_agent(model)
    generator = torch.Generator(device="cpu").manual_seed(77)

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    first = agent.act_outer_policy(torch.zeros(3), generator=generator)
    second = agent.act_outer_policy(torch.zeros(3), generator=generator)

    assert not torch.equal(first, second)
    assert model.encoded_shapes == [(1, 3), (1, 3)]
    assert all(deterministic is False for deterministic, _ in model.policy_calls)
    assert all(used is generator for _, used in model.policy_calls)
    assert tuple(module.training for module in model.modules()) == modes
    assert agent.last_inner_metrics == {}
    assert agent.last_inner_rollout_lengths == []
    assert random.getstate() == python_state
    _assert_numpy_rng_equal(np.random.get_state(), numpy_state)
    torch.testing.assert_close(torch.random.get_rng_state(), torch_state, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("model", "error"),
    [(_OuterModel(fail=True), RuntimeError), (_OuterModel(nonfinite=True), ValueError)],
)
def test_direct_outer_actor_restores_rng_and_mode_on_failure(model, error):
    model.train()
    model.child.eval()
    modes = tuple(module.training for module in model.modules())
    agent = _outer_agent(model)
    generator = torch.Generator(device="cpu").manual_seed(11)
    torch_state = torch.random.get_rng_state().clone()

    with pytest.raises(error):
        agent.act_outer_policy(torch.zeros(3), generator=generator)

    assert tuple(module.training for module in model.modules()) == modes
    torch.testing.assert_close(torch.random.get_rng_state(), torch_state, rtol=0, atol=0)
    assert agent.last_inner_metrics == {}
    assert agent.last_inner_rollout_lengths == []


def test_outer_action_telemetry_has_planned_only_denominator_and_zero_inner_work(
    monkeypatch,
):
    algorithm = object.__new__(AMBITDMPC2)
    algorithm._wandb_train_window = WandbAccumulator()
    algorithm._outer_policy_episode_selected = False
    algorithm._inner_steps_total = 0
    algorithm._inner_updates_total = 0
    algorithm._wandb_inner_seconds = 0.0
    algorithm._wandb_inner_actions = 0
    algorithm._wandb_inner_steps = 0
    algorithm._wandb_outer_policy_seconds = 0.0
    algorithm._wandb_outer_policy_actions = 0
    algorithm._wandb_train_seconds = 0.0
    algorithm.agent = SimpleNamespace(
        last_inner_metrics={"stale": 123.0},
        last_inner_rollout_lengths=[9],
    )

    # Seed/random actions contribute to neither side of the planned-action
    # fraction. One inner and one outer action therefore produce exactly 1/2.
    algorithm._record_action_metrics(planned=False, action_seconds=9.0)
    algorithm.agent.last_inner_metrics = {
        "inner_active": 1.0,
        "inner_rollouts": 0.0,
        "inner_steps": 0.0,
        "inner_updates": 0.0,
    }
    algorithm.agent.last_inner_rollout_lengths = []
    algorithm._record_action_metrics(planned=True, action_seconds=0.3)
    algorithm._outer_policy_episode_selected = True
    algorithm.agent.last_inner_metrics = {"stale": 999.0}
    algorithm.agent.last_inner_rollout_lengths = [99]
    algorithm._record_action_metrics(planned=True, action_seconds=0.2)

    payload = algorithm._wandb_train_window.pop()
    assert payload["train/outer_policy_action_fraction"] == pytest.approx(0.5)
    assert payload["train/outer_policy_actions"] == 1
    assert payload["train/inner_behavior_actions"] == 1
    assert payload["train/inner_actions"] == 1
    assert payload["train/inner_model_steps"] == 0
    assert payload["train/inner_active"] == pytest.approx(1 / 3)

    monkeypatch.setattr(
        TDMPC2Baseline,
        "_timing_wandb_payload",
        lambda self, updates_since_log: {},
    )
    timing = algorithm._timing_wandb_payload(0)
    assert timing["time/outer_policy_action_seconds"] == pytest.approx(0.2)
    assert timing["time/outer_policy_seconds_per_action"] == pytest.approx(0.2)
    assert timing["time/inner_action_seconds"] == pytest.approx(0.3)
    assert timing["time/inner_seconds_per_action"] == pytest.approx(0.3)


def _logging_stub(*, cadence, extras):
    algorithm = object.__new__(TDMPC2Baseline)
    algorithm._wandb_run = object()
    algorithm._wandb_every = 10
    algorithm._global_step = 10 if cadence else 9
    algorithm._episode_idx = 3
    algorithm._episode_return = 7.0
    algorithm._episode_len = 2
    algorithm._num_updates = 0
    algorithm._wandb_last_updates = 0
    algorithm._last_wandb_step = None
    algorithm._wandb_window_start_step = 0
    algorithm._wandb_train_window = WandbAccumulator()
    algorithm._wandb_reward_window = WandbAccumulator()
    algorithm._wandb_update_window = _DeviceMeanAccumulator()
    algorithm._wandb_reward_window.add_sum("rollout/reward_sentinel", 1.0)
    algorithm._wandb_train_window.add_sum("train/window_sentinel", 2.0)
    algorithm._wandb_update_window.update({
        "train/update_sentinel": torch.tensor(3.0)
    })
    algorithm._wandb_train_seconds = 4.0
    algorithm._wandb_planner_seconds = 5.0
    algorithm.cfg = SimpleNamespace(seed_steps=0)
    algorithm.buffer = SimpleNamespace(num_eps=1)
    algorithm._episode_payload_extras = MethodType(lambda self: extras, algorithm)
    algorithm._replay_wandb_payload = MethodType(lambda self: {}, algorithm)
    algorithm._resolve_reward_components = MethodType(lambda self, info: {}, algorithm)
    algorithm._timing_wandb_payload = MethodType(lambda self, updates: {}, algorithm)
    algorithm._extra_wandb_payload = MethodType(lambda self, updates: {}, algorithm)
    return algorithm


@pytest.mark.parametrize("cadence", [False, True])
def test_episode_payload_hook_rejects_collisions_on_sparse_and_combined_paths(
    monkeypatch,
    cadence,
):
    monkeypatch.setattr(tdmpc2_module, "log_wandb", lambda *args, **kwargs: None)
    collision = "train/window_sentinel" if cadence else "episode/index"
    algorithm = _logging_stub(cadence=cadence, extras={collision: 1})
    with pytest.raises(ValueError, match="collide"):
        algorithm._log_wandb_step(
            1.0,
            False,
            True,
            completed_episode=True,
        )
    assert algorithm._wandb_reward_window.snapshot() == {
        "rollout/reward_sentinel": 1.0
    }
    assert algorithm._wandb_train_window.snapshot() == {
        "train/window_sentinel": 2.0
    }
    assert algorithm._wandb_update_window.floats() == {
        "train/update_sentinel": 3.0
    }
    assert algorithm._wandb_train_seconds == 4.0
    assert algorithm._wandb_planner_seconds == 5.0


def test_ambi_episode_payload_and_run_name_contract():
    algorithm = object.__new__(AMBITDMPC2)
    algorithm.cfg = SimpleNamespace(seed=5)
    algorithm.custom_params = {}
    algorithm.run_params = {"name": "humanoid-walk-base-v1", "env": "fallback"}
    algorithm._outer_policy_episode_selected = True
    algorithm._outer_policy_episode_eligible = True
    assert algorithm._episode_payload_extras() == {
        "rollout/outer_policy_episode": 1,
        "rollout/outer_policy_episode_eligible": 1,
    }
    assert algorithm._wandb_run_name() == (
        "AMBITDMPC2-humanoid-walk-base-v1-seed5"
    )

    algorithm.custom_params = {"wandb_run_name": "explicit-name"}
    assert algorithm._wandb_run_name() == "explicit-name"
