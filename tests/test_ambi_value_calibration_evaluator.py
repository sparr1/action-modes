import random

import numpy as np
import pytest
import torch

from RL.tdmpc2_core import value_calibration
from RL.tdmpc2_core.value_calibration import ValueCalibrationEvaluator


class _Factory:
    def __init__(self, env):
        self.env = env
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.env


class _TwoStepEnv:
    def __init__(
        self,
        *,
        reward_from_action=False,
        reward_values=None,
        fail_on_step=False,
        consume_rng=False,
        boundary="truncated",
    ):
        if boundary not in {"terminated", "truncated"}:
            raise ValueError("boundary must be 'terminated' or 'truncated'.")
        self.reward_from_action = reward_from_action
        self.reward_values = reward_values
        self.fail_on_step = fail_on_step
        self.consume_rng = consume_rng
        self.boundary = boundary
        self.reset_seeds = []
        self.actions = []
        self.close_calls = 0
        self._step = 0

    @staticmethod
    def _consume_global_rng():
        random.random()
        np.random.random()
        torch.rand(())

    def reset(self, *, seed=None):
        if self.consume_rng:
            self._consume_global_rng()
        self.reset_seeds.append(seed)
        self._step = 0
        return np.array([1.0], dtype=np.float32), {}

    def step(self, action):
        if self.consume_rng:
            self._consume_global_rng()
        action_scalar = float(np.asarray(action).reshape(-1)[0])
        self.actions.append(action_scalar)
        if self.fail_on_step:
            raise RuntimeError("synthetic environment failure")
        self._step += 1
        if self.reward_from_action:
            reward = action_scalar
        elif self.reward_values is not None:
            reward = self.reward_values[self._step - 1]
        else:
            reward = float(self._step)
        at_boundary = self._step == 2
        terminated = at_boundary and self.boundary == "terminated"
        truncated = at_boundary and self.boundary == "truncated"
        next_observation = np.array([1.0 + self._step], dtype=np.float32)
        return next_observation, reward, terminated, truncated, {}

    def close(self):
        self.close_calls += 1


class _FakeModel(torch.nn.Module):
    def __init__(
        self,
        *,
        deterministic_action=0.0,
        stochastic_action_from_latent=False,
        q_offsets=(1.0, 3.0, 5.0),
        q_from_action=False,
        generator_action=False,
        consume_rng=False,
    ):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.deterministic_action = float(deterministic_action)
        self.stochastic_action_from_latent = stochastic_action_from_latent
        self.register_buffer("q_offsets", torch.tensor(q_offsets, dtype=torch.float32))
        self.q_from_action = q_from_action
        self.generator_action = generator_action
        self.consume_rng = consume_rng
        self.encode_shapes = []
        self.call_modes = []
        self.action_calls = []
        self.q_actions = []
        self.q_targets = []

    @staticmethod
    def _consume_global_rng():
        random.random()
        np.random.random()
        torch.rand(())

    def _record_mode(self):
        if self.consume_rng:
            self._consume_global_rng()
        self.call_modes.append(bool(self.training))

    def encode(self, observation):
        self._record_mode()
        self.encode_shapes.append(tuple(observation.shape))
        return observation.to(torch.float32)

    def pi_action(self, latent, *, deterministic, generator):
        self._record_mode()
        self.action_calls.append(
            {
                "deterministic": bool(deterministic),
                "generator_seed": int(generator.initial_seed()),
                "latent": latent.detach().cpu().clone(),
            }
        )
        if not deterministic and self.generator_action:
            return torch.randn(
                (latent.shape[0], 1),
                dtype=latent.dtype,
                device=latent.device,
                generator=generator,
            )
        if deterministic or not self.stochastic_action_from_latent:
            return torch.full(
                (latent.shape[0], 1),
                self.deterministic_action,
                dtype=latent.dtype,
                device=latent.device,
            )
        return latent[:, :1] + 0.25

    def q_values(self, latent, action, *, target):
        del latent
        self._record_mode()
        self.q_targets.append(bool(target))
        self.q_actions.append(action.detach().cpu().clone())
        base = action.reshape(1, 1, 1) if self.q_from_action else 0.0
        return base + self.q_offsets.reshape(-1, 1, 1)


def _observation_to_tensor(observation):
    return torch.as_tensor(observation, dtype=torch.float32)


def _assert_numpy_rng_equal(left, right):
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def test_paper_protocol_uses_full_mean_policy_rollouts_and_independent_q_probes(
    monkeypatch,
):
    env = _TwoStepEnv()
    factory = _Factory(env)
    model = _FakeModel(
        deterministic_action=0.5,
        q_offsets=(1.0, 3.0, 5.0, 7.0),
    )
    sampled_head_generator_seeds = []

    def fixed_head_permutation(num_heads, *, device, generator):
        assert num_heads == 4
        sampled_head_generator_seeds.append(int(generator.initial_seed()))
        return torch.tensor([3, 1, 0, 2], device=device)

    monkeypatch.setattr(value_calibration.torch, "randperm", fixed_head_permutation)
    evaluator = ValueCalibrationEvaluator(
        model=model,
        env_factory=factory,
        observation_to_tensor=_observation_to_tensor,
        unscale_action=lambda action: action,
        discount=0.5,
        samples=2,
        seed=91,
        protocols="paper_deterministic",
        device="cpu",
    )

    assert factory.calls == 0
    metrics = evaluator.evaluate()

    # Each MC episode returns 1 + .5 * 2 = 2.  The seeded pair is heads 3 and
    # 1, so each independent initial-state Q probe returns (7 + 3) / 2 = 5.
    assert metrics["eval/mc_value"] == pytest.approx(2.0)
    assert metrics["eval/q_value"] == pytest.approx(5.0)
    assert metrics["eval/q_minus_mc"] == pytest.approx(3.0)
    assert metrics["eval/mc_value_std"] == pytest.approx(0.0)
    assert metrics["eval/q_value_std"] == pytest.approx(0.0)
    assert metrics["eval/value_samples"] == pytest.approx(2.0)
    assert metrics["time/value_eval_seconds"] >= 0.0

    assert factory.calls == 1
    assert len(env.reset_seeds) == 4
    assert env.reset_seeds[0] is not None
    assert env.reset_seeds[1] is None
    assert env.reset_seeds[2] is not None
    assert env.reset_seeds[3] is None
    assert env.reset_seeds[0] != env.reset_seeds[2]
    assert len(env.actions) == 4
    assert len(model.q_actions) == 2
    assert model.q_targets == [False, False]
    assert all(call["deterministic"] for call in model.action_calls)
    assert all(shape == (1, 1) for shape in model.encode_shapes)
    assert all(mode is False for mode in model.call_modes)
    assert model.training is True
    assert len(sampled_head_generator_seeds) == 2
    assert len(set(sampled_head_generator_seeds)) == 2

    evaluator.close()
    evaluator.close()
    assert env.close_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        evaluator.evaluate()


@pytest.mark.parametrize("boundary", ["terminated", "truncated"])
def test_discounted_rollouts_stop_on_termination_or_truncation(boundary):
    env = _TwoStepEnv(reward_values=(2.0, 4.0), boundary=boundary)
    evaluator = ValueCalibrationEvaluator(
        model=_FakeModel(q_offsets=(1.0, 2.0)),
        env_factory=_Factory(env),
        observation_to_tensor=_observation_to_tensor,
        unscale_action=lambda action: action,
        discount=0.25,
        samples=1,
        seed=19,
        protocols="paper_deterministic",
        device="cpu",
    )

    metrics = evaluator.evaluate()

    assert metrics["eval/mc_value"] == pytest.approx(2.0 + 0.25 * 4.0)
    assert len(env.actions) == 2
    evaluator.close()


def test_stochastic_protocol_pairs_the_executed_first_action_with_all_online_heads():
    env = _TwoStepEnv(reward_from_action=True)
    factory = _Factory(env)
    model = _FakeModel(
        stochastic_action_from_latent=True,
        q_offsets=(1.0, 3.0, 5.0),
        q_from_action=True,
    )
    evaluator = ValueCalibrationEvaluator(
        model=model,
        env_factory=factory,
        observation_to_tensor=_observation_to_tensor,
        unscale_action=lambda action: action,
        discount=0.5,
        samples=2,
        seed=17,
        protocols=("stochastic_bellman",),
        device="cpu",
    )

    metrics = evaluator.evaluate()

    # a0=1.25 and a1=2.25, hence G=1.25 + .5*2.25=2.375.  Initial Q heads
    # are [2.25, 4.25, 6.25], with mean 4.25 and min 2.25.
    assert metrics["eval/stochastic_mc_value"] == pytest.approx(2.375)
    assert metrics["eval/stochastic_mc_value_std"] == pytest.approx(0.0)
    assert metrics["eval/stochastic_q_mean_all"] == pytest.approx(4.25)
    assert metrics["eval/stochastic_q_mean_all_std"] == pytest.approx(0.0)
    assert metrics["eval/stochastic_q_min_all"] == pytest.approx(2.25)
    assert metrics["eval/stochastic_q_minus_mc_mean_all"] == pytest.approx(1.875)
    assert metrics["eval/stochastic_q_rmse_mean_all"] == pytest.approx(1.875)
    assert metrics["eval/stochastic_q_head_std"] == pytest.approx(
        np.std([2.25, 4.25, 6.25], ddof=0)
    )
    assert metrics["eval/stochastic_q_minus_mc_min_all"] == pytest.approx(-0.125)
    assert metrics["eval/stochastic_q_rmse_min_all"] == pytest.approx(0.125)

    assert len(env.reset_seeds) == 2
    assert env.reset_seeds[0] is not None
    assert env.reset_seeds[1] is None
    assert env.actions == pytest.approx([1.25, 2.25, 1.25, 2.25])
    assert len(model.q_actions) == 2
    for episode, q_action in enumerate(model.q_actions):
        assert float(q_action.item()) == pytest.approx(env.actions[2 * episode])
    assert model.q_targets == [False, False]
    assert all(not call["deterministic"] for call in model.action_calls)
    action_seeds = [call["generator_seed"] for call in model.action_calls]
    assert action_seeds[0] == action_seeds[1]
    assert action_seeds[2] == action_seeds[3]
    assert action_seeds[0] != action_seeds[2]
    assert all(shape == (1, 1) for shape in model.encode_shapes)

    evaluator.close()


def test_repeated_evaluation_restarts_namespaced_rng_banks_without_global_draws():
    env = _TwoStepEnv(reward_from_action=True)
    model = _FakeModel(
        deterministic_action=0.25,
        q_offsets=(1.0, 2.0, 4.0, 8.0),
        q_from_action=True,
        generator_action=True,
    )
    evaluator = ValueCalibrationEvaluator(
        model=model,
        env_factory=_Factory(env),
        observation_to_tensor=_observation_to_tensor,
        unscale_action=lambda action: action,
        discount=0.75,
        samples=2,
        seed=31415,
        protocols=("paper_deterministic", "stochastic_bellman"),
        device="cpu",
    )

    random.seed(111)
    np.random.seed(222)
    torch.manual_seed(333)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()

    first = evaluator.evaluate()
    first_reset_pattern = list(env.reset_seeds)
    assert random.getstate() == python_state
    _assert_numpy_rng_equal(np.random.get_state(), numpy_state)
    assert torch.equal(torch.random.get_rng_state(), torch_state)

    env.reset_seeds.clear()
    second = evaluator.evaluate()
    second_reset_pattern = list(env.reset_seeds)
    assert random.getstate() == python_state
    _assert_numpy_rng_equal(np.random.get_state(), numpy_state)
    assert torch.equal(torch.random.get_rng_state(), torch_state)

    assert first_reset_pattern == second_reset_pattern
    assert len(first_reset_pattern) == 6
    assert first_reset_pattern[1::2] == [None, None, None]
    seeded_batches = first_reset_pattern[0::2]
    assert all(seed is not None for seed in seeded_batches)
    assert len(set(seeded_batches)) == 3
    for key in set(first) - {"time/value_eval_seconds"}:
        assert second[key] == pytest.approx(first[key])

    evaluator.close()


@pytest.mark.parametrize("fail_on_step", [False, True])
def test_mode_and_global_rng_are_restored_on_success_and_environment_error(
    fail_on_step,
):
    env = _TwoStepEnv(fail_on_step=fail_on_step, consume_rng=True)
    factory = _Factory(env)
    model = _FakeModel(
        stochastic_action_from_latent=True,
        q_from_action=True,
        consume_rng=True,
    )
    model.train(False)
    evaluator = ValueCalibrationEvaluator(
        model=model,
        env_factory=factory,
        observation_to_tensor=_observation_to_tensor,
        unscale_action=lambda action: action,
        discount=0.9,
        samples=1,
        seed=7,
        protocols="stochastic_bellman",
        device="cpu",
    )

    random.seed(1234)
    np.random.seed(5678)
    torch.manual_seed(9012)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    cuda_state = None
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        cuda_state = [state.clone() for state in torch.cuda.get_rng_state_all()]

    assert factory.calls == 0
    if fail_on_step:
        with pytest.raises(RuntimeError, match="synthetic environment failure"):
            evaluator.evaluate()
    else:
        metrics = evaluator.evaluate()
        assert metrics["eval/value_samples"] == 1.0

    assert factory.calls == 1
    assert model.training is False
    assert random.getstate() == python_state
    _assert_numpy_rng_equal(np.random.get_state(), numpy_state)
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    if cuda_state is not None:
        restored_cuda_state = torch.cuda.get_rng_state_all()
        assert len(restored_cuda_state) == len(cuda_state)
        assert all(
            torch.equal(restored, expected)
            for restored, expected in zip(restored_cuda_state, cuda_state)
        )
    assert all(mode is False for mode in model.call_modes)

    evaluator.close()
    evaluator.close()
    assert env.close_calls == 1


def test_nonfinite_actions_q_heads_and_accumulated_returns_fail_closed():
    cases = (
        (
            _FakeModel(deterministic_action=float("nan"), q_offsets=(1.0, 2.0)),
            _TwoStepEnv(),
            "non-finite action",
        ),
        (
            _FakeModel(q_offsets=(float("nan"), 2.0)),
            _TwoStepEnv(),
            "non-finite Q head",
        ),
        (
            _FakeModel(q_offsets=(1.0, 2.0)),
            _TwoStepEnv(reward_values=(1e308, 1e308)),
            "Monte Carlo return",
        ),
    )
    for model, env, message in cases:
        evaluator = ValueCalibrationEvaluator(
            model=model,
            env_factory=_Factory(env),
            observation_to_tensor=_observation_to_tensor,
            unscale_action=lambda action: action,
            discount=1.0,
            samples=1,
            seed=5,
            protocols="paper_deterministic",
            device="cpu",
        )
        with pytest.raises(ValueError, match=message):
            evaluator.evaluate()
        assert model.training is True
        evaluator.close()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"samples": 0}, "samples"),
        ({"protocols": ()}, "At least one"),
        ({"protocols": ("unknown",)}, "Unknown"),
        ({"protocols": ("paper_deterministic", "paper_deterministic")}, "unique"),
        ({"discount": 1.1}, "discount"),
    ],
)
def test_constructor_rejects_invalid_protocol_controls(kwargs, message):
    arguments = {
        "model": _FakeModel(),
        "env_factory": lambda: _TwoStepEnv(),
        "observation_to_tensor": _observation_to_tensor,
        "unscale_action": lambda action: action,
        "discount": 0.99,
        "samples": 1,
        "seed": 1,
        "protocols": ("paper_deterministic",),
        "device": "cpu",
    }
    arguments.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=message):
        ValueCalibrationEvaluator(**arguments)
