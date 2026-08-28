import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from RL.sac_core import SACAgent, SACConfig, SquashedGaussianActor
from RL.sac_value_calibration import (
    PAPER_DETERMINISTIC,
    STOCHASTIC_SOFT_BELLMAN,
    SACValueCalibrationEvaluator,
)


class _Actor(torch.nn.Module):
    def forward(self, observation, deterministic=False):
        assert deterministic is True
        return torch.full(
            (observation.shape[0], 1),
            0.4,
            dtype=observation.dtype,
            device=observation.device,
        )


class _ModeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))


class _FakeAgent:
    def __init__(
        self,
        *,
        alpha=0.25,
        pair_size=2,
        target_reduction="min_pair",
        nonfinite=None,
    ):
        self.device = torch.device("cpu")
        self.obs_dim = 1
        self.action_dim = 1
        self.actor = _Actor()
        self.critic = _ModeModule()
        self.critic_target = _ModeModule()
        self.q_backend = SimpleNamespace(num_q=2, pair_size=pair_size)
        self.config = SimpleNamespace(q_target_reduction=target_reduction)
        self.entropy_coefficient = torch.tensor([alpha], dtype=torch.float32)
        self.nonfinite = nonfinite
        self.sample_calls = []
        self.q_calls = []

    def sample_action_log_prob(self, observation, *, generator=None):
        assert generator is not None
        # Consume the private stream so repeated evaluator calls also test that
        # their namespaced generators restart independently of global RNG state.
        torch.rand((), generator=generator)
        self.sample_calls.append(
            (observation.detach().clone(), int(generator.initial_seed()))
        )
        action = observation[:, :1] * 0.1
        log_prob = -observation[:, :1]
        if self.nonfinite == "action":
            action = torch.full_like(action, float("nan"))
        if self.nonfinite == "log_prob":
            log_prob = torch.full_like(log_prob, float("nan"))
        return action, log_prob

    def q_values(self, observation, action, *, target=False):
        self.q_calls.append(
            {
                "observation": observation.detach().clone(),
                "action": action.detach().clone(),
                "target": bool(target),
            }
        )
        offsets = (5.0, 7.0) if target else (3.0, 5.0)
        if self.nonfinite == "q":
            offsets = (float("nan"), offsets[1])
        return tuple(
            torch.full(
                (observation.shape[0], 1),
                offset,
                dtype=observation.dtype,
                device=observation.device,
            )
            for offset in offsets
        )


class _TwoStepEnv:
    def __init__(
        self,
        *,
        boundary="truncated",
        fail=False,
        consume_global_rng=False,
    ):
        self.boundary = boundary
        self.fail = fail
        self.consume_global_rng = consume_global_rng
        self.reset_seeds = []
        self.actions = []
        self.close_calls = 0
        self._step = 0

    @staticmethod
    def _consume_rng():
        random.random()
        np.random.random()
        torch.rand(())

    def reset(self, *, seed=None):
        if self.consume_global_rng:
            self._consume_rng()
        self.reset_seeds.append(seed)
        self._step = 0
        return np.array([1.0], dtype=np.float32), {}

    def step(self, action):
        if self.consume_global_rng:
            self._consume_rng()
        if self.fail:
            raise RuntimeError("synthetic environment failure")
        self.actions.append(float(np.asarray(action).reshape(-1)[0]))
        self._step += 1
        at_boundary = self._step == 2
        terminated = at_boundary and self.boundary == "terminated"
        truncated = at_boundary and self.boundary == "truncated"
        return (
            np.array([1.0 + self._step], dtype=np.float32),
            (2.0, 4.0)[self._step - 1],
            terminated,
            truncated,
            {},
        )

    def close(self):
        self.close_calls += 1


class _Factory:
    def __init__(self, env):
        self.env = env
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.env


def _evaluator(agent=None, env=None, **overrides):
    arguments = {
        "agent": _FakeAgent() if agent is None else agent,
        "env_factory": _Factory(_TwoStepEnv() if env is None else env),
        "observation_to_array": lambda observation: observation,
        "unscale_action": lambda action: action,
        "discount": 0.5,
        "samples": 1,
        "seed": 17,
        "protocols": (STOCHASTIC_SOFT_BELLMAN,),
        "device": "cpu",
    }
    arguments.update(overrides)
    return SACValueCalibrationEvaluator(**arguments)


def _assert_numpy_rng_equal(left, right):
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def test_stochastic_soft_protocol_pairs_action_and_matches_entropy_and_tail_math():
    agent = _FakeAgent(alpha=0.25)
    env = _TwoStepEnv(boundary="truncated")
    evaluator = _evaluator(agent=agent, env=env)

    metrics = evaluator.evaluate()

    # a0=0.1 is queried and then executed. Its log pi=-1 is excluded because
    # this is Q(s0,a0). At t=1, log pi=-2 contributes the entropy bonus.
    assert float(agent.q_calls[0]["action"].item()) == pytest.approx(0.1)
    assert env.actions == pytest.approx([0.1, 0.2])
    assert agent.q_calls[0]["target"] is False
    assert agent.q_calls[1]["target"] is True
    assert float(agent.q_calls[1]["action"].item()) == pytest.approx(0.3)

    reward_mc = 2.0 + 0.5 * 4.0
    finite_soft = 2.0 + 0.5 * (4.0 - 0.25 * -2.0)
    tail = 0.5**2 * (5.0 - 0.25 * -3.0)
    corrected = finite_soft + tail
    assert metrics["eval/stochastic_reward_mc_value"] == pytest.approx(reward_mc)
    assert metrics["eval/stochastic_soft_mc_finite_value"] == pytest.approx(
        finite_soft
    )
    assert metrics["eval/stochastic_soft_mc_bootstrapped_value"] == pytest.approx(
        corrected
    )
    assert metrics["eval/stochastic_soft_truncation_tail"] == pytest.approx(tail)
    assert metrics["eval/stochastic_soft_truncation_fraction"] == 1.0
    assert metrics["eval/stochastic_soft_q_mean_all"] == pytest.approx(4.0)
    assert metrics["eval/stochastic_soft_q_min_all"] == pytest.approx(3.0)
    assert metrics["eval/stochastic_soft_q_head_std"] == pytest.approx(1.0)
    assert metrics[
        "eval/stochastic_soft_q_minus_mc_bootstrapped_mean_all"
    ] == pytest.approx(4.0 - corrected)
    assert metrics[
        "eval/stochastic_soft_q_rmse_bootstrapped_mean_all"
    ] == pytest.approx(abs(4.0 - corrected))
    assert metrics[
        "eval/stochastic_soft_q_minus_mc_bootstrapped_min_all"
    ] == pytest.approx(3.0 - corrected)
    assert metrics[
        "eval/stochastic_soft_q_rmse_bootstrapped_min_all"
    ] == pytest.approx(abs(3.0 - corrected))
    assert metrics["eval/stochastic_soft_alpha"] == pytest.approx(0.25)
    assert metrics["eval/value_samples"] == 1.0
    assert metrics["time/value_eval_seconds"] >= 0.0

    evaluator.close()


def test_true_termination_has_no_bootstrap_tail():
    evaluator = _evaluator(env=_TwoStepEnv(boundary="terminated"))

    metrics = evaluator.evaluate()

    assert metrics["eval/stochastic_soft_truncation_tail"] == 0.0
    assert metrics["eval/stochastic_soft_truncation_fraction"] == 0.0
    assert metrics["eval/stochastic_soft_mc_bootstrapped_value"] == pytest.approx(
        metrics["eval/stochastic_soft_mc_finite_value"]
    )
    evaluator.close()


def test_paper_protocol_retains_independent_deterministic_mc_and_q_probes():
    agent = _FakeAgent()
    env = _TwoStepEnv(boundary="truncated")
    evaluator = _evaluator(
        agent=agent,
        env=env,
        protocols=(PAPER_DETERMINISTIC,),
        samples=2,
    )

    metrics = evaluator.evaluate()

    assert metrics["eval/mc_value"] == pytest.approx(4.0)
    assert metrics["eval/mc_value_std"] == 0.0
    assert metrics["eval/q_value"] == pytest.approx(4.0)
    assert metrics["eval/q_value_std"] == 0.0
    assert metrics["eval/q_minus_mc"] == 0.0
    assert len(env.reset_seeds) == 4
    assert env.reset_seeds[0] is not None
    assert env.reset_seeds[1] is None
    assert env.reset_seeds[2] is not None
    assert env.reset_seeds[3] is None
    assert env.reset_seeds[0] != env.reset_seeds[2]
    assert all(call["target"] is False for call in agent.q_calls)
    evaluator.close()


@pytest.mark.parametrize("fail", [False, True])
def test_evaluate_restores_all_module_modes_and_global_rng(fail):
    agent = _FakeAgent()
    agent.actor.train(True)
    agent.critic.train(False)
    agent.critic_target.train(True)
    env = _TwoStepEnv(fail=fail, consume_global_rng=True)
    evaluator = _evaluator(agent=agent, env=env)

    random.seed(101)
    np.random.seed(202)
    torch.manual_seed(303)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()

    if fail:
        with pytest.raises(RuntimeError, match="synthetic environment failure"):
            evaluator.evaluate()
    else:
        evaluator.evaluate()

    assert agent.actor.training is True
    assert agent.critic.training is False
    assert agent.critic_target.training is True
    assert random.getstate() == python_state
    _assert_numpy_rng_equal(np.random.get_state(), numpy_state)
    torch.testing.assert_close(torch.random.get_rng_state(), torch_state, rtol=0, atol=0)
    evaluator.close()


def test_repeated_evaluation_restarts_private_streams_and_close_is_idempotent():
    agent = _FakeAgent()
    env = _TwoStepEnv()
    factory = _Factory(env)
    evaluator = _evaluator(agent=agent, env_factory=factory)
    assert factory.calls == 0

    first = evaluator.evaluate()
    first_seeds = [seed for _, seed in agent.sample_calls]
    agent.sample_calls.clear()
    second = evaluator.evaluate()
    second_seeds = [seed for _, seed in agent.sample_calls]

    assert factory.calls == 1
    assert first_seeds == second_seeds
    for key in set(first) - {"time/value_eval_seconds"}:
        assert second[key] == pytest.approx(first[key])

    evaluator.close()
    evaluator.close()
    assert env.close_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        evaluator.evaluate()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"samples": 0}, "samples"),
        ({"seed": -1}, "seed"),
        ({"discount": 1.1}, "discount"),
        ({"protocols": ()}, "At least one"),
        ({"protocols": ("unknown",)}, "Unknown"),
        (
            {"protocols": (PAPER_DETERMINISTIC, PAPER_DETERMINISTIC)},
            "unique",
        ),
    ],
)
def test_constructor_rejects_invalid_controls(overrides, message):
    with pytest.raises((TypeError, ValueError), match=message):
        _evaluator(**overrides)


def test_paper_protocol_requires_a_two_head_pair():
    with pytest.raises(ValueError, match="q_pair_size=2"):
        _evaluator(
            agent=_FakeAgent(pair_size=1),
            protocols=(PAPER_DETERMINISTIC,),
        )


@pytest.mark.parametrize(
    ("nonfinite", "message"),
    [
        ("action", "non-finite action"),
        ("log_prob", "log-probability"),
        ("q", "non-finite Q"),
    ],
)
def test_nonfinite_policy_and_critic_outputs_fail_closed(nonfinite, message):
    evaluator = _evaluator(agent=_FakeAgent(nonfinite=nonfinite))
    with pytest.raises(ValueError, match=message):
        evaluator.evaluate()
    evaluator.close()


def test_explicit_actor_generator_is_reproducible_without_global_rng_draws():
    torch.manual_seed(7)
    actor = SquashedGaussianActor(2, 1, (8,))
    observation = torch.tensor([[0.25, -0.5]])
    global_state = torch.random.get_rng_state().clone()
    first_generator = torch.Generator().manual_seed(1234)
    second_generator = torch.Generator().manual_seed(1234)

    first_action, first_log_prob = actor.action_log_prob(
        observation, generator=first_generator
    )
    torch.testing.assert_close(torch.random.get_rng_state(), global_state, rtol=0, atol=0)
    second_action, second_log_prob = actor.action_log_prob(
        observation, generator=second_generator
    )

    torch.testing.assert_close(first_action, second_action, rtol=0, atol=0)
    torch.testing.assert_close(first_log_prob, second_log_prob, rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), global_state, rtol=0, atol=0)


def test_agent_exposes_fixed_and_learned_alpha_and_target_selectable_q_values():
    fixed = SACAgent(
        2, 1, SACConfig(net_arch=(4,), ent_coef=0.3, device="cpu", seed=3)
    )
    learned = SACAgent(
        2, 1, SACConfig(net_arch=(4,), ent_coef="auto_0.7", device="cpu", seed=3)
    )
    torch.testing.assert_close(fixed.entropy_coefficient, torch.tensor(0.3))
    torch.testing.assert_close(learned.entropy_coefficient, torch.tensor([0.7]))

    with torch.no_grad():
        for parameter in fixed.critic.parameters():
            parameter.zero_()
        for parameter in fixed.critic_target.parameters():
            parameter.zero_()
        fixed.critic.qf1[-1].bias.fill_(1.0)
        fixed.critic.qf2[-1].bias.fill_(2.0)
        fixed.critic_target.qf1[-1].bias.fill_(3.0)
        fixed.critic_target.qf2[-1].bias.fill_(4.0)

    online = fixed.q_values(torch.zeros(2), torch.zeros(1))
    target = fixed.q_values(torch.zeros(2), torch.zeros(1), target=True)
    assert [float(value.item()) for value in online] == pytest.approx([1.0, 2.0])
    assert [float(value.item()) for value in target] == pytest.approx([3.0, 4.0])


def test_real_sac_agent_runs_both_protocols_with_the_frozen_metric_schema():
    agent = SACAgent(
        1,
        1,
        SACConfig(net_arch=(4,), ent_coef=0.2, device="cpu", seed=9),
    )
    modes_before = (
        agent.actor.training,
        agent.critic.training,
        agent.critic_target.training,
    )
    evaluator = _evaluator(
        agent=agent,
        env=_TwoStepEnv(),
        protocols=(PAPER_DETERMINISTIC, STOCHASTIC_SOFT_BELLMAN),
    )

    metrics = evaluator.evaluate()

    assert set(metrics) == {
        "eval/mc_value",
        "eval/mc_value_std",
        "eval/q_value",
        "eval/q_value_std",
        "eval/q_minus_mc",
        "eval/stochastic_reward_mc_value",
        "eval/stochastic_reward_mc_value_std",
        "eval/stochastic_soft_mc_finite_value",
        "eval/stochastic_soft_mc_finite_value_std",
        "eval/stochastic_soft_mc_bootstrapped_value",
        "eval/stochastic_soft_mc_bootstrapped_value_std",
        "eval/stochastic_soft_truncation_tail",
        "eval/stochastic_soft_truncation_fraction",
        "eval/stochastic_soft_q_mean_all",
        "eval/stochastic_soft_q_mean_all_std",
        "eval/stochastic_soft_q_min_all",
        "eval/stochastic_soft_q_head_std",
        "eval/stochastic_soft_q_minus_mc_bootstrapped_mean_all",
        "eval/stochastic_soft_q_rmse_bootstrapped_mean_all",
        "eval/stochastic_soft_q_minus_mc_bootstrapped_min_all",
        "eval/stochastic_soft_q_rmse_bootstrapped_min_all",
        "eval/stochastic_soft_alpha",
        "eval/value_samples",
        "time/value_eval_seconds",
    }
    assert all(np.isfinite(value) for value in metrics.values())
    assert (
        agent.actor.training,
        agent.critic.training,
        agent.critic_target.training,
    ) == modes_before
    evaluator.close()
