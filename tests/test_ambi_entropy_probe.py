"""Scientific pairing, actual-step attribution, and observer parity."""
import math

import numpy as np
import pytest
import torch

from RL.tdmpc2_core.common import math as td_math
from RL.tdmpc2_core.inner_improvement import InnerImprovementEngine
from tests.test_ambi_entropy_probe_invariants import _fixture
from utils.ambi_entropy_probe import (
    _gradient, _gradient_summary, _hash, _preserve_environment,
    entropy_components, probe_entropy_root,
)


@pytest.mark.parametrize("representation", ["scalar", "distributional"])
def test_observation_is_identical_to_unobserved_actual_engine_with_dropout(representation):
    wrapper = _fixture(representation)
    observation = np.asarray([0.2, -0.1, .7], np.float32)
    try:
        result = probe_entropy_root(wrapper, observation, seed=197,
                                    actor_steps=3, anchor_samples=8)
        model = wrapper.agent.model
        with _preserve_environment(model, wrapper.agent.device):
            with torch.no_grad():
                root = model.encode(torch.as_tensor(observation)[None]).detach()
            engine = InnerImprovementEngine(wrapper.agent)
            engine.rng = engine._new_rng(197)
            action, metrics, _ = engine.act(root, t0=True, eval_mode=True,
                                            collect_diagnostics=False)
            baseline = result["exact_engine"]
            np.testing.assert_array_equal(np.asarray(baseline["action"]), action.detach().numpy().squeeze())
            assert baseline["final_actor_sha256"] == _hash(engine._action_pool.actor.state_dict())
            assert baseline["final_critic_sha256"] == _hash(engine._action_pool.critic.state_dict())
            assert baseline["final_private_rng_sha256"] == _hash(engine.rng.training_state_dict())
            for key in ("inner_actor_entropy", "inner_actor_loss", "inner_alpha_final", "inner_temperature_loss"):
                assert baseline["metrics"][key] == float(metrics[key])
        assert len(baseline["actor_updates"]) == 4
        assert all(row["gradient_mode"].startswith("actual_actor_training_forward")
                   for row in baseline["actor_updates"])
        assert all(row["actual_update_critic_training"] for row in baseline["actor_updates"])
    finally:
        wrapper.env.close()


def test_counterfactuals_pair_every_sample_and_alpha_zero_remains_stochastic():
    wrapper = _fixture()
    try:
        result = probe_entropy_root(wrapper, [1., 0., 0.], seed=19,
                                    actor_steps=3, anchor_samples=8, exact_engine=False)
        assert not result["exact_engine"]["performed"]
        arms = result["actor_arms"]
        for key in ("paired_sample_id", "initial_actor_sha256", "frozen_critic_sha256", "fixed_replay_sha256"):
            assert len({arm[key] for arm in arms.values()}) == 1
        assert arms["tenth_auto"]["updates"][0]["alpha_before"] == pytest.approx(
            .1 * arms["inherited_auto"]["updates"][0]["alpha_before"])
        zero = arms["zero_entropy"]
        assert zero["policy_sampling"].startswith("stochastic")
        assert all(row["alpha_before"] == row["alpha_after"] == 0 for row in zero["updates"])
        assert all(row["temperature_before"] is None and row["temperature_after"] is None
                   for row in zero["updates"])
        assert all(row["temperature_grad_norm_pre_clip"] == 0 for row in zero["updates"])
        # The initial sampled policy is identical, even for the alpha-zero arm.
        assert len({arm["updates"][0]["training_batch_entropy"] for arm in arms.values()}) == 1
    finally:
        wrapper.env.close()


def test_entropy_decomposition_keeps_sampled_and_analytic_gaussian_terms_distinct():
    mu = torch.tensor([[2., 0.], [2., 0.]], dtype=torch.float64)
    logstd = torch.full_like(mu, math.log(.5))
    eps = torch.tensor([[0., 0.], [.1, -.2]], dtype=torch.float64)
    z = mu + logstd.exp() * eps
    _, _, logprob = td_math.squash(mu, z, td_math.gaussian_logprob(eps, logstd))
    result = entropy_components(dict(log_prob=logprob, log_std=logstd,
                                     pre_tanh_action=z, pre_tanh_mean=mu))
    assert result["entropy_mc"] == pytest.approx(
        result["entropy_gaussian_sampled"] + result["entropy_tanh_correction_mc"])
    assert result["entropy_gaussian_analytic"] == pytest.approx(2 * math.log(.5) + math.log(2 * math.pi * math.e))
    assert abs(result["entropy_gaussian_sampled"] - result["entropy_gaussian_analytic"]) > .5


def test_gradient_projection_uses_positive_objective_gradients_and_actual_delta():
    actor = torch.nn.Linear(2, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        actor.weight.copy_(torch.tensor([[1., 2.]]))
    theta = actor.weight.reshape(-1)
    q, entropy = theta[0] ** 2 + 3 * theta[1], 2 * theta[0] - theta[1]
    gq, gh = _gradient(q, actor), _gradient(entropy, actor)
    delta = torch.tensor([.03, -.02], dtype=torch.float64)
    result = _gradient_summary(gq, gh, alpha=.4, delta=delta)
    assert result["return_q_first_order_change"] == pytest.approx(0.)
    assert result["entropy_first_order_change"] == pytest.approx(.08)
    assert actor.weight.grad is None


def test_rejects_scaled_or_nonfresh_protocol_instead_of_silent_reinterpretation():
    wrapper = _fixture()
    try:
        wrapper.cfg.sac_actor_loss_scale_mode = "tdmpc2_percentile_range"
        with pytest.raises(ValueError, match="sac_actor_loss_scale_mode"):
            probe_entropy_root(wrapper, [1., 0., 0.], seed=1, actor_steps=1, anchor_samples=2)
    finally:
        wrapper.env.close()
