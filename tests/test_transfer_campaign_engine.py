"""Opt-in campaign state, relative prior anchors, and compiled routing."""

from copy import deepcopy

import pytest
import torch

from tests.test_aux_critic_transfer import critic_params
from tests.test_ambi_inner_decoupling import _assert_tree_equal, _clone_tree
from tests.test_ambi_root_local_sac import _model_from_params


def make_model(**overrides):
    options = dict(inner_critic_scope="action", inner_rollout_horizon=3,
                   inner_rounds=1, inner_batch_size=4)
    options.update(overrides)
    return _model_from_params(critic_params(**options))


def solve(model, donor=None, mechanism="fresh", *, compiled=False):
    options = {"allow_compile": compiled}
    if mechanism == "full" and donor is not None:
        options["learner_state"] = donor
    else:
        options["target"] = "online"
        if donor is not None:
            if mechanism in {"replay", "anchors"}:
                options.update(actor=donor["modules"]["actor"], critic=donor["modules"]["critic"])
            if mechanism == "replay":
                options["replay"] = donor["replay"]
            if mechanism == "behavior":
                options["collection_actor"] = donor["modules"]["actor"]
    if mechanism == "anchors":
        options.update(actor_prior_kl_coef=.1, critic_prior_l2_coef=.1)
    with model.agent.inner_engine.diagnostic_initialization(**options):
        action = model.agent.act(torch.tensor([1., .2, -.1]), t0=donor is None, eval_mode=True)
    return action, model.agent.inner_engine.export_diagnostic_state(
        include_optimizers=mechanism == "full", include_replay=mechanism in {"replay", "full"})


def test_lightweight_export_has_independent_device_storage_and_no_adam_copy(monkeypatch):
    model = make_model()
    try:
        _, donor = solve(model)
        engine = model.agent.inner_engine
        for component in ("actor", "critic", "temperature"):
            monkeypatch.setattr(getattr(engine._action_pool, component + "_optim"), "state_dict",
                                lambda: pytest.fail("Lightweight export copied Adam"))
        exported = engine.export_diagnostic_state()
        assert exported["optimizers"] is exported["replay"] is None
        for name in ("actor", "critic", "critic_target"):
            for key, value in exported["modules"][name].items():
                assert value.device == model.agent.device
                assert value.data_ptr() != getattr(engine._action_pool, name).state_dict()[key].data_ptr()
                value.zero_()
            _assert_tree_equal(getattr(engine._action_pool, name).state_dict(), donor["modules"][name])
    finally:
        model.close()


def test_anchor_equations_match_relative_parameter_distance_and_gaussian_kl():
    model = make_model()
    try:
        engine = model.agent.inner_engine
        outer = _clone_tree(model.agent.model.state_dict())
        actor, critic = deepcopy(engine._actor_base.state_dict()), deepcopy(engine._critic_base.state_dict())
        for state in (actor, critic):
            for value in state.values():
                value.add_(.02)
        with engine.diagnostic_initialization(actor=actor, critic=critic, target="online",
                actor_prior_kl_coef=.1, critic_prior_l2_coef=.1):
            with engine.rng.fork("initialization"):
                engine._prepare_workspace(t0=True)
            engine._apply_diagnostic_initialization()
            penalty, gradient_norm = engine._diagnostic_critic_anchor()
            expected = torch.stack([
                (parameter - prior.detach()).square().mean() / (prior.detach().square().mean() + 1e-6)
                for parameter, prior in zip(engine.state.critic.parameters(), engine._critic_base.parameters())
            ]).mean()
            torch.testing.assert_close(penalty, expected)
            gradients = torch.autograd.grad(.1 * penalty, tuple(engine.state.critic.parameters()))
            torch.testing.assert_close(gradient_norm, torch.stack([g.square().sum() for g in gradients]).sum().sqrt())
            z = model.agent.model.encode(torch.ones(2, 3))
            noise = torch.zeros(2, 1)
            anchored = engine._sac_actor_kernel(z, engine.alpha.detach(), noise, None, True)
            engine._diagnostic_actor_prior_kl_coef = 0.
            baseline = engine._sac_actor_kernel(z, engine.alpha.detach(), noise, None, True)
            current = model.agent.model.policy_stats(z, policy=engine.state.actor, **engine._inner_policy_kwargs())
            prior = model.agent.model.policy_stats(z, policy=engine._actor_base, **engine._actor_options)
            expected_kl = torch.distributions.kl_divergence(
                torch.distributions.Normal(current["pre_tanh_mean"], current["log_std"].exp()),
                torch.distributions.Normal(prior["pre_tanh_mean"], prior["log_std"].exp()),
            ).sum(-1).mean()
            torch.testing.assert_close(anchored[2] - baseline[2], .1 * expected_kl, rtol=1e-5, atol=1e-6)
            assert expected_kl > 0
        _assert_tree_equal(model.agent.model.state_dict(), outer)
        assert all(p.grad is None for p in engine._critic_base.parameters())
    finally:
        model.close()


def test_full_state_carries_target_cadence_and_preserves_adam_allocations():
    model = make_model(inner_critic_target_update_interval=3, inner_critic_target_tau=1.)
    try:
        _, donor = solve(model, mechanism="full")
        engine = model.agent.inner_engine
        identities = {component: [[id(value) for value in state.values() if torch.is_tensor(value)]
                                  for state in getattr(engine._action_pool, component + "_optim").state.values()]
                      for component in ("actor", "critic", "temperature")}
        assert model.agent.last_inner_metrics["inner_critic_target_updates"] == 0
        online_steps = []
        original = engine._sac_critic_step
        def record(*args, **kwargs):
            result = original(*args, **kwargs)
            online_steps.append(_clone_tree(engine.state.critic.state_dict()))
            return result
        engine._sac_critic_step = record
        _, second = solve(model, donor, "full")
        assert model.agent.last_inner_metrics["inner_critic_target_updates"] == 1
        assert model.agent.last_inner_metrics["inner_previous_replay_fraction"] == .25
        _assert_tree_equal(second["modules"]["critic_target"], online_steps[0])
        assert second["lifetimes"]["critic"] == 4
        for component in identities:
            actual = [[id(value) for value in state.values() if torch.is_tensor(value)]
                      for state in getattr(engine._action_pool, component + "_optim").state.values()]
            assert actual == identities[component]
    finally:
        model.close()


@pytest.mark.parametrize("mechanism", ["behavior", "replay", "full", "anchors"])
def test_compile_graphs_match_eager_campaign_and_keep_behavior_allocation(monkeypatch, mechanism):
    torch._dynamo.reset()
    original_compile = torch.compile
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: original_compile(fn, backend="eager", **kwargs))
    eager, compiled = make_model(), make_model(compile=True, compile_strict=True)
    try:
        compiled.agent.model.load_state_dict(eager.agent.model.state_dict())
        donors, behavior_ids = [None, None], []
        for _ in range(3):
            outputs = []
            for index, model in enumerate((eager, compiled)):
                action, donors[index] = solve(model, donors[index], mechanism, compiled=index == 1)
                outputs.append(action)
            torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
            _assert_tree_equal(donors[0], donors[1])
            _assert_tree_equal(eager.agent.inner_engine.rng.training_state_dict(),
                               compiled.agent.inner_engine.rng.training_state_dict())
            behavior = compiled.agent.inner_engine._diagnostic_collection_pool
            if behavior is not None:
                behavior_ids.append(id(behavior))
        if mechanism == "behavior":
            assert len(behavior_ids) == 2 and len(set(behavior_ids)) == 1
    finally:
        eager.close()
        compiled.close()
        torch._dynamo.reset()
