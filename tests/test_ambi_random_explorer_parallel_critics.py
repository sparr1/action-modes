"""Fused paired-update checks for the separate-critics explorer mode."""

from copy import deepcopy

import pytest
import torch

from tests.test_ambi_random_explorer_engine_invariants import (
    _assert_tree_equal,
    _explorer_model,
    _flat_parameters,
    _module_parameter_ids,
    _optimizer_parameter_ids,
    _optimizer_steps,
)
from tests.test_ambi_random_explorer_targets_and_resume import (
    _batch,
    _prepare_action_workspace,
)


def _constant_target_components(engine, monkeypatch):
    """Make the two bootstrap targets deterministic and visibly different."""

    def controlled_policy(z, *, policy, **kwargs):
        del kwargs
        primary = policy is engine.state.actor
        action = z.new_zeros(z.shape[0], int(engine.cfg.action_dim))
        log_prob = z.new_full((z.shape[0], 1), -1.0 if primary else -2.0)
        return action, {
            "pre_tanh_action": action,
            "pre_tanh_mean": action,
            "log_std": torch.zeros_like(action),
            "log_prob": log_prob,
        }

    def controlled_q(z, action, critic, *, reduction=None):
        del action, reduction
        if critic is engine.state.critic_target:
            return z.new_full((z.shape[0], 1), 3.0)
        assert critic is engine.state.explorer_critic_target
        return z.new_full((z.shape[0], 1), 7.0)

    monkeypatch.setattr(engine.model, "pi", controlled_policy)
    monkeypatch.setattr(engine, "_q_with", controlled_q)


def test_paired_critic_step_builds_both_losses_before_one_backward_and_steps(
    monkeypatch,
):
    model = _explorer_model(
        "separate_critics",
        inner_actor_updates_per_round=0,
        inner_explorer_actor_updates_per_round=0,
    )
    try:
        engine = _prepare_action_workspace(model)
        batch = _batch(engine, source=[0, 1, 0, 1, 0, 1, 0, 1])
        _constant_target_components(engine, monkeypatch)
        events = []
        online_inputs = {}

        original_predictions = engine.model.q_predictions

        def record_predictions(z, action, *args, **kwargs):
            critic = kwargs.get("qs")
            if critic is engine.state.critic:
                label = "primary"
            elif critic is engine.state.explorer_critic:
                label = "explorer"
            else:
                return original_predictions(z, action, *args, **kwargs)
            events.append(f"{label}_forward")
            online_inputs[label] = (z.data_ptr(), action.data_ptr())
            return original_predictions(z, action, *args, **kwargs)

        original_loss = engine.model.critic_loss
        loss_count = 0

        def record_loss(predictions, target, **kwargs):
            nonlocal loss_count
            label = "primary" if loss_count == 0 else "explorer"
            loss_count += 1
            events.append(f"{label}_loss")
            return original_loss(predictions, target, **kwargs)

        original_backward = torch.autograd.backward

        def record_backward(*args, **kwargs):
            events.append("backward")
            return original_backward(*args, **kwargs)

        primary_optimizer = engine.state.critic_optim
        explorer_optimizer = engine.state.explorer_critic_optim
        original_primary_step = primary_optimizer.step
        original_explorer_step = explorer_optimizer.step

        def primary_step(*args, **kwargs):
            events.append("primary_step")
            return original_primary_step(*args, **kwargs)

        def explorer_step(*args, **kwargs):
            events.append("explorer_step")
            return original_explorer_step(*args, **kwargs)

        monkeypatch.setattr(engine.model, "q_predictions", record_predictions)
        monkeypatch.setattr(engine.model, "critic_loss", record_loss)
        monkeypatch.setattr(torch.autograd, "backward", record_backward)
        monkeypatch.setattr(primary_optimizer, "step", primary_step)
        monkeypatch.setattr(explorer_optimizer, "step", explorer_step)

        with engine.rng.action_fork():
            with engine.rng.fork("bootstrap"):
                engine._separate_critics_step(
                    batch, update_primary=True, update_explorer=True
                )

        assert loss_count == 2
        assert events.count("backward") == 1
        first_mutation = min(
            events.index("primary_step"), events.index("explorer_step")
        )
        for required in (
            "primary_forward",
            "primary_loss",
            "explorer_forward",
            "explorer_loss",
            "backward",
        ):
            assert events.index(required) < first_mutation
        assert events.index("backward") > events.index("explorer_loss")
        assert online_inputs["primary"] == online_inputs["explorer"]

        primary_ids = _module_parameter_ids(engine.state.critic)
        explorer_ids = _module_parameter_ids(engine.state.explorer_critic)
        assert primary_ids.isdisjoint(explorer_ids)
        assert _optimizer_parameter_ids(primary_optimizer) == primary_ids
        assert _optimizer_parameter_ids(explorer_optimizer) == explorer_ids
        assert primary_ids.isdisjoint(_optimizer_parameter_ids(explorer_optimizer))
        assert explorer_ids.isdisjoint(_optimizer_parameter_ids(primary_optimizer))
        assert any(
            parameter.grad is not None and torch.count_nonzero(parameter.grad)
            for parameter in engine.state.critic_params
        )
        assert any(
            parameter.grad is not None and torch.count_nonzero(parameter.grad)
            for parameter in engine.state.explorer_critic_params
        )
        for module in (
            engine.state.actor,
            engine.state.explorer_actor,
            engine.state.critic_target,
            engine.state.explorer_critic_target,
            engine.model._pi,
            engine.model._Qs,
        ):
            assert all(parameter.grad is None for parameter in module.parameters())
    finally:
        model.env.close()


def test_fused_paired_critic_update_matches_independent_reference_steps(monkeypatch):
    model = _explorer_model(
        "separate_critics",
        inner_actor_updates_per_round=0,
        inner_explorer_actor_updates_per_round=0,
    )
    try:
        engine = _prepare_action_workspace(model)
        batch = _batch(engine, source=[0, 1, 0, 1, 0, 1, 0, 1])
        _constant_target_components(engine, monkeypatch)
        with torch.no_grad():
            engine.state.log_alpha.fill_(torch.tensor(0.2).log())
            engine.state.explorer_log_alpha.fill_(torch.tensor(0.7).log())

        reference_primary = deepcopy(engine.state.critic)
        reference_explorer = deepcopy(engine.state.explorer_critic)
        reference_primary_optimizer = engine._new_optimizer(
            reference_primary, "critic"
        )
        reference_explorer_optimizer = engine._new_optimizer(
            reference_explorer, "critic"
        )
        target_primary_before = deepcopy(engine.state.critic_target.state_dict())
        target_explorer_before = deepcopy(
            engine.state.explorer_critic_target.state_dict()
        )
        outer_before = deepcopy(engine.model._Qs.state_dict())

        with engine.rng.action_fork():
            with engine.rng.fork("bootstrap"):
                metrics = engine._separate_critics_step(
                    batch, update_primary=True, update_explorer=True
                )

        discount = float(model.agent.discount)
        continuation = discount * (1.0 - batch["terminated"])
        primary_target = batch["reward"] + continuation * (3.0 - 0.2 * -1.0)
        explorer_target = batch["reward"] + continuation * (7.0 - 0.7 * -2.0)

        reference_primary_predictions = engine.model.q_predictions(
            batch["z"], batch["action"], qs=reference_primary
        )
        reference_primary_loss = engine.model.critic_loss(
            reference_primary_predictions, primary_target
        )
        reference_primary_optimizer.zero_grad(set_to_none=True)
        reference_primary_loss.backward()
        reference_primary_grad_norm = torch.nn.utils.clip_grad_norm_(
            reference_primary.parameters(),
            float(engine.cfg.inner_critic_grad_clip_norm),
        )
        reference_primary_optimizer.step()

        reference_explorer_predictions = engine.model.q_predictions(
            batch["z"], batch["action"], qs=reference_explorer
        )
        reference_explorer_loss = engine.model.critic_loss(
            reference_explorer_predictions, explorer_target
        )
        reference_explorer_optimizer.zero_grad(set_to_none=True)
        reference_explorer_loss.backward()
        reference_explorer_grad_norm = torch.nn.utils.clip_grad_norm_(
            reference_explorer.parameters(),
            float(engine.cfg.inner_critic_grad_clip_norm),
        )
        reference_explorer_optimizer.step()

        torch.testing.assert_close(
            _flat_parameters(engine.state.critic),
            _flat_parameters(reference_primary),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            _flat_parameters(engine.state.explorer_critic),
            _flat_parameters(reference_explorer),
            rtol=0,
            atol=0,
        )
        _assert_tree_equal(
            engine.state.critic_optim.state_dict(),
            reference_primary_optimizer.state_dict(),
        )
        _assert_tree_equal(
            engine.state.explorer_critic_optim.state_dict(),
            reference_explorer_optimizer.state_dict(),
        )
        torch.testing.assert_close(metrics["critic_loss"], reference_primary_loss)
        torch.testing.assert_close(
            metrics["explorer_critic_loss"], reference_explorer_loss
        )
        torch.testing.assert_close(
            metrics["critic_grad_norm"], reference_primary_grad_norm
        )
        torch.testing.assert_close(
            metrics["explorer_critic_grad_norm"], reference_explorer_grad_norm
        )
        _assert_tree_equal(engine.state.critic_target.state_dict(), target_primary_before)
        _assert_tree_equal(
            engine.state.explorer_critic_target.state_dict(), target_explorer_before
        )
        _assert_tree_equal(engine.model._Qs.state_dict(), outer_before)
        assert engine.state.critic_steps == 1
        assert engine.state.explorer_critic_steps == 1
        assert engine.state.policy_evaluations == 2 * int(engine.cfg.inner_batch_size)
        assert engine.state.q_evaluations == 4 * int(engine.cfg.inner_batch_size)
    finally:
        model.env.close()


@pytest.mark.parametrize(
    ("update_primary", "update_explorer"),
    [(True, False), (False, True)],
)
def test_paired_critic_step_preserves_asymmetric_component_doses(
    update_primary,
    update_explorer,
    monkeypatch,
):
    model = _explorer_model(
        "separate_critics",
        inner_actor_updates_per_round=0,
        inner_explorer_actor_updates_per_round=0,
    )
    try:
        engine = _prepare_action_workspace(model)
        batch = _batch(engine, source=[0, 1, 0, 1, 0, 1, 0, 1])
        _constant_target_components(engine, monkeypatch)
        primary_before = _flat_parameters(engine.state.critic).clone()
        explorer_before = _flat_parameters(engine.state.explorer_critic).clone()
        disabled_params = (
            engine.state.explorer_critic_params
            if update_primary
            else engine.state.critic_params
        )
        for parameter in disabled_params:
            parameter.grad = torch.ones_like(parameter)

        with engine.rng.action_fork():
            with engine.rng.fork("bootstrap"):
                metrics = engine._separate_critics_step(
                    batch,
                    update_primary=update_primary,
                    update_explorer=update_explorer,
                )

        assert ("critic_loss" in metrics) is update_primary
        assert ("explorer_critic_loss" in metrics) is update_explorer
        assert engine.state.critic_steps == int(update_primary)
        assert engine.state.explorer_critic_steps == int(update_explorer)
        assert bool(engine.state.critic_optim.state) is update_primary
        assert bool(engine.state.explorer_critic_optim.state) is update_explorer
        assert torch.equal(
            primary_before, _flat_parameters(engine.state.critic)
        ) == (not update_primary)
        assert torch.equal(
            explorer_before, _flat_parameters(engine.state.explorer_critic)
        ) == (not update_explorer)
        assert all(parameter.grad is None for parameter in disabled_params)
        batch_size = int(engine.cfg.inner_batch_size)
        assert engine.state.policy_evaluations == 2 * batch_size
        assert engine.state.q_evaluations == 3 * batch_size
    finally:
        model.env.close()


def test_three_paired_critic_updates_preserve_rng_metrics_and_target_cadence():
    options = {
        "inner_critic_updates_per_round": 3,
        "inner_actor_updates_per_round": 0,
        "inner_explorer_actor_updates_per_round": 0,
        "inner_explorer_temperature_updates_per_round": 0,
        "inner_replay_sampling": "with_replacement",
    }
    first = _explorer_model("separate_critics", **options)
    second = _explorer_model("separate_critics", **options)
    try:
        global_rng = torch.random.get_rng_state().clone()
        first_action = first.agent.act(torch.zeros(3), collect_diagnostics=False)
        torch.testing.assert_close(torch.random.get_rng_state(), global_rng)
        second_action = second.agent.act(torch.zeros(3), collect_diagnostics=False)
        torch.testing.assert_close(torch.random.get_rng_state(), global_rng)
        torch.testing.assert_close(first_action, second_action, rtol=0, atol=0)

        first_engine = first.agent.inner_engine
        second_engine = second.agent.inner_engine
        first_pool = first_engine._action_pool
        second_pool = second_engine._action_pool
        first_metrics = first.agent.last_inner_metrics
        second_metrics = second.agent.last_inner_metrics

        assert first_metrics["inner_update_slots"] == 3
        assert first_metrics["inner_requested_update_slots"] == 3
        assert first_metrics["inner_replay_draws"] == 24
        assert first_metrics["inner_critic_optimizer_steps"] == 3
        assert first_metrics["inner_explorer_critic_optimizer_steps"] == 3
        assert first_metrics["inner_critic_target_updates"] == 3
        assert first_metrics["inner_explorer_critic_target_updates"] == 3
        assert first_metrics["inner_actor_optimizer_steps"] == 0
        assert first_metrics["inner_explorer_actor_optimizer_steps"] == 0
        assert first_metrics["inner_temperature_optimizer_steps"] == 0
        assert first_metrics["inner_explorer_temperature_optimizer_steps"] == 0
        assert _optimizer_steps(first_pool.critic_optim) == {3}
        assert _optimizer_steps(first_pool.explorer_critic_optim) == {3}
        assert (
            first_metrics["inner_primary_td_error_abs_count"]
            + first_metrics["inner_explorer_td_error_abs_count"]
            == 24
        )
        assert (
            first_metrics["inner_explorer_critic_primary_td_error_abs_count"]
            + first_metrics["inner_explorer_critic_explorer_td_error_abs_count"]
            == 24
        )
        assert first_metrics["inner_policy_evaluations"] == 59
        assert first_metrics["inner_q_evaluations"] == 98

        for component in (
            "critic",
            "critic_target",
            "explorer_critic",
            "explorer_critic_target",
        ):
            _assert_tree_equal(
                getattr(first_pool, component).state_dict(),
                getattr(second_pool, component).state_dict(),
            )
        _assert_tree_equal(
            first_pool.critic_optim.state_dict(),
            second_pool.critic_optim.state_dict(),
        )
        _assert_tree_equal(
            first_pool.explorer_critic_optim.state_dict(),
            second_pool.explorer_critic_optim.state_dict(),
        )
        _assert_tree_equal(
            first_engine.rng.training_state_dict(),
            second_engine.rng.training_state_dict(),
        )
        for key in (
            "inner_critic_loss",
            "inner_explorer_critic_loss",
            "inner_critic_grad_norm",
            "inner_explorer_critic_grad_norm",
        ):
            torch.testing.assert_close(
                torch.as_tensor(first_metrics[key]),
                torch.as_tensor(second_metrics[key]),
                rtol=0,
                atol=0,
            )
    finally:
        first.env.close()
        second.env.close()
