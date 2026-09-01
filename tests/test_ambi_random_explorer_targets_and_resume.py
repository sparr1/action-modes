"""Numerical target, gradient-path, selector-tail, and resume checks."""

from copy import deepcopy

import pytest
import torch

import RL.tdmpc2_core.inner_improvement as inner_improvement
from tests.test_ambi_random_explorer_engine_invariants import (
    _assert_tree_equal,
    _explorer_model,
    _flat_parameters,
)
from tests.test_ambi_root_local_sac import _tiny_component_model


def _prepare_action_workspace(model):
    engine = model.agent.inner_engine
    with engine.rng.action_fork():
        with engine.rng.fork("initialization"):
            engine._prepare_workspace(t0=False)
    return engine


def _batch(engine, *, source=None):
    batch_size = int(engine.cfg.inner_batch_size)
    batch = {
        "z": torch.linspace(
            -0.4,
            0.6,
            batch_size * int(engine.cfg.latent_dim),
            device=engine.device,
        ).reshape(batch_size, int(engine.cfg.latent_dim)),
        "action": torch.zeros(
            batch_size, int(engine.cfg.action_dim), device=engine.device
        ),
        "reward": torch.arange(
            1, batch_size + 1, dtype=torch.float32, device=engine.device
        ).reshape(-1, 1),
        "next_z": torch.linspace(
            0.7,
            -0.3,
            batch_size * int(engine.cfg.latent_dim),
            device=engine.device,
        ).reshape(batch_size, int(engine.cfg.latent_dim)),
        "terminated": torch.tensor(
            ([0, 1] * ((batch_size + 1) // 2))[:batch_size],
            dtype=torch.float32,
            device=engine.device,
        ).reshape(-1, 1),
    }
    if source is not None:
        batch["source"] = torch.as_tensor(
            source, dtype=torch.uint8, device=engine.device
        ).reshape(-1, 1)
    return batch


def _controlled_mixture_target(estimator, source):
    model = _explorer_model(
        "shared_mixture",
        inner_mixture_target_estimator=estimator,
        inner_batch_size=4,
    )
    patcher = pytest.MonkeyPatch()
    try:
        engine = _prepare_action_workspace(model)
        batch = _batch(engine, source=source)
        selected_actions = []
        targets = []
        original_critic_loss = engine.model.critic_loss

        def deterministic_components(z, *, policy, **kwargs):
            del kwargs
            value = 0.25 if policy is engine.state.actor else -0.5
            action = z.new_full((z.shape[0], int(engine.cfg.action_dim)), value)
            return action, {
                "pre_tanh_action": action,
                "pre_tanh_mean": action,
                "log_std": torch.zeros_like(action),
                "log_prob": torch.zeros(z.shape[0], 1, device=z.device),
            }

        def controlled_log_mu(pre_tanh_action, primary, explorer, weight):
            del primary, explorer, weight
            primary_rows = pre_tanh_action[:, :1] > 0
            return torch.where(
                primary_rows,
                pre_tanh_action.new_full((pre_tanh_action.shape[0], 1), -0.7),
                pre_tanh_action.new_full((pre_tanh_action.shape[0], 1), -1.3),
            )

        def controlled_bootstrap(z, action, **kwargs):
            del z, kwargs
            selected_actions.append(action.detach().clone())
            return 5.0 + action[:, :1]

        def capture_target(predictions, target):
            targets.append(target.detach().clone())
            return original_critic_loss(predictions, target)

        patcher.setattr(engine.model, "pi", deterministic_components)
        patcher.setattr(engine.model, "mixture_log_prob", controlled_log_mu)
        patcher.setattr(engine, "_bootstrap_q", controlled_bootstrap)
        patcher.setattr(engine.model, "critic_loss", capture_target)
        with engine.rng.action_fork():
            with engine.rng.fork("bootstrap"):
                engine._shared_mixture_critic_step(batch, torch.tensor(0.4))
        assert len(targets) == 1
        return {
            "target": targets[0].cpu(),
            "reward": batch["reward"].cpu(),
            "terminated": batch["terminated"].cpu(),
            "selected_actions": torch.cat(selected_actions).cpu(),
            "discount": float(model.agent.discount),
        }
    finally:
        patcher.undo()
        model.env.close()


@pytest.mark.parametrize("estimator", ["stratified", "weighted"])
def test_mixture_target_is_exact_masked_and_independent_of_replay_source(estimator):
    primary_first = _controlled_mixture_target(estimator, [0, 0, 1, 1])
    explorer_first = _controlled_mixture_target(estimator, [1, 1, 0, 0])

    torch.testing.assert_close(
        primary_first["target"], explorer_first["target"], rtol=0, atol=0
    )
    result = primary_first
    reward = result["reward"]
    terminated = result["terminated"]
    selected = result["selected_actions"]
    discount = result["discount"]

    if estimator == "stratified":
        assert selected.shape == (4, 1)
        assert int((selected == 0.25).sum()) == 2
        assert int((selected == -0.5).sum()) == 2
        log_mu = torch.where(
            selected > 0,
            selected.new_full(selected.shape, -0.7),
            selected.new_full(selected.shape, -1.3),
        )
        bootstrap = 5.0 + selected - 0.4 * log_mu
    else:
        assert selected.shape == (8, 1)
        torch.testing.assert_close(selected[:4], torch.full((4, 1), 0.25))
        torch.testing.assert_close(selected[4:], torch.full((4, 1), -0.5))
        primary_value = 5.25 - 0.4 * -0.7
        explorer_value = 4.5 - 0.4 * -1.3
        bootstrap = reward.new_full(reward.shape, 0.5 * (primary_value + explorer_value))

    expected = reward + discount * (1.0 - terminated) * bootstrap
    torch.testing.assert_close(result["target"], expected)
    torch.testing.assert_close(result["target"][terminated.bool()], reward[terminated.bool()])


def test_joint_mixture_primary_sample_backpropagates_into_both_actors_only(
    monkeypatch,
):
    model = _explorer_model(
        "shared_mixture",
        inner_batch_size=4,
        inner_critic_updates_per_round=0,
    )
    try:
        engine = _prepare_action_workspace(model)
        batch = _batch(engine, source=[0, 0, 1, 1])
        outer_actor = _flat_parameters(engine.model._pi).clone()
        outer_critic = _flat_parameters(engine.model._Qs).clone()
        original_mixture_log_prob = engine.model.mixture_log_prob
        calls = []

        def primary_component_only(pre_tanh_action, primary, explorer, weight):
            calls.append(pre_tanh_action)
            if len(calls) == 1:
                return original_mixture_log_prob(
                    pre_tanh_action, primary, explorer, weight
                )
            # Preserve a valid zero-valued graph for the explorer-sampled
            # component, leaving only P-sampled log(mu) to drive the loss.
            return pre_tanh_action.sum(dim=-1, keepdim=True) * 0.0

        def zero_q(z, action, **kwargs):
            del z, kwargs
            zero = action.sum(dim=-1, keepdim=True) * 0.0
            return zero.unsqueeze(0).expand(int(engine.cfg.num_q), -1, -1)

        monkeypatch.setattr(
            engine.model, "mixture_log_prob", primary_component_only
        )
        monkeypatch.setattr(engine.model, "Q", zero_q)
        for parameter in engine.state.critic.parameters():
            parameter.grad = None
        engine._shared_mixture_policy_step(
            batch,
            update_actor=True,
            update_temperature=False,
        )

        assert len(calls) == 2
        primary_gradients = [
            parameter.grad for parameter in engine.state.actor_params
            if parameter.grad is not None
        ]
        explorer_gradients = [
            parameter.grad for parameter in engine.state.explorer_actor_params
            if parameter.grad is not None
        ]
        assert primary_gradients and explorer_gradients
        assert any(torch.count_nonzero(gradient) for gradient in primary_gradients)
        # R receives this signal only through its density inside the P-sampled
        # exact mixture log-probability.
        assert any(torch.count_nonzero(gradient) for gradient in explorer_gradients)
        assert all(parameter.grad is None for parameter in engine.state.critic.parameters())
        assert all(parameter.grad is None for parameter in engine.model._pi.parameters())
        assert all(parameter.grad is None for parameter in engine.model._Qs.parameters())
        torch.testing.assert_close(
            _flat_parameters(engine.model._pi), outer_actor, rtol=0, atol=0
        )
        torch.testing.assert_close(
            _flat_parameters(engine.model._Qs), outer_critic, rtol=0, atol=0
        )
    finally:
        model.env.close()


def test_separate_critics_use_their_own_target_actor_alpha_and_terminal_mask(
    monkeypatch,
):
    model = _explorer_model("separate_critics", inner_batch_size=4)
    try:
        engine = _prepare_action_workspace(model)
        batch = _batch(engine, source=[0, 1, 0, 1])
        targets = []
        original_critic_loss = engine.model.critic_loss
        primary_before = _flat_parameters(engine.state.critic).clone()
        explorer_before = _flat_parameters(engine.state.explorer_critic).clone()
        outer_before = _flat_parameters(engine.model._Qs).clone()
        with torch.no_grad():
            engine.state.log_alpha.fill_(torch.tensor(0.2).log())
            engine.state.explorer_log_alpha.fill_(torch.tensor(0.7).log())

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

        def capture_target(predictions, target):
            targets.append(target.detach().clone())
            return original_critic_loss(predictions, target)

        monkeypatch.setattr(engine.model, "pi", controlled_policy)
        monkeypatch.setattr(engine, "_q_with", controlled_q)
        monkeypatch.setattr(engine.model, "critic_loss", capture_target)
        with engine.rng.action_fork():
            with engine.rng.fork("bootstrap"):
                engine._separate_critics_step(
                    batch, update_primary=True, update_explorer=True
                )

        assert len(targets) == 2
        discount = float(model.agent.discount)
        expected_primary = batch["reward"] + discount * (
            1.0 - batch["terminated"]
        ) * (3.0 - 0.2 * -1.0)
        expected_explorer = batch["reward"] + discount * (
            1.0 - batch["terminated"]
        ) * (7.0 - 0.7 * -2.0)
        torch.testing.assert_close(targets[0], expected_primary)
        torch.testing.assert_close(targets[1], expected_explorer)
        torch.testing.assert_close(
            targets[0][batch["terminated"].bool()],
            batch["reward"][batch["terminated"].bool()],
        )
        torch.testing.assert_close(
            targets[1][batch["terminated"].bool()],
            batch["reward"][batch["terminated"].bool()],
        )
        assert not torch.equal(primary_before, _flat_parameters(engine.state.critic))
        assert not torch.equal(
            explorer_before, _flat_parameters(engine.state.explorer_critic)
        )
        assert all(parameter.grad is None for parameter in engine.state.actor.parameters())
        assert all(
            parameter.grad is None for parameter in engine.state.explorer_actor.parameters()
        )
        torch.testing.assert_close(
            _flat_parameters(engine.model._Qs), outer_before, rtol=0, atol=0
        )
    finally:
        model.env.close()


def test_source_td_metrics_aggregate_by_exact_rows_when_slots_are_skewed():
    def source_metrics(errors, sources, *, prefix=""):
        errors = torch.as_tensor(errors, dtype=torch.float32)
        batch = {
            "source": torch.as_tensor(sources, dtype=torch.uint8).reshape(-1, 1)
        }
        values = errors.reshape(1, -1, 1)
        targets = torch.zeros(errors.numel(), 1)
        return inner_improvement.InnerImprovementEngine._source_td_metrics(
            batch, values, targets, prefix=prefix
        )

    history = []
    for errors, sources in (
        ([1.0, 2.0, 3.0], [0, 0, 0]),
        ([8.0, 2.0, 4.0, 6.0, 8.0], [0, 1, 1, 1, 1]),
        ([9.0], [1]),
    ):
        item = {"critic_loss": torch.tensor(1.0)}
        item.update(source_metrics(errors, sources))
        item.update(
            source_metrics(
                [2.0 * error for error in errors],
                sources,
                prefix="explorer_critic_",
            )
        )
        history.append(item)

    metrics = inner_improvement.InnerImprovementEngine._average_update_metrics(
        history
    )

    assert metrics["inner_primary_td_error_abs_count"] == 4
    assert metrics["inner_explorer_td_error_abs_count"] == 5
    assert metrics["inner_primary_td_error_abs_mean"] == pytest.approx(14.0 / 4.0)
    assert metrics["inner_explorer_td_error_abs_mean"] == pytest.approx(29.0 / 5.0)
    assert metrics["inner_explorer_critic_primary_td_error_abs_count"] == 4
    assert metrics["inner_explorer_critic_explorer_td_error_abs_count"] == 5
    assert metrics[
        "inner_explorer_critic_primary_td_error_abs_mean"
    ] == pytest.approx(28.0 / 4.0)
    assert metrics[
        "inner_explorer_critic_explorer_td_error_abs_mean"
    ] == pytest.approx(58.0 / 5.0)
    assert metrics["inner_critic_loss"] == pytest.approx(1.0)


def test_soft_handoff_masks_after_termination_and_bootstraps_outer_tail_once(
    monkeypatch,
):
    model = _explorer_model(
        "frozen_random",
        episodic=True,
        inner_rollout_horizon=2,
        inner_execution_handoff_samples=2,
        inner_critic_updates_per_round=0,
        inner_actor_updates_per_round=0,
        inner_execution_policy_source="outer_soft_handoff",
    )
    try:
        engine = _prepare_action_workspace(model)
        counts = {"prefix_pi": 0, "outer_pi": 0, "q": 0, "termination": 0}

        def controlled_pi(z, *, policy, **kwargs):
            del kwargs
            outer = policy is engine.model._pi
            counts["outer_pi" if outer else "prefix_pi"] += 1
            action = z.new_zeros(z.shape[0], int(engine.cfg.action_dim))
            return action, {
                "log_prob": z.new_full((z.shape[0], 1), -0.25 if outer else -0.5)
            }

        def controlled_q(z, action, **kwargs):
            del action, kwargs
            counts["q"] += 1
            return z.new_full((z.shape[0], 1), 100.0)

        def terminate(z, **kwargs):
            del kwargs
            counts["termination"] += 1
            return z.new_ones(z.shape[0], 1)

        monkeypatch.setattr(engine.model, "pi", controlled_pi)
        monkeypatch.setattr(
            engine.model, "joint_input", lambda z, action: z
        )
        monkeypatch.setattr(
            engine.model, "reward_from_joint", lambda joint: joint
        )
        monkeypatch.setattr(
            engine.model, "next_from_joint", lambda joint: joint + 1.0
        )
        monkeypatch.setattr(engine.model, "termination", terminate)
        monkeypatch.setattr(engine.model, "Q", controlled_q)
        monkeypatch.setattr(
            inner_improvement.td_math,
            "two_hot_inv",
            lambda reward, cfg: reward.new_full((reward.shape[0], 1), 2.0),
        )

        root_z = torch.zeros(1, int(engine.cfg.latent_dim))
        with engine.rng.action_fork():
            with engine.rng.fork("execution") as generator:
                score, model_steps = engine._outer_soft_handoff_scores(
                    root_z, engine.state.actor, generator
                )

        expected = 2.0 - engine.agent.alpha.detach() * -0.5
        torch.testing.assert_close(score, torch.ones_like(score) * expected)
        assert model_steps == 4
        assert counts == {
            "prefix_pi": 2,
            "outer_pi": 1,
            "q": 1,
            "termination": 2,
        }
    finally:
        model.env.close()


def test_soft_handoff_full_nonterminal_formula_and_common_random_numbers(
    monkeypatch,
):
    model = _explorer_model(
        "frozen_random",
        episodic=False,
        inner_rollout_horizon=2,
        inner_execution_handoff_samples=2,
        inner_critic_updates_per_round=0,
        inner_actor_updates_per_round=0,
        inner_execution_policy_source="outer_soft_handoff",
    )
    try:
        engine = _prepare_action_workspace(model)
        draws = {"primary": [], "explorer": [], "outer": []}

        def controlled_pi(z, *, policy, generator=None, **kwargs):
            del kwargs
            if policy is engine.state.actor:
                label, log_prob = "primary", -0.5
            elif policy is engine.state.explorer_actor:
                label, log_prob = "explorer", -0.5
            else:
                assert policy is engine.model._pi
                label, log_prob = "outer", -0.25
            draw = torch.randn(
                z.shape[0], 1, device=z.device, dtype=z.dtype, generator=generator
            )
            draws[label].append(draw.detach().clone())
            action = z.new_zeros(z.shape[0], int(engine.cfg.action_dim))
            return action, {
                "log_prob": z.new_full((z.shape[0], 1), log_prob),
            }

        monkeypatch.setattr(engine.model, "pi", controlled_pi)
        monkeypatch.setattr(engine.model, "joint_input", lambda z, action: z)
        monkeypatch.setattr(engine.model, "reward_from_joint", lambda joint: joint)
        monkeypatch.setattr(engine.model, "next_from_joint", lambda joint: joint + 1)
        monkeypatch.setattr(
            engine.model,
            "Q",
            lambda z, action, **kwargs: z.new_full((z.shape[0], 1), 10.0),
        )
        monkeypatch.setattr(
            inner_improvement.td_math,
            "two_hot_inv",
            lambda reward, cfg: reward.new_full((reward.shape[0], 1), 2.0),
        )

        root_z = torch.zeros(1, int(engine.cfg.latent_dim), device=engine.device)
        with engine.rng.action_fork():
            _, metrics, _ = engine._execute_two_policy(
                root_z, eval_mode=True, return_info=False
            )

        alpha = engine.agent.alpha.detach()
        discount = float(engine.agent.discount)
        prefix = 2.0 - alpha * -0.5
        tail = 10.0 - alpha * -0.25
        expected = (prefix + discount * prefix + discount**2 * tail).reshape(())
        torch.testing.assert_close(metrics["inner_selector_primary_score"], expected)
        torch.testing.assert_close(metrics["inner_selector_explorer_score"], expected)
        assert metrics["inner_selector_primary_wins"] == 1
        assert metrics["inner_selector_score_margin"] == 0
        assert metrics["inner_selector_model_steps"] == 8

        assert len(draws["primary"]) == len(draws["explorer"]) == 2
        assert len(draws["outer"]) == 2
        for primary, explorer in zip(draws["primary"], draws["explorer"]):
            torch.testing.assert_close(primary, explorer, rtol=0, atol=0)
        torch.testing.assert_close(draws["outer"][0], draws["outer"][1], rtol=0, atol=0)
    finally:
        model.env.close()


def test_fixed_q_counterfactual_logs_explorer_choice_independently_of_execution(
    monkeypatch,
):
    model = _explorer_model(
        "frozen_random",
        inner_critic_updates_per_round=0,
        inner_actor_updates_per_round=0,
        inner_execution_policy_source="primary",
    )
    try:
        engine = _prepare_action_workspace(model)
        root_z = torch.linspace(
            -0.5,
            0.5,
            int(engine.cfg.latent_dim),
            device=engine.device,
        ).reshape(1, -1)
        primary_mean = engine.model.policy_stats(
            root_z,
            policy=engine.state.actor,
            **engine._inner_policy_kwargs(),
        )["mean"].detach()
        explorer_mean = engine.model.policy_stats(
            root_z,
            policy=engine.state.explorer_actor,
            **engine._inner_policy_kwargs(),
        )["mean"].detach()
        assert not torch.equal(primary_mean, explorer_mean)

        q_calls = []

        def prefer_explorer(z, action, **kwargs):
            q_calls.append((action.detach().clone(), dict(kwargs)))
            return -(action - explorer_mean).square().sum(dim=-1, keepdim=True)

        monkeypatch.setattr(engine.model, "Q", prefer_explorer)
        with engine.rng.action_fork():
            executed, metrics, selected_policy = engine._execute_two_policy(
                root_z, eval_mode=True, return_info=False
            )

        expected_primary_q = -(primary_mean - explorer_mean).square().sum()
        expected_explorer_q = expected_primary_q.new_zeros(())
        expected_margin = expected_explorer_q - expected_primary_q
        expected_distance = torch.linalg.vector_norm(
            primary_mean - explorer_mean, dim=-1
        ).mean()

        assert selected_policy is engine.state.actor
        torch.testing.assert_close(executed, primary_mean, rtol=0, atol=0)
        torch.testing.assert_close(
            metrics["inner_fixed_q_counterfactual_primary_q"], expected_primary_q
        )
        torch.testing.assert_close(
            metrics["inner_fixed_q_counterfactual_explorer_q"], expected_explorer_q
        )
        torch.testing.assert_close(
            metrics["inner_fixed_q_counterfactual_margin"], expected_margin
        )
        assert metrics["inner_fixed_q_counterfactual_primary_wins"] == 0
        assert metrics["inner_fixed_q_counterfactual_explorer_wins"] == 1
        assert metrics["inner_fixed_q_counterfactual_explorer_rate"] == 1
        assert metrics["inner_fixed_q_counterfactual_execution_agreement"] == 0
        torch.testing.assert_close(
            metrics["inner_fixed_q_counterfactual_action_l2_to_executed"],
            expected_distance,
        )
        logged_action = torch.stack(
            [
                torch.as_tensor(
                    metrics[f"inner_fixed_q_counterfactual_action_{index}"]
                )
                for index in range(int(engine.cfg.action_dim))
            ]
        )
        torch.testing.assert_close(logged_action, explorer_mean[0], rtol=0, atol=0)
        assert metrics["inner_fixed_q_counterfactual_policy_evaluations"] == 2
        assert metrics["inner_fixed_q_counterfactual_q_evaluations"] == 2
        assert len(q_calls) == 2
        assert all(
            kwargs["target"] is True and kwargs["reduction"] == "min_all"
            for _, kwargs in q_calls
        )
    finally:
        model.env.close()


@pytest.mark.parametrize("source", ["primary", "explorer"])
def test_fixed_q_counterfactual_execution_restores_exact_module_modes(source):
    model = _explorer_model(
        "frozen_random",
        inner_critic_updates_per_round=0,
        inner_actor_updates_per_round=0,
        inner_execution_policy_source=source,
    )
    try:
        engine = _prepare_action_workspace(model)
        roots = (
            engine.state.actor,
            engine.state.explorer_actor,
            engine.model._target_Qs,
        )
        for root_index, root in enumerate(roots):
            for module_index, module in enumerate(root.modules()):
                module.training = bool((root_index + module_index) % 2)
        modes_before = tuple(
            tuple(module.training for module in root.modules()) for root in roots
        )
        assert all(any(modes) and not all(modes) for modes in modes_before)

        root_z = torch.linspace(
            -0.5,
            0.5,
            int(engine.cfg.latent_dim),
            device=engine.device,
        ).reshape(1, -1)
        with engine.rng.action_fork():
            engine._execute_two_policy(
                root_z, eval_mode=True, return_info=False
            )

        modes_after = tuple(
            tuple(module.training for module in root.modules()) for root in roots
        )
        assert modes_after == modes_before
    finally:
        model.env.close()


@pytest.mark.parametrize("source", ["explorer", "outer_q_gate"])
def test_behavior_metadata_comes_from_the_selected_explorer_component(
    source, monkeypatch
):
    model = _explorer_model(
        "frozen_random",
        inner_critic_updates_per_round=0,
        inner_actor_updates_per_round=0,
        inner_execution_policy_source=source,
    )
    engine = model.agent.inner_engine

    if source == "outer_q_gate":
        def prefer_explorer(z, action, **kwargs):
            del kwargs
            explorer_mean = engine.model.policy_stats(
                z,
                policy=engine.state.explorer_actor,
                **engine._inner_policy_kwargs(),
            )["mean"]
            return -(action - explorer_mean).square().sum(dim=-1, keepdim=True)

        monkeypatch.setattr(engine.model, "Q", prefer_explorer)

    try:
        _, behavior = model.agent.act(
            torch.zeros(3),
            eval_mode=False,
            collect_diagnostics=False,
            return_behavior_policy=True,
        )
        assert model.agent.last_inner_metrics["inner_selector_explorer_wins"] == 1
        pool = engine._action_pool
        root_z = engine.model.encode(torch.zeros(1, 3, device=engine.device)).detach()
        expected = engine.model.policy_stats(
            root_z,
            policy=pool.explorer_actor,
            **engine._inner_policy_kwargs(),
        )
        torch.testing.assert_close(
            behavior["pre_tanh_mean"], expected["pre_tanh_mean"][0].cpu()
        )
        torch.testing.assert_close(
            behavior["log_std"], expected["log_std"][0].cpu()
        )
    finally:
        model.env.close()


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile is unavailable")
def test_active_shared_mixture_real_compile_attempt_is_rng_safe():
    model = _explorer_model(
        "shared_mixture",
        compile=True,
        compile_strict=False,
        inner_mixture_target_estimator="weighted",
    )
    try:
        global_state = torch.random.get_rng_state().clone()
        action = model.agent.act(torch.zeros(3), collect_diagnostics=False)
        torch.testing.assert_close(
            torch.random.get_rng_state(), global_state, rtol=0, atol=0
        )
        assert torch.isfinite(action).all()
        pool = model.agent.inner_engine._action_pool
        # Non-strict compilation either creates a real Dynamo wrapper or records
        # its eager fallback; silently skipping the attempt is not allowed.
        for critic in (pool.critic, pool.critic_target):
            assert critic.compile_failed or critic._compiled_forward is not None
    finally:
        model.env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_active_explorer_cuda_preserves_cpu_and_all_cuda_global_rng_streams():
    model = _explorer_model("shared_mixture", device="cuda")
    try:
        cpu_before = torch.random.get_rng_state().clone()
        cuda_before = [state.clone() for state in torch.cuda.get_rng_state_all()]
        model.agent.act(torch.zeros(3), collect_diagnostics=False)
        torch.testing.assert_close(torch.random.get_rng_state(), cpu_before, rtol=0, atol=0)
        for actual, expected in zip(torch.cuda.get_rng_state_all(), cuda_before):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        model.env.close()


@pytest.mark.parametrize("mode", ["shared_mixture", "separate_critics"])
def test_active_explorer_identity_compile_matches_eager_action_metrics_and_rng(
    mode, monkeypatch
):
    monkeypatch.setattr(torch, "compile", lambda function, **kwargs: function)
    options = {
        "inner_mixture_target_estimator": "weighted",
        "inner_execution_policy_source": "mixture_sample",
    }
    eager = _explorer_model(mode, compile=False, **options)
    compiled = _explorer_model(
        mode,
        compile=True,
        compile_strict=True,
        **options,
    )
    try:
        global_state = torch.random.get_rng_state().clone()
        eager_action = eager.agent.act(torch.zeros(3), collect_diagnostics=False)
        torch.testing.assert_close(
            torch.random.get_rng_state(), global_state, rtol=0, atol=0
        )
        compiled_action = compiled.agent.act(
            torch.zeros(3), collect_diagnostics=False
        )
        torch.testing.assert_close(
            torch.random.get_rng_state(), global_state, rtol=0, atol=0
        )
        torch.testing.assert_close(eager_action, compiled_action, rtol=0, atol=0)

        eager_metrics = eager.agent.last_inner_metrics
        compiled_metrics = compiled.agent.last_inner_metrics
        assert set(eager_metrics) == set(compiled_metrics)
        for key in eager_metrics:
            if key.endswith("_seconds"):
                continue
            torch.testing.assert_close(
                torch.as_tensor(eager_metrics[key]),
                torch.as_tensor(compiled_metrics[key]),
                rtol=0,
                atol=0,
                msg=lambda message, key=key: f"{key}: {message}",
            )

        eager_pool = eager.agent.inner_engine._action_pool
        compiled_pool = compiled.agent.inner_engine._action_pool
        for component in ("actor", "critic", "critic_target", "explorer_actor"):
            _assert_tree_equal(
                getattr(eager_pool, component).state_dict(),
                getattr(compiled_pool, component).state_dict(),
            )
        for component in ("actor_optim", "critic_optim", "explorer_actor_optim"):
            _assert_tree_equal(
                getattr(eager_pool, component).state_dict(),
                getattr(compiled_pool, component).state_dict(),
            )
        if mode == "separate_critics":
            for component in ("explorer_critic", "explorer_critic_target"):
                _assert_tree_equal(
                    getattr(eager_pool, component).state_dict(),
                    getattr(compiled_pool, component).state_dict(),
                )
            _assert_tree_equal(
                eager_pool.explorer_critic_optim.state_dict(),
                compiled_pool.explorer_critic_optim.state_dict(),
            )
        _assert_tree_equal(
            eager.agent.inner_engine.rng.training_state_dict(),
            compiled.agent.inner_engine.rng.training_state_dict(),
        )
    finally:
        eager.env.close()
        compiled.env.close()


def test_disabled_explorer_explicit_defaults_preserve_action_rng_and_metrics():
    common = {
        "inner_rounds": 1,
        "inner_rollouts_per_round": 2,
        "inner_rollout_horizon": 1,
        "inner_batch_size": 2,
        "inner_replay_capacity": 2,
        "inner_critic_updates_per_round": 0,
        "inner_actor_updates_per_round": 0,
    }
    omitted = _tiny_component_model(**common)
    explicit = _tiny_component_model(
        **common,
        inner_explorer_mode="none",
        inner_prior_rollout_weight=0.5,
        inner_mixture_target_estimator="stratified",
        inner_explorer_actor_updates_per_round=None,
        inner_explorer_critic_updates_per_round=None,
        inner_explorer_temperature_updates_per_round=None,
        inner_execution_policy_source="primary",
        inner_execution_handoff_samples=8,
    )
    try:
        global_state = torch.random.get_rng_state().clone()
        omitted_action = omitted.agent.act(
            torch.zeros(3), collect_diagnostics=False
        )
        torch.testing.assert_close(
            torch.random.get_rng_state(), global_state, rtol=0, atol=0
        )
        explicit_action = explicit.agent.act(
            torch.zeros(3), collect_diagnostics=False
        )
        torch.testing.assert_close(
            torch.random.get_rng_state(), global_state, rtol=0, atol=0
        )
        torch.testing.assert_close(omitted_action, explicit_action, rtol=0, atol=0)
        for key in (
            "inner_model_steps",
            "inner_rollouts",
            "inner_buffer_size",
            "inner_policy_evaluations",
            "inner_q_evaluations",
            "inner_actor_optimizer_steps",
            "inner_critic_optimizer_steps",
            "inner_behavior_reward_sum_mean",
        ):
            torch.testing.assert_close(
                torch.as_tensor(omitted.agent.last_inner_metrics[key]),
                torch.as_tensor(explicit.agent.last_inner_metrics[key]),
                rtol=0,
                atol=0,
            )
        _assert_tree_equal(
            omitted.agent.inner_engine.rng.training_state_dict(),
            explicit.agent.inner_engine.rng.training_state_dict(),
        )
    finally:
        omitted.env.close()
        explicit.env.close()


def test_active_explorer_resume_continues_initialization_and_selector_randomness():
    options = {
        "inner_rollout_horizon": 1,
        "inner_replay_capacity": 4,
        "inner_batch_size": 4,
        "inner_critic_updates_per_round": 0,
        "inner_actor_updates_per_round": 0,
        "inner_execution_policy_source": "mixture_sample",
    }
    uninterrupted = _explorer_model("frozen_random", **options)
    restored = _explorer_model("frozen_random", **options)
    try:
        uninterrupted.agent.act(
            torch.zeros(3), eval_mode=True, collect_diagnostics=False
        )
        uninterrupted.agent.inner_engine.prepare_training_resume_boundary()
        checkpoint = deepcopy(
            uninterrupted.agent.inner_engine.training_state_dict()
        )
        restored.agent.inner_engine.load_training_state_dict(checkpoint)

        global_state = torch.random.get_rng_state().clone()
        expected_action = uninterrupted.agent.act(
            torch.zeros(3), eval_mode=True, collect_diagnostics=False
        )
        torch.testing.assert_close(
            torch.random.get_rng_state(), global_state, rtol=0, atol=0
        )
        actual_action = restored.agent.act(
            torch.zeros(3), eval_mode=True, collect_diagnostics=False
        )
        torch.testing.assert_close(
            torch.random.get_rng_state(), global_state, rtol=0, atol=0
        )
        torch.testing.assert_close(expected_action, actual_action, rtol=0, atol=0)
        torch.testing.assert_close(
            _flat_parameters(
                uninterrupted.agent.inner_engine._action_pool.explorer_actor
            ),
            _flat_parameters(restored.agent.inner_engine._action_pool.explorer_actor),
            rtol=0,
            atol=0,
        )
        assert (
            uninterrupted.agent.last_inner_metrics["inner_selector_primary_wins"]
            == restored.agent.last_inner_metrics["inner_selector_primary_wins"]
        )
        uninterrupted.agent.inner_engine.prepare_training_resume_boundary()
        restored.agent.inner_engine.prepare_training_resume_boundary()
        _assert_tree_equal(
            uninterrupted.agent.inner_engine.training_state_dict(),
            restored.agent.inner_engine.training_state_dict(),
        )
    finally:
        uninterrupted.env.close()
        restored.env.close()


def test_active_full_agent_exact_state_round_trips_and_rejects_population_change():
    options = {
        "inner_critic_updates_per_round": 0,
        "inner_actor_updates_per_round": 0,
    }
    source = _explorer_model("separate_critics", **options)
    restored = _explorer_model("separate_critics", **options)
    incompatible = _explorer_model(
        "separate_critics",
        inner_prior_rollout_weight=0.25,
        **options,
    )
    try:
        source.agent.act(torch.zeros(3), collect_diagnostics=False)
        source.agent.prepare_training_resume_boundary()
        saved = deepcopy(source.agent.training_state_dict())

        restored.agent.load_training_state_dict(saved)
        _assert_tree_equal(restored.agent.training_state_dict(), saved)

        incompatible.agent.prepare_training_resume_boundary()
        pristine = deepcopy(incompatible.agent.training_state_dict())
        with pytest.raises(ValueError, match="critic-target specification"):
            incompatible.agent.load_training_state_dict(saved)
        _assert_tree_equal(incompatible.agent.training_state_dict(), pristine)
    finally:
        source.env.close()
        restored.env.close()
        incompatible.env.close()
