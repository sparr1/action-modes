"""Runtime invariants for action-local adaptive parameter-noise exploration."""

import math

import pytest
import torch

import RL.tdmpc2_core.inner_improvement as inner_runtime
from tests.test_ambi_root_local_sac import _tiny_component_model


def _parameter_noise_model(**overrides):
    params = {
        "inner_rounds": 1,
        "inner_rollouts_per_round": 6,
        "inner_rollout_horizon": 2,
        "inner_batch_size": 4,
        "inner_replay_capacity": 12,
        "inner_critic_updates_per_round": 1,
        "inner_actor_updates_per_round": 1,
        "inner_explorer_mode": "adaptive_param_noise",
        "inner_prior_rollout_weight": 0.5,
        "inner_param_noise_actor_count": 3,
        "inner_param_noise_calibration_directions": 2,
        "inner_param_noise_calibration_batch_size": 4,
        "inner_param_noise_calibration_max_probes": 2,
        "inner_execution_policy_source": "primary",
    }
    params.update(overrides)
    return _tiny_component_model(**params)


def test_parameter_noise_runtime_preserves_budget_sources_and_primary_ownership(
    monkeypatch,
):
    model = _parameter_noise_model(
        inner_rounds=2,
        inner_replay_capacity=24,
    )
    engine = model.agent.inner_engine
    selected_primary = []
    original_execute = engine._execute_policy

    def record_execute(root_z, policy, **kwargs):
        selected_primary.append(policy is engine.state.actor)
        return original_execute(root_z, policy, **kwargs)

    def forbid_two_policy_execution(*args, **kwargs):
        del args, kwargs
        raise AssertionError("parameter noise must execute only the primary actor")

    monkeypatch.setattr(engine, "_execute_policy", record_execute)
    monkeypatch.setattr(
        engine, "_execute_two_policy", forbid_two_policy_execution
    )
    try:
        global_rng = torch.random.get_rng_state().clone()
        action = model.agent.act(torch.zeros(3), collect_diagnostics=False)
        torch.testing.assert_close(
            torch.random.get_rng_state(), global_rng, rtol=0, atol=0
        )
        assert torch.isfinite(action).all()
        assert selected_primary == [True]

        metrics = model.agent.last_inner_metrics
        assert metrics["inner_model_steps"] == 24
        assert metrics["inner_primary_rollouts"] == 6
        assert metrics["inner_explorer_rollouts"] == 6
        assert metrics["inner_primary_transitions"] == 12
        assert metrics["inner_explorer_transitions"] == 12
        assert metrics["inner_actor_optimizer_steps"] == 2
        assert metrics["inner_critic_optimizer_steps"] == 2
        assert metrics["inner_explorer_actor_optimizer_steps"] == 0
        assert metrics["inner_explorer_critic_optimizer_steps"] == 0
        assert metrics["inner_explorer_temperature_optimizer_steps"] == 0
        assert metrics["inner_selector_primary_wins"] == 1
        assert metrics["inner_selector_explorer_wins"] == 0

        pool = engine._action_pool
        assert pool.explorer_actor is None
        assert pool.explorer_actor_optim is None
        assert pool.explorer_critic is None
        sources = pool.replay.source[: pool.replay.size].reshape(-1).long()
        assert torch.bincount(sources, minlength=2).tolist() == [12, 12]

        assert metrics["inner_param_noise_actor_count"] == 3
        assert metrics["inner_param_noise_rollouts_per_actor"] == 1
        assert 2 <= metrics["inner_param_noise_calibration_probes"] <= 4
        calibration_evaluations = metrics[
            "inner_param_noise_calibration_policy_evaluations"
        ]
        assert calibration_evaluations >= (
            metrics["inner_param_noise_calibration_probes"] * 2 * 2
        )
        assert calibration_evaluations <= (
            metrics["inner_param_noise_calibration_probes"] * 2 * 2 * 4
        )
        assert metrics["inner_param_noise_behavior_action_rms_mean"] > 0
        assert (
            model.cfg.inner_param_noise_sigma_min
            <= metrics["inner_param_noise_sigma_final"]
            <= model.cfg.inner_param_noise_sigma_max
        )
        for key, value in metrics.items():
            if key.startswith("inner_param_noise_"):
                assert math.isfinite(float(value)), key

        parameter_spec = model.agent._inner_population_spec()["parameter_noise"]
        assert parameter_spec["actor_count"] == 3
        assert parameter_spec["rollouts_per_actor"] == 1
        assert parameter_spec["behavior_action"] == "policy_sample"
        assert parameter_spec["behavior_std_scale"] == pytest.approx(
            model.cfg.inner_behavior_std_scale
        )
        assert parameter_spec["perturbed_policy_output"] == "mean_only"
        assert parameter_spec["behavior_log_std_source"] == "clean_actor"
        assert parameter_spec["clean_log_std_mapping"] == (
            model.cfg.inner_log_std_mapping
        )
        assert parameter_spec["clean_log_std_min"] == pytest.approx(
            model.cfg.inner_log_std_min
        )
        assert parameter_spec["clean_log_std_max"] == pytest.approx(
            model.cfg.inner_log_std_max
        )
        assert parameter_spec["reset_per_action"] is True
        assert parameter_spec["recalibrate_per_round"] is True
        assert parameter_spec["calibration_relative_tolerance"] == 0.10
        assert parameter_spec["calibration_log_error_exponent"] == 0.5
        assert parameter_spec["calibration_update_ratio_min"] == 0.5
        assert parameter_spec["calibration_update_ratio_max"] == 2.0

        changed = _parameter_noise_model(inner_behavior_std_scale=2.0)
        try:
            assert changed.agent._inner_population_spec() != (
                model.agent._inner_population_spec()
            )
            assert changed.agent._critic_target_spec() != (
                model.agent._critic_target_spec()
            )
        finally:
            changed.env.close()
        changed_mapping = _parameter_noise_model(
            inner_log_std_mapping="tdmpc2_tanh",
            inner_log_std_min=-10,
            inner_log_std_max=2,
        )
        try:
            assert changed_mapping.agent._critic_target_spec() != (
                model.agent._critic_target_spec()
            )
        finally:
            changed_mapping.env.close()

        # Sigma and sampled perturbations are action-local and deliberately
        # absent from the exact boundary checkpoint.
        assert engine._parameter_noise_stddev is None
        engine.prepare_training_resume_boundary()
        state = engine.training_state_dict()
        assert state["version"] == 2
        assert state["explorer_mode"] == "adaptive_param_noise"
        assert "parameter_noise" not in state
    finally:
        model.env.close()


def test_parameter_noise_draws_fresh_grouped_actors_fixed_through_each_horizon(
    monkeypatch,
):
    model = _parameter_noise_model(
        inner_rounds=2,
        inner_rollouts_per_round=8,
        inner_rollout_horizon=3,
        inner_replay_capacity=48,
        inner_critic_updates_per_round=0,
        inner_actor_updates_per_round=0,
        inner_param_noise_actor_count=2,
        inner_param_noise_calibration_directions=3,
        inner_param_noise_calibration_max_probes=1,
    )
    sampled_populations = []
    sampled_streams = []
    behavior_deltas = []
    behavior_parameter_maps = []
    population_shapes = []
    population_chunks = []
    calibration_shapes = []
    calibration_chunks = []
    engine = model.agent.inner_engine
    original_sample = inner_runtime.sample_parameter_deltas
    original_population_mean = inner_runtime.population_actor_mean_raw
    original_action_rms = inner_runtime.parameter_noise_action_rms

    def record_sample(actor, spec, population_size, *, generator):
        deltas = original_sample(
            actor, spec, population_size, generator=generator
        )
        sampled_populations.append(population_size)
        sampled_streams.append(
            "initialization"
            if generator is engine.rng.generator("initialization")
            else "collection"
            if generator is engine.rng.generator("collection")
            else "unexpected"
        )
        if population_size == 2:
            behavior_deltas.append(
                {name: value.detach().clone() for name, value in deltas.items()}
            )
        return deltas

    def record_population_mean(
        actor, spec, batched_parameters, latents, *, chunk_size=None
    ):
        behavior_parameter_maps.append(batched_parameters)
        population_shapes.append(tuple(latents.shape))
        population_chunks.append(chunk_size)
        return original_population_mean(
            actor,
            spec,
            batched_parameters,
            latents,
            chunk_size=chunk_size,
        )

    def record_action_rms(
        actor, spec, batched_parameters, latents, *, chunk_size=None
    ):
        calibration_shapes.append(tuple(latents.shape))
        calibration_chunks.append(chunk_size)
        return original_action_rms(
            actor,
            spec,
            batched_parameters,
            latents,
            chunk_size=chunk_size,
        )

    monkeypatch.setattr(inner_runtime, "sample_parameter_deltas", record_sample)
    monkeypatch.setattr(
        inner_runtime, "population_actor_mean_raw", record_population_mean
    )
    monkeypatch.setattr(
        inner_runtime, "parameter_noise_action_rms", record_action_rms
    )
    try:
        model.agent.act(torch.zeros(3), collect_diagnostics=False)
        assert sampled_populations == [3, 2, 3, 2]
        assert sampled_streams == [
            "initialization",
            "collection",
            "initialization",
            "collection",
        ]
        assert calibration_shapes == [(3, 1, 8), (3, 4, 8)]
        assert calibration_chunks == [8, 8]
        assert population_shapes == [(2, 2, 8)] * 6
        assert population_chunks == [8] * 6
        assert len(behavior_parameter_maps) == 6
        assert all(
            mapping is behavior_parameter_maps[0]
            for mapping in behavior_parameter_maps[:3]
        )
        assert all(
            mapping is behavior_parameter_maps[3]
            for mapping in behavior_parameter_maps[3:]
        )
        assert behavior_parameter_maps[0] is not behavior_parameter_maps[3]
        assert len(behavior_deltas) == 2
        assert any(
            not torch.equal(
                behavior_deltas[0][name], behavior_deltas[1][name]
            )
            for name in behavior_deltas[0]
        )
    finally:
        model.env.close()


@pytest.mark.parametrize("mode", ["none", "frozen_random"])
def test_parameter_noise_metrics_are_absent_from_existing_mode_shapes(mode):
    model = _tiny_component_model(
        inner_rounds=1,
        inner_rollouts_per_round=4,
        inner_rollout_horizon=1,
        inner_batch_size=4,
        inner_replay_capacity=4,
        inner_critic_updates_per_round=0,
        inner_actor_updates_per_round=0,
        inner_explorer_mode=mode,
        inner_prior_rollout_weight=0.5,
        inner_execution_policy_source="primary",
    )
    try:
        model.agent.act(torch.zeros(3), collect_diagnostics=False)
        assert not any(
            key.startswith("inner_param_noise_")
            for key in model.agent.last_inner_metrics
        )
    finally:
        model.env.close()


def test_parameter_noise_without_concrete_explorer_has_no_fixed_q_diagnostics():
    model = _parameter_noise_model(
        inner_critic_updates_per_round=0,
        inner_actor_updates_per_round=0,
        inner_param_noise_calibration_max_probes=1,
    )
    try:
        model.agent.act(
            torch.zeros(3), collect_diagnostics=False, eval_mode=True
        )
        assert model.agent.inner_engine._action_pool.explorer_actor is None
        assert not any(
            key.startswith("inner_fixed_q_counterfactual_")
            for key in model.agent.last_inner_metrics
        )
    finally:
        model.env.close()


def test_parameter_noise_behavior_uses_noisy_mean_with_clean_logstd(
    monkeypatch,
):
    model = _parameter_noise_model(
        inner_rounds=1,
        inner_rollouts_per_round=8,
        inner_rollout_horizon=1,
        inner_replay_capacity=8,
        inner_critic_updates_per_round=0,
        inner_actor_updates_per_round=0,
        inner_param_noise_actor_count=2,
    )
    engine = model.agent.inner_engine
    try:
        with engine.rng.fork("initialization"):
            engine._prepare_workspace(t0=True)
        root_z = torch.zeros(1, model.cfg.latent_dim)
        observed_std_scales = []

        def fixed_population_mean(
            actor, spec, batched_parameters, latents, *, chunk_size=None
        ):
            del actor, spec, batched_parameters, chunk_size
            return latents.new_full((2, 2, model.cfg.action_dim), 0.5)

        def fixed_clean_stats(z, *, policy, std_scale, **kwargs):
            del policy, kwargs
            observed_std_scales.append(std_scale)
            return {
                "mean": z.new_zeros((z.shape[0], model.cfg.action_dim)),
                "pre_tanh_mean": z.new_zeros(
                    (z.shape[0], model.cfg.action_dim)
                ),
                "log_std": z.new_full(
                    (z.shape[0], model.cfg.action_dim), math.log(0.25)
                ),
            }

        def replica_noise(shape, *, dtype, device, generator):
            del generator
            noise = torch.zeros(shape, dtype=dtype, device=device)
            noise[:, 1, :] = 1.0
            return noise

        monkeypatch.setattr(
            inner_runtime, "population_actor_mean_raw", fixed_population_mean
        )
        monkeypatch.setattr(model.agent.model, "policy_stats", fixed_clean_stats)
        monkeypatch.setattr(inner_runtime.torch, "randn", replica_noise)
        with engine.rng.fork("collection") as generator:
            result = engine._collect_parameter_noise_population(
                root_z,
                spec=object(),
                batched_parameters={},
                actor_count=2,
                rollouts_per_actor=2,
                horizon=1,
                generator=generator,
            )

        actions = engine.state.replay.action[: engine.state.replay.size]
        expected = torch.tensor(
            [math.tanh(0.5), math.tanh(0.75)] * 2,
            dtype=actions.dtype,
            device=actions.device,
        ).unsqueeze(-1).expand_as(actions)
        torch.testing.assert_close(actions, expected)
        # Both replicas use the same perturbed mean at the shared root, while
        # independent SAC epsilon makes their sampled actions distinct.
        torch.testing.assert_close(actions[0], actions[2])
        torch.testing.assert_close(actions[1], actions[3])
        assert not torch.equal(actions[0], actions[1])
        assert observed_std_scales == [model.cfg.inner_behavior_std_scale]
        assert result["transition_count"] == 4
        assert engine.state.policy_evaluations == 8
    finally:
        model.env.close()


def test_parameter_noise_episodic_replay_is_depth_major_and_drops_dead_rows(
    monkeypatch,
):
    model = _parameter_noise_model(
        episodic=True,
        inner_rounds=1,
        inner_rollouts_per_round=8,
        inner_rollout_horizon=3,
        inner_replay_capacity=24,
        inner_critic_updates_per_round=0,
        inner_actor_updates_per_round=0,
        inner_param_noise_actor_count=2,
    )
    engine = model.agent.inner_engine
    try:
        with engine.rng.fork("initialization"):
            engine._prepare_workspace(t0=True)
        root_z = torch.zeros(1, model.cfg.latent_dim)
        spec = engine._parameter_noise_actor_spec()
        termination_calls = 0

        def terminate_in_two_depths(next_z):
            nonlocal termination_calls
            if termination_calls == 0:
                values = next_z.new_tensor([[1.0], [0.0], [1.0], [0.0]])
            else:
                values = next_z.new_ones((next_z.shape[0], 1))
            termination_calls += 1
            return values

        monkeypatch.setattr(engine.model, "termination", terminate_in_two_depths)
        with engine.rng.fork("collection") as generator:
            deltas = inner_runtime.sample_parameter_deltas(
                engine.state.actor,
                spec,
                2,
                generator=generator,
            )
            parameters = inner_runtime.make_perturbed_actor_parameters(
                engine.state.actor,
                spec,
                deltas,
                model.cfg.inner_param_noise_sigma_init,
            )
            result = engine._collect_parameter_noise_population(
                root_z,
                spec=spec,
                batched_parameters=parameters,
                actor_count=2,
                rollouts_per_actor=2,
                horizon=3,
                generator=generator,
            )

        assert termination_calls == 2
        assert result["lengths"].tolist() == [1, 2, 1, 2]
        assert result["terminated"].tolist() == [True, True, True, True]
        assert result["transition_count"] == 6
        assert engine.state.replay.size == 6
        assert engine.state.replay.source[:6].reshape(-1).tolist() == [1] * 6
        # The first horizon slice contains all actor-major rows; the next
        # contains only the still-alive replica from each actor.
        assert engine.state.replay.terminated[:6].reshape(-1).tolist() == [
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            1.0,
        ]
    finally:
        model.env.close()


def test_parameter_noise_sigma_resets_per_action_and_warm_starts_only_rounds(
    monkeypatch,
):
    model = _parameter_noise_model(
        inner_rounds=2,
        inner_replay_capacity=24,
        inner_critic_updates_per_round=0,
        inner_actor_updates_per_round=0,
        inner_param_noise_calibration_max_probes=1,
    )
    engine = model.agent.inner_engine
    starts = []
    finishes = []
    original_calibrate = engine._calibrate_parameter_noise

    def record_calibration(root_z, generator):
        starts.append(float(engine._parameter_noise_stddev))
        spec = original_calibrate(root_z, generator)
        finishes.append(float(engine._parameter_noise_stddev))
        return spec

    monkeypatch.setattr(engine, "_calibrate_parameter_noise", record_calibration)
    try:
        model.agent.act(torch.zeros(3), collect_diagnostics=False)
        model.agent.act(torch.zeros(3), collect_diagnostics=False)

        initial = model.cfg.inner_param_noise_sigma_init
        assert starts[0] == pytest.approx(initial)
        assert starts[1] == pytest.approx(finishes[0])
        assert starts[2] == pytest.approx(initial)
        assert starts[3] == pytest.approx(finishes[2])
        assert engine._parameter_noise_stddev is None
    finally:
        model.env.close()


def test_parameter_noise_calibration_converges_honors_tolerance_and_bounds(
    monkeypatch,
):
    model = _parameter_noise_model(
        inner_rounds=1,
        inner_replay_capacity=12,
        inner_critic_updates_per_round=0,
        inner_actor_updates_per_round=0,
        inner_param_noise_target_action_rms=0.1,
        inner_param_noise_sigma_init=0.001,
        inner_param_noise_sigma_min=1e-6,
        inner_param_noise_sigma_max=0.1,
        inner_param_noise_calibration_max_probes=8,
    )
    engine = model.agent.inner_engine
    current_sigma = None

    def record_parameters(actor, spec, deltas, stddev):
        nonlocal current_sigma
        del actor, spec, deltas
        current_sigma = float(stddev)
        return {}

    def proportional_rms(
        actor, spec, batched_parameters, latents, *, chunk_size=None
    ):
        del actor, spec, batched_parameters, chunk_size
        return latents.new_tensor(current_sigma * 10.0)

    monkeypatch.setattr(
        inner_runtime, "make_perturbed_actor_parameters", record_parameters
    )
    monkeypatch.setattr(
        inner_runtime, "parameter_noise_action_rms", proportional_rms
    )
    try:
        with engine.rng.fork("initialization"):
            engine._prepare_workspace(t0=True)
        root_z = torch.zeros(1, model.cfg.latent_dim)

        with engine.rng.fork("initialization") as generator:
            engine._calibrate_parameter_noise(root_z, generator)
        assert 1 < engine._parameter_noise_calibration_probes <= 8
        assert engine._parameter_noise_calibration_hits == [1.0]
        final_rms = float(engine._parameter_noise_calibration_rms[-1])
        assert final_rms == pytest.approx(0.1, rel=0.10)

        # A measurement already inside the ten-percent deadband performs one
        # probe and leaves the warm-started sigma unchanged.
        engine._reset_parameter_noise_action_state()
        starting_sigma = engine._parameter_noise_stddev

        def inside_tolerance(
            actor, spec, batched_parameters, latents, *, chunk_size=None
        ):
            del actor, spec, batched_parameters, chunk_size
            return latents.new_tensor(0.091)

        monkeypatch.setattr(
            inner_runtime, "parameter_noise_action_rms", inside_tolerance
        )
        with engine.rng.fork("initialization") as generator:
            engine._calibrate_parameter_noise(root_z, generator)
        assert engine._parameter_noise_calibration_probes == 1
        assert engine._parameter_noise_calibration_hits == [1.0]
        assert engine._parameter_noise_stddev == starting_sigma

        # An unreachable target stops immediately when the bounded update
        # reaches sigma_max, and records the miss rather than overshooting.
        engine.cfg.inner_param_noise_sigma_max = 0.002
        engine._reset_parameter_noise_action_state()

        def below_target(
            actor, spec, batched_parameters, latents, *, chunk_size=None
        ):
            del actor, spec, batched_parameters, chunk_size
            return latents.new_tensor(1e-6)

        monkeypatch.setattr(
            inner_runtime, "parameter_noise_action_rms", below_target
        )
        with engine.rng.fork("initialization") as generator:
            engine._calibrate_parameter_noise(root_z, generator)
        assert engine._parameter_noise_calibration_probes == 1
        assert engine._parameter_noise_calibration_hits == [0.0]
        assert engine._parameter_noise_sigma_bound_hits == [1.0]
        assert engine._parameter_noise_stddev == pytest.approx(0.002)
    finally:
        model.env.close()
