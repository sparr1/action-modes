"""Cumulative SAC scheduling has explicit budgets and scientific identities."""

import pytest

from tests.test_ambi_root_local_sac import _build_cfg
from utils.eval_series_data import planner_identity
from utils.resume_identity import scientific_trial_parameters


def frequency_cfg(**overrides):
    options = dict(
        inner_operator="sac", inner_rounds=3, inner_rollouts_per_round=4,
        inner_rollout_horizon=1, inner_updates_per_round=3,
        inner_actor_update_interval=2, inner_critic_target_update_interval=3,
        inner_temperature_mode="auto",
    )
    options.update(overrides)
    return _build_cfg(**options)


@pytest.mark.parametrize("g,p,t,j", [(20, 1, 2, 2), (20, 5, 2, 2),
                                   (10, 1, 2, 2), (3, 2, 3, 3),
                                   (3, 20, 3, 3), (3, 2, 3, 0)])
@pytest.mark.parametrize("temperature", ["auto", "fixed", "inherit_outer"])
def test_frequency_budgets_use_complete_critic_clock(g, p, t, j, temperature):
    cfg = frequency_cfg(inner_rounds=j, inner_updates_per_round=g,
                        inner_actor_update_interval=p,
                        inner_critic_target_update_interval=t,
                        inner_temperature_mode=temperature)
    critic = j * g
    actor = critic // p
    alpha = actor if temperature == "auto" else 0
    assert cfg.inner_schedule_mode == "canonical"
    assert not cfg.inner_component_update_schedule
    assert cfg.inner_critic_updates_per_round is None
    assert cfg.inner_actor_updates_per_round is None
    assert cfg.inner_critic_updates_per_action == critic
    assert cfg.inner_actor_updates_per_action == actor
    assert cfg.inner_temperature_updates_per_action == alpha
    assert cfg.inner_expected_update_slots == critic
    assert cfg.inner_nominal_updates_per_round == g
    assert cfg.inner_total_optimizer_steps_per_action == critic + actor + alpha
    assert cfg.inner_updates_per_iteration is None
    assert cfg.inner_critic_target_update_interval == t
    assert cfg.inner_nominal_critic_utd == (g / 4 if j else 0)


@pytest.mark.parametrize("key", ["inner_actor_update_interval", "inner_updates_per_round",
                                 "inner_critic_target_update_interval"])
@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, 2.0, "2", "auto"])
def test_frequency_controls_reject_non_positive_integers(key, value):
    with pytest.raises(ValueError, match="positive integer"):
        frequency_cfg(**{key: value})


def test_frequency_requires_explicit_fixed_shared_budget():
    with pytest.raises(ValueError, match="inner_updates_per_round.*positive integer"):
        _build_cfg(inner_actor_update_interval=2)


@pytest.mark.parametrize("overrides", [
    {"inner_critic_updates_per_round": 3, "inner_actor_updates_per_round": 1},
    {"inner_critic_updates_per_action": 9},
    {"inner_actor_updates_per_action": 4},
    {"inner_temperature_updates_per_action": 4},
    {"inner_model_step_budget": 12},
    {"inner_iterations": 3},
    {"inner_steps_per_update": 1},
    {"inner_operator": "td3"},
    {"inner_sac_return_estimator": "retrace"},
    {"inner_update_timing": "step"},
    {"inner_component_update_order": "interleaved"},
    {"inner_replay_strategy": "ere"},
    {"inner_explorer_mode": "shared_mixture"},
    {"inner_outer_replay_fraction": 0.5},
    {"inner_actor_adaptation": "frozen"},
    {"inner_critic_adaptation": "frozen"},
])
def test_frequency_rejects_unsupported_schedule_combinations(overrides):
    with pytest.raises(ValueError):
        frequency_cfg(**overrides)


@pytest.mark.parametrize("component", ["actor", "critic", "temperature", "replay",
                                      "actor_optimizer", "critic_optimizer",
                                      "temperature_optimizer"])
def test_frequency_requires_every_inner_lifetime_to_be_action_local(component):
    with pytest.raises(ValueError, match="inner_actor_update_interval"):
        frequency_cfg(**{f"inner_{component}_scope": "run"})


def test_null_aliases_allow_overriding_an_existing_component_recipe():
    cfg = frequency_cfg(inner_critic_updates_per_round=None,
                        inner_actor_updates_per_round=None,
                        inner_critic_updates_per_action=None,
                        inner_actor_updates_per_action=None,
                        inner_temperature_updates_per_action=None,
                        inner_model_step_budget=None, inner_steps_per_update=None,
                        inner_iterations=None, inner_rollouts=None,
                        inner_horizon=None, inner_updates_per_iteration=None)
    assert cfg.inner_critic_updates_per_action == 9
    assert cfg.inner_actor_updates_per_action == 4
    assert cfg.inner_expected_update_slots == 9


def test_inactive_interval_preserves_historical_joint_temperature_budget():
    # The old joint path may optimize alpha while the actor itself is frozen.
    cfg = frequency_cfg(inner_actor_update_interval=None,
                        inner_actor_adaptation="frozen")
    assert cfg.inner_actor_updates_per_action == 0
    assert cfg.inner_critic_updates_per_action == 9
    assert cfg.inner_temperature_updates_per_action == 9
    assert cfg.inner_updates_per_iteration == 3


def test_frequency_identity_is_explicit_and_inactive_default_is_compatible():
    base = {"inner_operator": "sac", "inner_rounds": 3,
            "inner_updates_per_round": 3}

    def planner(params):
        return planner_identity(params, {}, "AMBITDMPC2/AMBITDMPC2", "tanh_mean")

    def resume(params):
        return scientific_trial_parameters({"alg": "AMBITDMPC2/AMBITDMPC2",
                                            "alg_params": params})

    inactive = {**base, "inner_actor_update_interval": None}
    active = {**base, "inner_actor_update_interval": 2}
    assert planner(base) == planner(inactive)
    assert resume(base) == resume(inactive)
    assert "inner_actor_update_interval" not in resume(base)["alg_params"]
    assert planner(base) != planner(active)
    assert resume(base) != resume(active)
    assert planner(active) != planner({**active, "inner_actor_update_interval": 3})
    assert resume(active) != resume({**active, "inner_actor_update_interval": 3})
    assert planner(active)["settings"]["inner_actor_update_interval"] == 2


@pytest.mark.parametrize("interval", [None, 2])
def test_native_tdambi_rejects_active_frequency_schedule(interval):
    import gymnasium as gym
    from RL.TDAMBI import TDAMBI

    algorithm = object.__new__(TDAMBI)
    algorithm.env = gym.make("Pendulum-v1", max_episode_steps=5)
    algorithm.run_params = {"seed": 3, "device": "cpu", "env": "test-env", "total_steps": 12}
    algorithm.custom_params = {}
    try:
        params = dict(device="cpu", inner_updates_per_round=3,
                      inner_actor_update_interval=interval)
        if interval is None:
            assert algorithm._build_cfg(params).inner_actor_update_interval is None
        else:
            with pytest.raises(ValueError, match="canonical paired-update schedule"):
                algorithm._build_cfg(params)
    finally:
        algorithm.env.close()


@pytest.mark.parametrize("saved_schedule", ["component", "frequency"])
def test_checkpoint_sidecar_roundtrip_and_component_override(tmp_path, saved_schedule):
    import json
    from utils.ambi_research import resolve_preset
    from utils.checkpoint_context import load_checkpoint_context
    from utils.checkpointing import checkpoint_metadata

    params = dict(inner_operator="sac", inner_rounds=3, inner_rollouts_per_round=4,
                  inner_rollout_horizon=1, inner_temperature_mode="auto")
    if saved_schedule == "component":
        params.update(inner_critic_updates_per_round=3, inner_actor_updates_per_round=1)
    else:
        params.update(inner_updates_per_round=3, inner_actor_update_interval=2,
                      inner_critic_target_update_interval=3)
    original = _build_cfg(**params)
    checkpoint = tmp_path / "checkpoint.pt"
    metadata = checkpoint_metadata(
        kind="periodic", step=50, episode=1, best_score=None, best_window=100,
        trial_run_params={"alg": "AMBITDMPC2/AMBITDMPC2", "env": "Pendulum-v1",
                          "alg_params": params}, experiment_params={},
    )
    checkpoint.with_suffix(".pt.metadata.json").write_text(json.dumps(metadata))
    context = load_checkpoint_context(checkpoint)
    restored = _build_cfg(**context.trial_run_params["alg_params"])
    assert restored.inner_actor_update_interval == original.inner_actor_update_interval
    assert restored.inner_actor_updates_per_action == original.inner_actor_updates_per_action
    matrix = {
        "schema_version": 1, "base_alg_config": "checkpoint",
        "shared_alg_params": {
            "inner_critic_updates_per_round": None,
            "inner_actor_updates_per_round": None,
            "inner_updates_per_round": 3, "inner_actor_update_interval": 2,
            "inner_critic_target_update_interval": 3,
        },
        "comparisons": {"clock": {"reference": "frequency",
                                    "variants": {"frequency": {"alg_params": {}}}}},
    }
    resolved = resolve_preset(tmp_path / "matrix.json", "clock/frequency",
                              matrix=matrix, checkpoint_context=context)
    cfg = _build_cfg(**resolved["algorithm_config"]["alg_params"])
    assert cfg.inner_critic_updates_per_action == 9
    assert cfg.inner_actor_updates_per_action == cfg.inner_temperature_updates_per_action == 4
    assert cfg.inner_expected_update_slots == 9
    assert context.trial_run_params["alg_params"] == params
