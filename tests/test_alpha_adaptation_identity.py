"""Temperature interventions preserve default identities and remain distinguishable."""

import pytest

from tests.test_ambi_config_decoupling import _build_cfg
from utils.eval_series_data import planner_identity
from utils.resume_identity import scientific_trial_parameters


def test_repeat_default_is_compatible_but_nondefault_and_carry_are_distinct():
    base = dict(inner_operator="sac", inner_temperature_mode="auto",
                inner_temperature_scope="action", inner_temperature_optimizer_scope="action")
    def identity(config):
        return planner_identity(config, {}, "AMBITDMPC2/AMBITDMPC2", "tanh_sample")
    original = identity(base)
    assert identity({**base, "inner_temperature_updates_per_actor": 1}) == original
    repeated = identity({**base, "inner_temperature_updates_per_actor": 100})
    assert repeated != original
    assert repeated["settings"]["inner_temperature_updates_per_actor"] == 100
    carried = identity({**base, "inner_temperature_scope": "episode",
                        "inner_temperature_optimizer_scope": "episode"})
    assert carried != original and carried != repeated
    trial = dict(alg="AMBITDMPC2/AMBITDMPC2", seed=55, alg_params=base)
    explicit = {**trial, "alg_params": {**base, "inner_temperature_updates_per_actor": 1}}
    changed = {**trial, "alg_params": {**base, "inner_temperature_updates_per_actor": 100}}
    assert scientific_trial_parameters(trial) == scientific_trial_parameters(explicit)
    assert scientific_trial_parameters(trial) != scientific_trial_parameters(changed)


@pytest.mark.parametrize("scope", ["episode", "run"])
def test_auxiliary_allows_only_temperature_and_adam_carry(scope):
    cfg = _build_cfg(aux_return_mode="sac", inner_temperature_scope=scope,
                     inner_temperature_optimizer_scope=scope)
    assert cfg.inner_actor_scope == cfg.inner_critic_scope == cfg.inner_replay_scope == "action"


@pytest.mark.parametrize("overrides", [
    {"inner_temperature_optimizer_scope": "action"},
    {"inner_temperature_scope": "action"},
    {"inner_temperature_scope": "run", "inner_temperature_optimizer_scope": "episode"},
    {"inner_temperature_mode": "fixed"},
    {"inner_entropy_enabled": False},
    {"inner_actor_scope": "episode"},
    {"inner_critic_scope": "episode"},
    {"inner_replay_scope": "episode"},
    {"inner_actor_optimizer_scope": "episode"},
    {"inner_critic_optimizer_scope": "episode"},
])
def test_auxiliary_rejects_unsupported_persistence(overrides):
    params = dict(aux_return_mode="sac", inner_temperature_scope="episode",
                  inner_temperature_optimizer_scope="episode")
    params.update(overrides)
    with pytest.raises(ValueError):
        _build_cfg(**params)


def test_bundle_allows_temperature_carry_only_for_full_episodes():
    from evaluate_ambi_checkpoint import _validate_benchmark_scopes
    params = dict(inner_operator="sac", inner_temperature_mode="auto", inner_entropy_enabled=True,
                  inner_temperature_scope="episode", inner_temperature_optimizer_scope="episode",
                  inner_actor_scope="action", inner_critic_scope="action", inner_replay_scope="action")
    _validate_benchmark_scopes(params, episode_evaluation=True)
    with pytest.raises(ValueError, match="full-episode"):
        _validate_benchmark_scopes(params, episode_evaluation=False)
    for key, value in (("inner_actor_scope", "episode"), ("inner_critic_scope", "run"),
                       ("inner_replay_scope", "episode"), ("inner_temperature_optimizer_scope", "action"),
                       ("inner_temperature_mode", "fixed"), ("inner_entropy_enabled", False)):
        with pytest.raises(ValueError, match="action-local"):
            _validate_benchmark_scopes({**params, key: value}, episode_evaluation=True)
    reset = {**params, "inner_temperature_scope": "action", "inner_temperature_optimizer_scope": "action"}
    _validate_benchmark_scopes(reset, episode_evaluation=False)
