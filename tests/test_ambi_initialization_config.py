"""Independent initialization choices preserve prior behavior and identities."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
import gymnasium as gym

from RL.TDAMBI import TDAMBI
from tests.test_ambi_config_decoupling import _build_cfg
from tests.test_tdambi_checkpoint import tiny_native_params
from utils import ambi_research
from utils.eval_series_data import planner_identity
from utils.resume_identity import scientific_trial_parameters


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("inner_actor_initialization", "inner_critic_initialization")
SCOPES = (
    "actor", "critic", "temperature", "replay",
    "actor_optimizer", "critic_optimizer", "temperature_optimizer",
)
ALGORITHM = "AMBITDMPC2/AMBITDMPC2"


def test_initialization_defaults_preserve_prior_configuration():
    omitted = _build_cfg()
    explicit = _build_cfg(**{field: "PRIOR" for field in FIELDS})
    assert vars(omitted) == vars(explicit)
    assert all(getattr(omitted, field) == "prior" for field in FIELDS)


@pytest.mark.parametrize("actor", ["prior", "random"])
@pytest.mark.parametrize("critic", ["prior", "random"])
@pytest.mark.parametrize("adaptation", ["clone", "lora_rl"])
def test_initialization_choices_are_independent_of_dense_or_lora_adaptation(
    actor, critic, adaptation,
):
    cfg = _build_cfg(
        inner_actor_initialization=actor.upper(),
        inner_critic_initialization=critic.upper(),
        inner_critic_adaptation=adaptation,
    )
    assert cfg.inner_actor_initialization == actor
    assert cfg.inner_critic_initialization == critic
    assert cfg.inner_actor_adaptation == "clone"
    assert cfg.inner_critic_adaptation == adaptation


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("value", [None, True, 1, "scratch", "clone", ""])
def test_initialization_rejects_unknown_values(field, value):
    with pytest.raises(ValueError, match=field):
        _build_cfg(**{field: value})


@pytest.mark.parametrize("component", ["actor", "critic"])
def test_random_frozen_component_keeps_zero_updates(component):
    cfg = _build_cfg(**{
        f"inner_{component}_initialization": "random",
        f"inner_{component}_adaptation": "frozen",
    })
    assert getattr(cfg, f"inner_{component}_updates_per_action") == 0


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("operator", ["none", "td3", "mppi"])
def test_random_initialization_requires_sac(field, operator):
    with pytest.raises(ValueError, match="Random inner initialization"):
        _build_cfg(**{field: "random", "inner_operator": operator})


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("mode", ["frozen_random", "shared_mixture", "separate_critics"])
def test_random_initialization_rejects_explorer_populations(field, mode):
    with pytest.raises(ValueError, match="Random inner initialization"):
        _build_cfg(**{field: "random", "inner_explorer_mode": mode})


@pytest.mark.parametrize("component", SCOPES)
@pytest.mark.parametrize("scope", ["episode", "run"])
def test_random_initialization_requires_all_action_scopes(component, scope):
    params = {"inner_actor_initialization": "random", f"inner_{component}_scope": scope}
    if component.endswith("_optimizer"):
        params[f"inner_{component.removesuffix('_optimizer')}_scope"] = scope
    with pytest.raises(ValueError, match="Random inner initialization"):
        _build_cfg(**params)


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("component", ["actor", "critic"])
def test_random_initialization_rejects_prior_writeback(field, component):
    with pytest.raises(ValueError, match="Random inner initialization"):
        _build_cfg(**{field: "random", f"inner_{component}_writeback_coef": 0.1})


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("probe", [
    {"value_equivalence_diagnostics": True},
    {"value_equivalence_loss_coef": 0.1},
])
def test_random_initialization_rejects_fresh_prior_value_equivalence(field, probe):
    with pytest.raises(ValueError, match="prior-initialized inner networks"):
        _build_cfg(**{field: "random", **probe})


def test_random_critic_requires_online_target_initialization():
    with pytest.raises(ValueError, match="inner_critic_target_initialization='online'"):
        _build_cfg(inner_critic_initialization="random",
                   inner_critic_target_initialization="outer_target")
    cfg = _build_cfg(inner_actor_initialization="random",
                     inner_critic_target_initialization="outer_target")
    assert cfg.inner_critic_initialization == "prior"
    assert cfg.inner_critic_target_initialization == "outer_target"


def test_td_ambi_training_recipe_supports_random_with_online_target():
    params = json.loads((ROOT / "configs/dmcontrol/algs/TD-AMBI.json").read_text())["alg_params"]
    params.update(inner_actor_initialization="random", inner_critic_initialization="random",
                  inner_critic_target_initialization="online", compile=False)
    cfg = _build_cfg(**params)
    assert cfg.inner_operator == "sac"
    assert all(getattr(cfg, field) == "random" for field in FIELDS)
    assert cfg.inner_actor_loss_scale_update == "per_update"


@pytest.mark.parametrize("field", FIELDS)
def test_native_tdambi_accepts_independent_random_initialization(field):
    algorithm = object.__new__(TDAMBI)
    algorithm.env = gym.make("Pendulum-v1", max_episode_steps=5)
    algorithm.run_params = {"device": "cpu", "seed": 3}
    try:
        cfg = algorithm._build_cfg(tiny_native_params(inner_rollout_horizon=2, **{field: "RANDOM"}))
        assert getattr(cfg, field) == "random"
        assert cfg.inner_operator == "tdambi"
        assert cfg.inner_critic_target_initialization == (
            "online" if field == "inner_critic_initialization" else "outer_target"
        )
    finally:
        algorithm.env.close()


@pytest.mark.parametrize("operator", ["none", "td3", "mppi"])
def test_non_sac_comparators_clear_inherited_random_initialization(monkeypatch, operator):
    path = ROOT / "configs/research/ambi_inner_decoupling.json"
    matrix = ambi_research.load_preset_matrix(path)
    original_load = ambi_research._load_json

    def random_base(base_path):
        base = original_load(base_path)
        base["alg_params"].update({field: "random" for field in FIELDS})
        return base

    monkeypatch.setattr(ambi_research, "_load_json", random_base)
    resolved = ambi_research.resolve_preset(path, f"inner_operator/{operator}", matrix)
    cfg = _build_cfg(**resolved["algorithm_config"]["alg_params"])
    assert cfg.inner_operator == operator
    assert all(getattr(cfg, field) == "prior" for field in FIELDS)


def _lineage(params):
    return scientific_trial_parameters({"alg": ALGORITHM, "alg_params": params})


def _planner(params):
    return planner_identity(params, {}, ALGORITHM, "tanh_mean")


@pytest.mark.parametrize("identity", [_lineage, _planner])
@pytest.mark.parametrize("adaptation", ["clone", "lora_rl"])
def test_prior_defaults_preserve_historical_identity(identity, adaptation):
    historical = {"inner_operator": "sac", "inner_critic_adaptation": adaptation}
    before = deepcopy(historical)
    expected = identity(historical)
    explicit = {**historical, **{field: "PRIOR" for field in FIELDS}}
    assert identity(explicit) == expected
    settings = expected["alg_params"] if identity is _lineage else expected["settings"]
    assert not set(FIELDS).intersection(settings)
    assert historical == before


@pytest.mark.parametrize("identity", [_lineage, _planner])
def test_random_actor_and_critic_have_distinct_case_normalized_identities(identity):
    baseline = {"inner_operator": "sac", "inner_critic_adaptation": "lora_rl"}
    results = []
    for actor, critic in (("prior", "prior"), ("random", "prior"),
                          ("prior", "random"), ("random", "random")):
        params = {**baseline, FIELDS[0]: actor, FIELDS[1]: critic}
        result = identity(params)
        assert result == identity({**baseline, FIELDS[0]: actor.upper(), FIELDS[1]: critic.upper()})
        results.append(json.dumps(result, sort_keys=True))
    assert len(set(results)) == 4
