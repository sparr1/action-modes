"""Freeze the original-prior settings pair while retaining current initialization."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from RL.AMBITDMPC2 import AMBITDMPC2
from RL.tdmpc2_core.common.soft_world_model import SoftWorldModel
from tests.test_ambi_prior_sac_study_configs import COMMON, _HumanoidSpaces, _unique_object


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "ambi_aux_return_original_prior_study"
CASES = tuple(f"ambi_aux_return_original_prior_{suffix}" for suffix in ("shared", "detached"))
GROUP = "ambi-aux-return-original-prior-20260916"


def _load(kind, name):
    return json.loads((ROOT / "configs/dmcontrol" / kind / f"{name}.json").read_text(),
                      object_pairs_hook=_unique_object)


def _resolve(name):
    manifest = _load("experiments", MANIFEST)
    config = _load("algs", name)
    run = {"name": name, **config, **manifest["overrides_alg"], "device": "cpu"}
    algorithm = object.__new__(AMBITDMPC2)
    algorithm.env = _HumanoidSpaces()
    algorithm.run_params = run
    algorithm.custom_params = deepcopy(run["alg_params"])
    algorithm.experiment_params = manifest
    algorithm.cfg = algorithm._build_cfg({"device": "cpu", **run["alg_params"]})
    return algorithm


@pytest.mark.parametrize("name", CASES)
def test_full_recipe_differs_from_existing_auxiliary_pair_only_as_authorized(name):
    suffix = "detached" if name.endswith("_detached") else "shared"
    old = _load("algs", f"ambi_aux_return_sac_clip_target21_{suffix}")
    actual = _load("algs", name)
    expected = deepcopy(old)
    expected["total_steps"] = 1_500_000
    expected["alg_params"].update(
        outer_q_actor_reduction="min_pair", inner_q_actor_reduction="min_pair",
        aux_return_outer_q_actor_reduction="min_pair", log_std_min=-20,
        inner_log_std_min=-20, wandb_group=GROUP,
        wandb_tags=actual["alg_params"]["wandb_tags"],
    )
    assert actual == expected
    tags = set(actual["alg_params"]["wandb_tags"])
    assert {"1p5m-decisions", "actor-q-min-pair", "log-std-bounds-minus20-plus2",
            "original-prior-u13m14st-settings", "current-critic-initialization",
            "strict-cuda-compile", "single-seed", "seed55"} <= tags
    assert not {"2m-decisions", "actor-q-mean-pair", "log-std-bounds-minus10-plus2"} & tags


@pytest.mark.parametrize("name", CASES)
def test_resolved_outer_recipe_matches_original_prior_settings_and_auxiliary_contract(name):
    learner = _resolve(name)
    cfg = learner.cfg
    # Original u13m14st: auto alpha starts at 1 and auto target is -action_dim.
    # At H=3 its reference-H=3 temporal branch equals divide_horizon.
    expected = {**COMMON, "compile_strict": True, "outer_q_actor_reduction": "min_pair",
                "log_std_min": -20}
    for key, value in expected.items():
        assert learner.custom_params[key] == value and getattr(cfg, key) == value, key
    assert cfg.ent_coef == "auto_1.0" and cfg.target_entropy == -cfg.action_dim == -21.
    assert cfg.log_std_mapping == cfg.inner_log_std_mapping == "direct_clamp"
    assert cfg.log_std_min == cfg.inner_log_std_min == -20
    assert cfg.log_std_max == cfg.inner_log_std_max == 2
    assert cfg.inner_q_actor_reduction == cfg.inner_q_target_reduction == "min_pair"
    assert cfg.temporal_loss_normalization == "divide_horizon"
    assert cfg.aux_return_mode == "sac" and cfg.critic_value_mode == "single"
    assert cfg.aux_return_detach_representation is name.endswith("_detached")
    assert cfg.aux_return_critic_coef == cfg.critic_coef == .1
    assert cfg.aux_return_critic_lr == cfg.actor_lr == cfg.critic_lr == 3e-4
    for key, value in {"outer_q_actor_reduction": "min_pair", "outer_q_target_reduction": "min_pair",
                       "log_std_min": -20, "log_std_max": 2, "sac_actor_loss_scale_mode": "none"}.items():
        assert cfg.aux_return_actor_cfg[key] == value
    assert (cfg.action_dim, cfg.obs_shape, cfg.episode_length) == (21, {"state": (67,)}, 500)
    assert cfg.latent_dim == cfg.mlp_dim == 512
    assert cfg.seed == 55 and cfg.steps == 1_500_000
    assert cfg.inner_model_step_budget == cfg.inner_actor_updates_per_action == cfg.inner_critic_updates_per_action == 0
    for key in ("inner_actor_source", "inner_critic_source", "inner_horizon_actor_source", "inner_horizon_critic_source"):
        assert getattr(cfg, key) == "sac"


def test_pair_varies_only_detachment_and_tags_and_retains_120_checkpoints():
    manifest = _load("experiments", MANIFEST)
    assert tuple(manifest["configs"]) == CASES
    assert manifest["trials"] == 1 and "alg_params" not in manifest["overrides_alg"]
    assert manifest["overrides_alg"]["total_steps"] == 1_500_000
    assert manifest["overrides_alg"]["seed"] == 55
    assert manifest["env_params"] == {"task": "humanoid-walk", "obs": "state", "render_mode": None}
    assert manifest["logs"] == "timestamp" and manifest["save_trials"] == "none"
    assert manifest["checkpoint_every"] == 25_000 and manifest["save_strat"] == "all"
    normalized = []
    for name in CASES:
        config = _load("algs", name)
        assert config["total_steps"] == 1_500_000 and config["seed"] == 55
        assert config["checkpoint_every"] == 25_000 and config["save_strat"] == "all"
        assert config["episodes"] is None
        config["alg_params"].pop("aux_return_detach_representation")
        config["alg_params"].pop("wandb_tags")
        normalized.append(config)
    assert normalized[0] == normalized[1]
    assert len(CASES) * 1_500_000 // 25_000 == 120


def test_wandb_identity_is_distinct_from_existing_auxiliary_campaign(monkeypatch):
    calls = []
    run = SimpleNamespace(finish=lambda: None, log=lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(
        init=lambda **kwargs: calls.append(kwargs) or run, define_metric=lambda *args, **kwargs: None,
    ))
    for name in CASES:
        assert _resolve(name)._init_wandb().raw_run is run
        assert calls[-1]["name"] == f"AMBITDMPC2-{name}-seed55"
        assert calls[-1]["config"]["config"]["seed"] == 55
        assert calls[-1]["group"] == GROUP
        assert calls[-1]["project"] == "ambi" and calls[-1]["mode"] == "online"
    assert len({call["name"] for call in calls}) == 2


def test_network_parameters_retain_current_initialization_in_both_arms():
    """Recipe changes affect action statistics/reductions, not initialization."""
    parameters, rng_states = [], []
    with torch.random.fork_rng(devices=[]):
        for name in ("ambi_aux_return_sac_clip_target21_shared", *CASES):
            cfg = deepcopy(_resolve(name).cfg)
            cfg.model_size = None
            cfg.enc_dim = cfg.mlp_dim = 32
            cfg.latent_dim = 16
            cfg.num_enc_layers = 2
            cfg.simnorm_dim = 8
            cfg.num_bins = cfg.q_num_bins = 11
            torch.manual_seed(cfg.seed)
            model = SoftWorldModel(cfg)
            assert not hasattr(model, "_return_pi")
            for ensemble in (model._Qs, model._aux_return_Qs):
                for critic in ensemble:
                    assert torch.count_nonzero(critic[-1].weight) == 0
                    assert torch.count_nonzero(critic[-1].bias) > 0
            parameters.append({key: value.detach().clone() for key, value in model.named_parameters()})
            rng_states.append(torch.random.get_rng_state().clone())
    for params, rng in zip(parameters[1:], rng_states[1:]):
        assert params.keys() == parameters[0].keys()
        for key, value in params.items():
            torch.testing.assert_close(value, parameters[0][key], rtol=0, atol=0)
        torch.testing.assert_close(rng, rng_states[0], rtol=0, atol=0)
