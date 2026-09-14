"""Keep the six single-seed reward-critic backbones scientifically controlled."""

from copy import deepcopy
import json
import sys
from types import SimpleNamespace

import pytest

from tests import test_ambi_prior_sac_study_configs as sac_study


MANIFEST = "ambi_prior_reward_critic_study"
GROUP = "ambi-prior-reward-critic-20260914"
COUNTERPARTS = {
    name.replace("_sac_", "_reward_"): name for name in sac_study.CASES
}
FIXED_COUNTERPARTS = {
    f"ambi_prior_reward_{mapping}_fixed0p0001":
        f"ambi_prior_reward_{mapping}_target21"
    for mapping in ("clip", "smooth")
}
ALL_CASES = tuple(COUNTERPARTS) + tuple(FIXED_COUNTERPARTS)
CHANGED_PARAMS = {"outer_critic_target", "wandb_group", "wandb_tags"}


def _resolve(monkeypatch, name, trial=0, *, manifest=MANIFEST):
    # Reuse the real config resolver while selecting this study's own manifest.
    with monkeypatch.context() as patch:
        patch.setattr(sac_study, "MANIFEST", manifest)
        return sac_study._resolve(name, trial)


@pytest.mark.parametrize("name,control", COUNTERPARTS.items())
def test_reward_critic_has_one_active_change_from_its_soft_control(
    monkeypatch, name, control,
):
    candidate = sac_study._load("algs", name)
    original = sac_study._load("algs", control)
    expected = deepcopy(original)
    expected["alg_params"]["outer_critic_target"] = "reward_only"
    expected["alg_params"]["wandb_group"] = GROUP
    replacements = {
        "sac-prior-parameterization": "reward-critic-prior",
        "critic-target-entropy-augmented": "critic-target-reward-only",
        "two-seed": "single-seed",
        "seeds55-56": "seed55",
    }
    expected["alg_params"]["wandb_tags"] = [
        replacements.get(tag, tag) for tag in original["alg_params"]["wandb_tags"]
    ]
    assert candidate == expected

    algorithm = _resolve(monkeypatch, name)
    baseline = _resolve(monkeypatch, control, manifest=sac_study.MANIFEST)
    cfg = algorithm.cfg
    assert {
        key: value for key, value in vars(cfg).items() if key not in CHANGED_PARAMS
    } == {
        key: value for key, value in vars(baseline.cfg).items()
        if key not in CHANGED_PARAMS
    }
    assert cfg.seed == 55
    assert cfg.outer_critic_target == "reward_only"
    assert cfg.outer_actor_entropy_mode == "squashed"
    assert cfg.ent_coef == "auto_1.0"
    assert cfg.sac_actor_loss_scale_mode == "none"
    assert cfg.outer_q_actor_reduction == "mean_pair"
    assert cfg.outer_q_target_reduction == "min_pair"
    assert (cfg.log_std_mapping, cfg.target_entropy) == sac_study.CASES[control]
    assert cfg.outer_policy_diagnostics and cfg.wandb_event_indexed
    # The inactive inner objective is unchanged; this is not an inner recipe.
    assert cfg.inner_sac_critic_target == "entropy_augmented"
    assert cfg.inner_operator == "none"
    assert cfg.inner_model_step_budget == cfg.inner_actor_updates_per_action == 0


@pytest.mark.parametrize("name,control", FIXED_COUNTERPARTS.items())
def test_fixed_temperature_changes_only_the_coefficient_and_its_tags(
    monkeypatch, name, control,
):
    candidate = sac_study._load("algs", name)
    original = sac_study._load("algs", control)
    expected = deepcopy(original)
    expected["alg_params"]["ent_coef"] = 0.0001
    replacements = {
        "entropy-autotemp": "entropy-fixed",
        "outer-alpha-auto-initial1": "outer-alpha-fixed0p0001",
        "entropy-target-minus21": "entropy-target-inactive",
    }
    expected["alg_params"]["wandb_tags"] = [
        replacements.get(tag, tag) for tag in original["alg_params"]["wandb_tags"]
    ]
    assert candidate == expected

    algorithm = _resolve(monkeypatch, name)
    baseline = _resolve(monkeypatch, control)
    changed = {"ent_coef", "wandb_tags"}
    assert {
        key: value for key, value in vars(algorithm.cfg).items()
        if key not in changed
    } == {
        key: value for key, value in vars(baseline.cfg).items()
        if key not in changed
    }
    assert algorithm.cfg.ent_coef == 0.0001
    # The target and temperature LR remain serialized but are inactive.
    assert algorithm.cfg.target_entropy == -21.0
    assert algorithm.cfg.ent_coef_lr == 3e-4


def test_reward_study_preserves_the_harness_and_has_six_single_seed_cells():
    manifest = sac_study._load("experiments", MANIFEST)
    original = sac_study._load("experiments", sac_study.MANIFEST)
    identity = {"study_type", "study_note", "configs", "trials"}
    assert {k: v for k, v in manifest.items() if k not in identity} == {
        k: v for k, v in original.items() if k not in identity
    }
    assert manifest["study_type"] != original["study_type"]
    assert manifest["study_note"] != original["study_note"]
    assert len(manifest["configs"]) == len(set(manifest["configs"])) == 6
    assert manifest["configs"] == list(ALL_CASES)
    assert manifest["trials"] == 1
    assert "alg_params" not in manifest["overrides_alg"]
    cells = []
    for name in manifest["configs"]:
        config = sac_study._load("algs", name)
        run = {**config, **manifest["overrides_alg"]}
        assert run["total_steps"] == 2_000_000
        assert run["checkpoint_every"] == 25_000
        assert run["save_strat"] == "all"
        assert run["total_steps"] // run["checkpoint_every"] == 80
        assert {"single-seed", "seed55"}.issubset(run["alg_params"]["wandb_tags"])
        assert not {"two-seed", "seeds55-56"}.intersection(
            run["alg_params"]["wandb_tags"]
        )
        cells.extend(
            (name, run["seed"] + trial) for trial in range(manifest["trials"])
        )
    assert len(cells) == len(set(cells)) == 6
    assert {seed for _, seed in cells} == {55}
    assert len(cells) * 2_000_000 // 25_000 == 480


def test_reward_run_names_and_wandb_metadata_are_distinct(monkeypatch):
    calls = []
    run = SimpleNamespace(finish=lambda: None, log=lambda *args, **kwargs: None)
    fake = SimpleNamespace(
        init=lambda **kwargs: calls.append(kwargs) or run,
        define_metric=lambda *args, **kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake)
    for name in ALL_CASES:
        algorithm = _resolve(monkeypatch, name)
        assert "wandb_run_name" not in algorithm.custom_params
        assert algorithm._init_wandb().raw_run is run
        call = calls[-1]
        assert call["name"] == f"AMBITDMPC2-{name}-seed55"
        assert call["group"] == GROUP
        assert call["config"]["config"]["outer_critic_target"] == "reward_only"
        assert call["config"]["config"]["seed"] == 55
        assert call["config"]["run_params"]["seed"] == 55
        assert call["config"]["config"]["ent_coef"] == (
            0.0001 if name in FIXED_COUNTERPARTS else "auto_1.0"
        )
    assert len({call["name"] for call in calls}) == 6


@pytest.mark.parametrize("name", ALL_CASES)
def test_checkpoint_metadata_identifies_reward_critic_and_dormant_soft_inner(
    monkeypatch, name,
):
    from main import _resolved_runtime_metadata
    from utils.checkpointing import checkpoint_metadata

    algorithm = _resolve(monkeypatch, name)
    params = deepcopy(algorithm.run_params)
    params["resolved_runtime"] = _resolved_runtime_metadata(
        algorithm, trial_run_params=params,
    )
    sidecar = checkpoint_metadata(
        kind="scheduled", step=25_000, episode=50, best_score=None,
        best_window=100, trial_run_params=params,
        experiment_params=algorithm.experiment_params,
    )
    restored = json.loads(json.dumps(sidecar, allow_nan=False))["trial_run_params"]
    assert restored["seed"] == 55
    assert restored["alg_params"]["wandb_group"] == GROUP
    for view in (restored["alg_params"], restored["resolved_runtime"]["critic"]):
        assert view["outer_critic_target"] == "reward_only"
        assert view["inner_sac_critic_target"] == "entropy_augmented"
    assert restored["resolved_runtime"]["outer_policy_diagnostics"] == {
        key: value for key, value in sac_study.COMMON.items()
        if key.startswith("outer_policy_diagnostics") or key == "wandb_event_indexed"
    }
    entropy = restored["resolved_runtime"]["actor_entropy"]["outer"]
    fixed = name in FIXED_COUNTERPARTS
    assert entropy["actor_entropy_mode"] == "squashed"
    assert entropy["target_entropy_semantics"] == "squashed_action_entropy"
    assert entropy["temperature_mode"] == ("fixed" if fixed else "auto")
    assert entropy["target_active"] is not fixed
    assert entropy["target_entropy"] == algorithm.cfg.target_entropy
    assert restored["alg_params"]["ent_coef"] == (0.0001 if fixed else "auto_1.0")
