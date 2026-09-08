"""Paper-transfer presets preserve the base-v2 scientific workload."""

import copy
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ALGS = ROOT / "configs/dmcontrol/algs"
EXPERIMENTS = ROOT / "configs/dmcontrol/experiments"
BASE = "ambi_humanoid_walk_base_v2"
VARIANTS = [("input_hidden", 64), ("input_hidden", 96), ("input_hidden", 128), ("hidden", 96)]


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        assert key not in result, f"Duplicate JSON key: {key}"
        result[key] = value
    return result


def _load(path):
    return json.loads(path.read_text(), object_pairs_hook=_unique_object)


@pytest.mark.parametrize("layers,rank", VARIANTS)
def test_critic_lora_changes_only_adaptation_and_run_labels(layers, rank):
    baseline = _load(ALGS / f"{BASE}.json")
    name = f"{BASE}_lora_rl_{layers}_r{rank}"
    actual = _load(ALGS / f"{name}.json")
    expected = copy.deepcopy(baseline)
    expected["alg_params"].update({
        "inner_critic_adaptation": "lora_rl",
        "inner_critic_lora_layers": layers,
        "inner_critic_lora_rank": rank,
        "inner_critic_lora_scale": 1.0,
        "inner_critic_lora_weight_decay": 0.0002,
        "wandb_run_name": (
            f"AMBITDMPC2-humanoid-walk-base-v2-j8-n32-g1-"
            f"lora-rl-{layers.replace('_', '-')}-r{rank}-seed55"
        ),
        "wandb_tags": [
            *baseline["alg_params"]["wandb_tags"],
            "inner-critic-lora-rl",
            f"lora-layers-{layers.replace('_', '-')}",
            f"lora-r{rank}",
            "lora-scale1",
            "lora-adapter-wd2e-4",
        ],
    })
    assert actual == expected
    params = actual["alg_params"]
    assert params["inner_actor_adaptation"] == "clone"
    assert not any(key.startswith("inner_actor_lora_") for key in params)
    assert "inner_critic_lora_dropout" not in params
    for component in ("actor", "critic", "temperature", "replay", "actor_optimizer",
                      "critic_optimizer", "temperature_optimizer"):
        assert params[f"inner_{component}_scope"] == "action"


@pytest.mark.parametrize("layers,rank", VARIANTS)
def test_critic_lora_manifest_preserves_base_v2_budget_and_evaluation(layers, rank):
    baseline = _load(EXPERIMENTS / f"{BASE}.json")
    name = f"{BASE}_lora_rl_{layers}_r{rank}"
    actual = _load(EXPERIMENTS / f"{name}.json")
    expected = copy.deepcopy(baseline)
    expected.update({
        "configs": [name],
        "study_type": "single_seed_exploratory_inner_critic_lora",
        "study_note": actual["study_note"],
    })
    assert actual == expected
    assert actual["study_note"].startswith(baseline["study_note"])
    assert f"rank-{rank}" in actual["study_note"]
    assert "not a replication or a demonstrated latency improvement" in actual["study_note"]
    assert (ALGS / f"{name}.json").is_file()


@pytest.mark.parametrize("layers,rank", VARIANTS)
def test_new_preset_passes_the_algorithm_configuration_resolver(layers, rank):
    from tests.test_ambi_config_decoupling import _build_cfg

    params = _load(ALGS / f"{BASE}_lora_rl_{layers}_r{rank}.json")["alg_params"]
    cfg = _build_cfg(**params)
    assert cfg.inner_critic_adaptation == "lora_rl"
    assert cfg.inner_critic_lora_layers == layers
    assert cfg.inner_critic_lora_rank == rank
    assert cfg.inner_model_step_budget == 768
    assert cfg.inner_critic_updates_per_action == 8


def test_obsolete_joint_lora_presets_are_retired_without_repurposing_names():
    for name in ("ambi_humanoid_walk_base_lora_r8", "ambi_humanoid_walk_base_lora_r16"):
        assert not (ALGS / f"{name}.json").exists()
        assert not (EXPERIMENTS / f"{name}.json").exists()
    retired = "AntLegAdaptAMBITDMPC2LoRA"
    assert not (ROOT / f"configs/algs/{retired}.json").exists()
    assert not (ROOT / f"configs/experiments/{retired}.json").exists()
    sweep = _load(ROOT / "configs/experiments/AntLegAdaptPaperSweep.json")
    assert sweep["configs"] == ["AntLegAdaptSAC", "AntLegAdaptTDMPC2", "AntLegAdaptAMBITDMPC2"]
    launcher = (ROOT / "run_ambi_ccv_leg_adapt_sweep.sh").read_text()
    assert retired not in launcher
    assert "#SBATCH --array=0-2" in launcher
