"""Keep the launch matrix paired and its complete checkpoint banks explicit."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from tests.test_ambi_root_local_sac import _build_cfg
from slurm import ambi_transfer_training as campaign

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "configs/dmcontrol/experiments/ambi_transfer_training.json").read_text())
NAMES = MANIFEST["configs"]
J8_PATH = ROOT / "configs/dmcontrol/experiments/ambi_transfer_training_j8.json"
J8_MANIFEST = json.loads(J8_PATH.read_text())
CAMPAIGNS = [(6, MANIFEST), (8, J8_MANIFEST)]


@pytest.mark.parametrize("rounds,manifest,index,name", [
    (rounds, manifest, index, name)
    for rounds, manifest in CAMPAIGNS
    for index, name in enumerate(manifest["configs"])
])
def test_full_training_science_and_checkpoint_contract(rounds, manifest, index, name):
    config = json.loads((ROOT / "configs/dmcontrol/algs" / (name + ".json")).read_text())
    assert config["seed"] == 55 and config["total_steps"] == 500000
    assert config["checkpoint_every"] == manifest["checkpoint_every"] == 25000
    assert config["save_strat"] == manifest["save_strat"] == "all"
    assert "alg_params" not in manifest["overrides_alg"]
    params = config["alg_params"]
    cfg = _build_cfg(**params)
    assert cfg.inner_rollout_horizon == 1 + index // 2
    assert cfg.inner_critic_bernoulli_p == (0. if index % 2 == 0 else .5)
    assert (cfg.inner_rounds, cfg.inner_rollouts_per_round, cfg.inner_batch_size) == (rounds, 128, 256)
    assert (cfg.inner_critic_updates_per_round, cfg.inner_actor_updates_per_round) == (16, 4)
    assert (cfg.inner_critic_updates_per_action, cfg.inner_actor_updates_per_action) == (rounds * 16, rounds * 4)
    assert cfg.inner_replay_capacity == 3072
    assert cfg.inner_finite_horizon and cfg.inner_sac_critic_target == "reward_only"
    assert cfg.inner_critic_source == cfg.inner_horizon_critic_source == "aux_return"
    assert cfg.inner_actor_source == cfg.inner_horizon_actor_source == "sac"
    assert cfg.aux_return_mode == "sac" and not cfg.aux_return_detach_representation
    assert cfg.critic_ensemble_backend == "vmap" and cfg.compile and cfg.compile_strict
    assert cfg.inner_temperature_mode == "auto" and cfg.inner_entropy_enabled
    assert cfg.inner_execution_action == "policy_sample"
    assert cfg.seed_steps == cfg.pretrain_steps == 2500 and cfg.utd == 1
    assert params["target_entropy"] == -10.5
    assert params["wandb_project"] == "ambi"
    assert not params.get("pretrained_model_path")


@pytest.mark.parametrize("rounds,manifest", CAMPAIGNS)
def test_only_horizon_and_transfer_change_between_the_six_cells(rounds, manifest):
    names = manifest["configs"]
    assert len(names) == len(set(names)) == 6
    normalized = []
    for name in names:
        config = json.loads((ROOT / "configs/dmcontrol/algs" / (name + ".json")).read_text())
        params = deepcopy(config["alg_params"])
        for key in ("inner_rollout_horizon", "inner_critic_bernoulli_p", "wandb_run_name", "wandb_tags"):
            params.pop(key)
        normalized.append(params)
    assert all(row == normalized[0] for row in normalized)


def test_launcher_requires_clean_pinned_source_and_successful_smoke():
    launcher = (ROOT / "slurm/run_ambi_transfer_training_hydra.sbatch").read_text()
    helper = (ROOT / "slurm/ambi_transfer_training.py").read_text()
    assert "git status --porcelain --untracked-files=all" in launcher
    assert "AMBI_EXPECTED_COMMIT" in launcher and "--no-requeue" in launcher
    assert "gate/receipt.json" in launcher
    assert 'receipt["binding"] == campaign_binding' in helper
    assert 'output.mkdir(parents=True, exist_ok=False)' in helper
    assert launcher.count('--manifest "$manifest"') == 2


def test_j8_changes_only_rounds_and_experiment_identity():
    for old_name, new_name in zip(NAMES, J8_MANIFEST["configs"]):
        assert new_name == old_name.replace("_j6_", "_j8_")
        old = json.loads((ROOT / "configs/dmcontrol/algs" / (old_name + ".json")).read_text())
        new = json.loads((ROOT / "configs/dmcontrol/algs" / (new_name + ".json")).read_text())
        expected = deepcopy(old)
        params = expected["alg_params"]
        params["inner_rounds"] = 8
        params["wandb_run_name"] = params["wandb_run_name"].replace("J6", "J8")
        params["wandb_group"] = "ambi-transfer-training-j8-20261014"
        params["wandb_tags"] = [
            tag.replace("J6", "J8").replace("ambi-transfer-training-20261014", "ambi-transfer-training-j8-20261014")
            for tag in params["wandb_tags"]
        ]
        assert new == expected
    expected_manifest = deepcopy(MANIFEST)
    expected_manifest["configs"] = [name.replace("_j6_", "_j8_") for name in NAMES]
    expected_manifest["study_note"] = expected_manifest["study_note"].replace("J6/C16", "J8/C16")
    assert J8_MANIFEST == expected_manifest


def test_existing_j6_campaign_files_are_unchanged():
    # These are the exact published J6 inputs; a new campaign cannot rewrite them.
    hashes = [
        "0462b5d87971be70161c8e4c6b6a6d103f066605b16e143d8f400acb105ec215",
        "4459a5b3260ba9d72c3c779243313dd9aa9091be75414540973ca995ec776af0",
        "dbb851f780175d9ed0a015277c3fad656ff2df89e2ecc5b14bb41a572c44b81c",
        "34f5f777b6972f801ed4bc9e86ebb1bb2da527bd87348827ecb20950370ba410",
        "99ead7895751957d5340e1c9f0ec7e43a22b9032923e728a0528423f4efa5c79",
        "7eb00b7836f2c2b00b39e26284a4d7d43d6787d446597f81e9857e8e96a044c8",
    ]
    for name, expected in zip(NAMES, hashes):
        assert campaign.digest(ROOT / "configs/dmcontrol/algs" / (name + ".json")) == expected
    assert campaign.digest(campaign.MANIFEST) == "e0f77cab0f49ecbd8d128e026b8eb7a7da7c79d5d597a9d087a0fe2dcb7b9686"


def test_selected_manifest_controls_recipe_and_gate_binding():
    default = campaign.binding()
    explicit_j6 = campaign.binding(campaign.MANIFEST.relative_to(ROOT))
    j8 = campaign.binding(J8_PATH.relative_to(ROOT))
    assert default == explicit_j6
    assert default["manifest"] != j8["manifest"]
    assert default["manifest_sha256"] != j8["manifest_sha256"]
    assert set(j8["config_sha256"]) == set(J8_MANIFEST["configs"])
    assert default["lock_sha256"] == j8["lock_sha256"]
    assert campaign.recipe(0)[2]["alg_params"]["inner_rounds"] == 6
    assert campaign.recipe(0, J8_PATH)[2]["alg_params"]["inner_rounds"] == 8


def test_j6_gpu_gate_cannot_launch_j8(tmp_path):
    gate = tmp_path / "receipt.json"
    gate.write_text(json.dumps({"passed": True, "binding": campaign.binding()}))
    output = tmp_path / "train"
    with pytest.raises(AssertionError, match="GPU gate does not match"):
        campaign.launch(0, output, gate, J8_PATH)
    assert not output.exists()


def test_selected_manifest_reaches_main_and_launch_record(tmp_path, monkeypatch):
    runtime = {"python": "3.10", "torch": "2.3.1", "cuda": "12.1", "executable": "/test/python"}
    monkeypatch.setattr(campaign, "runtime", lambda: runtime)
    monkeypatch.setenv("WANDB_API_KEY", "test-only")
    monkeypatch.setenv("WANDB_RUN_ID", "test-j8")
    gate = tmp_path / "receipt.json"
    gate.write_text(json.dumps({"passed": True, "binding": campaign.binding(J8_PATH),
                               "cases": [{"passed": True}] * 6, "runtime": runtime}))
    calls = []
    monkeypatch.setattr(campaign.os, "execv", lambda executable, argv: calls.append((executable, argv)))
    output = tmp_path / "train"
    campaign.launch(3, output, gate, J8_PATH.relative_to(ROOT))
    command = calls[0][1]
    assert command[command.index("--run") + 1] == str(J8_PATH)
    record = json.loads((output / "launch.json").read_text())
    assert record["config"] == "ambi_full_h2_j8_bernoulli50"
    assert record["run_params"]["alg_params"]["inner_rounds"] == 8
    assert record["binding"] == campaign.binding(J8_PATH)


def test_manifest_outside_pinned_checkout_is_rejected(tmp_path):
    path = tmp_path / "external.json"
    path.write_text("{}")
    with pytest.raises(ValueError):
        campaign.resolve_manifest(path)
