"""Freeze the three-cell training-horizon campaign and its fail-closed GPU receipt."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from RL.AMBITDMPC2 import AMBITDMPC2
from slurm import ambi_aux_return_horizon_campaign as campaign
from tests.test_ambi_prior_sac_study_configs import _HumanoidSpaces, _unique_object


ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 40


def _load(path):
    return json.loads(path.read_text(), object_pairs_hook=_unique_object)


def _resolve(name):
    manifest = _load(ROOT / campaign.MANIFEST)
    config = _load(ROOT / "configs/dmcontrol/algs" / (name + ".json"))
    run = {"name": name, **config, **manifest["overrides_alg"], "device": "cpu"}
    algorithm = object.__new__(AMBITDMPC2)
    algorithm.env = _HumanoidSpaces()
    algorithm.run_params = run
    algorithm.custom_params = deepcopy(run["alg_params"])
    algorithm.experiment_params = manifest
    algorithm.cfg = algorithm._build_cfg({"device": "cpu", **run["alg_params"]})
    return algorithm


@pytest.mark.parametrize("name", campaign.CASES)
def test_horizon_recipe_resolves_only_the_authorized_training_axis(name):
    learner = _resolve(name)
    cfg, params = learner.cfg, learner.custom_params
    horizon = campaign.HORIZONS[campaign.CASES.index(name)]
    baseline = _load(ROOT / "configs/dmcontrol/algs" / (campaign.BASELINE + ".json"))
    actual = _load(ROOT / "configs/dmcontrol/algs" / (name + ".json"))
    for key in ("train_unroll_horizon", "wandb_group", "wandb_tags"):
        actual["alg_params"][key] = baseline["alg_params"][key]
    assert actual == baseline
    assert cfg.train_unroll_horizon == cfg.horizon == horizon
    assert cfg.inner_rollout_horizon == cfg.outer_planning_horizon == 3
    assert cfg.aux_return_mode == "sac" and cfg.critic_value_mode == "single"
    assert cfg.aux_return_detach_representation is False
    assert cfg.target_entropy == -10.5 and cfg.log_std_mapping == "direct_clamp"
    assert cfg.log_std_min == -10 and cfg.log_std_max == 2
    assert cfg.seed == 55 and cfg.steps == 2_000_000
    assert cfg.rho == .5 and cfg.inner_operator == "none"
    assert cfg.inner_actor_updates_per_action == cfg.inner_critic_updates_per_action == 0
    assert params["compile"] is params["compile_strict"] is True


def test_three_fresh_cells_preserve_manifest_and_240_checkpoints():
    manifest, files = campaign.recipes()
    baseline = _load(ROOT / "configs/dmcontrol/experiments/ambi_aux_return_sac_study.json")
    assert {k: v for k, v in manifest.items() if k not in {"configs", "study_type", "study_note"}} == {
        k: v for k, v in baseline.items() if k not in {"configs", "study_type", "study_note"}}
    assert manifest["trials"] == 1 and manifest["logs"] == "timestamp"
    assert len(files) * manifest["overrides_alg"]["total_steps"] // manifest["checkpoint_every"] == 240


@pytest.mark.parametrize("horizon", campaign.HORIZONS)
def test_horizon_replay_contains_complete_real_slices(horizon):
    import numpy as np
    import torch
    from tests.test_ambi_aux_return_horizon_cuda_gate import _FixedHorizonReplay
    observations = np.arange(33 * 67, dtype=np.float32).reshape(33, 67)
    actions = np.arange(32 * 21, dtype=np.float64).reshape(32, 21)
    rewards = np.arange(32, dtype=np.float32)
    state = torch.random.get_rng_state().clone()
    replay = _FixedHorizonReplay(observations, actions, rewards, "cpu", horizon)
    obs, action, reward, terminated, task = replay.sample()
    assert torch.equal(state, torch.random.get_rng_state())
    assert obs.shape == (horizon + 1, 256, 67)
    assert action.shape == (horizon, 256, 21)
    assert reward.shape == terminated.shape == (horizon, 256, 1)
    assert replay.draws == 1 and task is None and not terminated.any()
    for column in (0, 25, 31, 255):
        start = column % (32 - horizon + 1)
        np.testing.assert_array_equal(obs[:, column], observations[start:start+horizon+1])
        np.testing.assert_array_equal(action[:, column], actions[start:start+horizon])
        np.testing.assert_array_equal(reward[:, column, 0], rewards[start:start+horizon])


def test_wandb_names_and_configs_are_unique_and_seed_correct(monkeypatch):
    calls = []
    run = SimpleNamespace(finish=lambda: None, log=lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(
        init=lambda **kwargs: calls.append(kwargs) or run, define_metric=lambda *args, **kwargs: None,
    ))
    for name in campaign.CASES:
        assert _resolve(name)._init_wandb().raw_run is run
        assert calls[-1]["name"] == f"AMBITDMPC2-{name}-seed55"
        assert calls[-1]["config"]["config"]["seed"] == 55
        assert calls[-1]["group"] == "ambi-aux-return-sac-horizons-20261002"
        assert calls[-1]["project"] == "ambi" and calls[-1]["mode"] == "online"
    assert len({call["name"] for call in calls}) == 3


def _receipt(index=0):
    binding = campaign.binding(SHA)
    name, horizon = campaign.CASES[index], campaign.HORIZONS[index]
    return {
        "schema": 1, "passed": True, "binding": binding, "index": index, "config": name,
        "case": {
            "config": name, "config_sha256": binding["config_sha256"][name], "source_commit": SHA,
            "train_unroll_horizon": horizon,
            "passed": True, "compile_strict": True, "checkpoint_roundtrip": True,
            "exact_checkpoint_roundtrip": True, "next_update_reproducible": True,
            "gradient_routing": True, "no_return_actor": True, "no_inner_updates": True,
            "stochastic_sac_collection": True, "cold_production_shape_updates": 2,
            "continuation_validation_executions": 2,
            "optimizer_updates": 3, "real_decisions": 32, "checkpoint_size_bytes": 42,
            "replay_seed": 55, "replay_sha256": "b" * 64,
            "production_overrides": {"wandb": False}, "production_warmup_and_pretraining_executed": False,
            "detach_representation": False,
            "auxiliary_gradient_l1": {"encoder": 1., "dynamics": 0. if horizon == 1 else 2., "critic": 3.},
            "compile_status": [{key: False for key in campaign.FLAGS} for _ in range(4)],
        },
        "fresh_training": {
            "passed": True, "source_commit": SHA, "config": name,
            "config_sha256": binding["config_sha256"][name], "train_unroll_horizon": horizon,
            "real_decisions": 512, "optimizer_updates": 13, "checkpoint_size_bytes": 42,
            "checkpoint_roundtrip": True, "checkpoint_sidecar": True,
            "compile_status": [{key: False for key in campaign.FLAGS}],
            "production_overrides": {"wandb": False, "seed_steps": 500, "pretrain_steps": 2,
                                     "buffer_size": 4096, "total_steps": 512, "checkpoint_every": 256},
        },
    }


@pytest.mark.parametrize("index", range(3))
def test_gpu_receipt_accepts_complete_horizon_evidence(index):
    receipt = _receipt(index)
    assert campaign.validate_receipt(receipt, SHA, index) is receipt


@pytest.mark.parametrize("failure", (
    "source", "manifest", "config_binding", "case_config", "wrong_index", "wrong_horizon", "missing_flag",
    "aux_online_fallback", "aux_target_fallback", "nonboolean_flag", "exact_resume", "next_update", "inner_updates",
    "extra_actor", "encoder_gradient", "h1_dynamics_gradient", "fresh_missing", "fresh_budget", "fresh_fallback",
    "fresh_config", "fresh_horizon", "cold_updates", "continuation_updates", "stochastic", "compile_observations",
))
def test_gpu_receipt_rejects_incomplete_or_incompatible_evidence(failure):
    receipt = _receipt()
    first = receipt["case"]
    if failure == "cold_updates": first["cold_production_shape_updates"] = 1
    elif failure == "continuation_updates": first["continuation_validation_executions"] = 0
    elif failure == "stochastic": first["stochastic_sac_collection"] = False
    elif failure == "compile_observations": first["compile_status"].pop()
    elif failure == "source": receipt["binding"]["source_commit"] = "f" * 40
    elif failure == "manifest": receipt["binding"]["manifest_sha256"] = "f" * 64
    elif failure == "config_binding": receipt["binding"]["config_sha256"][campaign.CASES[2]] = "f" * 64
    elif failure == "case_config": first["config_sha256"] = "f" * 64
    elif failure == "wrong_index": receipt["index"] = 1
    elif failure == "wrong_horizon": first["train_unroll_horizon"] = 3
    elif failure == "missing_flag": first["compile_status"][0].pop("aux_target")
    elif failure == "aux_online_fallback": first["compile_status"][0]["aux_online"] = True
    elif failure == "aux_target_fallback": first["compile_status"][0]["aux_target"] = True
    elif failure == "nonboolean_flag": first["compile_status"][0]["aux_online"] = 0
    elif failure == "exact_resume": first["exact_checkpoint_roundtrip"] = False
    elif failure == "next_update": first["next_update_reproducible"] = False
    elif failure == "inner_updates": first["no_inner_updates"] = False
    elif failure == "extra_actor": first["no_return_actor"] = False
    elif failure == "encoder_gradient": first["auxiliary_gradient_l1"]["encoder"] = 0.
    elif failure == "h1_dynamics_gradient": first["auxiliary_gradient_l1"]["dynamics"] = 1.
    elif failure == "fresh_missing": receipt.pop("fresh_training")
    elif failure == "fresh_budget": receipt["fresh_training"]["real_decisions"] = 32
    elif failure == "fresh_fallback": receipt["fresh_training"]["compile_status"][0]["aux_target"] = True
    elif failure == "fresh_config": receipt["fresh_training"]["config_sha256"] = "f" * 64
    elif failure == "fresh_horizon": receipt["fresh_training"]["train_unroll_horizon"] = 3
    with pytest.raises((AssertionError, KeyError)):
        campaign.validate_receipt(receipt, SHA, 0)


def test_launcher_syntax_and_requeue_rejection(tmp_path):
    launcher = ROOT / "slurm/run_ambi_aux_return_horizons_oscar.sbatch"
    subprocess.run(["bash", "-n", str(launcher)], check=True)
    output = tmp_path / "must-not-create"
    env = {**os.environ, "SLURM_JOB_ID": "123", "SLURM_RESTART_COUNT": "1",
           "AMBI_PROJECT_DIR": str(ROOT), "AMBI_PYTHON": sys.executable,
           "EXPECTED_ACTION_MODES_SHA": SHA, "AMBI_AUX_OUTPUT_ROOT": str(output), "AMBI_AUX_CAMPAIGN": "test"}
    result = subprocess.run(["bash", str(launcher)], env=env, capture_output=True, text=True)
    assert result.returncode == 2 and "cannot resume after requeue" in result.stderr
    assert not output.exists()


def test_optimized_python_cannot_bypass_receipt_validation():
    helper = ROOT / "slurm/ambi_aux_return_horizon_campaign.py"
    optimized = compile(helper.read_text(), str(helper), "exec", optimize=1)
    with pytest.raises(RuntimeError, match="requires Python assertions"):
        exec(optimized, {"__name__": "optimized_campaign", "__file__": str(helper)})


def test_launch_metadata_records_full_resolution_and_job_to_wandb_mapping(tmp_path, monkeypatch):
    receipt = _receipt(2)
    current_runtime = {"python_executable": sys.executable, "python": sys.version,
                       "torch": "test", "cuda": "test", "gpu": "test", "compute_capability": [0, 0]}
    receipt["runtime"] = current_runtime
    receipt_path = tmp_path / "receipt.json"
    campaign.write_new(receipt_path, receipt)
    monkeypatch.setattr(campaign, "runtime", lambda: current_runtime)
    monkeypatch.setenv("WANDB_API_KEY", "fixture-placeholder-not-a-real-key")
    monkeypatch.setenv("WANDB_RUN_ID", "auxh123x2")
    monkeypatch.setenv("AMBI_AUX_CAMPAIGN", "test-campaign")
    monkeypatch.setenv("SLURM_JOB_ID", "125")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "123")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "2")
    import gymnasium as gym
    monkeypatch.setattr(gym, "make", lambda *args, **kwargs: _HumanoidSpaces())
    campaign.prepare_training(receipt_path, SHA, 2, tmp_path)
    actual = _load(tmp_path / "launch.json")
    assert actual["config"] == campaign.CASES[2] and actual["seed"] == 55
    assert actual["binding"] == campaign.binding(SHA)
    assert actual["gate_receipt_sha256"] == campaign.sha256(receipt_path)
    assert actual["resolved_config"]["aux_return_mode"] == "sac"
    assert actual["resolved_config"]["aux_return_detach_representation"] is False
    assert actual["resolved_config"]["target_entropy"] == -10.5
    assert actual["resolved_config"]["train_unroll_horizon"] == 7
    assert actual["resolved_config"]["steps"] == 2_000_000
    assert actual["resolved_config"]["compile_strict"] is True
    assert actual["wandb"]["url"] == "https://wandb.ai/rwgao_b-brown-university/ambi/runs/auxh123x2"
    assert actual["wandb"]["name"] == f"AMBITDMPC2-{campaign.CASES[2]}-seed55"
    assert actual["slurm"]["SLURM_ARRAY_JOB_ID"] == "123"
    assert "fixture-placeholder" not in (tmp_path / "launch.json").read_text()
    with pytest.raises(FileExistsError):
        campaign.prepare_training(receipt_path, SHA, 2, tmp_path)
