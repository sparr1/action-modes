"""Freeze the four-cell prior-only campaign and its fail-closed GPU receipt."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from RL.AMBITDMPC2 import AMBITDMPC2
from slurm import ambi_aux_return_sac_campaign as campaign
from tests.test_ambi_prior_sac_study_configs import COMMON, _HumanoidSpaces, _unique_object


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
def test_full_recipe_resolves_only_authorized_auxiliary_study_changes(name):
    learner = _resolve(name)
    params, cfg = learner.custom_params, learner.cfg
    target = -10.5 if "target10p5" in name else -21.
    detached = name.endswith("_detached")
    baseline_name = "ambi_prior_sac_clip_target" + ("10p5" if target == -10.5 else "21")
    baseline = _load(ROOT / "configs/dmcontrol/algs" / (baseline_name + ".json"))
    allowed = {"compile_strict", "wandb_group", "wandb_tags", "critic_value_mode", "inner_rebase_persistent"}
    for key, value in baseline["alg_params"].items():
        if key not in allowed:
            assert params[key] == value, key
    for key, value in COMMON.items():
        expected = True if key == "compile_strict" else value
        assert params[key] == expected and getattr(cfg, key) == expected, key
    assert cfg.aux_return_mode == "sac" and cfg.critic_value_mode == "single"
    assert cfg.aux_return_detach_representation is detached
    assert cfg.aux_return_critic_coef == cfg.critic_coef == .1
    assert cfg.actor_lr == cfg.critic_lr == cfg.aux_return_critic_lr == 3e-4
    assert cfg.aux_return_actor_cfg["outer_q_actor_reduction"] == "mean_pair"
    assert cfg.aux_return_actor_cfg["outer_q_target_reduction"] == "min_pair"
    assert cfg.aux_return_actor_cfg["sac_actor_loss_scale_mode"] == "none"
    assert cfg.target_entropy == target and cfg.log_std_mapping == "direct_clamp"
    assert cfg.log_std_min == -10 and cfg.log_std_max == 2
    assert (cfg.action_dim, cfg.obs_shape, cfg.episode_length) == (21, {"state": (67,)}, 500)
    assert cfg.inner_model_step_budget == cfg.inner_actor_updates_per_action == cfg.inner_critic_updates_per_action == 0
    for key in ("inner_actor_source", "inner_critic_source", "inner_horizon_actor_source", "inner_horizon_critic_source"):
        assert getattr(cfg, key) == "sac"
    assert cfg.seed == 55 and cfg.steps == 2_000_000
    assert {"single-seed", "seed55", "no-inner", "aux-return-no-extra-actor", "strict-cuda-compile"} <= set(params["wandb_tags"])
    assert not {"two-seed", "seeds55-56", "sac-prior-parameterization"} & set(params["wandb_tags"])


def test_four_cells_vary_only_entropy_target_and_detach_with_320_checkpoints():
    manifest, files = campaign.recipes()
    assert manifest["trials"] == 1 and manifest["logs"] == "timestamp"
    assert manifest["env_params"] == {"task": "humanoid-walk", "obs": "state", "render_mode": None}
    assert manifest["checkpoint_every"] == 25_000 and manifest["save_strat"] == "all"
    normalized, axes = [], set()
    for name, path in files.items():
        config = _load(path)
        params = config["alg_params"]
        axes.add((params["target_entropy"], params["aux_return_detach_representation"]))
        assert config["checkpoint_every"] == 25_000 and config["save_strat"] == "all"
        assert config["total_steps"] == 2_000_000 and config["seed"] == 55
        normalized.append({**config, "alg_params": {key: value for key, value in params.items() if key not in {
            "target_entropy", "aux_return_detach_representation", "wandb_tags",
        }}})
    assert axes == {(target, detach) for target in (-10.5, -21.) for detach in (False, True)}
    assert all(config == normalized[0] for config in normalized)
    assert len(files) * 2_000_000 // 25_000 == 320


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
        assert calls[-1]["group"] == "ambi-aux-return-sac-20260915"
        assert calls[-1]["project"] == "ambi" and calls[-1]["mode"] == "online"
    assert len({call["name"] for call in calls}) == 4


def _receipt():
    binding = campaign.binding(SHA)
    cases = []
    for name in campaign.CASES:
        detached = name.endswith("_detached")
        cases.append({
            "config": name, "config_sha256": binding["config_sha256"][name], "source_commit": SHA,
            "passed": True, "compile_strict": True, "checkpoint_roundtrip": True,
            "exact_checkpoint_roundtrip": True, "next_update_reproducible": True,
            "gradient_routing": True, "no_return_actor": True, "no_inner_updates": True,
            "optimizer_updates": 3, "real_decisions": 32, "checkpoint_size_bytes": 42,
            "replay_seed": 55, "replay_sha256": "b" * 64,
            "production_overrides": {"wandb": False}, "production_warmup_and_pretraining_executed": False,
            "detach_representation": detached,
            "auxiliary_gradient_l1": {"encoder": 0. if detached else 1., "dynamics": 0. if detached else 2., "critic": 3.},
            "compile_status": [{key: False for key in campaign.FLAGS}],
        })
    return {
        "schema": 1, "passed": True, "binding": binding, "cases": cases,
        "fresh_training": {
            "passed": True, "source_commit": SHA, "config": campaign.CASES[0], "real_decisions": 512,
            "optimizer_updates": 13, "checkpoint_size_bytes": 42, "checkpoint_roundtrip": True,
            "checkpoint_sidecar": True, "compile_status": [{key: False for key in campaign.FLAGS}],
            "production_overrides": {"wandb": False, "seed_steps": 500, "pretrain_steps": 2,
                                     "buffer_size": 4096, "total_steps": 512, "checkpoint_every": 256},
        },
    }


def test_gpu_receipt_accepts_complete_paired_strict_compile_evidence():
    receipt = _receipt()
    assert campaign.validate_receipt(receipt, SHA) is receipt


@pytest.mark.parametrize("failure", (
    "source", "manifest", "config_binding", "case_config", "missing_cell", "duplicate_cell", "missing_flag",
    "aux_online_fallback", "aux_target_fallback", "nonboolean_flag", "exact_resume", "next_update", "inner_updates",
    "extra_actor", "attached_gradient", "detached_gradient", "fresh_missing", "fresh_budget", "fresh_fallback", "replay_mismatch",
))
def test_gpu_receipt_rejects_incomplete_or_incompatible_evidence(failure):
    receipt = _receipt()
    first = receipt["cases"][0]
    if failure == "source": receipt["binding"]["source_commit"] = "f" * 40
    elif failure == "manifest": receipt["binding"]["manifest_sha256"] = "f" * 64
    elif failure == "config_binding": receipt["binding"]["config_sha256"][campaign.CASES[3]] = "f" * 64
    elif failure == "case_config": first["config_sha256"] = "f" * 64
    elif failure == "missing_cell": receipt["cases"].pop()
    elif failure == "duplicate_cell": receipt["cases"][1] = deepcopy(first)
    elif failure == "missing_flag": first["compile_status"][0].pop("aux_target")
    elif failure == "aux_online_fallback": first["compile_status"][0]["aux_online"] = True
    elif failure == "aux_target_fallback": first["compile_status"][0]["aux_target"] = True
    elif failure == "nonboolean_flag": first["compile_status"][0]["aux_online"] = 0
    elif failure == "exact_resume": first["exact_checkpoint_roundtrip"] = False
    elif failure == "next_update": first["next_update_reproducible"] = False
    elif failure == "inner_updates": first["no_inner_updates"] = False
    elif failure == "extra_actor": first["no_return_actor"] = False
    elif failure == "attached_gradient": first["auxiliary_gradient_l1"]["dynamics"] = 0.
    elif failure == "detached_gradient": receipt["cases"][1]["auxiliary_gradient_l1"]["encoder"] = 1.
    elif failure == "fresh_missing": receipt.pop("fresh_training")
    elif failure == "fresh_budget": receipt["fresh_training"]["real_decisions"] = 32
    elif failure == "fresh_fallback": receipt["fresh_training"]["compile_status"][0]["aux_target"] = True
    elif failure == "replay_mismatch": receipt["cases"][1]["replay_sha256"] = "c" * 64
    with pytest.raises((AssertionError, KeyError)):
        campaign.validate_receipt(receipt, SHA)


def test_launcher_syntax_and_requeue_rejection(tmp_path):
    launcher = ROOT / "slurm/run_ambi_aux_return_sac_oscar.sbatch"
    subprocess.run(["bash", "-n", str(launcher)], check=True)
    output = tmp_path / "must-not-create"
    env = {**os.environ, "SLURM_JOB_ID": "123", "SLURM_RESTART_COUNT": "1",
           "AMBI_PROJECT_DIR": str(ROOT), "AMBI_PYTHON": sys.executable,
           "EXPECTED_ACTION_MODES_SHA": SHA, "AMBI_AUX_OUTPUT_ROOT": str(output), "AMBI_AUX_CAMPAIGN": "test"}
    result = subprocess.run(["bash", str(launcher)], env=env, capture_output=True, text=True)
    assert result.returncode == 2 and "cannot resume after requeue" in result.stderr
    assert not output.exists()


def test_optimized_python_cannot_bypass_receipt_validation():
    helper = ROOT / "slurm/ambi_aux_return_sac_campaign.py"
    optimized = compile(helper.read_text(), str(helper), "exec", optimize=1)
    with pytest.raises(RuntimeError, match="requires Python assertions"):
        exec(optimized, {"__name__": "optimized_campaign", "__file__": str(helper)})


def test_launch_metadata_records_full_resolution_and_job_to_wandb_mapping(tmp_path, monkeypatch):
    receipt = _receipt()
    current_runtime = {"python_executable": sys.executable, "python": sys.version,
                       "torch": "test", "cuda": "test", "gpu": "test", "compute_capability": [0, 0]}
    receipt["runtime"] = current_runtime
    receipt_path = tmp_path / "receipt.json"
    campaign.write_new(receipt_path, receipt)
    monkeypatch.setattr(campaign, "runtime", lambda: current_runtime)
    monkeypatch.setenv("WANDB_API_KEY", "fixture-placeholder-not-a-real-key")
    monkeypatch.setenv("WANDB_RUN_ID", "aux123x2")
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
    assert actual["resolved_config"]["target_entropy"] == -21.
    assert actual["resolved_config"]["steps"] == 2_000_000
    assert actual["resolved_config"]["compile_strict"] is True
    assert actual["wandb"]["url"] == "https://wandb.ai/rwgao_b-brown-university/ambi/runs/aux123x2"
    assert actual["wandb"]["name"] == f"AMBITDMPC2-{campaign.CASES[2]}-seed55"
    assert actual["slurm"]["SLURM_ARRAY_JOB_ID"] == "123"
    assert "fixture-placeholder" not in (tmp_path / "launch.json").read_text()
    with pytest.raises(FileExistsError):
        campaign.prepare_training(receipt_path, SHA, 2, tmp_path)
