"""Three-arm source protocol, smoke identity, launcher, and explicit CUDA canary."""

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "tests/benchmarks/ambixqc_backbone_replay_campaign.py"
LAUNCHER = ROOT / "slurm/run_ambixqc_backbone_replay_oscar.sbatch"
spec = importlib.util.spec_from_file_location("backbone_replay_campaign", HELPER)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


@pytest.mark.parametrize("arm", helper.ARMS)
def test_production_inputs_preserve_versioned_recipe_and_smoke_changes_only_length_publication(arm):
    name = f"{helper.STEM}_{arm}"
    original = helper.read_json(ROOT / "configs/dmcontrol/algs" / f"{name}.json")
    manifest = helper.read_json(ROOT / "configs/dmcontrol/experiments" / f"{name}.json")
    original["alg_params"]["wandb_run_name"] = "test"
    production, prod_manifest = helper.prepare_inputs("production", arm, "test")
    assert production == original and prod_manifest == manifest
    smoke, smoke_manifest = helper.prepare_inputs("smoke", arm, "test")
    original["total_steps"] = 4000
    original["alg_params"].update(wandb=False, wandb_mode="disabled")
    manifest["checkpoint_every"] = 1000
    assert smoke == original and smoke_manifest == manifest
    assert smoke_manifest["save_replay_buffer"] is True
    assert smoke["alg_params"]["seed_steps"] == smoke["alg_params"]["pretrain_steps"] == 2500


@pytest.mark.parametrize("key,value", [
    ("aux_return_mode", "off"), ("aux_return_detach_representation", True),
    ("inner_operator", "xqc"), ("compile", True), ("seed_steps", 100),
    ("aux_return_critic_coef", 1.), ("xqc_critic_net_arch", [8]),
])
def test_runner_rejects_changed_scientific_protocol(monkeypatch, key, value):
    read = helper.read_json

    def changed(path):
        data = read(path)
        if "alg_params" in data:
            data["alg_params"][key] = value
        return data

    monkeypatch.setattr(helper, "read_json", changed)
    with pytest.raises(ValueError, match="scientific configuration changed"):
        helper.prepare_inputs("smoke", "aux_shared", "test")


def test_steady_timing_excludes_pretraining_and_checkpoints():
    result = helper.timing_record(10., {3000: 110., 3900: 128.})
    assert result["warmup_and_startup_seconds"] == 100.
    assert result["steady_decisions"] == 900
    assert result["steady_seconds_per_decision"] == pytest.approx(.02)
    assert result["estimated_1m_hours"] == pytest.approx((100 + 997000 * .02) / 3600)
    with pytest.raises(ValueError, match="steady window"):
        helper.timing_record(10., {3000: 110.})


def _gate(tmp_path):
    for index, arm in enumerate(helper.ARMS):
        directory = tmp_path / f"run-{arm}"
        directory.mkdir()
        helper.write_json(directory / "validation.json", {
            "schema": "ambixqc-backbone-replay-validation-v1", "mode": "smoke",
            "arm": arm, "total_steps": 4000, "source_sha": "a" * 40,
            "all_finite": True, "final_raw_replay_matches_training": True,
            "optimizer_fused": {"critic": True}, "checkpoints": [{}] * 4,
            "timing": helper.timing_record(0., {3000: 100., 3900: 118. + index * 9}),
            "protocol_sha256": helper.protocol_digest(arm),
        })
    return helper.summarize_smoke(tmp_path)


def test_smoke_gate_requires_all_arms_exact_commit_config_and_immutable_validation(tmp_path, monkeypatch):
    gate = _gate(tmp_path)
    path = tmp_path / "smoke-gate.json"
    assert gate["passed"] is True
    assert gate["arms"]["aux_detached"]["steady_ratio_to_baseline"] == pytest.approx(2.)
    helper.validate_smoke_gate(path, "aux_shared", "a" * 40)
    with pytest.raises(ValueError, match="exact commit"):
        helper.validate_smoke_gate(path, "aux_shared", "b" * 40)
    with monkeypatch.context() as scoped:
        scoped.setattr(helper, "protocol_digest", lambda arm: "changed")
        with pytest.raises(ValueError, match="configuration"):
            helper.validate_smoke_gate(path, "aux_shared", "a" * 40)
    validation = tmp_path / "run-baseline/validation.json"
    validation.write_text(validation.read_text() + "\n")
    with pytest.raises(ValueError, match="artifact has changed"):
        helper.validate_smoke_gate(path, "aux_shared", "a" * 40)


def test_smoke_summary_refuses_failed_validation(tmp_path):
    _gate(tmp_path)
    path = tmp_path / "run-aux_shared/validation.json"
    state = json.loads(path.read_text())
    state["final_raw_replay_matches_training"] = False
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="Incomplete smoke"):
        helper.summarize_smoke(tmp_path)


def test_full_size_runner_refuses_cpu_before_creating_outputs(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="one allocated CUDA GPU"):
        helper.run("smoke", "baseline", tmp_path / "output", "test")
    assert not (tmp_path / "output").exists()


def test_oscar_launcher_is_guarded_uses_node_local_caches_and_separate_arm_outputs():
    text = LAUNCHER.read_text()
    for required in (
        "#SBATCH --gres=gpu:l40s:1", "#SBATCH --cpus-per-task=6", "#SBATCH --mem=48G",
        "#SBATCH --time=18:00:00", "#SBATCH --no-requeue", "require_clean_sha",
        'require_lock "$PROJECT_DIR/environments/dmcontrol/uv.lock"',
        'require_lock "$ENV_PROJECT/uv.lock"', "--untracked-files=all", "AMBIXQC_SMOKE_GATE",
        '"$JOB_ROOT/run-$selected"', '"$JOB_ROOT/run-$ARM"', "TASK_SUFFIX",
        "WANDB_CONFIG_DIR", "WANDB_CACHE_DIR", "WANDB_DATA_DIR", "WANDB_ARTIFACT_DIR",
        "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "MPLCONFIGDIR", "TMPDIR",
        "CUBLAS_WORKSPACE_CONFIG=:4096:8 AMBI_XQC_REQUIRE_CUDA=1", "WANDB_RESUME=never",
        "tests/test_ambixqc_aux_training.py", "tests/test_ambixqc_replay_archive.py",
        "--mode summarize", '"$JOB_ROOT/smoke-gate.json"', '"$JOB_ROOT/PASS"',
    ):
        assert required in text
    assert "uv sync" not in text and "pip install" not in text
    assert "export CUBLAS_WORKSPACE_CONFIG" not in text
    subprocess.run(["/bin/bash", "-n", str(LAUNCHER)], check=True, close_fds=False)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_explicit_cuda_tiny_backbone_auxiliary_replay_and_detached_primary_equivalence(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        if os.environ.get("AMBI_XQC_REQUIRE_CUDA") == "1":
            pytest.fail("The Oscar gate requires real CUDA hardware.")
        pytest.skip("CUDA hardware is unavailable")
    from run_ambixqc_inner_evaluation import deterministic_evaluation
    from test_ambixqc_aux_training import _primary_state
    from test_ambixqc_prior_checkpoint import _wrapper
    from test_ambixqc_replay_archive import _assert_equal, _resident_rows

    outcomes = {}
    with deterministic_evaluation(device=device):
        for arm in helper.ARMS:
            model = _wrapper(
                device=device, inner_operator="none", xqc_optimizer_backend="auto",
                aux_return_mode="off" if arm == "baseline" else "xqc",
                aux_return_detach_representation=arm != "aux_shared",
            )
            # Initialize Gym's otherwise lazily OS-seeded streams before
            # comparing complete seeded control runs, as in the CPU canary.
            model.env.reset(seed=3)
            model.env.action_space.seed(3)
            directory = tmp_path / arm
            model.set_checkpointing(5, directory, "tiny", save_strat=("all",))
            model.enable_replay_archive(directory, "tiny")
            try:
                assert next(model.agent.parameters()).device.type == device
                model.learn(total_timesteps=10)
                checkpoint = model.save(directory, "final")
                assert helper.validate_live_replay(model.buffer, checkpoint)
                prior_bank_state = deepcopy(model.agent.checkpoint_state())
                helper.prior_bank.assert_finite(prior_bank_state)
                if arm != "baseline":
                    assert model.agent.aux_return.update_step == model.agent.num_updates > 0
                outcomes[arm] = {
                    "primary": _primary_state(model.agent),
                    "replay": _resident_rows(model.buffer),
                    "device_rng": (torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state()).clone(),
                }
            finally:
                model.flush_checkpoints()
                model.close_replay_archive()
                model.env.close()
        _assert_equal(outcomes["baseline"], outcomes["aux_detached"])
        base = outcomes["baseline"]["primary"]["module"]
        shared = outcomes["aux_shared"]["primary"]["module"]
        assert any(not torch.equal(value, shared[key]) for key, value in base.items() if key.startswith("model."))
