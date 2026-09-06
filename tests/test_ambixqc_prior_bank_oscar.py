import importlib.util
from pathlib import Path
import subprocess

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / "tests/benchmarks/ambixqc_prior_checkpoint_bank.py"
LAUNCHER = ROOT / "slurm/run_ambixqc_prior_checkpoint_bank_oscar.sbatch"
spec = importlib.util.spec_from_file_location("ambixqc_prior_bank_helper", HELPER_PATH)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def test_production_input_copy_changes_only_operational_name():
    algorithm, manifest = helper.prepare_inputs("production", "test-run")
    expected = helper.read_json(ROOT / "configs/dmcontrol/algs" / f"{helper.STEM}.json")
    expected["alg_params"]["wandb_run_name"] = "test-run"
    assert algorithm == expected
    assert manifest == helper.read_json(ROOT / "configs/dmcontrol/experiments" / f"{helper.STEM}.json")


def test_smoke_preserves_full_architecture_warmup_and_inner_settings():
    production, manifest = helper.prepare_inputs("production", "test")
    smoke, smoke_manifest = helper.prepare_inputs("smoke", "test")
    production["total_steps"] = 3000
    production["alg_params"].update(wandb=False, wandb_mode="disabled")
    manifest["checkpoint_every"] = 1000
    assert smoke == production
    assert smoke_manifest == manifest
    assert smoke["alg_params"]["seed_steps"] == 2500
    assert smoke["alg_params"]["pretrain_steps"] == 2500
    assert smoke["alg_params"]["xqc_actor_net_arch"] == [256] * 4
    assert smoke["alg_params"]["xqc_critic_net_arch"] == [512] * 4


@pytest.mark.parametrize("key,value", [
    ("inner_operator", "xqc"), ("eval_freq", 1000), ("compile", True),
    ("seed_steps", 500), ("pretrain_steps", 1),
    ("xqc_actor_net_arch", [8]), ("wandb_project", "wrong-project"),
])
def test_source_protocol_guard_rejects_changed_scientific_inputs(monkeypatch, key, value):
    original = helper.read_json

    def changed(path):
        result = original(path)
        if "alg_params" in result:
            result["alg_params"][key] = value
        return result

    monkeypatch.setattr(helper, "read_json", changed)
    with pytest.raises(ValueError, match="scientific configuration changed"):
        helper.prepare_inputs("smoke", "test")


def _checkpoint(root, step):
    path = root / f"prior-{step}.pt"
    state = {
        "checkpoint_version": 2, "semantic_signature": {"collection_operator": "none"},
        "inner": {"action_index": 0}, "reward_normalizer": {"count": float(step)},
        "num_updates": max(0, step - 1), "module": {"weights": torch.ones(2)},
    }
    torch.save(state, path)
    helper.write_json(str(path) + ".metadata.json", {
        "checkpoint": {"step": step}, "trial_run_params": {
            "seed": 55, "total_steps": 3000, "alg_params": {"inner_operator": "none"},
        },
    })
    return path, state


def test_smoke_checkpoint_bank_requires_three_finite_metadata_pairs(tmp_path):
    for step in (1000, 2000, 3000):
        _checkpoint(tmp_path, step)
    records = helper.validate_checkpoint_bank(tmp_path, total_steps=3000, cadence=1000)
    assert [record["step"] for record in records] == [1000, 2000, 3000]
    assert all(len(record["sha256"]) == 64 and record["bytes"] > 0 for record in records)


def test_smoke_checkpoint_bank_rejects_missing_and_nonfinite_state(tmp_path):
    path, state = _checkpoint(tmp_path, 1000)
    with pytest.raises(ValueError, match="missing, duplicate, or unexpected"):
        helper.validate_checkpoint_bank(tmp_path, total_steps=3000, cadence=1000)
    _checkpoint(tmp_path, 2000)
    _checkpoint(tmp_path, 3000)
    state["module"]["weights"][0] = float("nan")
    torch.save(state, path)
    with pytest.raises(ValueError, match="Non-finite tensor"):
        helper.validate_checkpoint_bank(tmp_path, total_steps=3000, cadence=1000)


def test_full_size_smoke_refuses_unallocated_cpu_before_creating_outputs(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / "smoke"
    with pytest.raises(RuntimeError, match="one allocated CUDA GPU"):
        helper.run("smoke", output, "test")
    assert not output.exists()


def test_oscar_launcher_resources_runtime_storage_and_smoke_gate():
    contents = LAUNCHER.read_text()
    for required in (
        "#SBATCH --partition=gpu", "#SBATCH --gres=gpu:l40s:1",
        "#SBATCH --cpus-per-task=6", "#SBATCH --mem=48G",
        "#SBATCH --time=72:00:00", "#SBATCH --no-requeue",
        'require_lock "$PROJECT_DIR/environments/dmcontrol/uv.lock"',
        'require_lock "$ENV_PROJECT/uv.lock"',
        "require_clean_sha", "--untracked-files=all", "required_gib=32",
        "refusing to overwrite job artifacts", "results must be outside the checkout",
        "WANDB_MODE=disabled", "WANDB_MODE=online", "WANDB_RESUME=never",
        "WANDB_CACHE_DIR", "tests/test_ambixqc_prior_checkpoint.py",
        "tests/test_ambixqc_checkpoint_evaluation.py", '"$JOB_ROOT/pytest.log"',
        '"$JOB_ROOT/run/validation.json"',
    ):
        assert required in contents
    assert "uv sync" not in contents and "pip install" not in contents
    assert "sbatch " not in contents.replace("override sbatch --time=02:00:00", "")
    subprocess.run(["/bin/bash", "-n", str(LAUNCHER)], check=True, close_fds=False)
