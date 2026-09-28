"""Six-cell Hydra protocol, independent replay capacity, and CUDA archive gate."""

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import gymnasium as gym
import pytest
import torch

from RL.AMBIXQC import AMBIXQC
from utils.checkpointing import resolve_checkpoint_config

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "tests/benchmarks/ambixqc_backbone_replay_500k_campaign.py"
LAUNCHER = ROOT / "slurm/run_ambixqc_backbone_replay_500k_hydra.sbatch"
spec = importlib.util.spec_from_file_location("backbone_replay_500k_campaign", HELPER)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


@pytest.mark.parametrize("cell", helper.CELLS)
def test_six_cells_preserve_established_backbones_except_budget_ratio_capacity_identity(cell):
    arm, ratio = helper.split_cell(cell)
    production, manifest = helper.prepare_inputs("production", cell, "test")
    previous = helper.read_json(ROOT / f"configs/dmcontrol/algs/ambixqc_humanoid_walk_backbone_replay_1m_{arm}.json")
    comparable = deepcopy(production)
    comparable["total_steps"] = previous["total_steps"]
    for obj in (previous, comparable):
        for key in ("wandb_group", "wandb_run_name", "wandb_tags", "xqc_utd", "replay_capacity"):
            obj["alg_params"].pop(key, None)
    assert comparable == previous
    assert production["total_steps"] == 500_000
    assert production["alg_params"]["xqc_utd"] == ratio
    assert production["alg_params"]["utd"] == 1
    assert production["alg_params"]["buffer_size"] == production["alg_params"]["replay_capacity"] == 1_000_000
    policy = resolve_checkpoint_config(production, manifest)
    assert policy.save_replay_buffer and policy.every == 25_000 and policy.strategies == ("all",)
    assert production["total_steps"] // policy.every == 20
    smoke, smoke_manifest = helper.prepare_inputs("smoke", cell, "test")
    production["total_steps"] = helper.SMOKE_STEPS
    production["alg_params"].update(wandb=False, wandb_mode="disabled")
    manifest["checkpoint_every"] = 1000
    assert smoke == production and smoke_manifest == manifest


@pytest.mark.parametrize("cell", helper.CELLS)
def test_resolved_schedule_and_uncapped_replay(cell):
    config, _ = helper.prepare_inputs("production", cell, "test")
    wrapper = object.__new__(AMBIXQC)
    wrapper.env = gym.make("Pendulum-v1")
    wrapper.run_params = {**config, "device": "cpu"}
    wrapper.experiment_params = {}
    wrapper.custom_params = config["alg_params"]
    try:
        cfg = wrapper._build_cfg({**config["alg_params"], "device": "cpu"})
    finally:
        wrapper.env.close()
    ratio = helper.split_cell(cell)[1]
    assert cfg.steps == 500_000 and cfg.xqc_lr_transition_steps == 500_000 * ratio
    assert cfg.replay_capacity == 1_000_000
    assert cfg.inner_model_step_budget == cfg.inner_expected_update_slots == 0


@pytest.mark.parametrize("key,value", [
    ("xqc_utd", 1), ("replay_capacity", 500_000), ("buffer_size", 100),
    ("utd", 2), ("inner_operator", "xqc"), ("aux_return_detach_representation", True),
    ("compile", True), ("xqc_critic_net_arch", [8]),
])
def test_rejects_changed_scientific_protocol(monkeypatch, key, value):
    read = helper.read_json
    def changed(path):
        data = read(path)
        if "alg_params" in data:
            data["alg_params"][key] = value
        return data
    monkeypatch.setattr(helper, "read_json", changed)
    with pytest.raises(ValueError, match="scientific configuration changed"):
        helper.prepare_inputs("smoke", "aux_shared_utd2", "test")


def _smoke_files(tmp_path):
    for index, cell in enumerate(helper.CELLS):
        ratio = helper.split_cell(cell)[1]
        job = tmp_path / f"smoke-{cell}-seed55-job12-task{index}"
        directory = job / f"run-{cell}"
        directory.mkdir(parents=True)
        (job / "PASS").write_text("PASS\n")
        helper.write_json(directory / "validation.json", {
            "schema": "ambixqc-backbone-replay-500k-validation-v1", "mode": "smoke",
            "arm": cell, "total_steps": 4000, "source_sha": "a" * 40,
            "xqc_utd": ratio, "xqc_updates": 3999 * ratio,
            "replay_capacity": 1_000_000, "lr_transition_steps": 4000 * ratio,
            "all_finite": True, "final_raw_replay_matches_training": True,
            "optimizer_fused": {"critic": True}, "checkpoints": [{}] * 4,
            "storage": helper.storage_estimate([{"bytes": 100_000_000}]),
            "timing": helper.timing_record(0., {3000: 100., 3900: 118. + index * 9}),
            "protocol_sha256": helper.protocol_digest(cell),
        })


def test_parallel_smoke_gate_binds_all_six_cells_commit_config_and_artifact(tmp_path, monkeypatch):
    _smoke_files(tmp_path)
    gate = helper.summarize_smoke(tmp_path)
    path = tmp_path / "smoke-gate.json"
    assert gate["passed"] is True and len(gate["arms"]) == 6
    assert gate["required_campaign_free_bytes"] == sum(entry["required_free_bytes"] for entry in gate["arms"].values())
    helper.validate_smoke_gate(path, "baseline_utd2", "a" * 40)
    with pytest.raises(ValueError, match="exact commit"):
        helper.validate_smoke_gate(path, "baseline_utd2", "b" * 40)
    with monkeypatch.context() as scoped:
        scoped.setattr(helper, "protocol_digest", lambda cell: "changed")
        with pytest.raises(ValueError, match="configuration"):
            helper.validate_smoke_gate(path, "baseline_utd2", "a" * 40)
    validation = Path(gate["arms"]["aux_shared_utd2"]["validation"])
    validation.write_text(validation.read_text() + "\n")
    with pytest.raises(ValueError, match="artifact has changed"):
        helper.validate_smoke_gate(path, "baseline_utd2", "a" * 40)


@pytest.mark.parametrize("problem", ["missing", "duplicate", "not_passed", "wrong_ratio"])
def test_smoke_summary_rejects_incomplete_or_incompatible_cells(tmp_path, problem):
    _smoke_files(tmp_path)
    path = next(tmp_path.glob("smoke-baseline_utd2-*/run-*/validation.json"))
    if problem == "missing":
        path.unlink()
    elif problem == "duplicate":
        duplicate = tmp_path / "smoke-baseline_utd2-seed55-job99-task3/run-baseline_utd2/validation.json"
        duplicate.parent.mkdir(parents=True)
        duplicate.write_text(path.read_text())
    elif problem == "not_passed":
        (path.parent.parent / "PASS").unlink()
    else:
        value = json.loads(path.read_text())
        value["xqc_updates"] = 3999
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        helper.summarize_smoke(tmp_path)
    assert not (tmp_path / "smoke-gate.json").exists()


def test_storage_budget_uses_largest_real_checkpoint_and_one_replay_archive():
    estimate = helper.storage_estimate([{"bytes": 50_000_000}, {"bytes": 100_000_000}])
    expected = 20 * 100_000_000 + 1000 * 501 * 384
    assert estimate["estimated_production_bytes"] == expected
    assert estimate["required_free_bytes"] >= expected * 1.5 + 2 * 1024 ** 3


def test_steady_timing_uses_500k_and_excludes_checkpoint_boundaries():
    result = helper.timing_record(10., {3000: 110., 3900: 128.})
    assert result["steady_decisions"] == 900
    assert result["estimated_500k_hours"] == pytest.approx((100 + 497000 * .02) / 3600)


def test_hydra_launcher_pins_node_guards_source_and_uses_parallel_cell_ownership():
    text = LAUNCHER.read_text()
    for required in (
        "#SBATCH --partition=gpus", "#SBATCH --nodelist=gpu2501", "#SBATCH --gres=gpu:1",
        "#SBATCH --cpus-per-task=6", "#SBATCH --mem=48G", "#SBATCH --time=24:00:00",
        '"$(hostname -s)" == gpu2501', "#SBATCH --no-requeue", "require_clean_sha",
        'require_lock "$PROJECT_DIR/environments/dmcontrol/uv.lock"',
        'require_lock "$ENV_PROJECT/uv.lock"', "--untracked-files=all", "AMBIXQC_SMOKE_GATE",
        "AMBIXQC_MIN_FREE_GIB:-8", '"$JOB_ROOT/run-$ARM"', "TASK_SUFFIX",
        "WANDB_CONFIG_DIR", "WANDB_CACHE_DIR", "WANDB_DATA_DIR", "WANDB_ARTIFACT_DIR",
        "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "MPLCONFIGDIR", "TMPDIR",
        "CUBLAS_WORKSPACE_CONFIG=:4096:8 AMBI_XQC_REQUIRE_CUDA=1", "WANDB_RESUME=never",
        "tests/test_ambixqc_update_ratio.py", "tests/test_tdmpc2_replay_capacity.py",
        '"$JOB_ROOT/PASS"', '"$SLURM_ARRAY_TASK_ID" == 0', "--array=0-5%6",
    ):
        assert required in text
    assert "uv sync" not in text and "pip install" not in text
    assert "export CUBLAS_WORKSPACE_CONFIG" not in text
    subprocess.run(["/bin/bash", "-n", str(LAUNCHER)], check=True, close_fds=False)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("ratio", [1, 2])
def test_tiny_backbone_archives_counters_and_detached_primary_equivalence(tmp_path, device, ratio):
    if device == "cuda" and not torch.cuda.is_available():
        if os.environ.get("AMBI_XQC_REQUIRE_CUDA") == "1":
            pytest.fail("The Hydra gate requires real CUDA hardware.")
        pytest.skip("CUDA hardware is unavailable")
    from run_ambixqc_inner_evaluation import deterministic_evaluation
    from test_ambixqc_aux_training import _primary_state
    from test_ambixqc_prior_checkpoint import _wrapper
    from test_ambixqc_replay_archive import _assert_equal, _resident_rows
    from utils.replay_archive import load_checkpoint_replay

    outcomes = {}
    with deterministic_evaluation(device=device):
        for arm in ("baseline", "aux_shared", "aux_detached"):
            model = _wrapper(
                device=device, inner_operator="none", xqc_optimizer_backend="auto",
                xqc_utd=ratio, replay_capacity=64,
                aux_return_mode="off" if arm == "baseline" else "xqc",
                aux_return_detach_representation=arm != "aux_shared",
            )
            model.env.reset(seed=3)
            model.env.action_space.seed(3)
            directory = tmp_path / arm
            model.set_checkpointing(5, directory, "tiny", save_strat=("all",))
            model.enable_replay_archive(directory, "tiny")
            try:
                model.learn(total_timesteps=10)
                checkpoint = model.save(directory, "final")
                assert helper.validate_live_replay(model.buffer, checkpoint)
                assert model.buffer.capacity == load_checkpoint_replay(checkpoint).capacity == 64
                state = deepcopy(model.agent.checkpoint_state())
                helper.prior_bank.assert_finite(state)
                assert state["xqc_workspace"]["update_step"] == state["num_updates"] * ratio > 0
                assert state["semantic_signature"]["xqc_utd"] == ratio
                if arm != "baseline":
                    assert model.agent.aux_return.update_step == model.agent.num_updates * ratio
                outcomes[arm] = {
                    "primary": _primary_state(model.agent), "replay": _resident_rows(model.buffer),
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
