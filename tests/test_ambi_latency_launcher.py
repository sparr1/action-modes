import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/research/ambi_latency_benchmark.json"
LAUNCHER = ROOT / "slurm/run_ambi_latency_oscar.sbatch"


def _load_config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _expected_counters(J, N, G, *, H=3, B=64):
    transitions = J * N * H
    slots = J * G
    replay_draws = slots * B
    return {
        "inner_rollouts": J * N,
        "inner_model_steps": transitions,
        "inner_total_model_steps": transitions,
        "inner_buffer_capacity": transitions,
        "inner_update_slots": slots,
        "inner_critic_optimizer_steps": slots,
        "inner_actor_optimizer_steps": slots,
        "inner_temperature_optimizer_steps": 0,
        "inner_critic_target_updates": slots,
        "inner_replay_draws": replay_draws,
        "inner_policy_evaluations": transitions + 2 * replay_draws + 1,
        "inner_q_evaluations": 3 * replay_draws,
    }


def test_latency_matrix_is_the_deduplicated_humanoid_g4_design():
    config = _load_config()
    settings = config["settings"]
    cells = config["cells"]
    triples = {(cell["J"], cell["N"], cell["G"]) for cell in cells}

    assert config["schema_version"] == 1
    assert config["benchmark"] == "ambi-inner-latency"
    assert config["base"]["algorithm_config"] == (
        "../dmcontrol/algs/ambi_humanoid_walk_updates_g4.json"
    )
    assert config["base"]["environment"] == {
        "id": "DMControl-v0",
        "params": {
            "task": "humanoid-walk",
            "obs": "state",
            "render_mode": None,
        },
    }
    assert (settings["H"], settings["B"]) == (3, 64)
    assert (
        settings["cold_calls"],
        settings["warmup_calls"],
        settings["measured_calls"],
        settings["observation_bank_size"],
        settings["blocks"],
    ) == (1, 49, 200, 64, 3)
    assert settings["device"] == "cuda"
    assert settings["action_mode"] == "training"
    assert settings["environment_seed"] == 55
    assert settings["controller_seed"] == 55
    assert settings["block_order_seed"] == 20260817
    assert settings["wandb"] is False
    assert settings["collect_diagnostics"] is False

    assert len(cells) == len(triples) == 15
    assert len({cell["name"] for cell in cells}) == 15
    assert all("block" not in cell for cell in cells)

    expected_families = {
        "G": {(2, 32, G) for G in (0, 2, 4, 8, 16)},
        "N": {(2, N, 4) for N in (8, 16, 32, 64, 128)},
        "natural_J": {(J, 32, 4) for J in (1, 2, 4, 8)},
        "matched_work_J": {
            (1, 64, 8),
            (2, 32, 4),
            (4, 16, 2),
            (8, 8, 1),
        },
    }
    for family, expected in expected_families.items():
        actual = {
            (cell["J"], cell["N"], cell["G"])
            for cell in cells
            if family in cell["families"]
        }
        assert actual == expected

    for cell in cells:
        assert cell["expected_counters"] == _expected_counters(
            cell["J"], cell["N"], cell["G"]
        )


def test_matched_work_cells_hold_transitions_and_update_slots_fixed():
    cells = _load_config()["cells"]
    matched = [cell for cell in cells if "matched_work_J" in cell["families"]]

    assert len(matched) == 4
    assert {cell["J"] * cell["N"] * 3 for cell in matched} == {192}
    assert {cell["J"] * cell["G"] for cell in matched} == {8}


def test_oscar_launcher_is_a_bounded_process_isolated_array():
    contents = LAUNCHER.read_text(encoding="utf-8")

    for directive in (
        "#SBATCH --partition=gpu",
        "#SBATCH --gres=gpu:l40s:1",
        "#SBATCH --cpus-per-task=6",
        "#SBATCH --mem=32G",
        "#SBATCH --time=03:00:00",
        "#SBATCH --array=0-2%2",
        "#SBATCH --output=logs/%x-%A_%a.out",
        "#SBATCH --error=logs/%x-%A_%a.err",
    ):
        assert directive in contents

    assert "AMBI_BENCHMARK_PYTHON" in contents
    assert "environments/dmcontrol/.venv/bin/python" in contents
    assert "AMBI_LATENCY_CHECKPOINT" in contents
    assert "AMBI_LATENCY_OUTPUT_ROOT" in contents
    assert "AMBI_LATENCY_EXPECTED_COMMIT" in contents
    assert "AMBI_LATENCY_EXPECTED_COMMIT must be a full lowercase 40-character commit SHA" in contents
    assert "does not match expected" in contents
    assert "AMBI_LATENCY_OUTPUT_ROOT must be outside the source checkout" in contents
    assert "MUJOCO_GL=egl" in contents
    assert "WANDB_MODE=disabled" in contents
    assert "--list-cells" in contents
    assert "len(raw_rows) != 15" in contents
    assert "random.Random(order_seed + block).shuffle(rows)" in contents
    assert 'for row in "${CELLS[@]}"' in contents
    assert '--cell "$J,$N,$G"' in contents
    assert "TORCHINDUCTOR_CACHE_DIR" in contents
    assert "TRITON_CACHE_DIR" in contents
    assert "CUDA_CACHE_PATH" in contents
    assert "benchmark requires a clean dedicated checkout" in contents

    for forbidden in (
        "main.py",
        "--resume-mode",
        "--lineage-dir",
        "conda activate",
        "WANDB_MODE=online",
        "#SBATCH --requeue",
    ):
        assert forbidden not in contents
