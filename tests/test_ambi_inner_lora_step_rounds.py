"""LoRA step-round evaluations preserve the dense protocol and cell ownership."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest

from tests.test_ambi_inner_benchmark_launcher import launch_env
from tests.test_ambi_inner_interval_sweep import COMPETING_KEYS
from tests.test_checkpoint_research_configs import checkpoint_context, _build_cfg
from utils.ambi_benchmark import protocol_for
from utils.ambi_research import load_preset_matrix, normalize_selectors, resolve_preset


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "configs/research/ambi_humanoid_inner_step_rounds.json"
MATRIX = ROOT / "configs/research/ambi_humanoid_inner_lora_step_rounds.json"
LAUNCHER = ROOT / "slurm/run_ambi_inner_lora_step_rounds_oscar.sbatch"
LORA_SETTINGS = {
    "inner_critic_adaptation": "lora_rl",
    "inner_critic_lora_layers": "input_hidden",
    "inner_critic_lora_rank": 96,
    "inner_critic_lora_scale": 1.0,
    "inner_critic_lora_weight_decay": 0.0002,
}


@pytest.mark.parametrize("rounds", [1, 3, 6])
def test_only_five_critic_lora_settings_change_from_dense_step_rounds(checkpoint_context, rounds):
    matrix = load_preset_matrix(MATRIX)
    baseline = load_preset_matrix(BASELINE)
    assert normalize_selectors(matrix) == [f"lora_step_rounds/j{j}" for j in (1, 3, 6)]
    assert matrix["source_run"] == baseline["source_run"] == "rwgao_b-brown-university/ambi/u13m14st"
    assert matrix["base_alg_config"] == baseline["base_alg_config"] == "checkpoint"
    assert matrix["shared_alg_params"] == {**baseline["shared_alg_params"], **LORA_SETTINGS}
    assert {k: v for k, v in matrix["evaluation"].items() if k != "default_presets"} == {
        k: v for k, v in baseline["evaluation"].items() if k != "default_presets"
    }
    assert matrix["evaluation"]["seeds"] == [101, 102, 103, 104, 105]
    assert matrix["evaluation"]["controller_seed"] == 55
    assert matrix["evaluation"]["max_steps"] == 500
    assert matrix["comparisons"]["lora_step_rounds"]["variants"][f"j{rounds}"]["alg_params"] == {
        "inner_rounds": rounds,
    }

    checkpoint_context.trial_run_params["alg_params"].update({k: 99 for k in COMPETING_KEYS})
    before = deepcopy(checkpoint_context)
    selected = resolve_preset(MATRIX, f"lora_step_rounds/j{rounds}", checkpoint_context=checkpoint_context)
    original = resolve_preset(BASELINE, f"step_rounds/j{rounds}", checkpoint_context=checkpoint_context)
    expected = deepcopy(original["algorithm_config"])
    expected["alg_params"].update(LORA_SETTINGS)
    assert selected["algorithm_config"] == expected
    assert not COMPETING_KEYS.intersection(selected["algorithm_config"]["alg_params"])
    assert selected["environment"] == original["environment"]
    assert protocol_for(selected, 55, 500) == protocol_for(original, 55, 500)
    assert protocol_for(selected, 55, 500)["action_rule"] == "tanh_mean"
    cfg = _build_cfg(selected["algorithm_config"])
    assert cfg.inner_actor_adaptation == "clone"
    for key, value in LORA_SETTINGS.items():
        assert getattr(cfg, key) == value
    assert cfg.inner_update_timing == "step"
    assert cfg.inner_steps_per_update == cfg.inner_rollouts_per_round == cfg.inner_batch_size == 512
    assert cfg.inner_rollout_horizon == 3 and cfg.inner_rounds == rounds
    assert cfg.inner_model_step_budget == 1536 * rounds
    assert cfg.inner_expected_update_slots == 3 * rounds
    assert (cfg.inner_critic_updates_per_action, cfg.inner_actor_updates_per_action,
            cfg.inner_temperature_updates_per_action) == (3 * rounds,) * 3
    assert cfg.inner_replay_capacity == 9216 and cfg.inner_replay_sampling == "with_replacement"
    assert (cfg.inner_actor_lr, cfg.inner_critic_lr, cfg.inner_temperature_lr) == (5e-5, 1e-4, 3e-4)
    assert cfg.inner_critic_target_tau == 0.01 and cfg.inner_critic_target_update_interval == 1
    assert cfg.inner_bootstrap_source == "inner_target" and cfg.inner_finite_horizon is False
    assert cfg.inner_outer_replay_fraction == cfg.inner_actor_writeback_coef == cfg.inner_critic_writeback_coef == 0
    for component in ("actor", "critic", "temperature", "replay", "actor_optimizer",
                      "critic_optimizer", "temperature_optimizer"):
        assert getattr(cfg, f"inner_{component}_scope") == "action"
    assert checkpoint_context == before


def _launch_env(launch_env):
    env = dict(launch_env)
    env.pop("AMBI_ROUND_VALUES", None)
    for step in range(50000, 500001, 50000):
        Path(env["AMBI_CHECKPOINT_PREFIX"] + str(step)).touch()
        Path(env["AMBI_CHECKPOINT_PREFIX"] + str(step) + ".metadata.json").touch()
        prior = Path(env["AMBI_BENCHMARK_REFERENCE_ROOT"]) / f"step_{step}/prior"
        prior.mkdir(parents=True, exist_ok=True)
        (prior / "manifest.json").write_text("{}")
    return env


def _flag(call, name):
    return call[call.index(name) + 1]


def _calls(env):
    path = Path(env["TEST_CALLS"])
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_launcher_owns_thirty_complete_checkpoint_round_cells(launch_env):
    env = _launch_env(launch_env)
    cells, outputs = set(), set()
    for index in range(30):
        env["SLURM_ARRAY_TASK_ID"] = str(index)
        subprocess.run(["bash", str(LAUNCHER)], env=env, check=True, capture_output=True, text=True)
        calls = _calls(env)
        assert len(calls) == 2 * (index + 1)
        evaluate, report = calls[-2:]
        step, rounds = (index % 10 + 1) * 50000, (6, 3, 1)[index // 10]
        cells.add((step, rounds))
        assert evaluate[0] == "evaluate_ambi_checkpoint.py"
        assert _flag(evaluate, "--checkpoint") == env["AMBI_CHECKPOINT_PREFIX"] + str(step)
        assert _flag(evaluate, "--matrix") == str(MATRIX.relative_to(ROOT))
        assert [evaluate[i + 1] for i, arg in enumerate(evaluate) if arg == "--preset"] == [
            f"lora_step_rounds/j{rounds}"]
        assert evaluate[evaluate.index("--seeds") + 1:evaluate.index("--max-steps")] == [
            "101", "102", "103", "104", "105"]
        assert _flag(evaluate, "--max-steps") == "500"
        assert _flag(evaluate, "--controller-seed") == "55"
        assert _flag(evaluate, "--eval-run-map") == env["EVAL_RUN_MAP"]
        assert _flag(evaluate, "--checkpoint-inventory") == env["CHECKPOINT_MANIFEST"]
        prior = str(Path(env["AMBI_BENCHMARK_REFERENCE_ROOT"]) / f"step_{step}/prior")
        output = str(Path(env["AMBI_BENCHMARK_OUTPUT_ROOT"]) / f"step_{step}/j{rounds}/inner")
        assert _flag(evaluate, "--reference-bundle") == prior
        assert _flag(evaluate, "--bundle-dir") == output
        assert not {"--wandb", "--save-root-bank", "--bank-only"} & set(evaluate)
        assert report[0] == "report_ambi_benchmark.py"
        assert [report[i + 1] for i, arg in enumerate(report) if arg == "--bundle"] == [prior, output]
        outputs.add(output)
    assert cells == {(step, rounds) for step in range(50000, 500001, 50000) for rounds in (1, 3, 6)}
    assert len(outputs) == 30
    repeated = subprocess.run(["bash", str(LAUNCHER)], env=env, capture_output=True, text=True)
    assert repeated.returncode != 0 and "output already exists" in repeated.stderr
    assert len(_calls(env)) == 60


def test_round_subset_runs_only_j6_and_rejects_unmapped_indices(launch_env):
    env = _launch_env(launch_env)
    env["AMBI_ROUND_VALUES"] = "6"
    for index in range(10):
        env["SLURM_ARRAY_TASK_ID"] = str(index)
        subprocess.run(["bash", str(LAUNCHER)], env=env, check=True, capture_output=True, text=True)
        evaluate = _calls(env)[-2]
        assert _flag(evaluate, "--preset") == "lora_step_rounds/j6"
        assert _flag(evaluate, "--checkpoint") == env["AMBI_CHECKPOINT_PREFIX"] + str((index + 1) * 50000)
    env["SLURM_ARRAY_TASK_ID"] = "10"
    result = subprocess.run(["bash", str(LAUNCHER)], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and "array index exceeds" in result.stderr
    assert len(_calls(env)) == 20


@pytest.mark.parametrize("index,rounds", [(5, 6), (15, 3), (25, 1)])
def test_smoke_makes_only_private_three_decision_prior_without_publication(launch_env, index, rounds):
    env = _launch_env(launch_env)
    env.pop("EVAL_RUN_MAP")
    env.pop("AMBI_BENCHMARK_REFERENCE_ROOT")
    env["SLURM_ARRAY_TASK_ID"] = str(index)
    subprocess.run(["bash", str(LAUNCHER), "--smoke"], env=env, check=True, capture_output=True, text=True)
    prior, inner, report = _calls(env)
    assert _flag(prior, "--preset") == "named_run/prior"
    assert _flag(inner, "--preset") == f"lora_step_rounds/j{rounds}"
    for call in (prior, inner):
        assert _flag(call, "--checkpoint") == env["AMBI_CHECKPOINT_PREFIX"] + "300000"
        assert call[call.index("--seeds") + 1:call.index("--max-steps")] == ["101", "102"]
        assert _flag(call, "--max-steps") == "3"
        assert "--eval-run-map" not in call and "--wandb" not in call
    private_prior = str(Path(env["AMBI_BENCHMARK_OUTPUT_ROOT"]) / f"step_300000/j{rounds}/prior")
    assert _flag(prior, "--bundle-dir") == _flag(inner, "--reference-bundle") == private_prior
    assert report[0] == "report_ambi_benchmark.py"


@pytest.mark.parametrize("index", [0, 5, 9])
def test_production_never_regenerates_a_missing_prior_even_at_50k(launch_env, index):
    env = _launch_env(launch_env)
    env["SLURM_ARRAY_TASK_ID"] = str(index)
    step = (index + 1) * 50000
    (Path(env["AMBI_BENCHMARK_REFERENCE_ROOT"]) / f"step_{step}/prior/manifest.json").unlink()
    result = subprocess.run(["bash", str(LAUNCHER)], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and "production never regenerates" in result.stderr
    assert _calls(env) == []
    assert not Path(env["AMBI_BENCHMARK_OUTPUT_ROOT"]).exists()


@pytest.mark.parametrize("round_values", ["", "2", "1 1", "6 6", "6 3 1 6", "6\n3"])
def test_round_selection_errors_fail_before_any_output(launch_env, round_values):
    env = _launch_env(launch_env)
    env["AMBI_ROUND_VALUES"] = round_values
    result = subprocess.run(["bash", str(LAUNCHER)], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert _calls(env) == []
    assert not Path(env["AMBI_BENCHMARK_OUTPUT_ROOT"]).exists()


@pytest.mark.parametrize("failure", ["missing_map", "sha", "dirty"])
def test_production_requires_explicit_run_map_and_clean_exact_source(launch_env, failure):
    env = _launch_env(launch_env)
    if failure == "missing_map":
        env.pop("EVAL_RUN_MAP")
        message = "explicit New/Append"
    elif failure == "sha":
        env["EXPECTED_ACTION_MODES_SHA"] = "different-sha"
        message = "commit mismatch"
    else:
        git = Path(env["PATH"].split(":")[0]) / "git"
        git.write_text('#!/bin/sh\ncase "$1" in rev-parse) echo test-sha;; status) echo " M changed.py";; esac\n')
        message = "checkout must be clean"
    result = subprocess.run(["bash", str(LAUNCHER)], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and message in result.stderr
    assert _calls(env) == []
    assert not Path(env["AMBI_BENCHMARK_OUTPUT_ROOT"]).exists()


def test_launcher_keeps_l40s_resources_without_an_arbitrary_concurrency_cap():
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True, capture_output=True)
    directives = [line for line in LAUNCHER.read_text().splitlines() if line.startswith("#SBATCH ")]
    assert "#SBATCH --array=0-29" in directives
    assert "#SBATCH --gres=gpu:l40s:1" in directives
    assert "#SBATCH --cpus-per-task=6" in directives
    assert "#SBATCH --mem=32G" in directives
    assert "#SBATCH --time=01:00:00" in directives
    assert "#SBATCH --no-requeue" in directives
    assert not any("--account" in line or "--qos" in line for line in directives)
