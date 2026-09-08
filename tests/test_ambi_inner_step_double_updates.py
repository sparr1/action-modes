"""Two-update step evaluations change dose while retaining the original pairing."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest

import evaluate_ambi_checkpoint as evaluator
from tests.test_ambi_benchmark_evaluation import checkpoint_matrix, events
from tests.test_ambi_inner_benchmark_launcher import launch_env
from tests.test_ambi_inner_interval_sweep import COMPETING_KEYS
from tests.test_checkpoint_research_configs import checkpoint_context, _build_cfg
from utils.ambi_benchmark import protocol_for
from utils.ambi_research import load_preset_matrix, normalize_selectors, resolve_preset
from utils.eval_series import concise_curve_label
from utils.eval_series_data import planner_identity


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs/research/ambi_humanoid_inner_step_double_updates.json"
ORIGINAL = ROOT / "configs/research/ambi_humanoid_inner_step_rounds.json"
LAUNCHER = ROOT / "slurm/run_ambi_inner_step_double_updates_oscar.sbatch"
SELECTORS = [f"step_double_updates/j{rounds}" for rounds in (1, 3, 6)]


def test_double_update_matrix_changes_only_interval_and_retains_paired_protocol(
    checkpoint_context,
):
    matrix = load_preset_matrix(MATRIX)
    original = load_preset_matrix(ORIGINAL)
    assert normalize_selectors(matrix) == SELECTORS
    assert matrix["source_run"] == original["source_run"] == "rwgao_b-brown-university/ambi/u13m14st"
    assert matrix["base_alg_config"] == original["base_alg_config"] == "checkpoint"
    assert {k: v for k, v in matrix["evaluation"].items() if k != "default_presets"} == {
        k: v for k, v in original["evaluation"].items() if k != "default_presets"
    }
    assert matrix["evaluation"]["seeds"] == [101, 102, 103, 104, 105]
    assert matrix["evaluation"]["max_steps"] == 500
    assert matrix["evaluation"]["controller_seed"] == 55
    expected_shared = deepcopy(original["shared_alg_params"])
    expected_shared["inner_steps_per_update"] = 256
    assert matrix["shared_alg_params"] == expected_shared

    before = deepcopy(checkpoint_context)
    for rounds, selector in zip((1, 3, 6), SELECTORS):
        selected = resolve_preset(MATRIX, selector, matrix, checkpoint_context=checkpoint_context)
        baseline = resolve_preset(ORIGINAL, f"step_rounds/j{rounds}", original,
                                  checkpoint_context=checkpoint_context)
        expected = deepcopy(baseline["algorithm_config"])
        expected["alg_params"]["inner_steps_per_update"] = 256
        assert selected["algorithm_config"] == expected
        assert selected["environment"] == baseline["environment"]
        assert protocol_for(selected, 55, 500) == protocol_for(baseline, 55, 500)
        assert protocol_for(selected, 55, 500)["action_rule"] == "tanh_mean"
    assert checkpoint_context == before


@pytest.mark.parametrize("rounds", [1, 3, 6])
def test_double_update_counts_and_curve_identity_are_distinct(checkpoint_context, rounds):
    checkpoint_context.trial_run_params["alg_params"].update({k: 99 for k in COMPETING_KEYS})
    before = deepcopy(checkpoint_context)
    selected = resolve_preset(MATRIX, f"step_double_updates/j{rounds}",
                              checkpoint_context=checkpoint_context)
    assert not COMPETING_KEYS.intersection(selected["algorithm_config"]["alg_params"])
    cfg = _build_cfg(selected["algorithm_config"])
    assert cfg.inner_update_timing == "step"
    assert cfg.inner_steps_per_update == 256
    assert cfg.inner_rollouts_per_round == cfg.inner_batch_size == 512
    assert cfg.inner_rollout_horizon == 3 and cfg.inner_rounds == rounds
    assert cfg.inner_model_step_budget == 1536 * rounds
    assert cfg.inner_replay_capacity == 9216
    assert cfg.inner_expected_update_slots == 6 * rounds
    assert cfg.inner_nominal_updates_per_round == 6
    assert (cfg.inner_critic_updates_per_action, cfg.inner_actor_updates_per_action,
            cfg.inner_temperature_updates_per_action) == (6 * rounds,) * 3
    assert cfg.inner_bootstrap_source == "inner_target"
    assert cfg.inner_finite_horizon is False and cfg.inner_outer_replay_fraction == 0
    assert cfg.inner_actor_writeback_coef == cfg.inner_critic_writeback_coef == 0
    for component in ("actor", "critic", "temperature", "replay",
                      "actor_optimizer", "critic_optimizer", "temperature_optimizer"):
        assert getattr(cfg, f"inner_{component}_scope") == "action"

    baseline = resolve_preset(ORIGINAL, f"step_rounds/j{rounds}",
                              checkpoint_context=checkpoint_context)
    old_cfg = _build_cfg(baseline["algorithm_config"])
    identity = planner_identity(vars(cfg), {}, "AMBITDMPC2/AMBITDMPC2", "tanh_mean")
    old_identity = planner_identity(vars(old_cfg), {}, "AMBITDMPC2/AMBITDMPC2", "tanh_mean")
    assert identity != old_identity
    assert identity["settings"]["inner_steps_per_update"] == 256
    registry = {"identity": {"backbone": "rwgao_b-brown-university/ambi/u13m14st",
                             "planner": identity}, "run_id": "test-dose"}
    label = concise_curve_label(registry)
    assert "s256" in label and "step updates" in label and f"J{rounds}" in label
    registry["identity"]["planner"] = old_identity
    assert label != concise_curve_label(registry)
    assert checkpoint_context == before


@pytest.mark.parametrize("rounds", [1, 3, 6])
def test_frozen_evaluator_updates_twice_after_each_parallel_step(
    checkpoint_matrix, tmp_path, rounds,
):
    checkpoint, matrix_path = checkpoint_matrix
    matrix = json.loads(matrix_path.read_text())
    matrix["shared_alg_params"].update(load_preset_matrix(MATRIX)["shared_alg_params"])
    # Preserve H3 and the two-updates-per-parallel-step ratio on a tiny model.
    matrix["shared_alg_params"].update(inner_rounds=rounds, inner_rollouts_per_round=2,
        inner_batch_size=2, inner_steps_per_update=1, inner_replay_capacity=36)
    matrix_path.write_text(json.dumps(matrix))
    bundle = tmp_path / "double-update-bundle"
    # The shared tiny checkpoint fixture was configured with a training H2.
    with pytest.warns(UserWarning, match="inner_rollout_horizon=3 exceeds train_unroll_horizon=2"):
        result = evaluator.evaluate_matrix(matrix_path, checkpoint, selectors=["budget/sac"],
                                           bundle_dir=bundle)
    run = result["results"][0]
    assert run["outer_state_unchanged"]
    assert run["outer_updates_before"] == run["outer_updates_after"]
    assert run["resolved_config"]["inner_finite_horizon"] is False
    assert run["resolved_config"]["inner_bootstrap_source"] == "inner_target"
    assert run["model_metrics"]["inner_model_steps"]["mean"] == 6 * rounds
    for component in ("critic", "actor", "temperature"):
        assert run["model_metrics"][f"inner_{component}_optimizer_steps"]["mean"] == 6 * rounds
    rows = [row for row in events(bundle) if row["phase"] != "decision"]
    solve_size = 1 + 9 * rounds
    assert len(rows) == 6 * solve_size  # Two seeds, three real decisions each.
    for index in range(0, len(rows), solve_size):
        solve = rows[index:index + solve_size]
        assert [row["phase"] for row in solve] == ["initial"] + [
            "collection", "update", "update"
        ] * (3 * rounds)
        collections = [row for row in solve if row["phase"] == "collection"]
        updates = [row for row in solve if row["phase"] == "update"]
        assert [row["critic_updates"] for row in collections] == list(range(0, 6 * rounds, 2))
        assert [row["metrics"]["collection_rollout_step"] for row in collections] == [1, 2, 3] * rounds
        assert [row["round_index"] for row in collections] == [j for j in range(1, rounds + 1) for _ in range(3)]
        for component in ("critic", "actor", "temperature"):
            assert [row[f"{component}_updates"] for row in updates] == list(range(1, 6 * rounds + 1))
            assert all(row[f"updated_{component}"] for row in updates)


def _double_update_env(launch_env):
    env = dict(launch_env)
    env.pop("AMBI_BENCHMARK_PRESETS")
    for step in range(50000, 500001, 50000):
        checkpoint = Path(env["AMBI_CHECKPOINT_PREFIX"] + str(step))
        checkpoint.touch()
        Path(str(checkpoint) + ".metadata.json").touch()
        prior = Path(env["AMBI_BENCHMARK_REFERENCE_ROOT"]) / f"step_{step}/prior"
        prior.mkdir(parents=True, exist_ok=True)
        (prior / "manifest.json").write_text("{}")
    return env


@pytest.mark.parametrize("index", [1, 2, 7, 10])
def test_launcher_reuses_every_prior_and_preserves_checkpoint_grid(launch_env, index):
    env = _double_update_env(launch_env)
    env["SLURM_ARRAY_TASK_ID"] = str(index)
    subprocess.run(["bash", str(LAUNCHER)], env=env, check=True, capture_output=True, text=True)
    calls = [json.loads(line) for line in Path(env["TEST_CALLS"]).read_text().splitlines()]
    assert len(calls) == 2
    evaluate, report = calls
    assert evaluate[0] == "evaluate_ambi_checkpoint.py" and report[0] == "report_ambi_benchmark.py"
    assert evaluate[evaluate.index("--checkpoint") + 1] == env["AMBI_CHECKPOINT_PREFIX"] + str(index * 50000)
    assert evaluate[evaluate.index("--matrix") + 1] == str(MATRIX.relative_to(ROOT))
    assert [evaluate[i + 1] for i, arg in enumerate(evaluate) if arg == "--preset"] == SELECTORS
    assert evaluate[evaluate.index("--max-steps") + 1] == "500"
    assert evaluate[evaluate.index("--seeds") + 1:evaluate.index("--max-steps")] == ["101", "102", "103", "104", "105"]
    assert evaluate[evaluate.index("--reference-bundle") + 1] == str(
        Path(env["AMBI_BENCHMARK_REFERENCE_ROOT"]) / f"step_{index * 50000}/prior")
    assert evaluate[evaluate.index("--eval-run-map") + 1] == env["EVAL_RUN_MAP"]
    assert not any("named_run/prior" in call or "--wandb" in call or "--save-root-bank" in call for call in calls)
    duplicate = subprocess.run(["bash", str(LAUNCHER)], env=env, capture_output=True, text=True)
    assert duplicate.returncode != 0 and "output already exists" in duplicate.stderr
    assert len(Path(env["TEST_CALLS"]).read_text().splitlines()) == 2


@pytest.mark.parametrize("index", [1, 6])
def test_launcher_requires_prior_even_at_50k_before_evaluation(launch_env, index):
    env = _double_update_env(launch_env)
    env["SLURM_ARRAY_TASK_ID"] = str(index)
    reference = Path(env["AMBI_BENCHMARK_REFERENCE_ROOT"]) / f"step_{index * 50000}/prior/manifest.json"
    reference.unlink()
    result = subprocess.run(["bash", str(LAUNCHER)], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and "paired prior reference missing" in result.stderr
    assert not Path(env["TEST_CALLS"]).exists()
    assert not (Path(env["AMBI_BENCHMARK_OUTPUT_ROOT"]) / f"step_{index * 50000}").exists()


def test_smoke_creates_separate_short_prior_and_never_publishes(launch_env):
    env = _double_update_env(launch_env)
    env.pop("EVAL_RUN_MAP")
    env.pop("AMBI_BENCHMARK_REFERENCE_ROOT")
    env["SLURM_ARRAY_TASK_ID"] = "6"
    subprocess.run(["bash", str(LAUNCHER), "--smoke"], env=env, check=True, capture_output=True, text=True)
    calls = [json.loads(line) for line in Path(env["TEST_CALLS"]).read_text().splitlines()]
    assert len(calls) == 3 and "named_run/prior" in calls[0]
    for call in calls[:2]:
        assert call[call.index("--max-steps") + 1] == "3"
        assert call[call.index("--seeds") + 1:call.index("--max-steps")] == ["101", "102"]
        assert "--eval-run-map" not in call and "--wandb" not in call
    prior = str(Path(env["AMBI_BENCHMARK_OUTPUT_ROOT"]) / "step_300000/prior")
    assert calls[0][calls[0].index("--bundle-dir") + 1] == prior
    assert calls[1][calls[1].index("--reference-bundle") + 1] == prior


def test_production_requires_explicit_new_run_assignment(launch_env):
    env = _double_update_env(launch_env)
    env.pop("EVAL_RUN_MAP")
    result = subprocess.run(["bash", str(LAUNCHER)], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and "explicit New" in result.stderr
    assert not Path(env["TEST_CALLS"]).exists()
