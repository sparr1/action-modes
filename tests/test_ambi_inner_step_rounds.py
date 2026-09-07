"""Frozen step-round experiments preserve pairing, timing, and launch scope."""
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
from utils.ambi_research import load_preset_matrix, normalize_selectors, resolve_preset
from utils.eval_series import concise_curve_label
from utils.eval_series_data import planner_identity

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs/research/ambi_humanoid_inner_step_rounds.json"
LAUNCHER = ROOT / "slurm/run_ambi_inner_step_rounds_oscar.sbatch"


def test_round_sweep_changes_only_rounds_and_preserves_source(checkpoint_context):
    matrix = load_preset_matrix(MATRIX)
    assert normalize_selectors(matrix) == [f"step_rounds/j{j}" for j in (1, 3, 6)]
    assert matrix["source_run"] == "rwgao_b-brown-university/ambi/u13m14st"
    assert matrix["evaluation"]["seeds"] == [101, 102, 103, 104, 105]
    assert matrix["evaluation"]["max_steps"] == 500
    assert matrix["evaluation"]["controller_seed"] == 55
    checkpoint_context.trial_run_params["alg_params"].update({k: 99 for k in COMPETING_KEYS})
    before = deepcopy(checkpoint_context)
    previous = None
    labels = []
    for j in (1, 3, 6):
        run = resolve_preset(MATRIX, f"step_rounds/j{j}", matrix,
                             checkpoint_context=checkpoint_context)["algorithm_config"]
        params = run["alg_params"]
        assert not COMPETING_KEYS.intersection(params)
        assert params["inner_rounds"] == j
        comparable = deepcopy(run)
        comparable["alg_params"].pop("inner_rounds")
        if previous is not None:
            assert comparable == previous
        previous = comparable
        cfg = _build_cfg(run)
        assert cfg.inner_update_timing == "step"
        assert cfg.inner_steps_per_update == cfg.inner_rollouts_per_round == cfg.inner_batch_size == 512
        assert cfg.inner_rollout_horizon == 3
        assert cfg.inner_model_step_budget == 1536 * j
        assert cfg.inner_replay_capacity == 9216
        assert cfg.inner_expected_update_slots == 3 * j
        assert cfg.inner_critic_updates_per_action == cfg.inner_actor_updates_per_action == cfg.inner_temperature_updates_per_action == 3 * j
        assert cfg.inner_bootstrap_source == "inner_target"
        assert cfg.inner_actor_lr == 5e-5 and cfg.inner_critic_lr == 1e-4
        assert cfg.inner_temperature_lr == 3e-4
        assert cfg.inner_actor_writeback_coef == cfg.inner_critic_writeback_coef == 0
        assert cfg.inner_finite_horizon is False and cfg.inner_outer_replay_fraction == 0
        settings = dict(vars(cfg))
        registry = {"identity": {"backbone": matrix["source_run"],
                    "planner": {"type": "sac", "settings": settings}}, "run_id": "1234test"}
        label = concise_curve_label(registry)
        assert "step updates" in label and f"J{j}" in label
        labels.append(label)
    assert len(set(labels)) == 3
    assert checkpoint_context == before


def test_step_identity_distinguishes_timing_without_changing_legacy_round_identity():
    config = {"inner_operator": "sac", "inner_steps_per_update": 512}
    legacy = planner_identity(config, {}, "AMBITDMPC2/AMBITDMPC2", "tanh_mean")
    explicit_round = planner_identity({**config, "inner_update_timing": "round"}, {},
                                      "AMBITDMPC2/AMBITDMPC2", "tanh_mean")
    step = planner_identity({**config, "inner_update_timing": "step"}, {},
                            "AMBITDMPC2/AMBITDMPC2", "tanh_mean")
    assert legacy == explicit_round and step != legacy
    assert step["settings"]["inner_update_timing"] == "step"


@pytest.mark.parametrize("rounds", [1, 3, 6])
def test_frozen_evaluator_keeps_step_timing_and_full_trace(checkpoint_matrix, tmp_path, rounds):
    checkpoint, matrix_path = checkpoint_matrix
    matrix = json.loads(matrix_path.read_text())
    matrix["shared_alg_params"].update(inner_rounds=rounds,
        inner_critic_updates_per_round=None, inner_actor_updates_per_round=None,
        inner_updates_per_round=None, inner_steps_per_update=2, inner_update_timing="step",
        inner_replay_capacity=24)
    matrix_path.write_text(json.dumps(matrix))
    bundle = tmp_path / "step-bundle"
    result = evaluator.evaluate_matrix(matrix_path, checkpoint, selectors=["budget/sac"],
                                       bundle_dir=bundle)
    run = result["results"][0]
    assert run["outer_state_unchanged"]
    assert run["resolved_config"]["inner_update_timing"] == "step"
    assert run["model_metrics"]["inner_critic_optimizer_steps"]["mean"] == 2 * rounds
    assert run["model_metrics"]["inner_model_steps"]["mean"] == 4 * rounds
    rows = [row for row in events(bundle) if row["phase"] != "decision"]
    assert len(rows) == 6 * (1 + 4 * rounds)
    # Two paired seeds x three decisions; each solve restarts at its root.
    for index in range(0, len(rows), 1 + 4 * rounds):
        solve = rows[index:index + 1 + 4 * rounds]
        assert [r["phase"] for r in solve] == ["initial"] + ["collection", "update"] * (2 * rounds)


def _step_env(launch_env):
    env = dict(launch_env)
    env.pop("AMBI_BENCHMARK_PRESETS")
    prefix = env["AMBI_CHECKPOINT_PREFIX"]
    for step in range(50000, 500001, 50000):
        Path(prefix + str(step)).touch()
        Path(prefix + str(step) + ".metadata.json").touch()
        if step != 50000:
            prior = Path(env["AMBI_BENCHMARK_REFERENCE_ROOT"]) / f"step_{step}/prior"
            prior.mkdir(parents=True, exist_ok=True)
            (prior / "manifest.json").write_text("{}")
    return env


@pytest.mark.parametrize("index", [1, 2, 3, 10])
def test_launcher_50k_grid_reuses_completed_priors_and_stages_only_selected_sac(launch_env, index):
    env = _step_env(launch_env)
    env["SLURM_ARRAY_TASK_ID"] = str(index)
    subprocess.run(["bash", str(LAUNCHER)], env=env, check=True, capture_output=True, text=True)
    calls = [json.loads(line) for line in Path(env["TEST_CALLS"]).read_text().splitlines()]
    assert len(calls) == (3 if index == 1 else 2)
    if index == 1:
        assert "named_run/prior" in calls[0]
        assert "--eval-run-map" not in calls[0]
    evaluate = calls[-2]
    assert evaluate[evaluate.index("--checkpoint") + 1].endswith(str(index * 50000))
    assert [evaluate[i + 1] for i, arg in enumerate(evaluate) if arg == "--preset"] == [
        "step_rounds/j1", "step_rounds/j3", "step_rounds/j6"]
    assert "--eval-run-map" in evaluate and "--reference-bundle" in evaluate
    assert not any("--wandb" in call or "--save-root-bank" in call for call in calls)
    assert evaluate[evaluate.index("--max-steps") + 1] == "500"
    duplicate = subprocess.run(["bash", str(LAUNCHER)], env=env, capture_output=True, text=True)
    assert duplicate.returncode != 0 and "output already exists" in duplicate.stderr


def test_smoke_uses_separate_short_prior_and_never_publishes(launch_env):
    env = _step_env(launch_env)
    env.pop("EVAL_RUN_MAP")
    env["SLURM_ARRAY_TASK_ID"] = "6"
    subprocess.run(["bash", str(LAUNCHER), "--smoke"], env=env, check=True, capture_output=True, text=True)
    calls = [json.loads(line) for line in Path(env["TEST_CALLS"]).read_text().splitlines()]
    assert len(calls) == 3 and "named_run/prior" in calls[0]
    for call in calls[:2]:
        assert call[call.index("--max-steps") + 1] == "3"
        assert call[call.index("--seeds") + 1:call.index("--max-steps")] == ["101", "102"]
        assert "--eval-run-map" not in call and "--wandb" not in call


def test_production_requires_explicit_new_run_assignment(launch_env):
    env = _step_env(launch_env)
    env.pop("EVAL_RUN_MAP")
    result = subprocess.run(["bash", str(LAUNCHER)], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and "explicit New" in result.stderr
    assert not Path(env["TEST_CALLS"]).exists()
