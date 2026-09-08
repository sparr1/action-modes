"""Faster target tracking changes only inner tau in the original N512 sweep."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest
import torch

import evaluate_ambi_checkpoint as evaluator
from RL.tdmpc2_core.inner_improvement import InnerImprovementEngine
from tests.test_ambi_benchmark_evaluation import checkpoint_matrix, events
from tests.test_ambi_inner_benchmark_launcher import launch_env
from tests.test_ambi_inner_interval_sweep import COMPETING_KEYS
from tests.test_ambi_inner_step_double_updates import _double_update_env
from tests.test_checkpoint_research_configs import checkpoint_context, _build_cfg
from utils.ambi_benchmark import protocol_for
from utils.ambi_research import load_preset_matrix, normalize_selectors, resolve_preset
from utils.eval_series_data import planner_identity


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs/research/ambi_humanoid_inner_fast_target_step_rounds.json"
BASELINE = ROOT / "configs/research/ambi_humanoid_inner_step_rounds.json"
LAUNCHER = ROOT / "slurm/run_ambi_inner_fast_target_step_rounds_oscar.sbatch"
SELECTORS = [f"fast_target_step_rounds/j{rounds}" for rounds in (1, 3, 6)]


@pytest.mark.parametrize("rounds", [1, 3, 6])
def test_only_inner_target_tau_changes_from_original_paired_sweep(checkpoint_context, rounds):
    matrix = load_preset_matrix(MATRIX)
    baseline = load_preset_matrix(BASELINE)
    assert normalize_selectors(matrix) == SELECTORS
    assert matrix["source_run"] == baseline["source_run"] == "rwgao_b-brown-university/ambi/u13m14st"
    assert matrix["base_alg_config"] == baseline["base_alg_config"] == "checkpoint"
    assert {k: v for k, v in matrix["evaluation"].items() if k != "default_presets"} == {
        k: v for k, v in baseline["evaluation"].items() if k != "default_presets"
    }
    assert matrix["evaluation"]["seeds"] == [101, 102, 103, 104, 105]
    assert matrix["evaluation"]["controller_seed"] == 55
    assert matrix["evaluation"]["max_steps"] == 500
    assert baseline["shared_alg_params"]["inner_critic_target_tau"] == 0.01
    assert matrix["shared_alg_params"] == {
        **baseline["shared_alg_params"], "inner_critic_target_tau": 0.1,
    }

    # An inherited outer tau or competing legacy schedule must not alter this ablation.
    checkpoint_context.trial_run_params["alg_params"].update({k: 99 for k in COMPETING_KEYS})
    checkpoint_context.trial_run_params["alg_params"]["tau"] = 0.0037
    before = deepcopy(checkpoint_context)
    selected = resolve_preset(MATRIX, f"fast_target_step_rounds/j{rounds}",
                              checkpoint_context=checkpoint_context)
    original = resolve_preset(BASELINE, f"step_rounds/j{rounds}",
                              checkpoint_context=checkpoint_context)
    expected = deepcopy(original["algorithm_config"])
    expected["alg_params"]["inner_critic_target_tau"] = 0.1
    assert selected["algorithm_config"] == expected
    assert not COMPETING_KEYS.intersection(selected["algorithm_config"]["alg_params"])
    assert selected["environment"] == original["environment"]
    assert protocol_for(selected, 55, 500) == protocol_for(original, 55, 500)
    assert protocol_for(selected, 55, 500)["action_rule"] == "tanh_mean"
    cfg = _build_cfg(selected["algorithm_config"])
    old_cfg = _build_cfg(original["algorithm_config"])
    assert cfg.tau == old_cfg.tau == 0.0037
    assert cfg.inner_critic_target_tau == 0.1
    assert cfg.inner_critic_target_update_interval == 1
    assert cfg.inner_update_timing == "step"
    assert cfg.inner_steps_per_update == cfg.inner_rollouts_per_round == cfg.inner_batch_size == 512
    assert cfg.inner_rollout_horizon == 3 and cfg.inner_rounds == rounds
    assert cfg.inner_model_step_budget == 1536 * rounds
    assert cfg.inner_replay_capacity == 9216 and cfg.inner_replay_sampling == "with_replacement"
    assert cfg.inner_expected_update_slots == 3 * rounds
    assert (cfg.inner_critic_updates_per_action, cfg.inner_actor_updates_per_action,
            cfg.inner_temperature_updates_per_action) == (3 * rounds,) * 3
    assert cfg.inner_bootstrap_source == "inner_target" and cfg.inner_finite_horizon is False
    assert cfg.inner_actor_writeback_coef == cfg.inner_critic_writeback_coef == 0
    for component in ("actor", "critic", "temperature", "replay",
                      "actor_optimizer", "critic_optimizer", "temperature_optimizer"):
        assert getattr(cfg, f"inner_{component}_scope") == "action"
    identity = planner_identity(vars(cfg), {}, "AMBITDMPC2/AMBITDMPC2", "tanh_mean")
    old_identity = planner_identity(vars(old_cfg), {}, "AMBITDMPC2/AMBITDMPC2", "tanh_mean")
    assert identity != old_identity
    old_identity["settings"]["inner_critic_target_tau"] = 0.1
    assert identity == old_identity
    assert checkpoint_context == before


@pytest.mark.parametrize("rounds", [1, 3, 6])
def test_evaluator_really_moves_fresh_action_targets_by_point_one_each_critic_step(
    checkpoint_matrix, tmp_path, monkeypatch, rounds,
):
    checkpoint, matrix_path = checkpoint_matrix
    matrix = json.loads(matrix_path.read_text())
    matrix["shared_alg_params"].update(load_preset_matrix(MATRIX)["shared_alg_params"])
    # Scale N and B together, retaining H3, J, and one update per parallel timestep.
    matrix["shared_alg_params"].update(inner_rounds=rounds, inner_rollouts_per_round=2,
        inner_batch_size=2, inner_steps_per_update=2, inner_replay_capacity=36)
    matrix_path.write_text(json.dumps(matrix))
    prepare = InnerImprovementEngine._prepare_workspace
    update = InnerImprovementEngine._maybe_update_targets
    resets, target_steps, movements = [], [], []

    def inspect_reset(engine, *, t0):
        prepare(engine, t0=t0)
        state = engine.state
        assert state.critic_steps == state.critic_lifetime_steps == state.critic_target_steps == 0
        for outer, critic, target in zip(engine.model._Qs.parameters(), state.critic.parameters(),
                                         state.critic_target.parameters()):
            assert torch.equal(outer, critic) and torch.equal(outer, target)
            assert outer.data_ptr() != critic.data_ptr() != target.data_ptr()
        resets.append(0)

    def inspect_target_update(engine, *, critic_updated, actor_updated):
        state = engine.state
        assert critic_updated and actor_updated
        before = [p.detach().clone() for p in state.critic_target.parameters()]
        online = [p.detach().clone() for p in state.critic.parameters()]
        count = state.critic_target_steps
        update(engine, critic_updated=critic_updated, actor_updated=actor_updated)
        assert state.critic_target_steps == count + 1 == state.critic_steps
        for old, source, actual in zip(before, online, state.critic_target.parameters()):
            torch.testing.assert_close(actual, torch.lerp(old, source, 0.1), rtol=1e-6, atol=1e-8)
        movements.append(any(not torch.equal(old, actual)
                             for old, actual in zip(before, state.critic_target.parameters())))
        target_steps.append(state.critic_target_steps)
        resets[-1] += 1

    monkeypatch.setattr(InnerImprovementEngine, "_prepare_workspace", inspect_reset)
    monkeypatch.setattr(InnerImprovementEngine, "_maybe_update_targets", inspect_target_update)
    bundle = tmp_path / "fast-target"
    with pytest.warns(UserWarning, match="inner_rollout_horizon=3 exceeds train_unroll_horizon=2"):
        result = evaluator.evaluate_matrix(matrix_path, checkpoint, selectors=["budget/sac"],
                                           bundle_dir=bundle)
    run = result["results"][0]
    assert run["outer_state_unchanged"]
    assert run["outer_updates_before"] == run["outer_updates_after"]
    # The evaluator performs one unscored warmup and six recorded action solves.
    assert resets == [3 * rounds] * 7
    assert target_steps == list(range(1, 3 * rounds + 1)) * 7
    assert all(movements)
    assert run["model_metrics"]["inner_model_steps"]["mean"] == 6 * rounds
    assert run["model_metrics"]["inner_critic_target_updates"]["mean"] == 3 * rounds
    for component in ("critic", "actor", "temperature"):
        assert run["model_metrics"][f"inner_{component}_optimizer_steps"]["mean"] == 3 * rounds
    rows = [row for row in events(bundle) if row["phase"] != "decision"]
    solve_size = 1 + 6 * rounds
    assert len(rows) == 6 * solve_size
    for index in range(0, len(rows), solve_size):
        solve = rows[index:index + solve_size]
        assert [row["phase"] for row in solve] == ["initial"] + ["collection", "update"] * (3 * rounds)
        collections = [row for row in solve if row["phase"] == "collection"]
        updates = [row for row in solve if row["phase"] == "update"]
        assert [row["metrics"]["collection_rollout_step"] for row in collections] == [1, 2, 3] * rounds
        for component in ("critic", "actor", "temperature"):
            assert [row[f"{component}_updates"] for row in updates] == list(range(1, 3 * rounds + 1))


def _flag(call, flag):
    return call[call.index(flag) + 1]


def test_launcher_produces_thirty_unique_complete_paired_cells(launch_env):
    env = _double_update_env(launch_env)
    cells, outputs = set(), set()
    for index in range(30):
        env["SLURM_ARRAY_TASK_ID"] = str(index)
        subprocess.run(["bash", str(LAUNCHER)], env=env, check=True, capture_output=True, text=True)
        calls = [json.loads(line) for line in Path(env["TEST_CALLS"]).read_text().splitlines()]
        assert len(calls) == 2 * (index + 1)
        evaluate, report = calls[-2:]
        step, rounds = (index % 10 + 1) * 50000, (6, 3, 1)[index // 10]
        cells.add((step, rounds))
        assert evaluate[0] == "evaluate_ambi_checkpoint.py"
        assert _flag(evaluate, "--checkpoint") == env["AMBI_CHECKPOINT_PREFIX"] + str(step)
        assert _flag(evaluate, "--matrix") == str(MATRIX.relative_to(ROOT))
        assert [evaluate[i + 1] for i, arg in enumerate(evaluate) if arg == "--preset"] == [
            f"fast_target_step_rounds/j{rounds}"]
        assert evaluate[evaluate.index("--seeds") + 1:evaluate.index("--max-steps")] == [
            "101", "102", "103", "104", "105"]
        assert _flag(evaluate, "--max-steps") == "500"
        prior = str(Path(env["AMBI_BENCHMARK_REFERENCE_ROOT"]) / f"step_{step}/prior")
        output = str(Path(env["AMBI_BENCHMARK_OUTPUT_ROOT"]) / f"step_{step}/j{rounds}/inner")
        assert _flag(evaluate, "--reference-bundle") == prior
        assert _flag(evaluate, "--bundle-dir") == output
        assert _flag(evaluate, "--eval-run-map") == env["EVAL_RUN_MAP"]
        assert report[0] == "report_ambi_benchmark.py"
        assert [report[i + 1] for i, arg in enumerate(report) if arg == "--bundle"] == [prior, output]
        outputs.add(output)
    assert cells == {(step, rounds) for step in range(50000, 500001, 50000) for rounds in (1, 3, 6)}
    assert len(outputs) == 30
    repeated = subprocess.run(["bash", str(LAUNCHER)], env=env, capture_output=True, text=True)
    assert repeated.returncode != 0 and "output already exists" in repeated.stderr
    assert len(Path(env["TEST_CALLS"]).read_text().splitlines()) == 60


@pytest.mark.parametrize("index,rounds", [(5, 6), (15, 3), (25, 1)])
def test_smoke_indices_keep_all_three_300k_cells_short_and_unpublished(launch_env, index, rounds):
    env = _double_update_env(launch_env)
    env.pop("EVAL_RUN_MAP")
    env.pop("AMBI_BENCHMARK_REFERENCE_ROOT")
    env["SLURM_ARRAY_TASK_ID"] = str(index)
    subprocess.run(["bash", str(LAUNCHER), "--smoke"], env=env, check=True, capture_output=True, text=True)
    prior, inner, report = [json.loads(line) for line in Path(env["TEST_CALLS"]).read_text().splitlines()]
    assert _flag(prior, "--preset") == "named_run/prior"
    assert _flag(inner, "--preset") == f"fast_target_step_rounds/j{rounds}"
    for call in (prior, inner):
        assert _flag(call, "--checkpoint") == env["AMBI_CHECKPOINT_PREFIX"] + "300000"
        assert call[call.index("--seeds") + 1:call.index("--max-steps")] == ["101", "102"]
        assert _flag(call, "--max-steps") == "3"
        assert "--eval-run-map" not in call and "--wandb" not in call
    private_prior = str(Path(env["AMBI_BENCHMARK_OUTPUT_ROOT"]) / f"step_300000/j{rounds}/prior")
    assert _flag(prior, "--bundle-dir") == _flag(inner, "--reference-bundle") == private_prior
    assert report[0] == "report_ambi_benchmark.py"


def test_launcher_keeps_l40s_and_no_arbitrary_concurrency_cap():
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True, capture_output=True)
    directives = [line for line in LAUNCHER.read_text().splitlines() if line.startswith("#SBATCH ")]
    assert "#SBATCH --array=0-29" in directives
    assert "#SBATCH --gres=gpu:l40s:1" in directives
    assert "#SBATCH --no-requeue" in directives
