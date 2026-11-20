import json
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from slurm.metaworld_campaign import CONFIG_ROOT, ROOT, TASKS, main, plan, prepare, production_algorithm, read_json


def test_prepared_campaign_has_four_ambi_single_task_seed_cells(tmp_path):
    result = plan(CONFIG_ROOT, tmp_path / "results")
    assert result["cell_count"] == 4
    assert {cell["task"] for cell in result["cells"]} == {
        "assembly", "hand-insert", "push", "stick-pull"
    }
    assert len({cell["evaluation_csv"] for cell in result["cells"]}) == 4
    for task in TASKS:
        cells = [cell for cell in result["cells"] if cell["task"] == task]
        assert [(cell["algorithm"], cell["trial_index"], cell["seed"]) for cell in cells] == [
            ("ambi_state", 0, 55)
        ]
        for cell in cells:
            assert (cell["decisions"], cell["raw_training_steps"]) == (1_000_000, 2_000_000)
            command = shlex.split(cell["command"])
            assert Path(command[command.index(str(ROOT / "main.py")) - 1]) == (
                ROOT / "environments/metaworld/python"
            )
            assert command[command.index("--num-runs") + 1] == "1"
            assert command[command.index("--trial-index") + 1] == str(cell["trial_index"])
            assert command[command.index("--alg-index") + 1] == "0"
            assert f"TDMPC2_EVAL_CSV={cell['evaluation_csv']}" in command
            assert "WANDB_MODE=disabled" in command
            assert "--resume-mode" not in command
    assert not (tmp_path / "results").exists()


def test_production_wandb_config_preserves_scientific_recipe(tmp_path):
    base = read_json(CONFIG_ROOT / "algs/ambi_state.json")
    names = set()
    for task in TASKS:
        path = tmp_path / f"{task}.json"
        main(["production-config", "--task", task, "--output", str(path)])
        run = read_json(path)
        assert run == production_algorithm(task)
        expected = base["alg_params"]
        params = run["alg_params"]
        assert {k: v for k, v in run.items() if k != "alg_params"} == {
            k: v for k, v in base.items() if k != "alg_params"}
        assert {k: v for k, v in params.items() if not k.startswith("wandb")} == {
            k: v for k, v in expected.items() if not k.startswith("wandb")}
        names.add(params["wandb_run_name"])
        assert task in params["wandb_tags"] and "seed55" in params["wandb_tags"]
        assert params["wandb_event_indexed"] and params["wandb_step_every"] == 1000
        assert (params["wandb"], params["wandb_mode"]) == (True, "online")
        assert (params["wandb_entity"], params["wandb_project"], params["wandb_group"]) == (
            "rwgao_b-brown-university", "ambi", "metaworld-tdmpc2-4task-s55")
    assert len(names) == len(TASKS)
    assert base["alg_params"]["wandb"] is False


def test_exact_wandb_initialization_receives_production_identity(tmp_path):
    from tests.test_wandb_resume import _FakeWandb
    from utils.wandb_resume import WandbResumeContext
    from utils.wandb_utils import init_wandb

    params = production_algorithm("stick-pull")["alg_params"]
    fake = _FakeWandb()
    run = init_wandb(params, default_project="unset", run_name="unset", wandb_module=fake,
                     resume_context=WandbResumeContext.new(run_id="test-cell", directory=tmp_path))
    call, = fake.init_calls
    assert call["entity"] == "rwgao_b-brown-university" and call["project"] == "ambi"
    assert call["group"] == "metaworld-tdmpc2-4task-s55"
    assert call["name"] == "Meta-World Stick Pull | AMBI post-tanh | seed55"
    assert call["id"] == "test-cell" and call["resume"] == "never"
    run.abort()


def test_reference_config_retains_shared_tdmpc2_settings():
    baseline = read_json(CONFIG_ROOT / "algs/tdmpc2_state.json")
    dmc = read_json(CONFIG_ROOT.parent / "dmcontrol/algs/tdmpc2_state_benchmark.json")
    params = baseline["alg_params"]
    for key in (
        "model_size", "buffer_size", "batch_size", "reward_coef", "value_coef", "consistency_coef",
        "rho", "lr", "enc_lr_scale", "grad_clip_norm", "tau", "discount_denom", "discount_min",
        "discount_max", "utd", "train_unroll_horizon", "outer_planning_horizon", "inner_rollout_horizon",
        "mpc", "iterations", "num_samples", "num_elites", "num_pi_trajs", "min_std", "max_std",
        "temperature", "entropy_coef", "dropout", "compile",
    ):
        assert params[key] == dmc["alg_params"][key], key
    assert params["seed_steps"] == params["pretrain_steps"] == 1000
    assert params["eval_freq"] == 25_000 and params["eval_episodes"] == 10
    assert params["wandb"] is False and params["wandb_mode"] == "disabled"
    assert "discount" not in params  # The standard heuristic resolves 100 decisions to 0.95.


def _write_algorithm(tmp_path, **changes):
    run = read_json(CONFIG_ROOT / "algs/tdmpc2_state.json")
    run.update(changes)
    path = tmp_path / "reviewed_ambi.json"
    path.write_text(json.dumps(run))
    return path, run


def test_prepare_preserves_full_scientific_recipe_and_input_file(tmp_path):
    params = read_json(CONFIG_ROOT / "algs/tdmpc2_state.json")["alg_params"]
    params.update(inner_rounds=8, inner_target_entropy=-2, critic_geometry_diagnostics=True,
                  custom_nested={"retain": [1, 2, 3]})
    path, original = _write_algorithm(tmp_path, alg="AMBITDMPC2/AMBITDMPC2", alg_params=params)
    saved = path.read_bytes()
    files = prepare([path], tmp_path / "configs", trials=2, decisions=200_000, base_seed=55)
    result = files["algs/reviewed_ambi.json"]
    assert result["alg_params"] == original["alg_params"]
    assert result["total_steps"] == 200_000 and result["seed"] == 55
    assert path.read_bytes() == saved
    assert not (tmp_path / "configs").exists()
    for task in TASKS:
        assert files[f"experiments/{task}.json"]["trials"] == 2
        assert "overrides_alg" not in files[f"experiments/{task}.json"]
    cells = plan(tmp_path / "configs", tmp_path / "outputs", files=files)["cells"]
    assert len(cells) == 8 and {cell["seed"] for cell in cells} == {55, 56}


@pytest.mark.parametrize("changes,error", [
    ({"env": "DMControl-v0"}, "reviewed algorithm"),
    ({"alg_params": {"obs": "rgb"}}, "state observations"),
    ({"total_steps": 10}, "episode boundary"),
])
def test_plan_rejects_wrong_environment_or_partial_episode_budget(tmp_path, changes, error):
    files = prepare([CONFIG_ROOT / "algs/tdmpc2_state.json"], tmp_path / "configs")
    files["algs/tdmpc2_state.json"].update(changes)
    with pytest.raises(ValueError, match=error):
        plan(tmp_path / "configs", tmp_path / "outputs", files=files)


@pytest.mark.parametrize("key,value,error", [
    ("eval_inner_comparison", True, "snapshot evaluation"),
    ("wandb", True, "Preparation requires"),
    ("eval_csv_path", "/tmp/shared.csv", "separate CSV"),
    ("critic_geometry_diagnostics", False, "critic geometry"),
])
def test_prepare_rejects_unsafe_reporting_or_missing_diagnostics(tmp_path, key, value, error):
    params = read_json(CONFIG_ROOT / "algs/tdmpc2_state.json")["alg_params"]
    params[key] = value
    path, _ = _write_algorithm(tmp_path, alg="AMBITDMPC2/AMBITDMPC2", alg_params=params)
    with pytest.raises(ValueError, match=error):
        prepare([path], tmp_path / "configs")


def test_cli_dry_run_and_prepare_never_create_result_directories(tmp_path, capsys):
    output, results = tmp_path / "configs", tmp_path / "results"
    arguments = ["prepare", "--algorithm-config", str(CONFIG_ROOT / "algs/tdmpc2_state.json"),
                 "--output-dir", str(output), "--results-dir", str(results)]
    main(arguments + ["--dry-run"])
    preview = json.loads(capsys.readouterr().out)
    assert preview["prepared"] is False and preview["cell_count"] == 4
    assert not output.exists() and not results.exists()
    main(arguments)
    written = json.loads(capsys.readouterr().out)
    assert written["prepared"] is True and written["cells"] == preview["cells"]
    assert len(list(output.rglob("*.json"))) == 5
    assert not results.exists()
    before = {path: path.read_bytes() for path in output.rglob("*.json")}
    with pytest.raises(FileExistsError):
        main(arguments)
    assert before == {path: path.read_bytes() for path in output.rglob("*.json")}


def test_duplicate_json_keys_are_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"seed": 1, "seed": 2}')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        read_json(path)


def test_partial_shallow_algorithm_override_is_rejected(tmp_path):
    files = prepare([CONFIG_ROOT / "algs/tdmpc2_state.json"], tmp_path / "configs")
    files["experiments/assembly.json"]["overrides_alg"] = {"alg_params": {"wandb": False}}
    with pytest.raises(ValueError, match="shallow manifest"):
        plan(tmp_path / "configs", tmp_path / "outputs", files=files)


@pytest.mark.parametrize("task", TASKS)
def test_main_resolves_a_single_cell_without_mutating_the_base_recipe(
    monkeypatch, tmp_path, task
):
    """Exercise the actual CLI's manifest/seed/checkpoint routing with no RL work."""
    import main as training_main

    algorithm_path = CONFIG_ROOT / "algs/ambi_state.json"
    original = algorithm_path.read_bytes()
    captured = {}

    class NoTrainingModel:
        supports_composable_checkpointing = True

        def set_checkpointing(self, **kwargs):
            captured["checkpointing"] = kwargs

        def learn(self, **kwargs):
            captured["budget"] = kwargs

    domain = SimpleNamespace(
        action_space=SimpleNamespace(seed=lambda seed: None),
        observation_space=SimpleNamespace(seed=lambda seed: None),
        close=lambda: None,
    )

    def construct_env(run, experiment):
        captured["env_params"] = experiment["env_params"]
        return domain

    def construct_algorithm(path, params, env, **kwargs):
        captured["run"] = kwargs["full_run_params"]
        captured["params"] = params
        return NoTrainingModel(), False, "AMBITDMPC2"

    monkeypatch.setattr(training_main, "build_env", construct_env)
    monkeypatch.setattr(training_main, "initialize_alg", construct_algorithm)
    monkeypatch.setattr(sys, "argv", [
        "main.py", "--run", str(CONFIG_ROOT / "experiments" / f"{task}.json"),
        "--alg-dir", str(CONFIG_ROOT / "algs"), "--trial-index", "0", "--num-runs", "1",
        "--alg-index", "0",
        "--log-dir", str(tmp_path / "artifacts"),
    ])
    training_main.main()
    assert captured["run"]["seed"] == 55
    assert captured["run"]["env"] == "MetaWorld-v0"
    assert captured["env_params"]["task"] == task
    assert captured["env_params"]["protocol"] == "metaworld-tdmpc2-04be337-state-final-success-v1"
    assert captured["params"] == json.loads(original)["alg_params"]
    assert captured["budget"] == {"total_timesteps": 1_000_000}
    assert captured["checkpointing"]["save_freq"] == 25_000
    assert set(captured["checkpointing"]["save_strat"]) == {"all", "best", "latest"}
    assert algorithm_path.read_bytes() == original
