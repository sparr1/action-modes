import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALGORITHM_PATH = (
    ROOT / "configs/dmcontrol/algs/tdmpc2_state_benchmark.json"
)
MANIFEST_ROOT = ROOT / "configs/dmcontrol/experiments/state"
LAUNCHER_PATH = ROOT / "slurm/run_tdmpc2_dmcontrol_state.sbatch"
HUMANOID_ALGORITHM_PATH = (
    ROOT / "configs/dmcontrol/algs/tdmpc2_humanoid_walk_state.json"
)
HUMANOID_MANIFEST_PATH = (
    ROOT
    / "configs/dmcontrol/experiments/state/humanoid_walk.json"
)
HUMANOID_LAUNCHER_PATH = (
    ROOT / "slurm/run_tdmpc2_humanoid_walk_state.sbatch"
)
HUMANOID_SMOKE_ALGORITHM_PATH = (
    ROOT / "configs/dmcontrol/algs/tdmpc2_humanoid_walk_state_smoke.json"
)
HUMANOID_SMOKE_MANIFEST_PATH = (
    ROOT / "configs/dmcontrol/experiments/humanoid_walk_state_smoke.json"
)

TASKS = {
    "cartpole_swingup": "cartpole-swingup",
    "cheetah_run": "cheetah-run",
    "cup_catch": "cup-catch",
    "finger_spin": "finger-spin",
    "reacher_easy": "reacher-easy",
    "walker_walk": "walker-walk",
}


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path):
    return json.loads(
        path.read_text(), object_pairs_hook=_reject_duplicate_keys
    )


def test_tdmpc2_state_benchmark_freezes_the_official_single_task_budget():
    config = _load_json(ALGORITHM_PATH)
    params = config["alg_params"]

    assert config["seed"] == 1
    assert config["env"] == "DMControl-v0"
    assert config["alg"] == "TDMPC2/TDMPC2Baseline"
    assert config["device"] == "cuda"
    assert config["total_steps"] == 4_000_000
    assert config["episodes"] is None
    assert params["obs"] == "state"
    assert params["model_size"] == 5
    assert params["buffer_size"] == 1_000_000
    assert params["batch_size"] == 256
    assert params["rho"] == 0.5
    assert params["seed_steps"] == params["pretrain_steps"] == 2_500
    assert params["utd"] == 1
    assert params["eval_freq"] == 100_000
    assert params["eval_episodes"] == 10
    assert (
        params["train_unroll_horizon"],
        params["outer_planning_horizon"],
        params["inner_rollout_horizon"],
    ) == (3, 3, 3)
    assert (
        params["iterations"],
        params["num_samples"],
        params["num_elites"],
        params["num_pi_trajs"],
    ) == (6, 512, 64, 24)
    assert params["mpc"] is True
    assert params["compile"] is False
    assert params["wandb"] is True
    assert params["wandb_entity"] == "rwgao_b-brown-university"
    assert params["wandb_project"] == "ambi"
    assert params["wandb_mode"] == "online"
    assert params["wandb_group"] == "tdmpc2-dmc6-state-4m"
    assert params["wandb_step_every"] == 1_000

    expected_official = {
        "reward_coef": 0.1,
        "value_coef": 0.1,
        "termination_coef": 1.0,
        "consistency_coef": 20.0,
        "lr": 3e-4,
        "enc_lr_scale": 0.3,
        "grad_clip_norm": 20.0,
        "tau": 0.01,
        "discount_denom": 5,
        "discount_min": 0.95,
        "discount_max": 0.995,
        "min_std": 0.05,
        "max_std": 2.0,
        "temperature": 0.5,
        "entropy_coef": 1e-4,
        "dropout": 0.01,
    }
    assert {key: params[key] for key in expected_official} == expected_official


def test_dmc6_manifests_resolve_one_shared_config_over_three_seeds():
    assert {path.stem for path in MANIFEST_ROOT.glob("*.json")} == (
        set(TASKS) | {"humanoid_walk"}
    )

    algorithm = _load_json(ALGORITHM_PATH)
    for stem, task in TASKS.items():
        manifest = _load_json(MANIFEST_ROOT / f"{stem}.json")
        resolved = {**algorithm, **manifest["overrides_alg"]}

        assert manifest["study_type"] == "tdmpc2_dmc6_state_benchmark"
        assert "ten online evaluation episodes every 100k" in manifest["study_note"]
        assert "Model checkpoints" in manifest["study_note"]
        assert "W&B syncs online from node-local temporary storage" in manifest["study_note"]
        assert resolved["seed"] == 1
        assert resolved["total_steps"] == 4_000_000
        assert manifest["env_params"] == {
            "task": task,
            "obs": "state",
            "render_mode": None,
        }
        assert manifest["trials"] == 3
        assert manifest["configs"] == ["tdmpc2_state_benchmark"]
        assert manifest["logs"] == "none"
        assert manifest["save_trials"] == "none"
        assert manifest["checkpoint_every"] is None
        assert manifest["save_strat"] == "none"
        assert "checkpoint_best_window" not in manifest


def test_hydra_launcher_maps_all_18_jobs_without_pinning_a_node():
    contents = LAUNCHER_PATH.read_text()

    assert "#SBATCH --array=0-17%4" in contents
    assert "--nodelist" not in contents
    assert "#SBATCH -w" not in contents
    assert "environments/dmcontrol/.venv/bin/python" in contents
    assert "uv run" not in contents
    assert "conda activate" not in contents
    assert "export MUJOCO_GL=egl" in contents
    assert "--alg-dir configs/dmcontrol/algs" in contents
    assert "TASK_INDEX=$((SLURM_ARRAY_TASK_ID / 3))" in contents
    assert "TRIAL_INDEX=$((SLURM_ARRAY_TASK_ID % 3))" in contents
    assert '--trial-index "$TRIAL_INDEX"' in contents
    assert "--num-runs 1" in contents
    assert "SLURM_ARRAY_JOB_ID" in contents
    assert "SLURM_ARRAY_TASK_ID" in contents
    assert "seed_${SEED}" in contents
    assert 'export TDMPC2_EVAL_CSV="$EVAL_CSV"' in contents
    assert "export WANDB_MODE=online" in contents
    assert 'export WANDB_DIR="$WANDB_LOCAL_ROOT"' in contents
    assert 'export WANDB_CACHE_DIR="$WANDB_LOCAL_ROOT/cache"' in contents
    assert "export WANDB_DISABLE_CODE=true" in contents
    assert "results/dmcontrol/tdmpc2_state" in contents
    assert 'mkdir -p "$RUN_ROOT"' not in contents
    for stem in TASKS:
        assert stem in contents


def test_humanoid_walk_uses_the_published_long_horizon_protocol():
    config = _load_json(HUMANOID_ALGORITHM_PATH)
    manifest = _load_json(HUMANOID_MANIFEST_PATH)
    params = config["alg_params"]

    assert config["seed"] == 1
    assert config["total_steps"] == 14_000_000
    assert config["alg"] == "TDMPC2/TDMPC2Baseline"
    assert config["device"] == "cuda"
    assert manifest["env_params"] == {
        "task": "humanoid-walk",
        "obs": "state",
        "render_mode": None,
    }
    assert manifest["trials"] == 3
    assert manifest["configs"] == ["tdmpc2_humanoid_walk_state"]
    assert manifest["logs"] == "none"
    assert manifest["save_trials"] == "none"
    assert manifest["checkpoint_every"] is None
    assert manifest["save_strat"] == "none"
    assert "checkpoint_best_window" not in manifest

    expected = {
        "model_size": 5,
        "episodic": False,
        "buffer_size": 1_000_000,
        "batch_size": 256,
        "reward_coef": 0.1,
        "value_coef": 0.1,
        "termination_coef": 1.0,
        "consistency_coef": 20.0,
        "rho": 0.5,
        "lr": 3e-4,
        "enc_lr_scale": 0.3,
        "grad_clip_norm": 20.0,
        "tau": 0.01,
        "discount_denom": 5,
        "discount_min": 0.95,
        "discount_max": 0.995,
        "seed_steps": 2_500,
        "pretrain_steps": 2_500,
        "utd": 1,
        "eval_freq": 100_000,
        "eval_episodes": 10,
        "train_unroll_horizon": 3,
        "outer_planning_horizon": 3,
        "inner_rollout_horizon": 3,
        "mpc": True,
        "iterations": 6,
        "num_samples": 512,
        "num_elites": 64,
        "num_pi_trajs": 24,
        "min_std": 0.05,
        "max_std": 2.0,
        "temperature": 0.5,
        "log_std_min": -10,
        "log_std_max": 2,
        "entropy_coef": 1e-4,
        "num_bins": 101,
        "vmin": -10,
        "vmax": 10,
        "num_channels": 32,
        "dropout": 0.01,
        "simnorm_dim": 8,
        "compile": False,
        "wandb": True,
        "wandb_entity": "rwgao_b-brown-university",
        "wandb_project": "ambi",
        "wandb_mode": "online",
        "wandb_group": "tdmpc2-humanoid-walk-state-14m",
        "wandb_step_every": 1_000,
    }
    assert {key: params[key] for key in expected} == expected


def test_humanoid_walk_launcher_maps_three_storage_guarded_a6000_seeds():
    contents = HUMANOID_LAUNCHER_PATH.read_text()

    assert "#SBATCH --constraint=rtx_a6000" in contents
    assert "#SBATCH --gres=gpu:1" in contents
    assert "#SBATCH --time=21-00:00:00" in contents
    assert "#SBATCH --array=0-2%3" in contents
    assert "#SBATCH --output=slurm/%x-%A_%a.out" in contents
    assert "--nodelist" not in contents
    assert "environments/dmcontrol/.venv/bin/python" in contents
    assert "humanoid_walk.json" in contents
    assert "Agent decisions: 14000000" in contents
    assert "AVAILABLE_KB" in contents
    assert "MUJOCO_GL=egl" in contents
    assert "PYTHONUNBUFFERED=1" in contents
    assert 'TRIAL_INDEX="$SLURM_ARRAY_TASK_ID"' in contents
    assert "SEED=$((TRIAL_INDEX + 1))" in contents
    assert 'export TDMPC2_EVAL_CSV="$EVAL_CSV"' in contents
    assert "export WANDB_MODE=online" in contents
    assert 'export WANDB_DIR="$WANDB_LOCAL_ROOT"' in contents
    assert 'export WANDB_CACHE_DIR="$WANDB_LOCAL_ROOT/cache"' in contents
    assert "export WANDB_DISABLE_CODE=true" in contents
    assert "results/dmcontrol/tdmpc2_state/humanoid_walk" in contents
    assert 'mkdir -p "$RUN_ROOT"' not in contents
    assert "torch.cuda.device_count()" in contents
    assert "--alg-index 0" in contents
    assert '--trial-index "$TRIAL_INDEX"' in contents
    assert "--num-runs 1" in contents


def test_humanoid_walk_smoke_is_small_and_cannot_be_mistaken_for_benchmark():
    config = _load_json(HUMANOID_SMOKE_ALGORITHM_PATH)
    manifest = _load_json(HUMANOID_SMOKE_MANIFEST_PATH)
    params = config["alg_params"]

    assert manifest["study_type"] == "functional_smoke_test"
    assert "not benchmark settings" in manifest["study_note"]
    assert manifest["env_params"]["task"] == "humanoid-walk"
    assert manifest["configs"] == ["tdmpc2_humanoid_walk_state_smoke"]
    assert manifest["checkpoint_every"] == 500
    assert manifest["save_strat"] == ["latest"]
    assert config["seed"] == 1
    assert config["total_steps"] == 502
    assert params["obs"] == "state"
    assert params["model_size"] == 5
    assert params["rho"] == 0.5
    assert params["seed_steps"] == 500
    assert params["pretrain_steps"] == 1
    assert params["buffer_size"] == 2_000
    assert params["batch_size"] == 16
    assert (
        params["iterations"],
        params["num_samples"],
        params["num_elites"],
        params["num_pi_trajs"],
    ) == (2, 64, 8, 4)
