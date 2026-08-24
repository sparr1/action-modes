import json
import stat
import subprocess
from pathlib import Path

import gymnasium as gym

from RL.AMBIXQC import AMBIXQC


ROOT = Path(__file__).resolve().parents[1]
ALGORITHM_PATH = ROOT / "configs/dmcontrol/algs/ambixqc_humanoid_walk_state.json"
MANIFEST_PATH = (
    ROOT / "configs/dmcontrol/experiments/ambixqc_humanoid_walk_state.json"
)
LAUNCHER_PATH = ROOT / "slurm/run_ambixqc_humanoid_walk_hydra.sbatch"
SUBMITTER_PATH = ROOT / "slurm/submit_ambixqc_humanoid_walk_hydra.sh"


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )


def test_full_humanoid_config_freezes_the_first_gpu_screen():
    config = _load_json(ALGORITHM_PATH)
    params = config["alg_params"]

    assert config == {
        **config,
        "seed": 55,
        "env": "DMControl-v0",
        "alg": "AMBIXQC/AMBIXQC",
        "device": "cuda",
        "total_steps": 1_000_000,
        "episodes": None,
    }
    assert params["obs"] == "state"
    assert params["model_size"] == 5
    assert params["buffer_size"] == 1_000_000
    assert params["batch_size"] == 256
    assert params["seed_steps"] == params["pretrain_steps"] == 2_500
    assert params["utd"] == 1
    assert params["eval_freq"] == 100_000
    assert params["eval_episodes"] == 10
    assert params["compile"] is params["compile_strict"] is False

    assert params["train_unroll_horizon"] == 3
    assert params["inner_rollout_horizon"] == 3
    assert params["temporal_loss_normalization"] == "reference_weighted_mean"
    assert params["temporal_loss_reference_horizon"] == 3
    assert params["rho"] == 0.5
    assert params["reward_coef"] == params["value_coef"] == 0.1
    assert params["consistency_coef"] == 20.0

    assert params["xqc_actor_net_arch"] == [256] * 4
    assert params["xqc_critic_net_arch"] == [512] * 4
    assert params["xqc_num_atoms"] == 101
    assert (params["xqc_vmin"], params["xqc_vmax"]) == (-5.0, 5.0)
    assert params["xqc_policy_delay"] == 3
    assert params["xqc_reward_normalization"] is True
    assert params["mpc"] is False

    assert (
        params["inner_rounds"],
        params["inner_rollouts_per_round"],
        params["inner_rollout_horizon"],
        params["inner_updates_per_round"],
    ) == (2, 32, 3, 4)
    assert params["inner_replay_capacity"] == 2 * 32 * 3
    assert params["inner_batch_size"] == 64
    assert params["inner_replay_sampling"] == "with_replacement"

    assert params["wandb"] is True
    assert params["wandb_mode"] == "online"
    assert params["wandb_entity"] == "rwgao_b-brown-university"
    assert params["wandb_project"] == "ambi"
    assert params["wandb_group"] == "ambixqc-humanoid-walk-state-1m"
    assert params["inner_diagnostics_every"] == params["wandb_step_every"] == 1000
    assert "wandb_run_name" not in params
    assert "1m-decisions" in params["wandb_tags"]


def test_full_humanoid_config_resolves_through_the_real_wrapper():
    config = _load_json(ALGORITHM_PATH)
    algorithm = object.__new__(AMBIXQC)
    algorithm.env = gym.make("Pendulum-v1")
    algorithm.run_params = {
        "seed": config["seed"],
        "device": "cpu",
        "env": config["env"],
        "total_steps": config["total_steps"],
    }
    algorithm.experiment_params = {}
    algorithm.custom_params = config["alg_params"]
    try:
        cfg = algorithm._build_cfg(
            {"device": "cpu", **config["alg_params"]}
        )
    finally:
        algorithm.env.close()

    assert cfg.steps == cfg.xqc_lr_transition_steps == 1_000_000
    assert cfg.inner_model_step_budget == 192
    assert cfg.inner_expected_update_slots == 8
    assert cfg.inner_actor_updates_per_action == 3
    assert cfg.inner_temperature_updates_per_action == 3
    assert cfg.compile is cfg.compile_strict is False


def test_manifest_is_one_seed_exploratory_and_keeps_portable_checkpoints():
    manifest = _load_json(MANIFEST_PATH)

    assert manifest["study_type"] == "ambixqc_humanoid_walk_single_seed_exploratory"
    assert "first full GPU execution" in manifest["study_note"]
    assert "not a statistical or paper reproduction" in manifest["study_note"]
    assert "Exact trainer resume is not supported" in manifest["study_note"]
    assert manifest["overrides_alg"] == {"env": "DMControl-v0"}
    assert manifest["env_params"] == {
        "task": "humanoid-walk",
        "obs": "state",
        "render_mode": None,
    }
    assert manifest["trials"] == 1
    assert manifest["configs"] == ["ambixqc_humanoid_walk_state"]
    assert manifest["logs"] == "none"
    assert manifest["save_trials"] == "none"
    assert manifest["checkpoint_every"] == 100_000
    assert manifest["save_strat"] == ["best", "latest"]
    assert manifest["checkpoint_best_window"] == 100


def test_launcher_is_one_unpinned_long_hydra_gpu_job():
    contents = LAUNCHER_PATH.read_text(encoding="utf-8")

    assert "#SBATCH --partition=gpus" in contents
    assert "#SBATCH --gres=gpu:1" in contents
    assert "#SBATCH --cpus-per-task=8" in contents
    assert "#SBATCH --mem=48G" in contents
    assert "#SBATCH --time=30-00:00:00" in contents
    assert "#SBATCH --nodes=1" in contents
    assert "#SBATCH --ntasks=1" in contents
    assert "#SBATCH --array" not in contents
    assert "#SBATCH --nodelist" not in contents
    assert "#SBATCH -w" not in contents

    assert "environments/dmcontrol/.venv/bin/python" in contents
    assert 'readonly PYTHON="${AMBIXQC_PYTHON:-$DEFAULT_PYTHON}"' in contents
    assert "AMBIXQC_PYTHON must resolve to an absolute path" in contents
    assert "uv run" not in contents
    assert "conda activate" not in contents
    assert "export MUJOCO_GL=egl" in contents
    assert "export WANDB_MODE=online" in contents
    assert 'export TDMPC2_EVAL_CSV="$EVAL_CSV"' in contents
    assert "ambixqc_humanoid_walk_state.json" in contents
    assert '--trial-index 0' in contents
    assert "--num-runs 1" in contents
    assert "Agent decisions: 1000000" in contents
    assert "Inner schedule: J2/N32/H3/G4" in contents


def test_launcher_guards_exact_source_lock_artifacts_and_outputs():
    contents = LAUNCHER_PATH.read_text(encoding="utf-8")

    assert "EXPECTED_ACTION_MODES_SHA" in contents
    assert "^[0-9a-f]{40}$" in contents
    assert "status --porcelain=v1 --untracked-files=all" in contents
    assert contents.count(
        'require_clean_sha "$PROJECT_DIR_REAL" "$EXPECTED_ACTION_MODES_SHA"'
    ) == 2
    assert "f123ba99aadde092401c0e912dbeb88994f00ae420680c69c18003965485efe6" in contents
    assert 'require_lock_hash "$PROJECT_DIR_REAL/environments/dmcontrol/uv.lock"' in contents
    assert "Durable results must be outside the Git checkout" in contents
    assert 'if [[ -e "$RUN_ROOT" || -L "$RUN_ROOT" ]]' in contents
    assert "Refusing to overwrite an existing run directory" in contents
    assert 'if [[ -e "$JOB_SCRATCH" || -L "$JOB_SCRATCH" ]]' in contents
    assert 'exec > >(tee "$RUN_ROOT/job.log") 2>&1' in contents
    assert "rm -rf" not in contents

    assert 'expected_steps = [0, *range(100_000, 1_000_001, 100_000)]' in contents
    assert 'int(row["seed"]) != 55' in contents
    assert "math.isfinite" in contents
    assert "before collecting a real transition" in contents

    assert 'export WANDB_RUN_ID="$WANDB_UNIQUE_ID"' in contents
    assert "export WANDB_RESUME=never" in contents
    assert 'config["alg_params"]["wandb_run_name"] = run_name' in contents
    assert "job$SLURM_JOB_ID" in contents
    assert '--alg-dir "$SCRATCH_ALG_DIR"' in contents


def test_submitter_prefers_gpu2301_then_gpu2201_and_submits_one_job():
    contents = SUBMITTER_PATH.read_text(encoding="utf-8")

    assert 'readonly PREFERRED_NODE="gpu2301"' in contents
    assert 'readonly FALLBACK_NODE="gpu2201"' in contents
    assert 'scontrol show node -o "$node"' in contents
    assert "preferred_slots >= 1" in contents
    assert "fallback_slots >= 1" in contents
    assert 'readonly SELECTED_NODE="$PREFERRED_NODE"' in contents
    assert 'readonly SELECTED_NODE="$FALLBACK_NODE"' in contents
    assert '--nodelist="$SELECTED_NODE"' in contents
    assert "sbatch" in contents
    assert "--parsable" in contents
    assert '--export="$export_spec"' in contents
    assert "EXPECTED_ACTION_MODES_SHA=$EXPECTED_ACTION_MODES_SHA" in contents
    assert "AMBIXQC_RESULTS_ROOT=$AMBIXQC_RESULTS_ROOT" in contents
    assert "AMBIXQC_RUN_ID=$AMBIXQC_RUN_ID" in contents
    assert "AMBIXQC_PYTHON=$AMBIXQC_PYTHON" in contents
    assert "--python)" in contents
    assert "--python must be an executable at an absolute path" in contents
    assert "--dry-run" in contents
    assert "scancel" not in contents

    first_sbatch = contents.index("  sbatch\n")
    assert contents.index('require_clean_sha "$PROJECT_DIR"') < first_sbatch
    assert contents.index('if [[ -e "$STUDY_ROOT"') < first_sbatch


def test_launcher_and_submitter_are_executable_valid_bash():
    for path in (LAUNCHER_PATH, SUBMITTER_PATH):
        assert path.stat().st_mode & stat.S_IXUSR
        subprocess.run(["bash", "-n", str(path)], check=True)
