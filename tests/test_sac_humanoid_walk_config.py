import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALGORITHM_PATH = (
    ROOT / "configs/dmcontrol/algs/sac_humanoid_walk_tdmpc_table5.json"
)
MANIFEST_PATH = (
    ROOT
    / "configs/dmcontrol/experiments/sac_humanoid_walk_tdmpc_table5.json"
)
LAUNCHER_PATH = ROOT / "slurm/run_sac_humanoid_walk_tdmpc_table5.sbatch"


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


def test_humanoid_sac_freezes_the_expressible_tdmpc_table5_hyperparameters():
    config = _load_json(ALGORITHM_PATH)
    params = config["alg_params"]

    assert config == {
        "seed": 1,
        "env": "DMControl-v0",
        "alg": "SAC/SAC",
        "device": "cuda",
        "alg_params": params,
        "total_steps": 7_000_000,
        "episodes": None,
    }
    assert "seed" not in params
    assert params == {
        "verbose": 1,
        "buffer_size": 7_000_000,
        "learning_starts": 1_000,
        "batch_size": 512,
        "train_freq": 1,
        "gradient_steps": 1,
        "learning_rate": 1e-3,
        "actor_lr": 1e-3,
        "critic_lr": 1e-3,
        "alpha_lr": 1e-4,
        "actor_betas": [0.9, 0.999],
        "critic_betas": [0.9, 0.999],
        "alpha_betas": [0.5, 0.999],
        "gamma": 0.99,
        "tau": 0.01,
        "ent_coef": "auto_0.1",
        "target_entropy": "auto",
        "target_update_interval": 2,
        "net_arch": [1024, 1024],
        "log_std_min": -5,
        "log_std_max": 2,
        "q_representation": "scalar",
        "num_q": 2,
        "q_pair_size": 2,
        "q_target_reduction": "min_pair",
        "q_actor_reduction": "min_pair",
        "eval_freq": 50_000,
        "eval_episodes": 10,
        "wandb": True,
        "wandb_entity": "rwgao_b-brown-university",
        "wandb_project": "ambi",
        "wandb_mode": "online",
        "wandb_group": "sac-humanoid-walk-state-tdmpc-table5-hparams-7m",
        "wandb_step_every": 1_000,
        "wandb_tags": [
            "sac",
            "native-sac",
            "dmcontrol",
            "humanoid-walk",
            "state",
            "tdmpc-table5-hparams",
            "direct-state",
            "sb3-like-equations",
        ],
    }


def test_manifest_maps_three_seeds_and_disables_large_local_artifacts():
    config = _load_json(ALGORITHM_PATH)
    manifest = _load_json(MANIFEST_PATH)

    assert manifest["study_type"] == (
        "native_sac_humanoid_walk_tdmpc_table5_benchmark"
    )
    assert "14 million raw simulator steps" in manifest["study_note"]
    assert "not a byte-for-byte Yarats SAC reproduction" in manifest["study_note"]
    assert "latent-100 encoder" in manifest["study_note"]
    assert "SB3-like" in manifest["study_note"]
    assert "separate W&B runs in one group" in manifest["study_note"]
    assert "not a Table-5 training hyperparameter" in manifest["study_note"]
    assert manifest["overrides_alg"] == {"env": "DMControl-v0"}
    assert manifest["env_params"] == {
        "task": "humanoid-walk",
        "obs": "state",
        "render_mode": None,
    }
    assert manifest["trials"] == 3
    assert [config["seed"] + index for index in range(manifest["trials"])] == [
        1,
        2,
        3,
    ]
    assert manifest["configs"] == ["sac_humanoid_walk_tdmpc_table5"]
    assert manifest["logs"] == "none"
    assert manifest["save_trials"] == "none"
    assert manifest["checkpoint_every"] is None
    assert manifest["save_strat"] == "none"
    assert manifest["log_info"] is False
    assert manifest["log_type"] == "summary"


def test_recipe_uses_the_dmcontrol_repeat_and_native_relu_network():
    from torch import nn

    from RL.sac_core import make_feature_mlp
    from domains.dmcontrol import _ACTION_REPEAT

    config = _load_json(ALGORITHM_PATH)
    architecture = config["alg_params"]["net_arch"]
    network = make_feature_mlp(3, architecture)

    assert _ACTION_REPEAT == 2
    assert config["total_steps"] * _ACTION_REPEAT == 14_000_000
    assert [type(layer) for layer in network] == [
        nn.Linear,
        nn.ReLU,
        nn.Linear,
        nn.ReLU,
    ]


def test_hydra_launcher_is_a_storage_guarded_serial_a6000_array():
    contents = LAUNCHER_PATH.read_text()

    assert "#SBATCH --constraint=rtx_a6000" in contents
    assert "#SBATCH --gres=gpu:1" in contents
    assert "#SBATCH --mem=32G" in contents
    assert "#SBATCH --time=21-00:00:00" in contents
    assert "#SBATCH --array=0-2%1" in contents
    assert "#SBATCH --output=slurm/%x-%A_%a.out" in contents
    assert "--nodelist" not in contents
    assert "AMBI_DMCONTROL_PYTHON" in contents
    assert "environments/dmcontrol/.venv/bin/python" in contents
    assert "uv run" not in contents
    assert "conda activate" not in contents
    assert "AVAILABLE_KB" in contents
    assert "export MUJOCO_GL=egl" in contents
    assert 'export SAC_EVAL_CSV="$EVAL_CSV"' in contents
    assert "export WANDB_MODE=online" in contents
    assert 'export WANDB_DIR="$WANDB_LOCAL_ROOT"' in contents
    assert 'export WANDB_CACHE_DIR="$WANDB_LOCAL_ROOT/cache"' in contents
    assert 'export WANDB_DATA_DIR="$WANDB_LOCAL_ROOT/data"' in contents
    assert 'export WANDB_ARTIFACT_DIR="$WANDB_LOCAL_ROOT/artifacts"' in contents
    assert "export WANDB_DISABLE_CODE=true" in contents
    assert "sac_humanoid_walk_tdmpc_table5.json" in contents
    assert "Agent decisions: 7000000" in contents
    assert "Raw simulator steps: 14000000" in contents
    assert "Replay capacity: 7000000" in contents
    assert 'TRIAL_INDEX="$SLURM_ARRAY_TASK_ID"' in contents
    assert "SEED=$((TRIAL_INDEX + 1))" in contents
    assert "results/dmcontrol/sac_tdmpc_table5/humanoid_walk" in contents
    assert 'mkdir -p "$RUN_ROOT"' not in contents
    assert 'srun "$PYTHON" main.py' in contents
    assert "--alg-index 0" in contents
    assert '--trial-index "$TRIAL_INDEX"' in contents
    assert "--num-runs 1" in contents
    assert "--resume" not in contents
