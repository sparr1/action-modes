import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG_PATH = (
    ROOT / "configs/dmcontrol/algs/tdmpc2_humanoid_walk_state.json"
)
ABLATION_CONFIG_PATH = (
    ROOT
    / "configs/dmcontrol/algs/tdmpc2_humanoid_walk_state_all_policy_rollouts.json"
)
MANIFEST_PATH = (
    ROOT
    / "configs/dmcontrol/experiments/tdmpc2_humanoid_walk_all_policy_rollouts.json"
)
LAUNCHER_PATH = (
    ROOT / "slurm/run_tdmpc2_humanoid_walk_all_policy_rollouts.sbatch"
)


def _load_json(path):
    return json.loads(path.read_text())


def test_ablation_changes_only_policy_proposal_count_and_wandb_metadata():
    baseline = _load_json(BASE_CONFIG_PATH)
    ablation = _load_json(ABLATION_CONFIG_PATH)
    expected = copy.deepcopy(baseline)

    expected["alg_params"]["num_pi_trajs"] = 512
    expected["alg_params"]["wandb_group"] = (
        "tdmpc2-humanoid-walk-state-14m-all-policy-rollouts"
    )
    expected["alg_params"]["wandb_tags"] = [
        "tdmpc2",
        "dmcontrol",
        "humanoid-walk",
        "state",
        "benchmark",
        "all-policy-rollouts",
        "num-pi-trajs-512",
    ]

    assert ablation == expected
    assert ablation["total_steps"] == 14_000_000
    assert ablation["alg_params"]["num_pi_trajs"] == (
        ablation["alg_params"]["num_samples"]
    )


def test_ablation_manifest_preserves_the_three_seed_baseline_protocol():
    baseline = _load_json(
        ROOT / "configs/dmcontrol/experiments/state/humanoid_walk.json"
    )
    manifest = _load_json(MANIFEST_PATH)

    for key in (
        "overrides_alg",
        "env_params",
        "trials",
        "logs",
        "save_trials",
        "checkpoint_every",
        "save_strat",
        "log_info",
        "log_type",
    ):
        assert manifest[key] == baseline[key]
    assert manifest["trials"] == 3
    assert manifest["configs"] == [
        "tdmpc2_humanoid_walk_state_all_policy_rollouts"
    ]
    assert "only scientific hyperparameter change" in manifest["study_note"]


def test_ablation_launcher_has_long_runtime_and_runtime_node_selection():
    contents = LAUNCHER_PATH.read_text()

    assert "#SBATCH --time=30-00:00:00" in contents
    assert "#SBATCH --array=0-2%3" in contents
    assert "#SBATCH --gres=gpu:1" in contents
    assert "#SBATCH --constraint" not in contents
    assert "#SBATCH --nodelist" not in contents
    assert "sbatch --nodelist=gpu2301" in contents
    assert "Agent decisions: 14000000" in contents
    assert "Policy proposal trajectories: 512 / 512" in contents
    assert "tdmpc2_humanoid_walk_all_policy_rollouts.json" in contents
    assert 'TRIAL_INDEX="$SLURM_ARRAY_TASK_ID"' in contents
    assert "SEED=$((TRIAL_INDEX + 1))" in contents
    assert "environments/dmcontrol/.venv/bin/python" in contents
    assert "MUJOCO_GL=egl" in contents
    assert "--num-runs 1" in contents


def test_outer_planner_runs_with_no_gaussian_proposal_slots():
    import gymnasium as gym
    import numpy as np

    from RL.TDMPC2 import TDMPC2Baseline

    env = gym.make("Pendulum-v1", max_episode_steps=5)
    params = {
        "device": "cpu",
        "model_size": None,
        "enc_dim": 32,
        "mlp_dim": 32,
        "latent_dim": 16,
        "num_enc_layers": 2,
        "num_q": 2,
        "simnorm_dim": 8,
        "num_bins": 11,
        "vmin": -5,
        "vmax": 5,
        "batch_size": 4,
        "train_unroll_horizon": 2,
        "outer_planning_horizon": 2,
        "inner_rollout_horizon": 2,
        "buffer_size": 100,
        "seed_steps": 4,
        "pretrain_steps": 2,
        "compile": False,
        "episodic": False,
        "discount": 0.99,
        "iterations": 2,
        "num_samples": 16,
        "num_elites": 4,
        "num_pi_trajs": 16,
        "wandb": False,
        "dropout": 0.0,
    }
    model = TDMPC2Baseline(
        "all-policy-planner-test",
        env,
        params,
        {"seed": 3, "device": "cpu", "env": "test-env", "total_steps": 1},
        {},
    )
    obs, _ = env.reset(seed=3)
    action, _ = model.predict(obs, deterministic=True, episode_start=True)

    assert action.shape == env.action_space.shape
    assert np.isfinite(action).all()
    assert model.cfg.num_pi_trajs == model.cfg.num_samples == 16
    env.close()
