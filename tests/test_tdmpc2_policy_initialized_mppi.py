import copy
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG_PATH = (
    ROOT / "configs/dmcontrol/algs/tdmpc2_humanoid_walk_state.json"
)
ABLATION_CONFIG_PATH = (
    ROOT
    / "configs/dmcontrol/algs/"
    "tdmpc2_humanoid_walk_state_mppi_policy_init512_gaussian_refine.json"
)
MANIFEST_PATH = (
    ROOT
    / "configs/dmcontrol/experiments/"
    "tdmpc2_humanoid_walk_mppi_policy_init512_gaussian_refine.json"
)
LAUNCHER_PATH = (
    ROOT
    / "slurm/"
    "run_tdmpc2_humanoid_walk_mppi_policy_init512_gaussian_refine.sbatch"
)


def _load_json(path):
    return json.loads(path.read_text())


def _make_tiny_model(*, num_pi_trajs, first_iteration_only=None):
    import gymnasium as gym

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
        "num_pi_trajs": num_pi_trajs,
        "wandb": False,
        "dropout": 0.0,
    }
    if first_iteration_only is not None:
        params["num_pi_trajs_first_iteration_only"] = first_iteration_only
    model = TDMPC2Baseline(
        "policy-initialized-mppi-test",
        env,
        params,
        {"seed": 3, "device": "cpu", "env": "test-env", "total_steps": 1},
        {},
    )
    return model, env


def test_ablation_changes_only_the_policy_proposal_schedule_and_wandb_metadata():
    baseline = _load_json(BASE_CONFIG_PATH)
    ablation = _load_json(ABLATION_CONFIG_PATH)
    expected = copy.deepcopy(baseline)

    expected["alg_params"].update(
        {
            "num_pi_trajs": 512,
            "num_pi_trajs_first_iteration_only": True,
            "wandb_group": (
                "tdmpc2-humanoid-walk-state-14m-"
                "policy-init512-gaussian-refine"
            ),
            "wandb_tags": [
                "tdmpc2",
                "dmcontrol",
                "humanoid-walk",
                "state",
                "benchmark",
                "policy-init-512-gaussian-refine",
                "first-population-all-policy-prior-candidates",
            ],
        }
    )

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
        "tdmpc2_humanoid_walk_state_mppi_policy_init512_gaussian_refine"
    ]
    assert "iteration 0 contains 512 stochastic policy-prior candidate" in (
        manifest["study_note"]
    )
    assert "iterations 1-7 each draw all 512 proposals" in manifest["study_note"]
    assert "first 2,500 random seed-collection decisions" in manifest["study_note"]
    assert "Q(z_H, pi(z_H))" in manifest["study_note"]
    assert "13,312 per invocation (+7.9%)" in manifest["study_note"]


def test_ablation_launcher_has_long_runtime_and_explicit_schedule():
    contents = LAUNCHER_PATH.read_text()

    assert "#SBATCH --time=30-00:00:00" in contents
    assert "#SBATCH --array=0-2%3" in contents
    assert "#SBATCH --gres=gpu:1" in contents
    assert "#SBATCH --constraint" not in contents
    assert "#SBATCH --nodelist" not in contents
    assert "sbatch --nodelist=gpu2301" in contents
    assert "Agent decisions: 14000000" in contents
    assert (
        "First population: 512 policy-prior / 0 Gaussian candidate trajectories"
        in contents
    )
    assert (
        "Refinement populations: 0 policy-prior / 512 Gaussian candidate "
        "trajectories" in contents
    )
    assert "tdmpc2_humanoid_walk_mppi_policy_init512_gaussian_refine.json" in contents
    assert 'TRIAL_INDEX="$SLURM_ARRAY_TASK_ID"' in contents
    assert "SEED=$((TRIAL_INDEX + 1))" in contents
    assert "environments/dmcontrol/.venv/bin/python" in contents
    assert "MUJOCO_GL=egl" in contents
    assert "--num-runs 1" in contents


@pytest.mark.parametrize(
    ("num_pi_trajs", "first_iteration_only", "expected_proposal_shapes"),
    [
        (16, True, [(2, 0, 1), (2, 16, 1)]),
        (4, None, [(2, 12, 1), (2, 12, 1)]),
    ],
    ids=("policy-initialization", "upstream-default-schedule"),
)
def test_planner_uses_policy_only_for_initialization_when_requested(
    monkeypatch,
    num_pi_trajs,
    first_iteration_only,
    expected_proposal_shapes,
):
    import numpy as np
    import torch

    model, env = _make_tiny_model(
        num_pi_trajs=num_pi_trajs,
        first_iteration_only=first_iteration_only,
    )
    original_randn = torch.randn
    proposal_shapes = []

    def tracked_randn(*shape, **kwargs):
        if len(shape) == 3:
            proposal_shapes.append(tuple(shape))
        return original_randn(*shape, **kwargs)

    monkeypatch.setattr(torch, "randn", tracked_randn)
    obs, _ = env.reset(seed=3)
    action, _ = model.predict(obs, deterministic=True, episode_start=True)

    assert action.shape == env.action_space.shape
    assert np.isfinite(action).all()
    assert proposal_shapes == expected_proposal_shapes
    assert model.cfg.num_pi_trajs_first_iteration_only is bool(
        first_iteration_only
    )
    env.close()


@pytest.mark.parametrize("invalid", [0, 1, "true", None])
def test_first_iteration_schedule_requires_a_strict_boolean(invalid):
    import gymnasium as gym

    from RL.TDMPC2 import TDMPC2Baseline

    env = gym.make("Pendulum-v1", max_episode_steps=5)
    with pytest.raises(
        ValueError,
        match="num_pi_trajs_first_iteration_only must be a boolean",
    ):
        TDMPC2Baseline(
            "invalid-policy-initialization-test",
            env,
            {
                "device": "cpu",
                "num_pi_trajs_first_iteration_only": invalid,
            },
            {"seed": 3, "device": "cpu", "env": "test-env", "total_steps": 1},
            {},
        )
    env.close()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {
                "mpc": False,
                "num_pi_trajs": 4,
                "num_pi_trajs_first_iteration_only": True,
            },
            "requires mpc=true",
        ),
        (
            {
                "num_pi_trajs": 0,
                "num_pi_trajs_first_iteration_only": True,
            },
            "requires num_pi_trajs>0",
        ),
        (
            {
                "iterations": 1,
                "num_pi_trajs": 4,
                "num_pi_trajs_first_iteration_only": True,
            },
            "requires at least two effective planning iterations",
        ),
        ({"mpc": "true"}, "mpc must be a boolean"),
    ],
    ids=("mpc-disabled", "no-policy-trajectories", "no-refinement", "mpc-type"),
)
def test_policy_initialization_schedule_rejects_incoherent_planners(
    overrides,
    message,
):
    import gymnasium as gym

    from RL.TDMPC2 import TDMPC2Baseline

    env = gym.make("Pendulum-v1", max_episode_steps=5)
    with pytest.raises(ValueError, match=message):
        TDMPC2Baseline(
            "invalid-policy-initialization-planner-test",
            env,
            {"device": "cpu", **overrides},
            {"seed": 3, "device": "cpu", "env": "test-env", "total_steps": 1},
            {},
        )
    env.close()
