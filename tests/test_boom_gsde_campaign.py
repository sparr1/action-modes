"""The requested BOOM x0.01 recipe changes only gSDE collection persistence."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from tensordict import TensorDict

from RL.AMBITDMPC2 import AMBITDMPC2
from RL.tdmpc2_core.common.buffer import Buffer
from slurm.boom_gsde_oscar import MANIFEST, SCOPES, check_active_boom, parity_cases, recipe
from tests.test_ambi_boom_gsde import _agent


@pytest.mark.parametrize("index", range(3))
def test_requested_boom_gsde_config_resolves_full_budget_and_acceleration(index):
    _, manifest, run = recipe(index, MANIFEST)
    params = {**run["alg_params"], "device": "cpu", "wandb": False}
    algorithm = object.__new__(AMBITDMPC2)
    algorithm.env = SimpleNamespace(
        observation_type="state",
        observation_space=gym.spaces.Box(-np.inf, np.inf, shape=(67,), dtype=np.float32),
        action_space=gym.spaces.Box(-1., 1., shape=(21,), dtype=np.float32),
        spec=SimpleNamespace(max_episode_steps=500),
    )
    algorithm.run_params = {**run, "device": "cpu"}
    algorithm.custom_params = params
    cfg = algorithm._build_cfg(params)
    assert cfg.actor_distribution == "gsde" and cfg.gsde_feature_source == "actor"
    assert cfg.gsde_log_std_init == -3. and cfg.gsde_mean_clip == 2.
    assert cfg.inner_gsde_noise_scope == SCOPES[index]
    assert cfg.inner_gsde_resample_steps == cfg.gsde_execution_resample_steps == 8
    assert cfg.outer_actor_objective == "boom" and cfg.outer_boom_coef == .0042
    assert cfg.outer_boom_min_std == .05 and cfg.store_behavior_policy
    assert cfg.outer_boom_compile_mode == "reduce-overhead"
    assert cfg.outer_actor_entropy_mode == cfg.inner_actor_entropy_mode == "squashed"
    assert cfg.ent_coef == cfg.inner_temperature == .0021
    assert cfg.inner_temperature_mode == "inherit_outer"
    assert cfg.inner_temperature_updates_per_action == 0
    assert cfg.inner_rounds == 8 and cfg.inner_rollout_horizon == 3
    assert cfg.inner_rollouts_per_round == 128 and cfg.inner_batch_size == 256
    assert cfg.inner_actor_updates_per_action == 32 and cfg.inner_critic_updates_per_action == 128
    assert cfg.steps == 1000000 and cfg.seed == 55
    assert run["checkpoint_every"] == manifest["checkpoint_every"] == 25000
    assert cfg.outer_critic_target == cfg.inner_sac_critic_target == "reward_only"
    assert cfg.inner_actor_loss_scale_scope == "action" and cfg.inner_actor_loss_scale_update == "per_update"
    assert cfg.inner_finite_horizon and cfg.inner_solve_interval == 1
    assert cfg.compile and cfg.compile_strict and cfg.inner_cuda_graph
    assert cfg.critic_ensemble_backend == "vmap" and cfg.critic_geometry_diagnostics
    assert cfg.critic_geometry_every == 1000


def test_only_scope_and_publication_identity_differ_between_arms():
    rows = []
    for index in range(3):
        _, _, run = recipe(index, MANIFEST)
        params = deepcopy(run["alg_params"])
        for key in ("inner_gsde_noise_scope", "wandb_run_name", "wandb_tags"):
            params.pop(key)
        rows.append(params)
    assert rows[0] == rows[1] == rows[2]


@pytest.mark.parametrize("bad_tag", ["skipped", "error", "failure"])
def test_gpu_evidence_requires_four_passed_cuda_cases(tmp_path, bad_tag):
    path = tmp_path / "parity.xml"
    cases = ''.join(f'<testcase name="test_cuda_case_{i}" />' for i in range(4))
    path.write_text('<testsuites><testsuite>' + cases + '</testsuite></testsuites>')
    assert len(parity_cases(path)) == 4
    path.write_text('<testsuites><testsuite>' + cases.replace(
        'name="test_cuda_case_0" />', f'name="test_cuda_case_0"><{bad_tag} /></testcase>',
    ) + '</testsuite></testsuites>')
    with pytest.raises(AssertionError):
        parity_cases(path)


def test_active_boom_probe_uses_collected_labels_and_exercises_threshold(monkeypatch):
    agent = _agent()
    rows = 8
    valid = torch.ones(rows, dtype=torch.bool)
    valid[:2] = False  # Episode sentinel and the last random seed action.
    staged = TensorDict({
        "obs": torch.randn(rows, 3), "action": torch.zeros(rows, 1),
        "reward": torch.ones(rows), "terminated": torch.zeros(rows),
        "behavior_pre_tanh_mean": torch.full((rows, 1), .3),
        "behavior_log_std": torch.zeros(rows, 1), "behavior_policy_valid": valid,
    }, batch_size=[rows])
    learner = SimpleNamespace(agent=agent, buffer=Buffer(agent.cfg),
                              _episode_len=rows - 1, _episode_staging=staged)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    result = check_active_boom(learner)
    assert result["passed"] and result["gate_active"] == 1
    assert result["valid_fraction"] > 0
    assert result["actor_grad_norm"] > 0


def test_launcher_disables_wandb_for_validation_and_requires_clean_pinned_source():
    path = Path(__file__).resolve().parents[1] / "slurm/run_boom_gsde_oscar.sbatch"
    script = path.read_text()
    assert '"$(git rev-parse HEAD)" == "$AMBI_EXPECTED_COMMIT"' in script
    assert 'git status --porcelain --untracked-files=all' in script
    assert script.count('export WANDB_MODE=disabled') == 2
    assert 'WANDB_RESUME=never' in script
    assert '-m slurm.boom_gsde_workspace' in script
