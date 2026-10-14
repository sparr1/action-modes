"""Keep the launch matrix paired and its complete checkpoint banks explicit."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from tests.test_ambi_root_local_sac import _build_cfg

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "configs/dmcontrol/experiments/ambi_transfer_training.json").read_text())
NAMES = MANIFEST["configs"]


@pytest.mark.parametrize("index,name", enumerate(NAMES))
def test_full_training_science_and_checkpoint_contract(index, name):
    config = json.loads((ROOT / "configs/dmcontrol/algs" / (name + ".json")).read_text())
    assert config["seed"] == 55 and config["total_steps"] == 500000
    assert config["checkpoint_every"] == MANIFEST["checkpoint_every"] == 25000
    assert config["save_strat"] == MANIFEST["save_strat"] == "all"
    assert "alg_params" not in MANIFEST["overrides_alg"]
    params = config["alg_params"]
    cfg = _build_cfg(**params)
    assert cfg.inner_rollout_horizon == 1 + index // 2
    assert cfg.inner_critic_bernoulli_p == (0. if index % 2 == 0 else .5)
    assert (cfg.inner_rounds, cfg.inner_rollouts_per_round, cfg.inner_batch_size) == (6, 128, 256)
    assert (cfg.inner_critic_updates_per_action, cfg.inner_actor_updates_per_action) == (96, 24)
    assert cfg.inner_replay_capacity == 3072
    assert cfg.inner_finite_horizon and cfg.inner_sac_critic_target == "reward_only"
    assert cfg.inner_critic_source == cfg.inner_horizon_critic_source == "aux_return"
    assert cfg.inner_actor_source == cfg.inner_horizon_actor_source == "sac"
    assert cfg.aux_return_mode == "sac" and not cfg.aux_return_detach_representation
    assert cfg.critic_ensemble_backend == "vmap" and cfg.compile and cfg.compile_strict
    assert cfg.inner_temperature_mode == "auto" and cfg.inner_entropy_enabled
    assert cfg.inner_execution_action == "policy_sample"
    assert cfg.seed_steps == cfg.pretrain_steps == 2500 and cfg.utd == 1
    assert params["target_entropy"] == -10.5
    assert params["wandb_project"] == "ambi"
    assert not params.get("pretrained_model_path")


def test_only_horizon_and_transfer_change_between_the_six_cells():
    assert len(NAMES) == len(set(NAMES)) == 6
    normalized = []
    for name in NAMES:
        config = json.loads((ROOT / "configs/dmcontrol/algs" / (name + ".json")).read_text())
        params = deepcopy(config["alg_params"])
        for key in ("inner_rollout_horizon", "inner_critic_bernoulli_p", "wandb_run_name", "wandb_tags"):
            params.pop(key)
        normalized.append(params)
    assert all(row == normalized[0] for row in normalized)


def test_launcher_requires_clean_pinned_source_and_successful_smoke():
    launcher = (ROOT / "slurm/run_ambi_transfer_training_hydra.sbatch").read_text()
    helper = (ROOT / "slurm/ambi_transfer_training.py").read_text()
    assert "git status --porcelain --untracked-files=all" in launcher
    assert "AMBI_EXPECTED_COMMIT" in launcher and "--no-requeue" in launcher
    assert "gate/receipt.json" in launcher
    assert 'receipt["binding"] == binding()' in helper
    assert 'output.mkdir(parents=True, exist_ok=False)' in helper
