"""Stand changes the environment task while preserving the source recipe."""
import json
from pathlib import Path
from tests.test_ambi_root_local_sac import _build_cfg

ROOT = Path(__file__).resolve().parents[1]

def test_source_recipe_is_preserved():
    folder = ROOT / "configs/dmcontrol"
    source = json.loads((folder / "algs/ambi_gsde_squashed_real_steps.json").read_text())
    target = json.loads((folder / "algs/ambi_gsde_squashed_real_steps_humanoid_stand.json").read_text())
    params = target["alg_params"].copy()
    for config in (source, target):
        config["alg_params"] = {k: v for k, v in config["alg_params"].items() if k not in {"wandb_group", "wandb_run_name", "wandb_tags"}}
    assert source == target
    original = json.loads((folder / "experiments/ambi_gsde_noise_scopes.json").read_text())
    manifest = json.loads((folder / "experiments/ambi_gsde_squashed_real_steps_humanoid_stand.json").read_text())
    original["env_params"]["task"] = "humanoid-stand"
    for key in ("overrides_alg", "env_params", "trials", "logs", "save_trials", "checkpoint_every", "save_strat", "log_info", "log_type"):
        assert original[key] == manifest[key], key
    assert manifest["configs"] == ["ambi_gsde_squashed_real_steps_humanoid_stand"]
    cfg = _build_cfg(**params)
    assert (cfg.inner_rollout_horizon, cfg.inner_rounds, cfg.inner_solve_interval) == (3, 8, 1)
    assert cfg.critic_geometry_diagnostics and cfg.critic_geometry_every == 1000
    assert cfg.inner_actor_loss_scale_scope == "action" and cfg.inner_actor_loss_scale_update == "per_update"
    assert cfg.compile and cfg.compile_strict and cfg.inner_cuda_graph
    assert cfg.ent_coef == cfg.inner_temperature == 0.0021
    assert cfg.inner_behavior_action == "policy_sample"
    assert cfg.inner_execution_action == "policy_sample"
    assert cfg.actor_distribution == "gsde" and cfg.inner_gsde_noise_scope == "real_steps"
    assert cfg.inner_gsde_resample_steps == cfg.gsde_execution_resample_steps == 8
