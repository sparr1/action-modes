"""Preserve the source run when removing only BOOM's alignment contribution."""
from tests.test_ambi_root_local_sac import _build_cfg
import json
from pathlib import Path


def test_coefficient_zero_preserves_source_recipe_and_resolves():
    root = Path(__file__).resolve().parents[1]
    folder = root / "configs/dmcontrol"
    source = json.loads((folder / "algs/ambi_boom_mean_execution.json").read_text())
    target = json.loads((folder / "algs/ambi_boom_mean_no_loss.json").read_text())
    params = target["alg_params"].copy()
    source["alg_params"]["outer_boom_coef"] = 0.0
    for config in (source, target):
        config["alg_params"] = {k: v for k, v in config["alg_params"].items() if not k.startswith("wandb")}
    assert target == source
    original = json.loads((folder / "experiments/ambi_boom_mean_execution.json").read_text())
    manifest = json.loads((folder / "experiments/ambi_boom_mean_no_loss.json").read_text())
    for key in ("overrides_alg", "env_params", "trials", "logs", "save_trials",
                "checkpoint_every", "save_strat", "log_info", "log_type"):
        assert original[key] == manifest[key]
    assert manifest["configs"] == ["ambi_boom_mean_no_loss"]
    cfg = _build_cfg(**params)
    assert cfg.outer_actor_objective == "boom" and cfg.outer_boom_coef == 0.0
    assert cfg.inner_execution_action == "mean"
    assert cfg.inner_behavior_action == "policy_sample"
    assert cfg.critic_geometry_diagnostics and cfg.critic_geometry_every == 1000
    assert cfg.compile and cfg.compile_strict and cfg.inner_cuda_graph
