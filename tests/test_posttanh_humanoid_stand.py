"""Humanoid Stand changes the source experiment's task without changing its learner."""
import json
from pathlib import Path

from tests.test_ambi_root_local_sac import _build_cfg

ROOT = Path(__file__).resolve().parents[1]


def test_only_task_and_publication_metadata_change():
    folder = ROOT / "configs/dmcontrol"
    source = json.loads((folder / "algs/ambi_inner_entropy_posttanh_outer_adapt.json").read_text())
    target = json.loads((folder / "algs/ambi_posttanh_outer_adapt_humanoid_stand.json").read_text())
    params = target["alg_params"].copy()
    for config in (source, target):
        config["alg_params"] = {k: v for k, v in config["alg_params"].items() if not k.startswith("wandb")}
    assert target == source
    original = json.loads((folder / "experiments/ambi_inner_entropy.json").read_text())
    manifest = json.loads((folder / "experiments/ambi_posttanh_outer_adapt_humanoid_stand.json").read_text())
    original["env_params"]["task"] = "humanoid-stand"
    for key in ("overrides_alg", "env_params", "trials", "logs", "save_trials",
                "checkpoint_every", "save_strat", "log_info", "log_type"):
        assert original[key] == manifest[key]
    assert manifest["configs"] == ["ambi_posttanh_outer_adapt_humanoid_stand"]
    cfg = _build_cfg(**params)
    assert cfg.inner_execution_action == "policy_sample"
    assert cfg.inner_actor_loss_scale_scope == "action"
    assert cfg.inner_actor_loss_scale_update == "per_update"
    assert cfg.critic_geometry_diagnostics and cfg.critic_geometry_every == 1000
    assert cfg.compile and cfg.compile_strict and cfg.inner_cuda_graph
