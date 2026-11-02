"""Keep the six-arm experiment fixed while enabling graph execution/telemetry.

The fingerprints below were computed from the approved configs at 557493c.
They exclude only publication fields, observational critic geometry controls,
and the two explicit graph selectors. Tests do not require Git history at run
time, so shallow checkouts retain the same scientific-contract checks.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from utils.pave_identity import (
    PAVE_COEFFICIENT_KEYS,
    resolve_critic_geometry_controls,
    resolve_pave_controls,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/dmcontrol/experiments/ambi_inner_entropy.json"
APPROVED_CONFIGS = {
    "ambi_inner_entropy_pretanh_outer_frozen": "178224ccd037c588dc429cba4bc4852c1280ac52f53004bde0d2942187a915ef",
    "ambi_inner_entropy_pretanh_outer_adapt": "43600e6a9694ea355abc1f9fabd6433ca18639811d6df7bc2adb361ada7c0e7c",
    "ambi_inner_entropy_pretanh_carry_adapt": "a232993fa68a1b3324bfbd6fa5143498b384d9b2975b593b987751d57f67a46b",
    "ambi_inner_entropy_posttanh_outer_frozen": "86174009d045503db2b600dae1039cc5a1e56ccdeb91500f38723149ae8f7958",
    "ambi_inner_entropy_posttanh_outer_adapt": "9c0d0867606013720b92d0a296f57715b1e9afbc31a15ea13f94d5cc1e1e8ff2",
    "ambi_inner_entropy_posttanh_carry_adapt": "a4606ff156f4d9847caedd6a3c6ba2b0e80e7131fef4f7de4bfc57656352796a",
}


def read_config(name):
    return json.loads((ROOT / f"configs/dmcontrol/algs/{name}.json").read_text())


def scientific_fingerprint(config):
    stable = deepcopy(config)
    stable["alg_params"] = {
        key: value
        for key, value in stable["alg_params"].items()
        if not key.startswith(("wandb", "critic_geometry_"))
        and key not in {"inner_cuda_graph", "outer_pave_cuda_graph"}
    }
    return hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def test_accelerated_manifest_keeps_all_six_cells_and_checkpoint_bank():
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["configs"] == list(APPROVED_CONFIGS)
    assert manifest["trials"] == 1
    assert manifest["env_params"] == {
        "task": "humanoid-walk", "obs": "state", "render_mode": None,
    }
    assert manifest["overrides_alg"] == {
        "seed": 55, "device": "cuda", "env": "DMControl-v0",
        "total_steps": 1_000_000, "episodes": None,
    }
    assert manifest["checkpoint_every"] == 25_000
    assert manifest["save_strat"] == "all"
    assert manifest["save_trials"] == manifest["logs"] == "none"


@pytest.mark.parametrize("name,approved_fingerprint", APPROVED_CONFIGS.items())
def test_graph_change_preserves_the_approved_scientific_recipe(name, approved_fingerprint):
    config = read_config(name)
    assert scientific_fingerprint(config) == approved_fingerprint, name
    params = config["alg_params"]
    assert (config["seed"], config["total_steps"]) == (55, 1_000_000)
    assert (config["checkpoint_every"], config["save_strat"]) == (25_000, "all")
    assert (params["inner_rollout_horizon"], params["inner_rounds"],
            params["inner_solve_interval"]) == (3, 8, 1)
    assert (params["inner_rollouts_per_round"], params["inner_batch_size"]) == (128, 256)
    assert params["inner_critic_updates_per_round"] * params["inner_rounds"] == 128
    assert params["inner_actor_updates_per_round"] * params["inner_rounds"] == 32
    assert params["inner_replay_capacity"] == 3_072
    assert params["inner_finite_horizon"] is True
    assert params["inner_bootstrap_source"] == "inner_target"
    assert params["outer_critic_target"] == params["inner_sac_critic_target"] == "reward_only"
    assert params["inner_behavior_action"] == params["inner_execution_action"] == "policy_sample"
    assert params["aux_return_mode"] == "off"


@pytest.mark.parametrize("name", APPROVED_CONFIGS)
def test_all_arms_enable_graphs_and_independent_critic_geometry(name):
    params = read_config(name)["alg_params"]
    assert params["inner_cuda_graph"] is True
    assert params["compile"] is params["compile_strict"] is True
    assert params["critic_ensemble_backend"] == "vmap"
    assert params["critic_geometry_diagnostics"] is True
    controls = resolve_critic_geometry_controls(params)
    assert controls["critic_geometry_every"] == 1_000
    assert controls["critic_geometry_batch_size"] == 64
    assert controls["critic_geometry_obs_noise_std"] == 0.01
    assert controls["critic_geometry_action_noise_std"] == 0.02
    # Observing geometry must not silently turn these baselines into PAVE runs.
    losses = resolve_pave_controls(params)
    assert all(losses[key] == 0.0 for key in PAVE_COEFFICIENT_KEYS)
    assert params["wandb"] is True and params["wandb_mode"] == "online"


@pytest.mark.parametrize("name", APPROVED_CONFIGS)
def test_entropy_and_scale_axes_remain_independent(name):
    params = read_config(name)["alg_params"]
    pretanh = "_pretanh_" in name
    entropy_mode, coefficient = (
        ("tdmpc2_scaled", 0.0001) if pretanh else ("squashed", 0.0021)
    )
    assert params["outer_actor_entropy_mode"] == params["inner_actor_entropy_mode"] == entropy_mode
    assert params["ent_coef"] == params["inner_temperature"] == coefficient
    assert params["inner_temperature_mode"] == "inherit_outer"
    assert params["sac_actor_loss_scale_mode"] == "tdmpc2_percentile_range"
    assert params["sac_actor_loss_scale_tau"] == 0.01
    expected_scope = "run" if name.endswith("carry_adapt") else "action"
    expected_update = "per_action" if name.endswith("outer_frozen") else "per_update"
    assert params["inner_actor_loss_scale_scope"] == expected_scope
    assert params["inner_actor_loss_scale_update"] == expected_update
    for field in ("inner_actor_scope", "inner_critic_scope", "inner_temperature_scope",
                  "inner_replay_scope", "inner_actor_optimizer_scope",
                  "inner_critic_optimizer_scope", "inner_temperature_optimizer_scope"):
        assert params[field] == "action"
