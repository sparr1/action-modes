import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALGORITHM_ROOT = ROOT / "configs/dmcontrol/algs"
EXPERIMENT_ROOT = ROOT / "configs/dmcontrol/experiments"
BASE = "ambi_humanoid_walk_base_min_all_q10"
VARIANTS = {
    "ambi_bounds": {
        "config": f"{BASE}_actor_tdmpc2_tanh_ambi_bounds",
        "log_std_min": -20,
        "run_name": (
            "AMBITDMPC2-humanoid-walk-base-v1-g4-min-all-q10-actor-"
            "tdmpc2-tanh-ambi-bounds-seed55"
        ),
        "tag": "actor-log-std-bounds-neg20-2",
    },
    "upstream_bounds": {
        "config": f"{BASE}_actor_tdmpc2_tanh_upstream_bounds",
        "log_std_min": -10,
        "run_name": (
            "AMBITDMPC2-humanoid-walk-base-v1-g4-min-all-q10-actor-"
            "tdmpc2-tanh-upstream-bounds-seed55"
        ),
        "tag": "actor-log-std-bounds-neg10-2",
    },
}


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load(path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )


def test_q10_tdmpc2_actor_configs_change_only_mapping_bounds_and_identity():
    baseline = _load(ALGORITHM_ROOT / f"{BASE}.json")
    for variant in VARIANTS.values():
        actual = _load(ALGORITHM_ROOT / f"{variant['config']}.json")
        expected = copy.deepcopy(baseline)
        expected["alg_params"].update(
            {
                "log_std_mapping": "tdmpc2_tanh",
                "log_std_min": variant["log_std_min"],
                "wandb_run_name": variant["run_name"],
                "wandb_tags": [
                    *baseline["alg_params"]["wandb_tags"],
                    "actor-log-std-tdmpc2-tanh",
                    variant["tag"],
                ],
            }
        )

        assert actual == expected
        params = actual["alg_params"]
        assert params["num_q"] == 10
        assert params["log_std_max"] == 2
        assert {
            params["outer_q_target_reduction"],
            params["outer_q_actor_reduction"],
            params["inner_q_target_reduction"],
            params["inner_q_actor_reduction"],
        } == {"min_all"}


def test_q10_tdmpc2_actor_pair_differs_only_in_bounds_and_identity():
    ambi = _load(ALGORITHM_ROOT / f"{VARIANTS['ambi_bounds']['config']}.json")
    upstream = _load(
        ALGORITHM_ROOT / f"{VARIANTS['upstream_bounds']['config']}.json"
    )
    ambi_params = copy.deepcopy(ambi["alg_params"])
    upstream_params = copy.deepcopy(upstream["alg_params"])
    for params in (ambi_params, upstream_params):
        params.pop("log_std_min")
        params.pop("wandb_run_name")
        params.pop("wandb_tags")

    assert ambi_params == upstream_params
    assert ambi["alg_params"]["log_std_min"] == -20
    assert upstream["alg_params"]["log_std_min"] == -10


def test_q10_tdmpc2_actor_manifests_are_isolated_single_seed_cells():
    for variant in VARIANTS.values():
        manifest = _load(EXPERIMENT_ROOT / f"{variant['config']}.json")
        assert manifest["study_type"] == (
            "single_seed_exploratory_tdmpc2_actor_bounds_ablation"
        )
        assert manifest["configs"] == [variant["config"]]
        assert manifest["trials"] == 1
        assert manifest["overrides_alg"]["seed"] == 55
        assert manifest["overrides_alg"]["total_steps"] == 14_000_000
        assert manifest["env_params"] == {
            "task": "humanoid-walk",
            "obs": "state",
            "render_mode": None,
        }
        assert manifest["logs"] == "timestamp"
        assert manifest["checkpoint_every"] == 100_000
