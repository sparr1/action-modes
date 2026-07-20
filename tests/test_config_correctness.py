import json
from pathlib import Path

import gymnasium as gym

from RL.TDMPC2 import TDMPC2Baseline
from utils.ambi_end_to_end import load_suite, render_condition_configs


ROOT = Path(__file__).resolve().parents[1]
AMBI_ROOT = ROOT / "configs/ambi"
AMBI_ALGS = AMBI_ROOT / "algs"
SUITE = AMBI_ROOT / "research/trial_matrix.json"


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json_strict(path):
    return json.loads(path.read_text(), object_pairs_hook=_reject_duplicate_keys)


def test_all_configs_are_valid_json_without_duplicate_keys():
    for path in sorted((ROOT / "configs").rglob("*.json")):
        _load_json_strict(path)


def test_baseline_comparison_is_explicit_single_seed_three_way_with_checkpoints():
    experiment = _load_json_strict(
        AMBI_ROOT / "experiments/canonical/baseline_comparison.json"
    )
    assert experiment["study_type"] == "single_seed_exploratory_screening"
    assert "no confidence" in experiment["study_note"].lower()
    assert experiment["overrides_alg"] == {
        "seed": 55,
        "device": "cuda",
        "env": "Ant-v4",
        "total_steps": 1_000_000,
        "episodes": None,
    }
    assert experiment["env_params"] == {
        "render_mode": None,
        "exclude_current_positions_from_observation": True,
        "max_episode_steps": 1000,
        "terminate_when_unhealthy": False,
    }
    assert experiment["trials"] == 1
    assert experiment["configs"] == [
        "native_sac_distributional_twin_q",
        "tdmpc2_baseline",
        "ambi_anchor",
    ]
    assert experiment["checkpoint_every"] == 100_000
    assert experiment["save_strat"] == ["best", "latest"]
    assert experiment["checkpoint_best_window"] == 100
    assert experiment["save_trials"] == "none"


def test_baseline_algorithms_make_q_and_horizon_contracts_visible():
    native = _load_json_strict(AMBI_ALGS / "native_sac_distributional_twin_q.json")
    native_params = native["alg_params"]
    assert native["alg"] == "SAC/SAC"
    assert native_params["q_representation"] == "distributional"
    assert native_params["num_q"] == 2
    assert native_params["q_pair_size"] == 2
    assert native_params["q_target_reduction"] == "min_pair"
    assert native_params["q_actor_reduction"] == "min_pair"
    assert (native_params["q_num_bins"], native_params["q_vmin"], native_params["q_vmax"]) == (
        101,
        -10,
        10,
    )

    tdmpc = _load_json_strict(AMBI_ALGS / "tdmpc2_baseline.json")["alg_params"]
    assert tdmpc["model_size"] == 5
    assert "num_q" not in tdmpc
    assert tdmpc["q_pair_size"] == 2
    assert tdmpc["train_unroll_horizon"] == 3
    assert tdmpc["outer_planning_horizon"] == 3
    assert tdmpc["inner_rollout_horizon"] == 3
    assert tdmpc["temporal_loss_normalization"] == "reference_weighted_mean"
    assert tdmpc["temporal_loss_reference_horizon"] == 3
    assert tdmpc["rho"] == 0.7

    ambi = _load_json_strict(AMBI_ALGS / "ambi_anchor.json")[
        "alg_params"
    ]
    assert ambi["mpc"] is False
    assert ambi["model_size"] == 5
    assert "num_q" not in ambi
    assert ambi["q_pair_size"] == 2
    assert ambi["outer_q_target_reduction"] == "min_pair"
    assert ambi["outer_q_actor_reduction"] == "min_pair"
    assert ambi["inner_q_target_reduction"] == "min_pair"
    assert ambi["inner_q_actor_reduction"] == "min_pair"
    assert ambi["inner_rounds"] == 4
    assert ambi["inner_rollouts_per_round"] == 64
    assert ambi["inner_rollout_horizon"] == 3
    assert ambi["inner_updates_per_round"] == "auto"
    assert ambi["inner_batch_size"] == 64
    assert ambi["inner_replay_capacity"] == 768


def test_all_full_manifests_are_one_seed_one_million_and_resolve_flat_configs():
    manifests = sorted((AMBI_ROOT / "experiments").rglob("*.json"))
    manifests = [path for path in manifests if "smoke" not in path.parts]
    assert manifests
    for path in manifests:
        experiment = _load_json_strict(path)
        assert experiment["study_type"] == "single_seed_exploratory_screening", path
        assert experiment["overrides_alg"]["seed"] == 55, path
        assert experiment["overrides_alg"]["total_steps"] == 1_000_000, path
        assert experiment["trials"] == 1, path
        assert experiment["checkpoint_every"] == 100_000, path
        assert experiment["save_strat"] == ["best", "latest"], path
        for name in experiment["configs"]:
            assert (AMBI_ALGS / f"{name}.json").is_file(), (path, name)


def test_end_to_end_matrix_is_the_deterministic_source_for_flat_ambi_configs():
    rendered = render_condition_configs(SUITE)
    assert len(rendered) == 21
    for filename, expected in rendered.items():
        assert (AMBI_ALGS / filename).read_text() == expected, filename


def test_every_declared_trial_budget_matches_its_ready_to_run_config():
    matrix = load_suite(SUITE)
    for condition in matrix["conditions"]:
        params = _load_json_strict(AMBI_ALGS / f"{condition['config']}.json")[
            "alg_params"
        ]
        assert params["train_unroll_horizon"] == condition["train"]
        assert params["outer_planning_horizon"] == condition["plan"]
        assert params["inner_rollout_horizon"] == condition["inner"]
        assert params["inner_rounds"] == condition["rounds"]
        assert params["inner_rollouts_per_round"] == condition["rollouts_per_round"]
        assert params["inner_updates_per_round"] == condition["updates_per_round"]
        assert params["inner_batch_size"] == condition["batch"]
        assert params["inner_replay_capacity"] == condition["replay_capacity"]
        assert condition["transitions_per_round"] == condition["inner"] * condition[
            "rollouts_per_round"
        ]
        assert condition["transitions_per_action"] == condition["rounds"] * condition[
            "transitions_per_round"
        ]
        assert condition["updates_per_action"] == condition["rounds"] * condition[
            "updates_per_round"
        ]
        assert condition["replay_rows_per_action"] == condition["updates_per_action"] * params[
            "inner_batch_size"
        ]
        assert condition["replay_capacity"] == condition["transitions_per_action"]


def test_fixed_budget_frontier_and_round_controls_have_the_requested_counts():
    conditions = load_suite(SUITE)["conditions"]
    frontier = [item for item in conditions if item["group"] == "fixed_budget"]
    assert [(item["inner"], item["rollouts_per_round"]) for item in frontier] == [
        (1, 192),
        (2, 96),
        (3, 64),
        (4, 48),
        (6, 32),
    ]
    assert {item["transitions_per_round"] for item in frontier} == {192}
    rounds = [item for item in conditions if item["group"] == "round_schedule"]
    assert [
        (item["rounds"], item["rollouts_per_round"], item["updates_per_round"])
        for item in rounds
    ] == [(1, 256, 768), (2, 128, 384), (4, 64, 192), (8, 32, 96)]
    assert {item["transitions_per_action"] for item in rounds} == {768}
    assert {item["updates_per_action"] for item in rounds} == {768}
    assert {item["replay_rows_per_action"] for item in rounds} == {49_152}


def test_all_unique_manifest_omits_behavioral_anchor_aliases():
    experiment = _load_json_strict(AMBI_ROOT / "experiments/all_unique_trials.json")
    names = experiment["configs"]
    assert len(names) == len(set(names))
    assert "ambi_anchor" in names
    assert {
        "ambi_horizon_train3_inner3",
        "ambi_updates192_per_round",
        "ambi_batch64",
        "ambi_schedule4_rounds",
        "ambi_fixed_budget_inner3_rollouts64",
        "ambi_breadth_depth_inner3_rollouts64",
        "ambi_fixed_budget_inner6_rollouts32",
        "ambi_breadth_depth_inner6_rollouts32",
    }.isdisjoint(names)

    fingerprints = []
    for name in names:
        config = _load_json_strict(AMBI_ALGS / f"{name}.json")
        if config["alg"] != "AMBITDMPC2/AMBITDMPC2":
            fingerprints.append((config["alg"],))
            continue
        params = config["alg_params"]
        updates = params["inner_updates_per_round"]
        if updates == "auto":
            updates = params["inner_rollouts_per_round"] * params["inner_rollout_horizon"]
        fingerprints.append(
            (
                config["alg"],
                params["train_unroll_horizon"],
                params["outer_planning_horizon"],
                params["inner_rollout_horizon"],
                params["inner_rounds"],
                params["inner_rollouts_per_round"],
                updates,
                params["inner_batch_size"],
                params["inner_replay_capacity"],
            )
        )
    assert len(fingerprints) == len(set(fingerprints))


def test_legacy_ambi_configs_no_longer_mix_horizon_schemas():
    for name in (
        "AntAMBITDMPC2.json",
        "AntAMBITDMPC2FullCopy.json",
        "AntAMBITDMPC2Debug.json",
    ):
        params = _load_json_strict(ROOT / "configs/algs" / name)["alg_params"]
        assert "horizon" not in params
        assert params["train_unroll_horizon"] == 3
        assert params["outer_planning_horizon"] == 3
        assert params["inner_rollout_horizon"] == 3


def test_smokes_cross_an_episode_boundary_and_full_inner_pilot_is_not_weakened():
    smoke_dir = AMBI_ROOT / "experiments/smoke"
    for path in smoke_dir.glob("*.json"):
        experiment = _load_json_strict(path)
        assert experiment["overrides_alg"]["total_steps"] == 20
        assert experiment["env_params"]["max_episode_steps"] == 16
        assert experiment["overrides_alg"]["total_steps"] > experiment["env_params"][
            "max_episode_steps"
        ]

    pilot = _load_json_strict(AMBI_ALGS / "smoke_ambi_anchor_throughput.json")
    params = pilot["alg_params"]
    assert pilot["total_steps"] - params["seed_steps"] >= 4
    assert params["inner_rounds"] == 4
    assert params["inner_rollouts_per_round"] == 64
    assert params["inner_rollout_horizon"] == 3
    assert params["inner_updates_per_round"] == "auto"
    assert params["inner_batch_size"] == 64
    assert params["inner_replay_capacity"] == 768

    tdmpc_run = _load_json_strict(AMBI_ALGS / "smoke_tdmpc2_train6_plan3.json")
    algorithm = object.__new__(TDMPC2Baseline)
    algorithm.env = gym.make("Pendulum-v1", max_episode_steps=16)
    algorithm.run_params = {
        **tdmpc_run,
        "seed": 55,
        "device": "cpu",
        "env": "test-env",
        "total_steps": 20,
    }
    try:
        cfg = algorithm._build_cfg({"device": "cpu", **tdmpc_run["alg_params"]})
    finally:
        algorithm.env.close()
    assert cfg.model_size == 5
    assert cfg.num_q == 5
    assert cfg.train_unroll_horizon == 6
    assert cfg.outer_planning_horizon == 3
    assert cfg.inner_rollout_horizon == 3


def test_ambi_launchers_use_consolidated_configs_and_algorithm_directory():
    expected = "configs/ambi/experiments/canonical/ambi_anchor.json"
    for name in ("run_ambi_ccv.sh", "run_ambi_hydra.sh", "run_ambi_oscar.sh"):
        contents = (ROOT / name).read_text()
        active_python = [
            line.strip()
            for line in contents.splitlines()
            if line.strip().startswith("python main.py") and not line.lstrip().startswith("#")
        ]
        assert active_python == ["python main.py \\"]
        assert expected in contents
        assert "--alg-dir configs/ambi/algs" in contents

        for line in contents.splitlines():
            if line.startswith(("#SBATCH --output=", "#SBATCH --error=")):
                path_template = line.split("=", 1)[1]
                assert (ROOT / Path(path_template).parent).is_dir()
