import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AMBI_ROOT = ROOT / "configs/ambi"
AMBI_ALGS = AMBI_ROOT / "algs"


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json_strict(path):
    return json.loads(path.read_text(), object_pairs_hook=_reject_duplicate_keys)


def _params(name):
    return _load_json_strict(AMBI_ALGS / f"{name}.json")["alg_params"]


def test_all_configs_are_valid_json_without_duplicate_keys():
    for path in sorted((ROOT / "configs").rglob("*.json")):
        _load_json_strict(path)


def test_ambi_tree_contains_only_the_anchor_and_two_requested_studies():
    assert {path.name for path in AMBI_ALGS.glob("*.json")} == {
        "ambi_anchor.json",
        "ambi_branch_n16.json",
        "ambi_branch_n64.json",
        "ambi_horizon_h1.json",
        "ambi_horizon_h6.json",
    }

    manifests = {
        path.relative_to(AMBI_ROOT / "experiments").as_posix()
        for path in (AMBI_ROOT / "experiments").rglob("*.json")
    }
    assert manifests == {
        "ambi_anchor.json",
        "ambi_branch_n16.json",
        "ambi_branch_n64.json",
        "ambi_horizon_h1.json",
        "ambi_horizon_h6.json",
    }


def test_anchor_has_the_small_fixed_training_dose():
    config = _load_json_strict(AMBI_ALGS / "ambi_anchor.json")
    params = config["alg_params"]

    assert config["alg"] == "AMBITDMPC2/AMBITDMPC2"
    assert config["seed"] == 55
    assert config["env"] == "Ant-v4"
    assert config["total_steps"] == 1_000_000
    assert params["compile"] is True
    assert params["compile_strict"] is False
    assert params["mpc"] is False
    assert (
        params["train_unroll_horizon"],
        params["outer_planning_horizon"],
        params["inner_rollout_horizon"],
    ) == (3, 3, 3)
    assert (
        params["inner_rounds"],
        params["inner_rollouts_per_round"],
        params["inner_updates_per_round"],
        params["inner_batch_size"],
    ) == (2, 32, 16, 64)
    assert params["inner_replay_capacity"] == 192
    assert params["inner_replay_sampling"] == "with_replacement"
    assert params["inner_replay_scope"] == "action"
    assert params["inner_actor_adaptation"] == "clone"
    assert params["inner_critic_adaptation"] == "clone"
    assert params["inner_temperature_mode"] == "inherit_outer"
    assert params["inner_bootstrap_source"] == "inner_target"
    assert params["inner_critic_target_tau"] == 0.005
    assert params["inner_critic_target_update_interval"] == 1


def test_all_manifests_are_single_seed_full_runs_and_resolve_configs():
    for path in sorted((AMBI_ROOT / "experiments").rglob("*.json")):
        experiment = _load_json_strict(path)
        assert experiment["study_type"] == "single_seed_exploratory_screening", path
        assert "no confidence" in experiment["study_note"].lower(), path
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
        assert experiment["checkpoint_every"] == 100_000
        assert experiment["save_strat"] == ["best", "latest"]
        assert experiment["checkpoint_best_window"] == 100
        assert experiment["save_trials"] == "none"
        assert experiment["configs"] == [path.stem]
        for name in experiment["configs"]:
            assert (AMBI_ALGS / f"{name}.json").is_file(), (path, name)


def test_every_algorithm_has_consistent_replay_and_update_budgets():
    for path in sorted(AMBI_ALGS.glob("*.json")):
        params = _load_json_strict(path)["alg_params"]
        transitions = (
            params["inner_rounds"]
            * params["inner_rollouts_per_round"]
            * params["inner_rollout_horizon"]
        )
        update_slots = (
            params["inner_rounds"] * params["inner_updates_per_round"]
        )
        replay_draws = update_slots * params["inner_batch_size"]

        assert params["inner_replay_capacity"] == transitions, path
        assert update_slots == 32, path
        assert replay_draws == 2048, path
        assert params["inner_replay_sampling"] == "with_replacement", path
        assert params["inner_temperature_mode"] == "inherit_outer", path


def test_branch_study_changes_only_branch_count_and_derived_capacity():
    names = ["ambi_branch_n16", "ambi_anchor", "ambi_branch_n64"]
    params = [_params(name) for name in names]
    assert [item["inner_rollouts_per_round"] for item in params] == [16, 32, 64]
    assert [item["inner_replay_capacity"] for item in params] == [96, 192, 384]

    controlled_keys = (
        "train_unroll_horizon",
        "outer_planning_horizon",
        "inner_rollout_horizon",
        "inner_rounds",
        "inner_updates_per_round",
        "inner_batch_size",
        "inner_temperature_mode",
    )
    for key in controlled_keys:
        assert [item[key] for item in params] == [params[0][key]] * 3

def test_horizon_study_uses_only_h1_h3_and_matched_train6_inner6():
    names = ["ambi_horizon_h1", "ambi_anchor", "ambi_horizon_h6"]
    params = [_params(name) for name in names]
    assert [
        (item["train_unroll_horizon"], item["inner_rollout_horizon"])
        for item in params
    ] == [(3, 1), (3, 3), (6, 6)]
    assert [item["inner_rollouts_per_round"] for item in params] == [32, 32, 32]
    assert [item["inner_replay_capacity"] for item in params] == [64, 192, 384]
    assert [item["inner_rounds"] for item in params] == [2, 2, 2]
    assert [item["inner_updates_per_round"] for item in params] == [16, 16, 16]
    assert [item["inner_batch_size"] for item in params] == [64, 64, 64]

    all_h6 = [
        path.stem
        for path in AMBI_ALGS.glob("*.json")
        if _load_json_strict(path)["alg_params"]["inner_rollout_horizon"] == 6
    ]
    assert all_h6 == ["ambi_horizon_h6"]
    assert params[-1]["train_unroll_horizon"] == 6

def test_fixed_update_dose_has_the_declared_data_reuse_gradient():
    expected = {
        "ambi_horizon_h1": (64, 32.0),
        "ambi_branch_n16": (96, 2048 / 96),
        "ambi_anchor": (192, 2048 / 192),
        "ambi_branch_n64": (384, 2048 / 384),
        "ambi_horizon_h6": (384, 2048 / 384),
    }
    for name, (transitions, average_draws) in expected.items():
        params = _params(name)
        generated = (
            params["inner_rounds"]
            * params["inner_rollouts_per_round"]
            * params["inner_rollout_horizon"]
        )
        replay_draws = (
            params["inner_rounds"]
            * params["inner_updates_per_round"]
            * params["inner_batch_size"]
        )
        assert generated == transitions
        assert replay_draws / generated == average_draws


def test_legacy_ambi_configs_do_not_mix_horizon_schemas():
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


def test_launchers_reference_only_existing_new_ambi_manifests():
    anchor_manifest = "configs/ambi/experiments/ambi_anchor.json"
    launcher_targets = {
        "run_ambi_hydra.sh": (anchor_manifest, "configs/ambi/algs"),
        "run_ambi_oscar.sh": (anchor_manifest, "configs/ambi/algs"),
        "run_ambi_ccv.sh": (
            "configs/experiments/AntLegAdaptPaperSweep.json",
            "configs/algs",
        ),
    }
    for name, (manifest, alg_dir) in launcher_targets.items():
        contents = (ROOT / name).read_text()
        assert manifest in contents
        assert f"--alg-dir {alg_dir}" in contents
        assert (ROOT / manifest).is_file()

    oscar_launcher = (ROOT / "run_ambi_oscar.sh").read_text()
    assert "#SBATCH --gres=gpu:l40s:1" in oscar_launcher

    slurm_targets = {
        "run_ambi_anchor.sbatch": anchor_manifest,
        "run_ambi_branch_n16.sbatch": "configs/ambi/experiments/ambi_branch_n16.json",
        "run_ambi_branch_n64.sbatch": "configs/ambi/experiments/ambi_branch_n64.json",
        "run_ambi_horizon_h1.sbatch": "configs/ambi/experiments/ambi_horizon_h1.json",
        "run_ambi_horizon_h6.sbatch": "configs/ambi/experiments/ambi_horizon_h6.json",
    }
    for filename, manifest in slurm_targets.items():
        contents = (ROOT / "slurm" / filename).read_text()
        assert manifest in contents
        assert "--alg-dir configs/ambi/algs" in contents
        assert (ROOT / manifest).is_file()

    submit = (ROOT / "slurm/submit_ambi_suite.sh").read_text()
    for filename in slurm_targets:
        assert f"slurm/{filename}" in submit


def test_relocated_comparators_and_frozen_matrix_still_resolve():
    matrix = _load_json_strict(ROOT / "configs/research/ambi_inner_decoupling.json")
    assert matrix["base_alg_config"] == "../algs/AntAMBITDMPC2.json"
    assert (ROOT / "configs/algs/AntAMBITDMPC2.json").is_file()

    for name in ("AntNativeDistributionalSACFiveQ", "AntTDMPC2Comparator"):
        assert (ROOT / "configs/algs" / f"{name}.json").is_file()
        manifest = _load_json_strict(
            ROOT / "configs/experiments" / f"{name}.json"
        )
        assert manifest["configs"] == [name]
