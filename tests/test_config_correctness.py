import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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


def test_ambi_tdmpc2_configs_preserve_continuing_task_and_inner_target_semantics():
    experiment = _load_json_strict(ROOT / "configs/experiments/AntAMBITDMPC2.json")
    assert experiment["env_params"]["terminate_when_unhealthy"] is False
    assert experiment["configs"] == ["AntAMBITDMPC2"]

    for name in ("AntAMBITDMPC2.json", "AntAMBITDMPC2Debug.json"):
        config = _load_json_strict(ROOT / "configs/algs" / name)
        assert config["alg"] == "AMBITDMPC2/AMBITDMPC2"
        assert config["alg_params"]["episodic"] is False
        assert config["alg_params"]["inner_critic_target_tau"] == 0.005
        assert (
            config["alg_params"]["inner_rollout_horizon"]
            <= config["alg_params"]["horizon"]
        )


def test_native_sac_q_representation_ablation_is_matched_across_five_seeds():
    experiment = _load_json_strict(
        ROOT / "configs/experiments/AntNativeSACQRepresentation.json"
    )

    assert experiment["configs"] == [
        "AntNativeSAC2",
        "AntNativeDistributionalSAC2",
    ]
    assert experiment["overrides_alg"]["seed"] == 55
    assert experiment["trials"] == 5


def test_ambi_launchers_run_the_learned_model_experiment():
    expected = "configs/experiments/AntAMBITDMPC2.json"
    for name in ("run_ambi_ccv.sh", "run_ambi_oscar.sh"):
        active_commands = [
            line.strip()
            for line in (ROOT / name).read_text().splitlines()
            if line.strip().startswith("python main.py")
        ]
        assert len(active_commands) == 1
        assert expected in active_commands[0]

        for line in (ROOT / name).read_text().splitlines():
            if line.startswith(("#SBATCH --output=", "#SBATCH --error=")):
                path_template = line.split("=", 1)[1]
                assert (ROOT / Path(path_template).parent).is_dir()
