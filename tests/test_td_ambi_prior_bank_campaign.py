"""Bank provenance and smoke gates reject incomplete or mismatched results."""

import gzip
import hashlib
import json

import pytest

from utils.td_ambi_prior_bank_campaign import (
    FAMILIES, SELECTORS, STEPS, prepare_inventory, validate_smoke,
)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def bank_inventory(tmp_path):
    banks = []
    for index, family in enumerate(FAMILIES):
        config = f"td_ambi_prior_{family}"
        directory = tmp_path / config
        prefix = f"model:{config}_0_"
        for step in STEPS:
            checkpoint = directory / f"{prefix}{step}"
            checkpoint.parent.mkdir(exist_ok=True)
            checkpoint.write_bytes(f"checkpoint-{index}-{step}".encode())
            _write(checkpoint.with_name(checkpoint.name + ".metadata.json"), {
                "schema_version": 1,
                "checkpoint": {"step": step, "kind": "periodic"},
                "trial_run_params": {
                    "name": config, "alg": "AMBITDMPC2/AMBITDMPC2", "seed": 55,
                    "alg_params": {
                        "outer_critic_target": "reward_only" if family.startswith("reward") else "entropy_augmented",
                        "sac_actor_loss_scale_mode": "tdmpc2_percentile_range" if family.endswith("qscale") else "none",
                        "outer_actor_entropy_mode": "tdmpc2_scaled", "inner_operator": "none",
                    },
                },
            })
        banks.append({"config": config, "models_directory": str(directory),
                      "filename_prefix": prefix,
                      "wandb_run_url": f"https://wandb.ai/entity/project/runs/run{index}"})
    path = tmp_path / "inventory.json"
    _write(path, {"banks": banks, "source_commit": "a" * 40})
    return path


@pytest.mark.parametrize("index", range(4))
def test_inventory_hashes_complete_bank_and_refuses_overwrite(bank_inventory, tmp_path, index):
    output = prepare_inventory(bank_inventory, index, tmp_path / "out")
    result = json.loads(output.read_text())
    assert result["source_run"] == f"entity/project/run{index}"
    assert result["source_commit"] == "a" * 40
    assert [row["step"] for row in result["checkpoints"]] == list(STEPS)
    for row in result["checkpoints"]:
        assert row["source_run"] == result["source_run"]
        assert row["sha256"] == hashlib.sha256(f"checkpoint-{index}-{row['step']}".encode()).hexdigest()
        assert len(row["metadata_sha256"]) == 64
    original = output.read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        prepare_inventory(bank_inventory, index, tmp_path / "out")
    assert output.read_bytes() == original
    second = prepare_inventory(bank_inventory, index, tmp_path / "out2")
    assert second.read_bytes() == original


@pytest.mark.parametrize("field,value", [("seed", 54), ("name", "wrong"), ("alg", "TDMPC2/TDMPC2")])
def test_inventory_rejects_wrong_checkpoint_provenance(bank_inventory, tmp_path, field, value):
    metadata_path = next((tmp_path / "td_ambi_prior_reward_qscale").glob("*.metadata.json"))
    metadata = json.loads(metadata_path.read_text())
    metadata["trial_run_params"][field] = value
    _write(metadata_path, metadata)
    with pytest.raises(ValueError, match=field):
        prepare_inventory(bank_inventory, 0, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_inventory_subset_is_sorted_and_missing_weights_fail(bank_inventory, tmp_path):
    output = prepare_inventory(bank_inventory, 0, tmp_path / "smoke", steps=[2_000_000, 25_000])
    assert [row["step"] for row in json.loads(output.read_text())["checkpoints"]] == [25_000, 2_000_000]
    path = tmp_path / "td_ambi_prior_reward_qscale" / "model:td_ambi_prior_reward_qscale_0_2000000"
    path.unlink()
    with pytest.raises(FileNotFoundError):
        prepare_inventory(bank_inventory, 0, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def _smoke(tmp_path, *, scaled=False):
    results, runs = [], []
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for selector in SELECTORS:
        sac = selector.startswith("inner/")
        adaptive_alpha = selector == "inner/adaptive" and not scaled
        metrics = {
            "inner_model_steps": 7680 if sac else 12336 if selector.endswith("mppi") else 0,
            "inner_critic_optimizer_steps": 15 if sac else 0,
            "inner_actor_optimizer_steps": 15 if sac else 0,
            "inner_temperature_optimizer_steps": 15 if adaptive_alpha else 0,
        }
        result = {"selector": selector, "outer_state_unchanged": True,
                  "outer_updates_before": 123, "outer_updates_after": 123,
                  "episodes": [{"seed": seed, "length": 3, "return": 2.0} for seed in [101, 102]],
                  "resolved_config": {"sac_actor_loss_scale_mode": "tdmpc2_percentile_range" if scaled else "none",
                                      "inner_temperature_mode": "auto" if adaptive_alpha else "inherit_outer",
                                      "inner_target_entropy": -441},
                  "model_metrics": {key: {"count": 6, "min": value, "max": value}
                                    for key, value in metrics.items()}}
        relative = selector.replace("/", "__") + ".jsonl.gz"
        with gzip.open(bundle / relative, "wt") as stream:
            for seed in [101, 102]:
                for decision in range(3):
                    stream.write(json.dumps({"phase": "decision", "episode_id": f"seed-{seed}",
                                             "decision_index": decision,
                                             "metrics": {f"decision/{key}": value for key, value in metrics.items()}}) + "\n")
        results.append(result)
        runs.append({"selector": selector, "status": "complete", "trace_files": [relative]})
    path = tmp_path / "results.json"
    _write(path, {"results": results})
    _write(bundle / "manifest.json", {"status": "complete", "runs": runs})
    return path, bundle


@pytest.mark.parametrize("scaled", [False, True])
def test_smoke_checks_all_decisions_and_scalar_protocol(tmp_path, scaled):
    path, bundle = _smoke(tmp_path, scaled=scaled)
    assert validate_smoke(path, bundle)["status"] == "passed"


@pytest.mark.parametrize("mutation,error", [
    ("outer", "outer state"), ("dose", "optimizer_steps"),
    ("seeds", "unpaired"), ("missing", "all four"), ("trace", "six decision traces"),
])
def test_smoke_rejects_unsafe_or_incomplete_result(tmp_path, mutation, error):
    path, bundle = _smoke(tmp_path)
    value = json.loads(path.read_text())
    if mutation == "outer":
        value["results"][0]["outer_state_unchanged"] = False
    elif mutation == "dose":
        value["results"][-1]["model_metrics"]["inner_temperature_optimizer_steps"]["min"] = 0
    elif mutation == "seeds":
        value["results"][-1]["episodes"][0]["seed"] = 103
    elif mutation == "missing":
        value["results"].pop()
    else:
        manifest = json.loads((bundle / "manifest.json").read_text())
        manifest["runs"][0]["trace_files"] = []
        _write(bundle / "manifest.json", manifest)
    _write(path, value)
    with pytest.raises(ValueError, match=error):
        validate_smoke(path, bundle)
