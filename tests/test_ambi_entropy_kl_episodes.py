"""Guard the paired joint-objective episode campaign before any GPU work."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from slurm import ambi_entropy_kl_episodes as campaign

REPO = Path(__file__).resolve().parents[1]


def matrices():
    return (json.loads((REPO / campaign.MATRIX).read_text()),
            json.loads((REPO / campaign.BASELINE_MATRIX).read_text()))


def test_only_fixed_beta_changes_from_sampled_entropy_baseline():
    matrix, baseline = matrices()
    campaign.validate_matrix(matrix, baseline)
    assert matrix["shared_alg_params"] == baseline["shared_alg_params"]
    params = matrix["shared_alg_params"]
    assert params["inner_entropy_enabled"] is True
    assert params["inner_temperature_mode"] == "auto"
    assert params["inner_temperature_lr"] == 0.0003
    assert params["inner_temperature_initialization"] == "inherit_outer"
    assert params["inner_outer_policy_kl_coef"] == 0.0
    assert params["inner_actor_lr"] == params["inner_critic_lr"] == 0.0003
    assert params["inner_rounds"] * params["inner_actor_updates_per_round"] == 32
    assert params["inner_rounds"] * params["inner_critic_updates_per_round"] == 128
    for name in campaign.CELLS:
        for multiplier in campaign.BETA_MULTIPLIERS:
            cell = campaign.intervention(name, multiplier)
            variant = matrix["comparisons"]["entropy_kl"]["variants"][cell["selector"].split("/")[1]]
            assert variant["alg_params"] == {
                "inner_outer_policy_kl_coef": campaign.OUTER_ALPHAS[name] * multiplier}
            assert cell["cell_id"] == f"{name}__beta{multiplier}"
            assert cell["temperature_lr"] == 0.0003


@pytest.mark.parametrize("key,value", [("inner_entropy_enabled", False),
    ("inner_temperature_lr", 0.03), ("inner_actor_lr", 0.003),
    ("inner_critic_updates_per_round", 48), ("inner_critic_bernoulli_p", 0.5)])
def test_matrix_rejects_second_intervention(key, value):
    matrix, baseline = matrices()
    matrix["shared_alg_params"][key] = value
    with pytest.raises(ValueError, match="Shared settings"):
        campaign.validate_matrix(matrix, baseline)


@pytest.mark.parametrize("change", ["beta", "entropy", "execution", "extra_cell"])
def test_matrix_rejects_wrong_variant(change):
    matrix, baseline = matrices()
    variants = matrix["comparisons"]["entropy_kl"]["variants"]
    variant = variants["sac_aux_100k_beta1"]
    if change == "beta":
        variant["alg_params"]["inner_outer_policy_kl_coef"] *= 2
    elif change == "entropy":
        variant["alg_params"]["inner_entropy_enabled"] = False
    elif change == "execution":
        variant["evaluation"]["execution_action"] = "mean"
    else:
        variants["sac_aux_100k_beta2"] = deepcopy(variant)
    with pytest.raises(ValueError):
        campaign.validate_matrix(matrix, baseline)


def pinned_campaign():
    profile = json.loads((REPO / campaign.PROFILE).read_text())
    payload = dict(schema_version=1, campaign_id=campaign.CAMPAIGN_ID, seeds=campaign.SEEDS,
                   controller_seed=55, max_steps=500, execution_action="sample", cells=[])
    for name in campaign.CELLS:
        for multiplier in campaign.BETA_MULTIPLIERS:
            payload["cells"].append({**profile["checkpoints"][name],
                **campaign.intervention(name, multiplier), "index": len(payload["cells"])})
    return payload, profile


def test_checkpoint_and_intervention_cannot_be_cross_assigned():
    payload, profile = pinned_campaign()
    campaign.validate_campaign(payload, profile)
    for key, value in [("checkpoint_id", "mppi_aux_400k"), ("beta_multiplier", 3),
                       ("beta", 1.0), ("temperature_lr", 0.03),
                       ("checkpoint_sha256", "bad"), ("selector", "entropy_kl/mppi_aux_400k_beta1")]:
        changed = deepcopy(payload)
        changed["cells"][0][key] = value
        with pytest.raises(ValueError, match="Cell checkpoint"):
            campaign.validate_campaign(changed, profile)
    payload["cells"].reverse()
    with pytest.raises(ValueError, match="Cell checkpoint"):
        campaign.validate_campaign(payload, profile)


def test_full_episode_and_smoke_have_separate_scope(tmp_path):
    cell = dict(checkpoint="/checkpoint", metadata="/metadata", selector="entropy_kl/sac_aux_100k_beta1",
                inventory_path="/inventory", reference_bundle="/prior-bundle")
    full = campaign.evaluator_command(cell, tmp_path)
    smoke = campaign.evaluator_command(cell, tmp_path, smoke_steps=8)
    assert full[full.index("--reference-bundle") + 1] == "/prior-bundle"
    assert "--reference-bundle" not in smoke
    assert full[full.index("--max-steps") + 1] == "500"
    assert full[full.index("--seeds") + 1:full.index("--controller-seed")] == [str(s) for s in campaign.SEEDS]
    assert smoke[smoke.index("--seeds") + 1:smoke.index("--controller-seed")] == ["101"]
    assert "--wandb" not in full and "--wandb" not in smoke
    for invalid in (0, True, 7, 500):
        with pytest.raises(ValueError, match="Smoke"):
            campaign.evaluator_command(cell, tmp_path, smoke_steps=invalid)


def test_existing_campaign_is_never_overwritten(tmp_path):
    (tmp_path / "campaign.json").write_text('{"keep": true}')
    with pytest.raises(FileExistsError):
        campaign.prepare(tmp_path)
    assert (tmp_path / "campaign.json").read_text() == '{"keep": true}'


def test_result_requires_joint_objective_and_all_diagnostic_counts(tmp_path, monkeypatch):
    from slurm import ambi_early_h3_j8_validation as validation
    cell = campaign.intervention("sac_aux_100k", 3)
    received = {}

    def validate(*args, **kwargs):
        received.update(kwargs)
        return {"selector": cell["selector"], "execution_action": "sample"}

    monkeypatch.setattr(validation, "validate_result", validate)
    metrics = {key: dict(count=2500, mean=0.1, min=0.0, max=0.2) for key in (
        "inner_final_outer_policy_kl", "inner_outer_policy_kl", "inner_actor_entropy",
        "inner_alpha_initial", "inner_alpha_final")}
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"results": [{"model_metrics": metrics}]}))
    campaign.validate_episode_result(tmp_path, cell)
    assert received["expected_temperature_steps"] == 32
    assert received["expected_seeds"] == campaign.SEEDS
    assert received["config_overrides"] == {"inner_entropy_enabled": True,
        "inner_temperature_mode": "auto", "inner_temperature_lr": 0.0003,
        "inner_outer_policy_kl_coef": cell["beta"]}
    metrics["inner_final_outer_policy_kl"]["count"] = 2499
    path.write_text(json.dumps({"results": [{"model_metrics": metrics}]}))
    with pytest.raises(ValueError, match="Incomplete joint-objective"):
        campaign.validate_episode_result(tmp_path, cell)


def test_result_rejects_mean_execution_after_generic_validation(tmp_path, monkeypatch):
    from slurm import ambi_early_h3_j8_validation as validation
    cell = campaign.intervention("sac_aux_100k", 1)
    monkeypatch.setattr(validation, "validate_result", lambda *a, **k:
        {"selector": cell["selector"], "execution_action": "mean"})
    with pytest.raises(ValueError, match="action execution"):
        campaign.validate_episode_result(tmp_path, cell)


def test_reference_or_inventory_change_blocks_gpu_start(tmp_path, monkeypatch):
    payload, _ = pinned_campaign()
    payload.update(source_commit="a" * 40, matrix_sha256="matrix", profile_sha256="profile")
    cell = payload["cells"][0]
    cell.update(reference_bundle="/prior", baseline_bundle="/baseline",
                reference_manifest_sha256="prior", baseline_manifest_sha256="baseline",
                inventory_path="/inventory", inventory_sha256="inventory")
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(campaign, "source", lambda: "a" * 40)
    monkeypatch.setattr(campaign, "verify_checkpoint", lambda *a: None)
    monkeypatch.setattr(campaign.subprocess, "run", lambda *a, **k: pytest.fail("GPU evaluation must not start"))
    digests = {str(REPO / campaign.MATRIX): "matrix", str(REPO / campaign.PROFILE): "profile",
               "/prior/manifest.json": "changed", "/baseline/manifest.json": "baseline", "/inventory": "inventory"}
    monkeypatch.setattr(campaign, "sha256", lambda p: digests[str(p)])
    with pytest.raises(RuntimeError, match="reused reference changed"):
        campaign.run(path, 0)
    digests["/prior/manifest.json"] = "prior"
    digests["/inventory"] = "changed"
    with pytest.raises(RuntimeError, match="inventory changed"):
        campaign.run(path, 0)


def test_launcher_uses_explicit_resources_and_disables_wandb():
    script = (REPO / "slurm/run_ambi_entropy_kl_episodes_oscar.sbatch").read_text()
    assert "SLURM_CPUS_PER_TASK:?" in script
    assert "EXPECTED_ACTION_MODES_SHA:?" in script
    assert "WANDB_MODE=disabled" in script
    assert "ambi_entropy_kl_episodes.py run" in script
    assert "#SBATCH --time" not in script
    assert "#SBATCH --cpus-per-task" not in script


def test_trace_hashes_pin_only_unique_files_inside_bundle(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    path = bundle / "trace.jsonl.gz"
    path.write_bytes(b"compressed trace fixture")
    manifest = bundle / "manifest.json"
    manifest.write_text(json.dumps({"runs": [{"trace_files": [path.name]}]}))
    assert campaign.trace_hashes(bundle) == {path.name: campaign.sha256(path)}
    manifest.write_text(json.dumps({"runs": [{"trace_files": [path.name, path.name]}]}))
    with pytest.raises(ValueError, match="duplicate trace"):
        campaign.trace_hashes(bundle)
    outside = tmp_path / "outside.jsonl.gz"
    outside.write_bytes(b"outside")
    manifest.write_text(json.dumps({"runs": [{"trace_files": ["../outside.jsonl.gz"]}]}))
    with pytest.raises(ValueError, match="Invalid trace"):
        campaign.trace_hashes(bundle)


def test_completed_worker_receipt_matches_publisher_contract(tmp_path, monkeypatch):
    payload, _ = pinned_campaign()
    payload.update(source_commit="a" * 40,
                   matrix_sha256=campaign.sha256(REPO / campaign.MATRIX),
                   profile_sha256=campaign.sha256(REPO / campaign.PROFILE))
    cell = payload["cells"][0]
    for kind in ("reference", "baseline"):
        bundle = tmp_path / kind
        bundle.mkdir()
        (bundle / "manifest.json").write_text(json.dumps({"reference": kind}))
        cell[kind + "_bundle"] = str(bundle)
        cell[kind + "_manifest_sha256"] = campaign.sha256(bundle / "manifest.json")
    inventory = tmp_path / "inventory.json"
    inventory.write_text("{}")
    cell.update(inventory_path=str(inventory), inventory_sha256=campaign.sha256(inventory),
                output_dir=str(tmp_path / "production" / cell["cell_id"]))
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(campaign, "source", lambda: "a" * 40)
    monkeypatch.setattr(campaign, "verify_checkpoint", lambda *a: None)

    def evaluate(command, **kwargs):
        result = Path(command[command.index("--output") + 1])
        result.write_text('{"complete": true}')
        bundle = Path(command[command.index("--bundle-dir") + 1])
        bundle.mkdir()
        (bundle / "trace.jsonl.gz").write_bytes(b"verified compressed trace")
        (bundle / "manifest.json").write_text(json.dumps({"runs": [{"trace_files": ["trace.jsonl.gz"]}]}))

    monkeypatch.setattr(campaign.subprocess, "run", evaluate)
    monkeypatch.setattr(campaign, "validate_episode_result", lambda output, selected, smoke:
                        {"passed": True, "selector": selected["selector"]})
    receipt = campaign.run(path, 0)
    assert receipt["validated"] is True and receipt["smoke"] is False
    for key in ("checkpoint_id", "cell_id", "beta_multiplier", "beta", "temperature_lr"):
        assert receipt[key] == cell[key]
    output = Path(cell["output_dir"])
    assert receipt["trace_sha256"] == {"trace.jsonl.gz": campaign.sha256(output / "bundle/trace.jsonl.gz")}
    assert receipt["results_sha256"] == campaign.sha256(output / "results.json")
    assert receipt["manifest_sha256"] == campaign.sha256(output / "bundle/manifest.json")
    assert json.loads((output / "receipt.json").read_text()) == receipt
    with pytest.raises(FileExistsError):
        campaign.run(path, 0)
