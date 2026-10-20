"""Keep the alpha-rate intervention isolated from other scientific settings."""
import json
from pathlib import Path

from slurm import ambi_alpha_lr_episodes as campaign


def test_only_temperature_learning_rate_changes_from_sampled_baseline():
    root = Path(__file__).resolve().parents[1]
    baseline = json.loads((root / "configs/research/ambi_early_h3_j8.json").read_text())
    matrix = json.loads((root / campaign.MATRIX).read_text())
    assert matrix["shared_alg_params"] == baseline["shared_alg_params"]
    for key in ("seeds", "controller_seed", "max_steps", "togo_return_rollouts", "togo_return_every"):
        assert matrix["evaluation"][key] == baseline["evaluation"][key]
    variants = matrix["comparisons"]["alpha_lr"]["variants"]
    assert set(variants) == {"lr_" + token for token in campaign.RATES}
    for token, rate in campaign.RATES.items():
        assert variants["lr_" + token]["alg_params"] == {"inner_temperature_lr": rate}
        assert variants["lr_" + token]["evaluation"] == {"execution_action": "sample"}
    assert matrix["shared_alg_params"]["inner_temperature_lr"] == 0.0003
    assert matrix["shared_alg_params"]["inner_entropy_enabled"] is True
    assert matrix["shared_alg_params"]["inner_outer_policy_kl_coef"] == 0


def test_probe_retains_original_root_bank_provenance():
    root = Path(__file__).resolve().parents[1] / "configs/research"
    prior = json.loads((root / "ambi_prior_kl_probe.json").read_text())
    probe = json.loads((root / "ambi_alpha_lr_probe.json").read_text())
    for key in prior:
        if key != "description":
            assert probe[key] == prior[key]
    assert probe["temperature_lrs"] == [0.0003, *campaign.RATES.values()]


def test_full_protocol_reuses_sampled_prior_and_smoke_is_separate(tmp_path):
    cell = dict(checkpoint="/checkpoint", metadata="/metadata", selector="alpha_lr/lr_0p03",
                inventory_path="/inventory", reference_bundle="/prior-bundle")
    full = campaign.evaluator_command(cell, tmp_path)
    smoke = campaign.evaluator_command(cell, tmp_path, smoke_steps=8)
    assert full[full.index("--reference-bundle") + 1] == "/prior-bundle"
    assert "--reference-bundle" not in smoke
    assert full[full.index("--max-steps") + 1] == "500"
    assert full[full.index("--seeds") + 1:full.index("--controller-seed")] == [str(s) for s in campaign.SEEDS]
    assert smoke[smoke.index("--seeds") + 1:smoke.index("--controller-seed")] == ["101"]
    assert "--wandb" not in full and "--wandb" not in smoke


def test_existing_campaign_is_never_overwritten(tmp_path):
    import pytest
    (tmp_path / "campaign.json").write_text('{"keep": true}')
    with pytest.raises(FileExistsError):
        campaign.prepare(tmp_path)
    assert (tmp_path / "campaign.json").read_text() == '{"keep": true}'
