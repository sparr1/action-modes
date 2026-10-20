"""Preserve the paired episode protocol while changing only the regularizer."""
import json
from pathlib import Path

from slurm import ambi_prior_kl_episodes as campaign


def test_episode_matrix_preserves_baseline_science_except_entropy_replacement():
    root = Path(__file__).resolve().parents[1]
    baseline = json.loads((root / "configs/research/ambi_early_h3_j8.json").read_text())
    kl = json.loads((root / campaign.MATRIX).read_text())
    expected = {**baseline["shared_alg_params"], "inner_entropy_enabled": False}
    assert kl["shared_alg_params"] == expected
    for key in ("seeds", "controller_seed", "max_steps", "togo_return_rollouts", "togo_return_every"):
        assert kl["evaluation"][key] == baseline["evaluation"][key]
    assert set(kl["comparisons"]["prior_kl"]["variants"]) == set(campaign.CELLS)
    for variant in kl["comparisons"]["prior_kl"]["variants"].values():
        assert set(variant["alg_params"]) == {"inner_outer_policy_kl_coef"}
        assert variant["alg_params"]["inner_outer_policy_kl_coef"] > 0
        assert variant["evaluation"] == {"execution_action": "sample"}


def test_prior_reference_is_used_only_for_complete_matched_protocol(tmp_path):
    cell = dict(checkpoint="/checkpoint", metadata="/metadata", selector="prior_kl/sac_aux_100k",
                inventory_path="/inventory", reference_bundle="/prior-bundle")
    full = campaign.evaluator_command(cell, tmp_path)
    smoke = campaign.evaluator_command(cell, tmp_path, smoke_steps=8)
    assert full[full.index("--reference-bundle") + 1] == "/prior-bundle"
    assert "--reference-bundle" not in smoke
    assert full[full.index("--max-steps") + 1] == "500"
    assert full[full.index("--seeds") + 1:full.index("--controller-seed")] == list(map(str, campaign.SEEDS))
    assert smoke[smoke.index("--seeds") + 1:smoke.index("--controller-seed")] == ["101"]
    assert "--wandb" not in full and "--wandb" not in smoke
