"""The four hidden-layer transfer examples retain the 575k study controls."""

import pytest

from tests.test_ambi_root_local_sac import _build_cfg
from tests.test_critic_transfer_presets import CHECKPOINT_SHA, ROOT, resolved, source_context
from utils.ambi_research import list_preset_selectors, load_preset_matrix, normalize_selectors


MATRICES = (
    ("ambi_critic_hidden_transfer_575k.json", "critic-hidden-transfer-v1", 1),
    ("ambi_critic_hidden_transfer_hold_h_575k.json", "critic-hidden-transfer-hold-h-v1", 3),
)


@pytest.mark.parametrize("filename,protocol,interval", MATRICES)
def test_hidden_examples_pin_backbone_protocol_seeds_and_default(filename, protocol, interval):
    matrix = load_preset_matrix(ROOT / "configs/research" / filename)
    assert matrix["study_protocol"] == protocol
    assert matrix["source_run"] == "rwgao_b-brown-university/ambi/aux6428346x0"
    assert matrix["checkpoint_steps"] == [575000]
    assert matrix["checkpoint_contract"] == {"step": 575000, "sha256": CHECKPOINT_SHA}
    assert set(list_preset_selectors(matrix)) == {
        "soft_soft/critic_hidden_warm", "return_return/critic_hidden_warm"}
    assert normalize_selectors(matrix) == ["return_return/critic_hidden_warm"]
    assert matrix["evaluation"] == {
        "controller_seed": 55, "seeds": [101, 102, 103, 104, 105], "max_steps": 500,
        "togo_return_rollouts": 32, "transfer_diagnostics": True,
        "default_presets": ["return_return/critic_hidden_warm"],
    }
    assert matrix["shared_alg_params"]["inner_solve_interval"] == interval


@pytest.mark.parametrize("filename,protocol,interval", MATRICES)
@pytest.mark.parametrize("group", ["soft_soft", "return_return"])
def test_hidden_examples_change_only_head_reset_from_full_critic_transfer(filename, protocol, interval, group, source_context):
    hidden = resolved(filename, f"{group}/critic_hidden_warm", source_context)
    full_filename = filename.replace("critic_hidden_transfer", "critic_transfer")
    full = resolved(full_filename, f"{group}/critic_warm", source_context)
    cfg = _build_cfg(**hidden)
    assert cfg.inner_critic_transfer_head == "random"
    assert cfg.inner_critic_scope == "episode"
    assert cfg.inner_actor_scope == "action"
    assert cfg.inner_rounds == 1 and cfg.inner_first_action_rounds is None
    assert cfg.inner_rollout_horizon == 3 and cfg.inner_solve_interval == interval
    assert cfg.inner_critic_updates_per_round == 16 and cfg.inner_actor_updates_per_round == 4
    assert cfg.inner_rollouts_per_round == 128 and cfg.inner_batch_size == 256
    assert hidden.pop("inner_critic_transfer_head") == "random"
    assert hidden == full


@pytest.mark.parametrize("group", ["soft_soft", "return_return"])
def test_hidden_hold_matrix_changes_only_cadence(group, source_context):
    every = resolved(MATRICES[0][0], f"{group}/critic_hidden_warm", source_context)
    held = resolved(MATRICES[1][0], f"{group}/critic_hidden_warm", source_context)
    assert every.pop("inner_solve_interval") == 1
    assert held.pop("inner_solve_interval") == 3
    assert every == held
