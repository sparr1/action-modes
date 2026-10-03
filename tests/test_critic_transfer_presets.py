"""Matched scientific controls for the eight frozen 575k critic examples."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from tests.test_ambi_root_local_sac import _build_cfg
from utils.ambi_research import (
    list_preset_selectors,
    load_preset_matrix,
    normalize_selectors,
    resolve_preset,
)
from utils.checkpoint_context import CheckpointContext


ROOT = Path(__file__).resolve().parents[1]
MATRICES = (
    ("ambi_critic_transfer_575k.json", "critic-transfer-v1", 1),
    ("ambi_critic_transfer_hold_h_575k.json", "critic-transfer-hold-h-v1", 3),
)
CHECKPOINT_SHA = "0c6955db7cb8555a67d7863344b70be68f4b3250814d131e647ee6f9ef01a042"
SELECTORS = {
    "soft_soft/fresh", "soft_soft/critic_warm",
    "return_return/fresh", "return_return/critic_warm",
}


@pytest.fixture
def source_context():
    path = ROOT / "configs/dmcontrol/algs/ambi_aux_return_sac_clip_target10p5_shared.json"
    return CheckpointContext(
        trial_run_params=json.loads(path.read_text()),
        experiment_params={"env_params": {
            "domain_name": "humanoid", "task_name": "walk", "max_episode_steps": 500,
        }},
        source=path,
        metadata={"checkpoint": {"step": 575000}},
    )


def resolved(filename, selector, context):
    return resolve_preset(
        ROOT / "configs/research" / filename, selector, checkpoint_context=context,
    )["algorithm_config"]["alg_params"]


@pytest.mark.parametrize("filename,protocol,interval", MATRICES)
def test_examples_pin_backbone_and_keep_coherent_paired_references(filename, protocol, interval):
    matrix = load_preset_matrix(ROOT / "configs/research" / filename)
    assert matrix["study_protocol"] == protocol
    assert matrix["source_run"] == "rwgao_b-brown-university/ambi/aux6428346x0"
    assert matrix["checkpoint_steps"] == [575000]
    assert matrix["checkpoint_contract"] == {"step": 575000, "sha256": CHECKPOINT_SHA}
    assert set(list_preset_selectors(matrix)) == SELECTORS
    assert normalize_selectors(matrix) == ["return_return/critic_warm"]
    assert all(group["reference"] == "fresh" for group in matrix["comparisons"].values())
    assert matrix["shared_alg_params"]["inner_solve_interval"] == interval
    assert matrix["evaluation"] == {
        "controller_seed": 55, "seeds": [101, 102, 103, 104, 105], "max_steps": 500,
        "togo_return_rollouts": 32, "transfer_diagnostics": True,
        "default_presets": ["return_return/critic_warm"],
    }


@pytest.mark.parametrize("filename,protocol,interval", MATRICES)
@pytest.mark.parametrize("selector", sorted(SELECTORS))
def test_every_example_resolves_valid_lifetimes_and_objectives(
    filename, protocol, interval, selector, source_context,
):
    source_before = deepcopy(source_context.trial_run_params)
    params = resolved(filename, selector, source_context)
    cfg = _build_cfg(**params)
    soft = selector.startswith("soft_soft/")
    warm = selector.endswith("/critic_warm")
    assert cfg.inner_critic_scope == ("episode" if warm else "action")
    for name in ("actor", "temperature", "replay", "actor_optimizer",
                 "critic_optimizer", "temperature_optimizer"):
        assert getattr(cfg, f"inner_{name}_scope") == "action"
    assert cfg.inner_actor_adaptation == cfg.inner_critic_adaptation == "clone"
    assert cfg.inner_actor_initialization == cfg.inner_critic_initialization == "prior"
    assert cfg.inner_critic_target_initialization == "online"
    assert not cfg.inner_rebase_persistent
    assert cfg.inner_actor_writeback_coef == cfg.inner_critic_writeback_coef == 0
    assert cfg.inner_actor_source == cfg.inner_horizon_actor_source == "sac"
    assert cfg.inner_critic_source == cfg.inner_horizon_critic_source == ("sac" if soft else "aux_return")
    assert cfg.inner_sac_critic_target == ("entropy_augmented" if soft else "reward_only")
    assert cfg.inner_terminal_entropy == ("outer" if soft else "none")
    assert cfg.inner_rounds == 1 and cfg.inner_first_action_rounds is None
    assert cfg.inner_rollout_horizon == 3 and cfg.inner_solve_interval == interval
    assert cfg.inner_rollouts_per_round == 128 and cfg.inner_batch_size == 256
    assert cfg.inner_critic_updates_per_round == 16 and cfg.inner_actor_updates_per_round == 4
    assert cfg.inner_replay_capacity == 3072 and cfg.inner_replay_sampling == "with_replacement"
    assert cfg.inner_component_update_order == "critic_first"
    assert cfg.inner_actor_lr == cfg.inner_critic_lr == cfg.inner_temperature_lr == 0.0003
    assert cfg.inner_critic_target_tau == 0.01
    assert cfg.inner_temperature_mode == "auto"
    assert cfg.inner_temperature_initialization == cfg.inner_target_entropy == "inherit_outer"
    assert cfg.inner_actor_entropy_mode == "squashed"
    assert cfg.inner_eval_execution_action == "mean"
    assert cfg.inner_sac_return_estimator == "one_step" and cfg.inner_finite_horizon
    assert cfg.compile and cfg.compile_strict and not cfg.wandb
    assert params["target_entropy"] == -10.5
    assert params["log_std_mapping"] == "direct_clamp"
    assert params["log_std_min"] == -10 and params["log_std_max"] == 2
    assert params["aux_return_mode"] == "sac" and not params["aux_return_detach_representation"]
    assert source_context.trial_run_params == source_before


@pytest.mark.parametrize("filename,protocol,interval", MATRICES)
@pytest.mark.parametrize("group", ("soft_soft", "return_return"))
def test_fresh_and_warm_differ_only_in_online_critic_lifetime(
    filename, protocol, interval, group, source_context,
):
    fresh = resolved(filename, f"{group}/fresh", source_context)
    warm = resolved(filename, f"{group}/critic_warm", source_context)
    assert fresh.pop("inner_critic_scope") == "action"
    assert warm.pop("inner_critic_scope") == "episode"
    assert fresh == warm


@pytest.mark.parametrize("selector", sorted(SELECTORS))
def test_hold_examples_change_only_solve_cadence(selector, source_context):
    every = resolved(MATRICES[0][0], selector, source_context)
    held = resolved(MATRICES[1][0], selector, source_context)
    assert every.pop("inner_solve_interval") == 1
    assert held.pop("inner_solve_interval") == 3
    assert every == held
