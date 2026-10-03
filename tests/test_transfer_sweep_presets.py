"""The H3 sweep has 24 unique selected conditions with shared fresh controls."""

from itertools import product

import pytest

from tests.test_ambi_root_local_sac import _build_cfg
from tests.test_critic_transfer_presets import (
    CHECKPOINT_SHA, ROOT, resolved, source_context,
)
from utils.ambi_research import load_preset_matrix, normalize_selectors


SWEEP_MATRICES = (
    ("ambi_critic_transfer_sweep_575k.json", "critic-transfer-v1", 1, "critic"),
    ("ambi_critic_transfer_hold_h_sweep_575k.json", "critic-transfer-hold-h-v1", 3, "critic"),
    ("ambi_actor_transfer_sweep_575k.json", "actor-transfer-v2", 1, "actor"),
    ("ambi_actor_transfer_hold_h_sweep_575k.json", "actor-transfer-hold-h-v1", 3, "actor"),
)


def matrix_for(filename):
    return load_preset_matrix(ROOT / "configs/research" / filename)


@pytest.mark.parametrize("filename,protocol,interval,component", SWEEP_MATRICES)
def test_sweep_defaults_pin_requested_protocol_workload_and_matching_references(
    filename, protocol, interval, component,
):
    matrix = matrix_for(filename)
    assert matrix["study_protocol"] == protocol
    assert matrix["source_run"] == "rwgao_b-brown-university/ambi/aux6428346x0"
    assert matrix["checkpoint_steps"] == [575000]
    assert matrix["checkpoint_contract"] == {"step": 575000, "sha256": CHECKPOINT_SHA}
    modes = ("fresh", "critic_warm") if component == "critic" else ("actor_warm",)
    assert set(normalize_selectors(matrix)) == {
        f"{source}_j{rounds}/{mode}"
        for source, rounds, mode in product(("soft_soft", "return_return"), (1, 8), modes)
    }
    assert matrix["shared_alg_params"]["inner_solve_interval"] == interval
    assert set(matrix["comparisons"]) == {
        f"{source}_j{rounds}" for source, rounds in product(("soft_soft", "return_return"), (1, 8))
    }
    for group in matrix["comparisons"].values():
        assert group["reference"] == "fresh"
        assert set(group["variants"]) == {"fresh", f"{component}_warm"}
    evaluation = matrix["evaluation"]
    assert evaluation["seeds"] == [101, 102, 103, 104, 105]
    assert evaluation["controller_seed"] == 55 and evaluation["max_steps"] == 500
    assert evaluation["transfer_diagnostics"] and evaluation["togo_return_rollouts"] == 32


def test_selected_sweep_covers_exact_cartesian_product_without_duplicate_fresh_cells(source_context):
    seen = []
    for filename, _, interval, _ in SWEEP_MATRICES:
        for selector in normalize_selectors(matrix_for(filename)):
            params = resolved(filename, selector, source_context)
            source = "soft" if params["inner_critic_source"] == "sac" else "return"
            mode = ("actor" if params["inner_actor_scope"] == "episode" else
                    "critic" if params["inner_critic_scope"] == "episode" else "fresh")
            seen.append((source, params["inner_rounds"], interval, mode))
    assert len(seen) == len(set(seen)) == 24
    assert set(seen) == set(product(("soft", "return"), (1, 8), (1, 3), ("fresh", "actor", "critic")))


def test_separate_prior_reference_has_same_pairing_and_no_inner_work(source_context):
    filename = "ambi_transfer_prior_reference_575k.json"
    matrix = matrix_for(filename)
    assert normalize_selectors(matrix) == ["reference/prior"]
    assert matrix["source_run"] == "rwgao_b-brown-university/ambi/aux6428346x0"
    assert matrix["checkpoint_contract"] == {"step": 575000, "sha256": CHECKPOINT_SHA}
    assert "study_protocol" not in matrix
    assert matrix["evaluation"] == {
        "controller_seed": 55, "seeds": [101, 102, 103, 104, 105],
        "max_steps": 500, "default_presets": ["reference/prior"],
    }
    params = resolved(filename, "reference/prior", source_context)
    cfg = _build_cfg(**params)
    assert cfg.inner_operator == "none"
    assert cfg.inner_rounds == 0 and cfg.inner_rollouts_per_round == 0
    assert cfg.inner_updates_per_round == 0 and cfg.inner_solve_interval == 1
    assert cfg.inner_actor_source == "sac" and cfg.inner_eval_execution_action == "mean"
    assert not cfg.inner_finite_horizon and not cfg.wandb
    assert cfg.inner_actor_writeback_coef == cfg.inner_critic_writeback_coef == 0
    for component in ("actor", "critic", "temperature", "replay", "actor_optimizer",
                      "critic_optimizer", "temperature_optimizer"):
        assert getattr(cfg, f"inner_{component}_scope") == "action"


@pytest.mark.parametrize("filename,protocol,interval,component", SWEEP_MATRICES)
@pytest.mark.parametrize("source,rounds", product(("soft_soft", "return_return"), (1, 8)))
def test_sweep_selected_network_is_the_only_changed_state_lifetime(
    filename, protocol, interval, component, source, rounds, source_context,
):
    group = f"{source}_j{rounds}"
    fresh = resolved(filename, f"{group}/fresh", source_context)
    warm = resolved(filename, f"{group}/{component}_warm", source_context)
    cfg = _build_cfg(**warm)
    assert cfg.inner_rounds == rounds and cfg.inner_first_action_rounds is None
    assert cfg.inner_rollout_horizon == 3 and cfg.inner_solve_interval == interval
    assert cfg.inner_rollouts_per_round == 128 and cfg.inner_batch_size == 256
    assert cfg.inner_replay_capacity == 3072
    assert rounds * cfg.inner_rollouts_per_round * cfg.inner_rollout_horizon <= cfg.inner_replay_capacity
    assert cfg.inner_critic_updates_per_round == 16 and cfg.inner_actor_updates_per_round == 4
    assert cfg.inner_component_update_order == "critic_first"
    assert cfg.inner_actor_adaptation == cfg.inner_critic_adaptation == "clone"
    assert cfg.inner_critic_target_initialization == "online" and not cfg.inner_rebase_persistent
    assert cfg.inner_eval_execution_action == "mean" and cfg.compile and cfg.compile_strict
    assert not cfg.wandb
    for other in ("actor", "critic", "temperature", "replay", "actor_optimizer",
                  "critic_optimizer", "temperature_optimizer"):
        assert getattr(cfg, f"inner_{other}_scope") == ("episode" if other == component else "action")
    soft = source == "soft_soft"
    assert cfg.inner_critic_source == cfg.inner_horizon_critic_source == ("sac" if soft else "aux_return")
    assert cfg.inner_sac_critic_target == ("entropy_augmented" if soft else "reward_only")
    assert cfg.inner_terminal_entropy == ("outer" if soft else "none")
    assert fresh.pop(f"inner_{component}_scope") == "action"
    assert warm.pop(f"inner_{component}_scope") == "episode"
    assert fresh == warm


@pytest.mark.parametrize("source,rounds,interval", product(("soft_soft", "return_return"), (1, 8), (1, 3)))
def test_fresh_comparator_is_identical_across_actor_and_critic_matrices(
    source, rounds, interval, source_context,
):
    suffix = "hold_h_" if interval == 3 else ""
    selector = f"{source}_j{rounds}/fresh"
    actor = resolved(f"ambi_actor_transfer_{suffix}sweep_575k.json", selector, source_context)
    critic = resolved(f"ambi_critic_transfer_{suffix}sweep_575k.json", selector, source_context)
    assert actor == critic


@pytest.mark.parametrize("component,source,rounds", product(("actor", "critic"), ("soft_soft", "return_return"), (1, 8)))
def test_solve_cadence_is_the_only_difference_between_sweep_cadence_matrices(
    component, source, rounds, source_context,
):
    selector = f"{source}_j{rounds}/{component}_warm"
    every = resolved(f"ambi_{component}_transfer_sweep_575k.json", selector, source_context)
    held = resolved(f"ambi_{component}_transfer_hold_h_sweep_575k.json", selector, source_context)
    assert every.pop("inner_solve_interval") == 1
    assert held.pop("inner_solve_interval") == 3
    assert every == held
