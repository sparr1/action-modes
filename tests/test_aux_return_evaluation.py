"""Frozen evaluation and rendering retain auxiliary state and route provenance."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import evaluate_ambi_checkpoint as evaluator
import render_checkpoint as renderer
from utils.ambi_research import PresetMatrixError, _validate_checkpoint_overrides
from utils.aux_return_identity import AUX_RETURN_SOURCE_DEFAULTS


def architecture(**config):
    return evaluator._critic_architecture_key({"algorithm_config": {"alg_params": config}})


@pytest.mark.parametrize("representation", ["scalar", "distributional"])
def test_frozen_architecture_supports_route_changes_without_module_changes(representation):
    base = {"q_representation": representation, "aux_return_mode": "return_actor"}
    assert architecture(**base) == architecture(**base, inner_critic_source="aux_return",
                                               inner_horizon_actor_source="return_actor")
    assert architecture(**base) != architecture(q_representation=representation)
    assert architecture(q_representation=representation) == architecture(
        q_representation=representation, aux_return_mode="off", **AUX_RETURN_SOURCE_DEFAULTS)
    assert architecture(**base) != architecture(q_representation=representation, aux_return_mode="sac")


def test_frozen_digest_catches_auxiliary_optimizer_and_scalar_mutation():
    learned = {"optimizer": {"step": torch.tensor(1.0)}, "scale": torch.tensor(3.0)}
    empty_state = SimpleNamespace(state_dict=lambda: {})
    agent = SimpleNamespace(model=empty_state, optim=empty_state, pi_optim=empty_state,
                            alpha=torch.tensor(0.1), ent_coef_optim=None, num_updates=1,
                            aux_return=SimpleNamespace(checkpoint_state=lambda: learned))
    model = SimpleNamespace(agent=agent)
    original = evaluator._outer_state_digest(model)
    learned["optimizer"]["step"].add_(1)
    assert evaluator._outer_state_digest(model) != original
    learned["optimizer"]["step"].sub_(1)
    assert evaluator._outer_state_digest(model) == original
    learned["scale"].add_(1)
    assert evaluator._outer_state_digest(model) != original


@pytest.mark.parametrize("mode", ["sac", "return_actor"])
def test_real_auxiliary_inner_solve_preserves_all_outer_state(mode):
    from tests.test_ambi_root_local_sac import _tiny_model

    actor = "return_actor" if mode == "return_actor" else "sac"
    model = _tiny_model(
        aux_return_mode=mode, aux_return_ent_coef=0,
        inner_actor_source=actor, inner_critic_source="aux_return",
        inner_horizon_actor_source=actor, inner_horizon_critic_source="aux_return",
        inner_rounds=1, inner_rollouts_per_round=4, inner_updates_per_round=1,
    )
    try:
        before = evaluator._outer_state_digest(model)
        action = model.agent.act(torch.zeros(3), collect_diagnostics=False)
        assert torch.isfinite(action).all()
        assert evaluator._outer_state_digest(model) == before
    finally:
        model.env.close()


def test_frozen_presets_can_change_routes_but_not_train_new_auxiliary_modules():
    _validate_checkpoint_overrides({"inner_actor_source": "return_actor",
                                    "inner_horizon_critic_source": "aux_return"}, {}, "test")
    with pytest.raises(PresetMatrixError, match="incompatible overrides"):
        _validate_checkpoint_overrides({"aux_return_mode": "return_actor"}, {}, "test")


def context():
    return renderer.RenderContext(
        trial_run_params={"alg": "AMBITDMPC2/AMBITDMPC2", "env": "Test-v0", "seed": 7,
                          "alg_params": {"aux_return_mode": "return_actor"}},
        experiment_params={"env_params": {}}, source=Path("metadata.json"), metadata=None,
    )


def test_render_override_does_not_change_saved_context():
    saved = context()
    before = deepcopy(saved.trial_run_params)
    run, _ = renderer._prepare_run_params(
        saved, backend="ambi_tdmpc2", device="cpu", controller_seed=7,
        source_overrides={"inner_horizon_actor_source": "return_actor",
                          "inner_horizon_critic_source": "aux_return"})
    assert run["alg_params"]["inner_horizon_actor_source"] == "return_actor"
    assert saved.trial_run_params == before
    with pytest.raises(renderer.RenderCheckpointError, match="require an AMBI"):
        renderer._prepare_run_params(saved, backend="tdmpc2", device="cpu", controller_seed=7,
                                     source_overrides={"inner_actor_source": "return_actor"})


def test_render_cli_accepts_all_four_sources():
    parsed = renderer.build_parser().parse_args([
        "model.pt", "--results-json", "results.json",
        "--inner-actor-source", "return_actor", "--inner-critic-source", "aux_return",
        "--inner-horizon-actor-source", "sac", "--inner-horizon-critic-source", "aux_return",
    ])
    assert parsed.inner_actor_source == "return_actor"
    assert parsed.inner_critic_source == "aux_return"
    assert parsed.inner_horizon_actor_source == "sac"
    assert parsed.inner_horizon_critic_source == "aux_return"


def test_render_results_describe_effective_routes():
    payload = renderer._results_payload(
        checkpoint=Path("model.pt"), context=context(), backend="ambi_tdmpc2",
        deterministic=True, max_steps=2,
        results=[renderer.EpisodeResult(episode=0, seed=7, episode_return=1.0,
                                        length=2, capped=True)],
        effective_config={"aux_return_mode": "return_actor", "inner_operator": "sac",
                          "inner_horizon_critic_source": "aux_return"},
    )
    assert payload["value_routing"]["inner_horizon_critic_source"] == "aux_return"
    assert payload["value_routing"]["inner_actor_source"] == "sac"


def test_frozen_episode_loads_auxiliary_checkpoint_with_new_routes(tmp_path):
    from tests.test_ambi_root_local_sac import _tiny_model, _tiny_params

    options = {"aux_return_mode": "return_actor", "aux_return_ent_coef": 0,
               "inner_rounds": 1, "inner_rollouts_per_round": 4,
               "inner_updates_per_round": 1}
    model = _tiny_model(**options)
    checkpoint = tmp_path / "auxiliary.pt"
    try:
        model.agent.save(checkpoint)
    finally:
        model.env.close()
    params = _tiny_params(**options, inner_critic_source="aux_return",
                          inner_horizon_actor_source="return_actor",
                          inner_horizon_critic_source="aux_return")
    resolved = {
        "selector": "routing/return_tail", "comparison": "routing", "variant": "return_tail",
        "reference": "sac", "description": "Same checkpoint with an auxiliary return tail",
        "algorithm_config": {"alg": "AMBITDMPC2/AMBITDMPC2", "device": "cpu", "seed": 3,
                             "env": "Pendulum-v1", "total_steps": 10, "alg_params": params},
        "environment": {"id": "Pendulum-v1", "params": {"max_episode_steps": 5}},
    }
    result = evaluator.evaluate_preset(resolved, checkpoint, [7], controller_seed=3, max_steps=1)
    assert result["outer_state_unchanged"]
    assert result["aux_return_spec"]["mode"] == "return_actor"
    assert result["value_routing"]["inner_horizon_critic_source"] == "aux_return"
    assert result["value_routing"]["inner_actor_source"] == "sac"
