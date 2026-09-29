"""Changing MPPI's terminal critic preserves policy, scale, BN and RNG ownership."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

import evaluate_ambi_checkpoint as evaluator
from RL.tdmpc2_core.xqc_mppi import FrozenXQCMPPIController
from test_ambixqc_core import _tiny_model, _batch, _tree_equal
from test_ambixqc_checkpoint_evaluation import checkpoint_case
from test_xqc_mppi import SMALL
from utils.ambi_benchmark import validate_evaluation_controller
from utils.ambi_research import PresetMatrixError
from utils.eval_series_data import planner_identity, descriptive_label


@pytest.mark.parametrize("source", ["xqc", "aux_return"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA hardware is unavailable"))])
def test_selected_online_critic_and_persistent_policy_are_frozen(source, device, monkeypatch):
    model = _tiny_model(device=device, inner_operator="none", aux_return_mode="xqc",
                        xqc_optimizer_backend="auto" if device == "cuda" else "single_tensor")
    try:
        agent = model.agent
        agent.observe_reward(3.0, False, False)
        agent._update(*(value.to(device) for value in _batch(agent)))
        model.load(deepcopy(agent.checkpoint_state()), frozen_evaluation=True)
        planner = FrozenXQCMPPIController(agent, SMALL, terminal_critic_source=source)
        assert planner.controller is agent.xqc_controller
        owner = agent.aux_return if source == "aux_return" else agent.xqc_controller
        other = agent.xqc_controller if source == "aux_return" else agent.aux_return
        assert planner.terminal_critic is owner.critic
        assert planner.reward_scale == agent.reward_normalizer.scale
        with monkeypatch.context() as patch:
            calls = []
            def values(z, action, *, bn_mode):
                calls.append(bn_mode)
                return torch.stack([z.new_full((len(z),), 2), z.new_full((len(z),), 4)])
            patch.setattr(owner.critic, "values", values)
            patch.setattr(other.critic, "values", lambda *a, **kw: pytest.fail("wrong online critic"))
            patch.setattr(owner.critic_target, "values", lambda *a, **kw: pytest.fail("target critic"))
            z = torch.zeros(5, agent.cfg.latent_dim, device=device)
            actual = planner._terminal_q(z, torch.zeros(5, 1, device=device),
                                         reduction="mean_all", generator=planner.generator)
            torch.testing.assert_close(actual, z.new_full((5, 1), 3 * planner.reward_scale))
            assert calls == ["running"]
        before = agent.frozen_outer_state()
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state() if device == "cuda" else None
        obs, _ = model.env.reset(seed=101)
        planner.reset(11)
        first = torch.stack([planner.act(obs) for _ in range(2)])
        planner.reset(22)
        planner.act(obs)
        planner.reset(11)
        repeat = torch.stack([planner.act(obs) for _ in range(2)])
        torch.testing.assert_close(first, repeat, rtol=0, atol=0)
        assert _tree_equal(before, agent.frozen_outer_state())
        assert torch.equal(cpu_rng, torch.get_rng_state())
        if cuda_rng is not None:
            assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
        assert agent.last_inner_metrics["inner_model_steps"] == 34
    finally:
        model.env.close()


def test_missing_return_critic_fails_instead_of_using_soft_critic():
    model = _tiny_model(inner_operator="none")
    try:
        model.load(deepcopy(model.agent.checkpoint_state()), frozen_evaluation=True)
        with pytest.raises(ValueError, match="trained auxiliary"):
            FrozenXQCMPPIController(model.agent, terminal_critic_source="aux_return")
        with pytest.raises(ValueError, match="terminal_critic_source"):
            FrozenXQCMPPIController(model.agent, terminal_critic_source="target")
    finally:
        model.env.close()


def return_matrix(path):
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "configs/research/ambixqc_humanoid_return_mppi_benchmark.json").read_text())
    data["evaluation"].update(seeds=[101, 102], max_steps=2)
    data["comparisons"]["controller"]["variants"]["mppi_return"]["evaluation_controller"]["params"] = SMALL
    path.write_text(json.dumps(data))
    return data


def test_metadata_rejects_missing_auxiliary_before_environment(checkpoint_case, tmp_path, monkeypatch):
    path, checkpoint = checkpoint_case
    return_matrix(path)
    monkeypatch.setattr(evaluator, "_make_env", lambda *a: pytest.fail("late auxiliary preflight"))
    with pytest.raises(PresetMatrixError, match="aux_return_mode"):
        evaluator.evaluate_matrix(path, checkpoint, bundle_dir=tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("checkpoint_case", [{"aux_return_mode": "xqc"}], indirect=True)
def test_return_bundle_reuses_prior_and_records_distinct_identity(checkpoint_case, tmp_path, monkeypatch):
    path, checkpoint = checkpoint_case
    matrix = return_matrix(path)
    prior_bundle = tmp_path / "prior"
    prior = evaluator.evaluate_matrix(path, checkpoint, selectors=["controller/prior"],
                                     bundle_dir=prior_bundle)["results"][0]
    result_dir = tmp_path / "return"
    result = evaluator.evaluate_matrix(path, checkpoint, bundle_dir=result_dir,
                                       reference_bundle=prior_bundle)["results"][0]
    assert result["outer_state_unchanged"]
    assert result["outer_updates_before"] == result["outer_updates_after"] == 4
    delta = [a["return"] - b["return"] for a, b in zip(result["episodes"], prior["episodes"])]
    assert result["paired_return_delta_vs_prior"]["mean"] == pytest.approx(sum(delta) / len(delta))
    manifest = json.loads((result_dir / "manifest.json").read_text())
    assert len(manifest["runs"]) == 1
    run = manifest["runs"][0]
    assert run["selector"] == "controller/mppi_return"
    actual = result["evaluation_controller"]
    assert actual["protocol"]["terminal_value_source"] == "online_aux_return_twin_mean"
    assert actual["protocol"]["terminal_value_semantics"] == "learned_reward_return_under_persistent_xqc_actor"
    config = {"evaluation_controller": matrix["comparisons"]["controller"]["variants"]["mppi_return"]["evaluation_controller"]}
    validate_evaluation_controller(config, actual)
    wrong = deepcopy(actual)
    wrong["protocol"]["terminal_value_source"] = "online_xqc_twin_mean"
    with pytest.raises(ValueError, match="terminal value source"):
        validate_evaluation_controller(config, wrong)
    identity = planner_identity({}, result, "AMBIXQC/AMBIXQC", result["action_rule"])
    assert identity["semantics"]["terminal_value_source"] == "online_aux_return_twin_mean"
    label = descriptive_label({"backbone": "project/return", "planner": identity,
                                "science": {"algorithm": "AMBIXQC/AMBIXQC"}})
    assert "return-only Q" in label
    # Metadata-only assignment must describe the same route as executed results.
    from utils import eval_series_data as data
    from utils.ambi_research import resolve_preset
    from utils.checkpoint_context import load_checkpoint_context
    resolved = resolve_preset(path, "controller/mppi_return",
                              checkpoint_context=load_checkpoint_context(checkpoint))
    monkeypatch.setattr(data, "_source", lambda *a, **kw: ("project/return", {}, {}))
    monkeypatch.setattr(data, "resolved_checkpoint_config", lambda *a, **kw: run["resolved_config"])
    monkeypatch.setattr(data, "scientific_identity", lambda *a, **kw: {})
    preflight = data.identity_for_ambi_checkpoint(
        manifest["checkpoint"], resolved, manifest["protocol"], [101, 102],
        manifest["code"], path=checkpoint)
    assert preflight["planner"] == identity
    resolved["evaluation_controller"]["terminal_critic_source"] = "xqc"
    soft = data.identity_for_ambi_checkpoint(
        manifest["checkpoint"], resolved, manifest["protocol"], [101, 102],
        manifest["code"], path=checkpoint)
    assert soft["planner"] != preflight["planner"]
