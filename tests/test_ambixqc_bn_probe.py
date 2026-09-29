"""Paired critic-BN diagnostics isolate learning, query statistics and RNG."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

import run_ambixqc_bn_probe as probe
from test_ambixqc_prior_checkpoint import _wrapper
from test_ambixqc_core import _batch


@pytest.fixture
def agent():
    model = _wrapper(aux_return_mode="xqc", inner_critic_source="aux_return",
                     inner_horizon_critic_source="aux_return", inner_critic_target="reward_only",
                     inner_terminal_bootstrap="outer", inner_actor_bn_mode="running")
    try:
        yield model.agent
    finally:
        model.env.close()


def test_fixed_root_grid_spreads_across_five_trajectories():
    positions = probe.root_positions()
    assert len(positions) == len(set(positions)) == 32
    assert [sum(s == seed for s, _ in positions) for seed in probe.source.SEEDS] == [7, 7, 6, 6, 6]
    for seed in probe.source.SEEDS:
        rows = [step for s, step in positions if s == seed]
        assert rows[0] == 0 and rows[-1] == 499 and rows == sorted(rows)
    assert probe.root_positions(1) == [(101, 0)]


def test_rng_guard_restores_even_after_failure():
    before = probe.tree_hash(probe.rng_state())
    with pytest.raises(RuntimeError):
        with probe.preserve_rng():
            torch.randn(4); np.random.rand(5)
            raise RuntimeError("intentional")
    assert probe.tree_hash(probe.rng_state()) == before


def test_paired_inputs_repeat_without_advancing_global_rng(agent):
    z = torch.zeros(1, agent.cfg.latent_dim)
    before = probe.tree_hash(probe.rng_state())
    one = probe.paired_inputs(agent, z, 12345, "root", count=8, max_updates=3)
    two = probe.paired_inputs(agent, z, 12345, "root", count=8, max_updates=3)
    other = probe.paired_inputs(agent, z, 23456, "root", count=8, max_updates=3)
    assert probe.tree_hash(one) == probe.tree_hash(two)
    assert probe.tree_hash(one) != probe.tree_hash(other)
    assert not torch.equal(one["train"]["actions"], one["heldout"]["actions"])
    assert probe.tree_hash(probe.rng_state()) == before


def test_actual_small_probe_is_paired_and_keeps_outer_actor_rng_frozen(agent):
    z = torch.zeros(1, agent.cfg.latent_dim)
    data = probe.paired_inputs(agent, z, 12345, "root", count=8, max_updates=3)
    before = probe.tree_hash(agent.frozen_outer_state())
    global_before = probe.tree_hash(probe.rng_state())
    inputs_before = probe.tree_hash(data)
    rows = probe.probe_pair(agent, data, inspection_steps=(0, 1, 3))
    assert len(rows) == 9
    assert probe.tree_hash(agent.frozen_outer_state()) == before
    assert probe.tree_hash(probe.rng_state()) == global_before
    assert probe.tree_hash(data) == inputs_before
    initial = [row for row in rows if row["critic_updates"] == 0]
    for key in ("heldout_running_q_mse", "running_joined_q_rmse", "one_actor_step_deployed_kl"):
        assert len({row[key] for row in initial}) == 1
    for row in rows:
        assert row["diagnostics_state_unchanged"]
        assert row["paired_inputs_unchanged"]
        assert row["one_actor_step_mean_action_l2"] >= 0
        assert row["target_clip_fraction"] >= 0
        assert np.isfinite(row["heldout_running_q_mse"])
        if row["mode"] != "batch_update":
            assert all(value == 0 for value in row["bn_drift_rms"].values())
    changed = [row for row in rows if row["mode"] == "batch_update" and row["critic_updates"] == 3]
    assert any(value > 0 for value in changed[0]["bn_drift_rms"].values())


def test_heldout_target_matches_atomwise_clipping_not_scalar_clipping():
    support = torch.tensor([-2., -1., 0., 1., 2.])
    probabilities = torch.tensor([[[.1, .1, .1, .2, .5]], [[.1, .1, .2, .1, .5]]])
    rewards = torch.tensor([1.6])
    target = probe.heldout_targets(rewards, probabilities.log(), support, 2., .9)
    # Head 1 has the lower expectation; transport its atoms independently.
    expected = (probabilities[1, 0] * (.8 + .9 * support).clamp(-2, 2)).sum()
    torch.testing.assert_close(target["heldout_target"][0], expected)
    assert target["heldout_target"][0] < target["heldout_unprojected_target"][0] < support[-1]
    assert target["target_clip_fraction"] > 0
    torch.testing.assert_close(target["heldout_target_probabilities"].sum(-1), torch.ones(1))
    for extreme, bound in [(20., 2.), (-20., -2.)]:
        clipped = probe.heldout_targets(torch.tensor([extreme]), probabilities.log(), support, 1., .9)
        torch.testing.assert_close(clipped["heldout_target"], torch.tensor([bound]))


def test_heldout_targets_match_actual_frozen_outer_tail_objective(agent):
    data = probe.paired_inputs(agent, torch.zeros(1, agent.cfg.latent_dim), 12345, "target", count=8)
    workspace = probe.clone_workspace(agent.xqc_controller, agent.aux_return.critic)
    batch = probe.LatentXQCBatch(**data["heldout"], discount=data["discount"])
    objective = workspace.controller.critic_objective(
        batch, next_noise=data["heldout_next_noise"], reward_scale=data["reward_scale"],
        outer_terminal_mask=torch.ones(8, dtype=torch.bool), outer_controller=agent.xqc_controller,
        outer_critic=agent.aux_return.critic, outer_critic_is_return=True,
        critic_target_kind="reward_only", critic_bn_mode="running")
    torch.testing.assert_close(objective.target_probabilities, data["heldout_target_probabilities"], atol=0, rtol=0)


@pytest.mark.parametrize("left,right,cosine,ratio", [
    ([0., 0.], [1., 0.], None, 0.), ([1., 0.], [0., 0.], None, None),
    ([0., 0.], [0., 0.], None, None), ([2., 0.], [-1., 0.], -1., 2.),
    ([1e-20, 0.], [1e-20, 0.], 1., 1.),
])
def test_gradient_comparison_does_not_invent_zero_norm_cosines(left, right, cosine, ratio):
    result = probe.gradient_comparison(torch.tensor(left), torch.tensor(right))
    assert result["cosine"] == cosine and result["norm_ratio"] == ratio
    assert result["cosine_defined"] == (cosine is not None)
    assert result["norm_ratio_defined"] == (ratio is not None)
    with pytest.raises(ValueError, match="finite"):
        probe.gradient_comparison(torch.tensor([float("nan")]), torch.ones(1))


def test_diagnostic_actor_step_inherits_trained_bn_and_reports_projected_actor(agent, monkeypatch):
    from RL.tdmpc2_core import xqc_controller
    agent._update(*_batch(agent))
    inherited_bn = probe.tree_hash(probe.bn_buffers(agent.xqc_controller.actor))
    assert any(torch.count_nonzero(value).item() for name, value in
               probe.bn_buffers(agent.xqc_controller.actor).items() if name.endswith("running_mean"))
    data = probe.paired_inputs(agent, torch.zeros(1, agent.cfg.latent_dim), 12345, "trained", count=8)
    workspace = probe.clone_workspace(agent.xqc_controller, agent.aux_return.critic)
    outer_before = probe.tree_hash(agent.frozen_outer_state())
    captured, projections = [], []
    real_clone, real_project = probe.clone_workspace, xqc_controller._project_unit_weights_
    def capture_clone(*args, **kwargs):
        result = real_clone(*args, **kwargs)
        assert probe.tree_hash(probe.bn_buffers(result.controller.actor)) == inherited_bn
        assert probe.tree_hash(result.controller.actor.state_dict()) == probe.tree_hash(agent.xqc_controller.actor.state_dict())
        captured.append(result)
        return result
    def capture_projection(weights):
        before = [weight.detach().clone() for weight in weights]
        result = real_project(weights)
        projections.append((before, [weight.detach().clone() for weight in weights]))
        return result
    monkeypatch.setattr(probe, "clone_workspace", capture_clone)
    monkeypatch.setattr(xqc_controller, "_project_unit_weights_", capture_projection)
    metrics, _ = probe.inspect(workspace, agent.xqc_controller, data,
                               probe.bn_buffers(agent.aux_return.critic), actor_lr=5e-3)
    assert len(captured) == len(projections) == 1
    before_projection, after_projection = projections[0]
    assert any(not torch.equal(a, b) for a, b in zip(before_projection, after_projection))
    for weight in after_projection:
        torch.testing.assert_close(weight.norm(dim=1), torch.ones(weight.shape[0]), atol=2e-7, rtol=2e-7)
    diagnostic = captured[0]
    assert diagnostic.actor_optimizer_steps == diagnostic.temperature_optimizer_steps == 1
    assert probe.tree_hash(probe.bn_buffers(diagnostic.controller.actor)) == inherited_bn
    with torch.no_grad():
        root = data["train"]["latents"][:1]
        old = agent.xqc_controller.actor.distribution(root, bn_mode="running")
        new = diagnostic.controller.actor.distribution(root, bn_mode="running")
        expected = float(probe.InnerXQCEngine._gaussian_kl(*new, *old).mean())
    assert metrics["one_actor_step_deployed_kl"] == expected
    assert probe.tree_hash(agent.frozen_outer_state()) == outer_before


def test_readonly_queries_and_actor_diagnostic_do_not_train_the_fitting_workspace(agent):
    data = probe.paired_inputs(agent, torch.zeros(1, agent.cfg.latent_dim), 12345, "root", count=8, max_updates=3)
    workspace = probe.clone_workspace(agent.xqc_controller, agent.aux_return.critic)
    before = probe.tree_hash({"model": workspace.controller.state_dict(), "optimizer": workspace.state_dict()})
    first, _ = probe.inspect(workspace, agent.xqc_controller, data, probe.bn_buffers(agent.aux_return.critic))
    repeated, _ = probe.inspect(workspace, agent.xqc_controller, data, probe.bn_buffers(agent.aux_return.critic))
    assert first == repeated
    assert before == probe.tree_hash({"model": workspace.controller.state_dict(), "optimizer": workspace.state_dict()})
    assert workspace.update_step == workspace.actor_optimizer_steps == workspace.temperature_optimizer_steps == 0


def selection_rows(roots=("r1", "r2"), seeds=(1, 2)):
    return [{"root_id": root, "solver_seed": seed, "mode": mode,
             "critic_updates": 3, "heldout_running_q_mse": {"batch_update": .1, "batch_no_update": .2, "running": .3}[mode]}
            for root in roots for seed in seeds for mode in probe.MODES]


def test_selection_compares_only_alternatives_and_tie_prefers_running():
    rows = selection_rows()
    selected, scores = probe.choose_mode(rows, ["r1", "r2"], [1, 2])
    assert selected == "batch_no_update"  # Control A is better but not an alternative.
    for row in rows:
        if row["mode"] == "running": row["heldout_running_q_mse"] = .2
    assert probe.choose_mode(rows, ["r1", "r2"], [1, 2])[0] == "running"


@pytest.mark.parametrize("problem", ["missing", "duplicate", "negative", "nan", "wrong_step"])
def test_selection_rejects_unpaired_or_nonfinite_rows(problem):
    rows = selection_rows()
    if problem == "missing": rows.pop()
    elif problem == "duplicate": rows.append(deepcopy(rows[0]))
    elif problem == "negative": rows[0]["heldout_running_q_mse"] = -1
    elif problem == "nan": rows[0]["heldout_running_q_mse"] = float("nan")
    else: rows[0]["critic_updates"] = 12
    with pytest.raises(ValueError): probe.choose_mode(rows, ["r1", "r2"], [1, 2])


def full_selection_fixture(tmp_path):
    roots = [{"root_id": f"seed-{seed}-decision-{step}", "seed": seed, "decision": step,
              "observation": [0.0]} for seed, step in probe.root_positions()]
    files, rows = [], []
    for root in roots:
        for seed in probe.SOLVER_SEEDS:
            relative = f"inputs/{root['root_id']}-{seed}.pt"
            path = tmp_path / relative
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(relative.encode())
            tensor_hash = probe.tree_hash(relative)
            files.append({"path": relative, "sha256": probe.source.file_sha256(path),
                          "root_id": root["root_id"], "solver_seed": seed, "tensor_sha256": tensor_hash})
            for step in probe.INSPECTION_STEPS:
                for mode in probe.MODES:
                    rows.append({"root_id": root["root_id"], "solver_seed": seed, "mode": mode,
                                 "critic_updates": step, "heldout_running_q_mse": .5,
                                 "diagnostics_state_unchanged": True, "paired_inputs_unchanged": True,
                                 "paired_inputs_tensor_sha256": tensor_hash})
    _, scores = probe.choose_mode(rows, [root['root_id'] for root in roots], probe.SOLVER_SEEDS)
    artifacts = {"roots": {"checkpoint_sha256": probe.source.CHECKPOINT_SHA, "roots": roots},
                 "paired_inputs": {"files": files},
                 "results": {"schema": probe.SCHEMA, "mode": "production", "root_count": 32, "source_sha": "a"*40,
                             "checkpoint_sha256": probe.source.CHECKPOINT_SHA,
                             "checkpoint_metadata_sha256": probe.CHECKPOINT_METADATA_SHA,
                             "target": probe.TARGET_KIND,
                             "inspection_steps": list(probe.INSPECTION_STEPS), "solver_seeds": list(probe.SOLVER_SEEDS),
                             "outer_state_unchanged": True, "global_rng_unchanged": True,
                             "outer_state_before_sha256": "b"*64, "outer_state_after_sha256": "b"*64,
                             "global_rng_sha256": "c"*64,
                             "records": rows}}
    selection = {"schema": probe.SELECTION_SCHEMA, "mode": "production", "source_sha": "a"*40,
                 "checkpoint_sha256": probe.source.CHECKPOINT_SHA, "root_count": 32,
                 "solver_seeds": list(probe.SOLVER_SEEDS), "inspection_steps": list(probe.INSPECTION_STEPS),
                 "outer_state_unchanged": True, "global_rng_unchanged": True,
                 "rule": "heldout_running_q_mse_at_3_updates", "selected_critic_bn_mode": "running", "aggregate_scores": scores}
    for key, artifact in artifacts.items():
        filename = key+".json"
        (tmp_path/filename).write_text(json.dumps(artifact))
        selection.update({key+"_file": filename, key+"_sha256": probe.source.file_sha256(tmp_path/filename)})
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(selection))
    return path, selection, artifacts


def test_complete_selection_artifacts_validate_and_tie_prefers_running(tmp_path):
    path, _, _ = full_selection_fixture(tmp_path)
    assert probe.validate_selection(path, source_sha="a"*40)["selected_critic_bn_mode"] == "running"
    with pytest.raises(ValueError): probe.validate_selection(path, source_sha="c"*40)


@pytest.mark.parametrize("problem", ["smoke", "hash", "paired_input", "missing_step", "duplicate_root", "selection", "nan", "target", "mutated_inputs"])
def test_selection_artifact_integrity_and_completeness(tmp_path, problem):
    path, selection, artifacts = full_selection_fixture(tmp_path)
    if problem == "smoke": selection["mode"] = "smoke"
    elif problem == "hash": selection["roots_sha256"] = "f"*64
    elif problem == "paired_input": (tmp_path/artifacts["paired_inputs"]["files"][0]["path"]).write_bytes(b"changed")
    elif problem == "selection": selection["selected_critic_bn_mode"] = "batch_no_update"
    else:
        key = "roots" if problem == "duplicate_root" else "results"
        if problem == "missing_step": artifacts[key]["records"].pop()
        elif problem == "duplicate_root": artifacts[key]["roots"][1] = deepcopy(artifacts[key]["roots"][0])
        elif problem == "target": artifacts[key]["target"] = "unprojected"
        elif problem == "mutated_inputs": artifacts[key]["records"][0]["paired_inputs_unchanged"] = False
        else: artifacts[key]["records"][0]["heldout_running_q_mse"] = float("nan")
        (tmp_path/selection[key+"_file"]).write_text(json.dumps(artifacts[key]))
        selection[key+"_sha256"] = probe.source.file_sha256(tmp_path/selection[key+"_file"])
    path.write_text(json.dumps(selection))
    with pytest.raises(ValueError): probe.validate_selection(path)


def test_smoke_driver_unpacks_frozen_loader_and_publishes_complete_artifacts(tmp_path, monkeypatch):
    import evaluate_ambi_checkpoint as evaluator
    import utils.ambi_research as research
    import utils.checkpoint_context as checkpoint_context
    import utils.ambi_benchmark as benchmark

    model = _wrapper(aux_return_mode="xqc", inner_critic_source="aux_return",
                     inner_horizon_critic_source="aux_return", inner_critic_target="reward_only",
                     inner_terminal_bootstrap="outer", inner_actor_bn_mode="running")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fixture")
    metadata = tmp_path / "checkpoint.pt.metadata.json"
    metadata.write_text("{}")
    real_hash = probe.source.file_sha256
    monkeypatch.setattr(probe.source, "file_sha256", lambda path: (
        probe.source.CHECKPOINT_SHA if str(path) == str(checkpoint) else
        probe.CHECKPOINT_METADATA_SHA if str(path) == str(metadata) else real_hash(path)))
    monkeypatch.setattr(benchmark, "code_identity", lambda: {"commit": "a"*40, "dirty": False})
    monkeypatch.setattr(checkpoint_context, "load_checkpoint_context", lambda path: {})
    monkeypatch.setattr(research, "resolve_preset", lambda *args, **kwargs: {"algorithm_config": {"alg_params": {}}})
    monkeypatch.setattr(evaluator, "_make_env", lambda resolved: model.env)
    monkeypatch.setattr(evaluator, "_initialize_frozen_model", lambda *args, **kwargs: (model, {}))
    observation, _ = model.env.reset(seed=101)
    monkeypatch.setattr(probe, "capture_roots", lambda *args, **kwargs: (
        [{"root_id": "seed-101-decision-0", "seed": 101, "decision": 0, "observation": observation.tolist()}], {"101": 0.0}))
    output = tmp_path / "output"
    before = probe.tree_hash(probe.rng_state())
    selection = probe.run(checkpoint, output, device="cpu", mode="smoke")
    assert selection["mode"] == "smoke" and (output / "PASS").is_file()
    assert len(json.loads((output / "results.json").read_text())["records"]) == 9
    assert probe.tree_hash(probe.rng_state()) == before
    with pytest.raises(ValueError): probe.validate_selection(output / "selection.json")
