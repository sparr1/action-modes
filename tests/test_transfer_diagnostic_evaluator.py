"""End-to-end saved-root diagnostics on an actual tiny frozen checkpoint."""

from copy import deepcopy
from collections import Counter
import json

import numpy as np
import pytest
import torch

import evaluate_ambi_transfer_diagnostics as evaluator
from tests.test_ambi_actor_transfer_evaluation import transfer_matrix
from tests.test_aux_critic_transfer import critic_params
from tests.test_ambi_root_local_sac import _model_from_params
from utils.ambi_benchmark import solver_seed
from utils.transfer_diagnostics import Reference, evaluating, expected_reduction, solve_fork, validate_controller
from utils.transfer_diagnostic_coverage import _fork
import utils.transfer_diagnostics as diagnostics


def arguments(transfer_matrix, output, *extra):
    checkpoint, matrix = transfer_matrix
    specification = json.loads(matrix.read_text())
    specification["shared_alg_params"] = dict(inner_finite_horizon=True,
                                               inner_first_action_rounds=None)
    matrix.write_text(json.dumps(specification))
    return evaluator.parser().parse_args([
        "--checkpoint", str(checkpoint), "--matrix", str(matrix),
        "--preset", "transfer/cold", "--output-dir", str(output),
        "--horizons", "1", "2", "5", "--rounds", "1",
        "--histories", "fresh", "joint", "--seeds", "101",
        "--decisions", "1", "--max-steps", "2", "--mc-rollouts", "2",
        "--action-count", "4", "--fit-states", "4", "--fit-steps", "1",
        *extra,
    ])


def read_json(path):
    return json.loads(path.read_text())


def direct_episode(args, resolved, history):
    """Run the same two real decisions with no diagnostic forks in between."""
    env = wrapped = None
    try:
        env = evaluator._make_env(resolved)
        wrapped, _ = evaluator._initialize_frozen_model(
            resolved, env, args.checkpoint, args.controller_seed, device=args.device,
        )
        engine = wrapped.agent.inner_engine
        engine.reset_for_evaluation(solver_seed(args.controller_seed, "source", 101))
        outer = evaluator._outer_state_digest(wrapped)
        observation, _ = env.reset(seed=101)
        donor, roots, total = None, [], 0.
        for _ in range(2):
            roots.append(np.asarray(observation).copy())
            rng = deepcopy(engine.rng.training_state_dict())
            action, _, donor = solve_fork(wrapped, observation, rng, donor=donor,
                branch=history if donor is not None else "fresh", capture_rounds=())
            observation, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            if terminated or truncated:
                break
        assert evaluator._outer_state_digest(wrapped) == outer
        return total, roots
    finally:
        evaluator._close_resources(wrapped, env)


def test_end_to_end_horizon_forks_preserve_source_and_write_complete_evidence(transfer_matrix, tmp_path):
    output = tmp_path / "diagnostics"
    args = arguments(transfer_matrix, output)
    checksum = evaluator._file_sha256(args.checkpoint)
    result = evaluator.run(args)
    assert result["status"] == "complete"
    assert result["root_count"] == 6
    assert result["outer_state_unchanged"] is True
    assert result["horizons"] == [1, 2, 5]
    assert result["options"]["target_cross"] is True
    assert evaluator._file_sha256(args.checkpoint) == checksum
    assert read_json(output / "manifest.json") == result
    assert read_json(output / "manifest.started.json")["status"] == "resolved"
    assert "not measured real-return improvements" in (output / "report.md").read_text()
    summary = read_json(output / "summary.json")
    assert {row["horizon"] for row in summary} == {1, 2, 5}
    assert {row["history"] for row in summary} == {"fresh", "joint"}
    model_rows = [row for row in summary if row["metric"] == "final_model_actor_objective"]
    assert {row["data_lane"] for row in model_rows} == {"common", "natural"}
    for setting in result["settings"]:
        horizon = setting["horizon"]
        for history in args.histories:
            directory = output / f"h{horizon}-j1/{history}/seed-101"
            episode = read_json(directory / "episode.json")
            root = read_json(directory / "decision-1/diagnostics.json")
            assert episode["steps"] == 2
            assert episode["captured_decisions"] == [1]
            assert episode["unreached_decisions"] == []
            direct_return, direct_roots = direct_episode(args, setting["resolved"], history)
            assert episode["real_return"] == pytest.approx(direct_return, abs=0, rel=0)
            np.testing.assert_array_equal(root["observation"], direct_roots[1])
            np.testing.assert_array_equal(root["previous_observation"], direct_roots[0])
            diagnostic = root["diagnostics"]
            assert diagnostic["horizon"] == horizon
            assert np.asarray(diagnostic["actions"]).shape == (4, 1)
            assert diagnostic["real"] == diagnostic["replanning"] == []
            assert len(diagnostic["branches"]) == 8
            common = [row for row in diagnostic["branches"] if row["data_lane"] == "common"]
            assert {row["branch"] for row in common} == {"fresh", "actor", "critic", "joint"}
            assert all(row["replay_sha256"] == common[0]["replay_sha256"] for row in common)
            assert len(common[0]["replay_sha256"]) == 1
            for branch in diagnostic["branches"]:
                assert [stage["stage"] for stage in branch["stages"]] == [
                    "initial", "after_first_critic_block", "after_first_actor_block", "post_round",
                ]
                assert branch["final"]["stage"] == "pre_reset"
                assert len(branch["final"]["critic_checks"]) == 6
                assert np.asarray(branch["final"]["q_heads"]).shape == (2, 4)
                assert len(branch["final"]["directional"]) == 2
                assert len(branch["final"]["current_action_directional"]) == 2
                assert all(check["support"] == "current_mean" for check in branch["final"]["current_action_directional"])
                # Critic-only fitting cannot change a fixed-coefficient policy
                # return estimate when policy weights and diagnostic noise agree.
                initial, after_critic = branch["stages"][:2]
                assert after_critic["model_actor_objective"] == initial["model_actor_objective"]
            coefficients = {stage["model_objective_entropy_coefficient"]
                            for branch in diagnostic["branches"] for stage in branch["stages"]}
            assert len(coefficients) == 1
            assert {(row["online"], row["target"]) for row in diagnostic["target_cross"]} == {
                ("prior", "prior"), ("prior", "carried"), ("carried", "prior"), ("carried", "carried"),
            }
            assert diagnostic["portability"]["horizon_shift"]["applicable"] == (horizon > 1)
            assert len(diagnostic["portability"]["support"]) == 6
            if horizon > 1:
                shift = diagnostic["portability"]["horizon_shift"]
                assert (shift["old_horizon"], shift["new_horizon"]) == (horizon - 1, horizon)
                assert len(shift["actionwise_target_change"]) == 4
            assert set(diagnostic["stationary"]["curves"]) == {
                "critic_prior", "critic_carried", "actor_prior", "actor_carried",
            }
            assert all(len(curve) == 2 for curve in diagnostic["stationary"]["curves"].values())


def test_full_trace_probes_are_saved_for_every_root_fork_without_changing_source(
    transfer_matrix, tmp_path, monkeypatch,
):
    args = arguments(transfer_matrix, tmp_path / "full-traces", "--full-trace-probes")
    args.horizons, args.rounds, args.histories = [2], [2], ["joint"]
    calls = []
    original = diagnostics.solve_fork

    def observe(*positional, **options):
        calls.append(dict(options))
        return original(*positional, **options)

    monkeypatch.setattr(diagnostics, "solve_fork", observe)
    monkeypatch.setattr(evaluator, "solve_fork", observe)
    result = evaluator.run(args)
    assert result["options"]["full_trace_probes"] is True
    assert sum(call.get("full_trace_probes", False) for call in calls) == 12
    assert sum(not call.get("full_trace_probes", False) for call in calls) == 2
    probed = [call for call in calls if call.get("full_trace_probes", False)]
    assert {call["probe_rollouts"] for call in probed} == {args.mc_rollouts}
    assert len({call["probe_seed"] for call in probed}) == 1
    directory = args.output_dir / "h2-j2/joint/seed-101"
    root = read_json(directory / "decision-1/diagnostics.json")
    rows = root["diagnostics"]["branches"] + root["diagnostics"]["target_cross"]
    assert len(rows) == 12
    for row in rows:
        _fork(row, 2, result["options"], "actual-checkpoint-root", Counter())
        events = row["trace_events"]
        assert events and all(isinstance(event["metrics"], dict) for event in events)
        for phase, key in (("transfer_probe", "transfer_root_q_inner_actor_mean_all"),
                           ("probe", "togo_return_mean")):
            probes = [event for event in events if event["phase"] == phase]
            assert [(event["stage"], event["round_index"]) for event in probes] == [
                ("initial", 0), ("before_first_actor_block", 1),
                ("after_first_actor_block", 1), ("post_round", 1), ("post_round", 2),
            ]
            assert all(np.isfinite(event["metrics"][key]) for event in probes)
        # Scalar per-update observations survive JSON serialization too.
        assert any("critic_loss" in event["metrics"] for event in events)
        assert any("actor_loss" in event["metrics"] for event in events)
    episode = read_json(directory / "episode.json")
    direct_return, direct_roots = direct_episode(args, result["settings"][0]["resolved"], "joint")
    assert episode["real_return"] == pytest.approx(direct_return, abs=0, rel=0)
    np.testing.assert_array_equal(root["observation"], direct_roots[1])
    assert result["outer_state_unchanged"] is True


@pytest.mark.parametrize("key,value", [
    ("horizons", [0]), ("horizons", [1, 1]), ("rounds", [0]),
    ("decisions", [2]), ("mc_rollouts", 1), ("action_count", 3),
    ("fit_states", 3), ("capture_rounds", [-1]), ("capture_rounds", [1, 1]),
])
def test_malformed_options_fail_before_creating_output(transfer_matrix, tmp_path, monkeypatch, key, value):
    output = tmp_path / "invalid"
    args = arguments(transfer_matrix, output)
    setattr(args, key, value)
    monkeypatch.setattr(evaluator, "_make_env", lambda *a: pytest.fail("Invalid request allocated environment"))
    with pytest.raises(ValueError):
        evaluator.run(args)
    assert not output.exists()


def test_dry_run_loads_no_network_or_output_and_existing_bundle_is_refused(transfer_matrix, tmp_path, monkeypatch):
    output = tmp_path / "dry"
    args = arguments(transfer_matrix, output, "--dry-run")
    monkeypatch.setattr(evaluator, "_make_env", lambda *a: pytest.fail("Dry/existing output allocated environment"))
    # The full integration test exercises real source identity. This test owns
    # output/model-allocation behavior, independently of platform Git spawning.
    monkeypatch.setattr(evaluator, "source_identity", lambda: {"git_head": "fixture"})
    result = evaluator.run(args)
    assert result["status"] == "resolved"
    assert [row["horizon"] for row in result["settings"]] == [1, 2, 5]
    assert not output.exists()
    output.mkdir()
    sentinel = output / "existing.txt"
    sentinel.write_text("keep")
    args.dry_run = False
    with pytest.raises(FileExistsError):
        evaluator.run(args)
    assert list(output.iterdir()) == [sentinel]
    assert sentinel.read_text() == "keep"


@pytest.mark.parametrize("objective", ["return", "soft"])
@pytest.mark.parametrize("reduction", ["mean_pair", "min_pair"])
def test_reference_h1_equals_manual_reward_and_configured_frozen_tail(objective, reduction):
    wrapped = _model_from_params(critic_params(objective, inner_critic_scope="action",
                                                inner_rollout_horizon=1,
                                                mppi_terminal_q_reduction=reduction))
    try:
        reference = Reference(wrapped, rollouts=3)
        model, engine = reference.model, reference.engine
        latent = reference.encode(np.asarray([[1., .2, -.1]], dtype=np.float32))
        actions = torch.tensor([[-.3], [.4]])
        seed, count, batch = 173, 3, 2
        global_rng = torch.random.get_rng_state().clone()
        with torch.no_grad(), evaluating(model):
            z = latent.expand(count * batch, -1)
            repeated_actions = actions.unsqueeze(0).expand(count, -1, -1).reshape(-1, 1)
            joint = model.joint_input(z, repeated_actions)
            rewards = model.decode_reward(model.reward_from_joint(joint))
            successor = model.next_from_joint(joint)
            generator = torch.Generator().manual_seed(solver_seed(seed, "tail"))
            noise = torch.randn((1, count, 1, 1), generator=generator).expand(-1, -1, batch, -1).reshape(-1, 1)
            if objective == "soft":
                assert wrapped.cfg.outer_actor_entropy_mode == "squashed"
                tail_action, info = model.pi(successor, noise=noise)
                critic = model._Qs
                coefficient = wrapped.agent.alpha.detach()
                if wrapped.agent.actor_loss_scale_enabled:
                    coefficient = coefficient * wrapped.agent.actor_loss_scale.detach().reshape(())
                bonus = -coefficient * info["log_prob"]
            else:
                tail_action, _ = model.pi(successor, policy=engine._horizon_actor,
                                         noise=noise, **engine._horizon_actor_options)
                critic, bonus = engine._horizon_critic, 0.
            heads = model.Q(successor, tail_action, qs=critic._forward_eager, reduction="all")
            assert heads.shape[0] == 2 and wrapped.cfg.mppi_terminal_q_reduction == reduction
            terminal = heads.mean(0) if reduction == "mean_pair" else heads.min(0).values
            manual = (rewards + reference.discount * (terminal + bonus)).reshape(count, batch)
        donor = deepcopy(engine._actor_base)
        with torch.no_grad():
            for parameter in donor.parameters():
                parameter.add_(.8)
        # H1 has no interior actor action/entropy, including with a huge coefficient.
        for actor in (engine._actor_base, donor):
            actual = reference.returns(latent, actions, actor, (1,), seed=seed, coefficient=999.)[1]
            torch.testing.assert_close(actual, manual, rtol=0, atol=0)
        torch.testing.assert_close(torch.random.get_rng_state(), global_rng, rtol=0, atol=0)
    finally:
        wrapped.close()


def test_pair_alias_and_horizon_noise_prefix_are_consistent():
    values = torch.tensor([0., 2., 8.])[:, None, None]
    torch.testing.assert_close(expected_reduction(values, "min"), torch.tensor([[[0.]], [[0.]], [[2.]]]).mean(0))
    wrapped = _model_from_params(critic_params("return", inner_critic_scope="action"))
    try:
        reference = Reference(wrapped, rollouts=3)
        z = reference.encode(np.asarray([[1., .2, -.1]], dtype=np.float32))
        actions = torch.tensor([[-.3], [.4]])
        actor = reference.engine._actor_base
        individual = reference.returns(z, actions, actor, (2,), seed=37)[2]
        extended = reference.returns(z, actions, actor, (1, 2, 7), seed=37)[2]
        torch.testing.assert_close(individual, extended, rtol=0, atol=0)
        wrapped.cfg.inner_bootstrap_source = "outer_online"
        with pytest.raises(ValueError, match="inner_bootstrap_source"):
            validate_controller(wrapped)
    finally:
        wrapped.close()
