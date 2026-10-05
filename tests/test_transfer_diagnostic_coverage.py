"""The runtime receipt rejects missing evidence despite enabled CLI flags."""
from itertools import product
import json

import pytest

from utils.transfer_diagnostic_coverage import BRANCHES, INTERVENTIONS, LANES, verify_full_bundle


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def full_bundle(tmp_path):
    """Small on-disk fixture with all six families and both H applicability cases."""
    options = dict(full_trace_probes=True, target_cross=True, data_lanes=list(LANES), capture_rounds=None,
                   mc_rollouts=2, action_count=4, fit_steps=1, fit_states=4, real_rollouts=2,
                   real_tail_steps=3, replan_steps=2, replan_repeats=2)
    value_check = dict(bias=0., rmse=0., centered_rmse=0., relative_rmse=0., top_action_regret=0.,
                       samples=2, actions=4, reference_mean=[0.] * 4, prediction_mean=[0.] * 4,
                       reference_se=[0.] * 4, spearman=None, pearson=None)
    direction = dict(predicted_derivative=0., reference_slope=0., reference_slope_se=0., gain=0., gain_se=0., samples=2)
    snapshot = dict(critic_checks=[dict(continuation=continuation, critic=critic, **value_check)
                    for continuation, critic in product(("prior", "carried", "current"), ("online", "target"))],
                    directional=[dict(support="prior_mean", epsilon=epsilon, **direction) for epsilon in (.03, .1)],
                    current_action_directional=[dict(support="current_mean", epsilon=epsilon, **direction) for epsilon in (.03, .1)],
                    model_actor_objective=0., critic_actor_objective=0., current_mean_critic_value=0.,
                    current_mean_reference_return=0., current_mean_reference_se=0., q_heads=[[0.] * 4] * 2)
    feature = dict(effective_rank=1., inactive_fraction=0.)
    stationary = {name: [dict(step=step, train_rmse=1., heldout_rmse=1., parameter_drift_l2=0.,
                        gradient_norm=0. if step else None, features=feature) for step in range(2)]
                  for name in ("critic_prior", "critic_carried", "actor_prior", "actor_carried")}
    episodes, root_paths = [], []
    for h, j in product((1, 2), (1, 3)):
        stage_pairs = [("initial", 0), ("after_first_critic_block", 1), ("after_first_actor_block", 1)]
        stage_pairs.extend(("post_round", index) for index in range(1, j + 1))
        probe_pairs = [("initial", 0), ("before_first_actor_block", 1), ("after_first_actor_block", 1)]
        probe_pairs.extend(("post_round", index) for index in range(1, j + 1))
        trace = [dict(phase="update", updated_actor=True, updated_critic=True, metrics={"loss": 0.})]
        for stage, index in probe_pairs:
            trace.extend([
                dict(phase="transfer_probe", measurement="transfer_boundary_root_probe", stage=stage,
                     round_index=index, metrics=dict(transfer_policy_kl_vs_prior=0., transfer_mean_action_delta_l2=0.,
                     transfer_actor_std_mean=1., transfer_root_q_inner_actor_mean_all=0.,
                     transfer_root_q_target_actor_mean_all=0., transfer_root_q_frozen_actor_mean_all=0.)),
                dict(phase="probe", measurement="post_update_togo_probe", stage=stage, round_index=index,
                     metrics=dict(togo_return_mean=0., togo_reward_mean=0., togo_bootstrap_mean=0.,
                                  togo_return_gain_vs_initial=0., togo_return_gain_vs_outer=0., probe_model_steps=2.)),
            ])
        fork = dict(stages=[dict(stage=stage, round=index, **snapshot) for stage, index in stage_pairs],
                    final=dict(stage="pre_reset", round=j, **snapshot), trace_events=trace)
        root_id = f"h{h}-j{j}/joint/seed-101/decision-1"
        directory = tmp_path / root_id
        real = [dict(branch=branch, replicate=replicate, prefix_complete=True, mc_complete=True,
                    prefix_decisions=h, tail_decisions=3, real_bootstrapped_return=0., real_mc_return=0.,
                    predicted_model_return=0., model_prefix_error=0., terminal_value_error=0., total_prediction_error=0.)
                for branch, replicate in product(BRANCHES, range(2))]
        replans = [dict(branch=branch, intervention=mode, future_replicate=replicate,
                       steps=2, steps_requested=2, requested_horizon_complete=True,
                       rewards=[0., 0.], actions=[[0.], [0.]], real_return=0., real_discounted_return=0.)
                   for branch, mode, replicate in product(BRANCHES, INTERVENTIONS, range(2))]
        shift = dict(applicable=h > 1)
        if h > 1:
            shift.update(old_horizon=h - 1, new_horizon=h, actionwise_target_change=[0.] * 4,
                         old_slice=value_check, new_slice=value_check)
        root = dict(root_id=root_id, horizon=h, rounds=j, history="joint", episode_seed=101, decision=1,
                    diagnostics=dict(horizon=h,
                        branches=[dict(data_lane=lane, branch=branch, replay_sha256=["digest"] * j, **fork)
                                  for lane, branch in product(LANES, BRANCHES)],
                        target_cross=[dict(online=online, target=target, **fork)
                                      for online, target in product(("prior", "carried"), repeat=2)],
                        portability=dict(horizon_shift=shift,
                            support=[dict(support=support, continuation=continuation, **value_check)
                                for support, continuation in product(("previous_root", "imagined_successor", "actual_successor"),
                                                                     ("prior", "carried"))]),
                        stationary=dict(curves=stationary), real=real, replanning=replans))
        root_paths.append(directory / "diagnostics.json")
        write(root_paths[-1], root)
        (directory / "donor.pt").write_bytes(b"Never deserialize coverage fixtures")
        write(directory / "simulator.json", dict(schema_version=1, sha256="digest", state={"fixture": True}))
        episode = dict(horizon=h, rounds=j, history="joint", seed=101, captured_decisions=[1], unreached_decisions=[])
        episodes.append(episode)
        write(directory.parent / "episode.json", episode)
    manifest = dict(status="complete", outer_state_unchanged=True, options=options, horizons=[1, 2],
                    rounds=[1, 3], histories=["joint"], episode_seeds=[101], decisions=[1],
                    episodes=episodes, root_count=4)
    write(tmp_path / "manifest.json", manifest)
    return tmp_path, root_paths


def change(path, mutate):
    value = json.loads(path.read_text())
    mutate(value)
    write(path, value)


def test_complete_bundle_receipt_counts_all_work(full_bundle):
    output, _ = full_bundle
    receipt = verify_full_bundle(output)
    assert receipt["verified"] is receipt["full_requested_workload"] is True
    assert receipt["roots"] == receipt["episodes"] == 4
    assert receipt["initialization_forks"] == 32
    assert receipt["target_cross_forks"] == 16
    assert receipt["snapshots_scored"] == 288
    assert receipt["critic_checks"] == 288 * 6
    assert receipt["directional_checks"] == 288 * 4
    assert receipt["transfer_boundary_root_probe"] == receipt["post_update_togo_probe"] == 240
    assert receipt["stationary_points"] == 32
    assert receipt["real_rollouts"] == 32
    assert receipt["replanning_rollouts"] == 96
    assert receipt["replanning_decisions"] == 192


@pytest.mark.parametrize("mutate,match", [
    (lambda m: m.update(status="resolved"), "not complete"),
    (lambda m: m.update(outer_state_unchanged=False), "frozen outer"),
    (lambda m: m["options"].update(full_trace_probes=False), "full_trace_probes"),
    (lambda m: m["options"].update(real_rollouts=0), "real_rollouts"),
    (lambda m: m["options"].update(capture_rounds=[1]), "all-round"),
    (lambda m: m["options"].update(data_lanes=["natural"]), "both data lanes"),
    (lambda m: m.update(root_count=3), "root count"),
])
def test_requested_flags_and_manifest_are_not_enough(full_bundle, mutate, match):
    output, _ = full_bundle
    change(output / "manifest.json", mutate)
    with pytest.raises(ValueError, match=match):
        verify_full_bundle(output)


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d["branches"].pop(), "branches"),
    (lambda d: d["branches"][0]["stages"].pop(), "learner stages"),
    (lambda d: d["branches"][0]["final"]["critic_checks"].pop(), "critic checks"),
    (lambda d: d["branches"][0]["final"]["directional"].pop(), "directional"),
    (lambda d: d["branches"][0].update(trace_events=[]), "trace events"),
    (lambda d: d["branches"][0]["trace_events"].pop(), "post_update_togo_probe"),
    (lambda d: d["branches"][0]["trace_events"][0].update(metrics={}), "update metrics"),
    (lambda d: d["branches"][1].update(replay_sha256=["different"]), "replay mismatch"),
    (lambda d: d["target_cross"].pop(), "target crossing"),
    (lambda d: d["portability"]["support"].pop(), "portability"),
    (lambda d: d["portability"]["horizon_shift"].update(applicable=True), "applicability"),
    (lambda d: d["stationary"]["curves"]["actor_carried"].pop(), "actor_carried"),
    (lambda d: d["real"].pop(), "real calibration"),
    (lambda d: d["real"][0].update(real_mc_return=None), "real_mc_return"),
    (lambda d: d["real"][0].update(tail_decisions=2), "ended before full timing workload"),
    (lambda d: d["replanning"].pop(), "replanning"),
    (lambda d: d["replanning"][0].update(steps=1), "ended before full timing workload"),
])
def test_missing_or_incomplete_saved_measurements_fail(full_bundle, mutate, match):
    output, paths = full_bundle
    change(paths[0], lambda root: mutate(root["diagnostics"]))
    with pytest.raises(ValueError, match=match):
        verify_full_bundle(output)


@pytest.mark.parametrize("filename", ["donor.pt", "simulator.json", "diagnostics.json", "episode.json"])
def test_missing_artifacts_fail(full_bundle, filename):
    output, paths = full_bundle
    directory = paths[0].parent.parent if filename == "episode.json" else paths[0].parent
    (directory / filename).unlink()
    with pytest.raises(ValueError, match="missing"):
        verify_full_bundle(output)


def test_unreached_root_fails_even_if_saved_root_exists(full_bundle):
    output, paths = full_bundle
    change(paths[0].parent.parent / "episode.json", lambda episode: episode.update(unreached_decisions=[1]))
    with pytest.raises(ValueError, match="unreached"):
        verify_full_bundle(output)
