from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from evaluate_ambi_transfer_campaign import listed_cells, parser, validate_args
from utils.transfer_campaign import (
    arm_initialization, blend_state, evaluate_episode, load_campaign, resolved_cell,
    summarize_episodes, validate_arm,
)


CAMPAIGN = Path(__file__).resolve().parents[1] / "configs/research/ambi_transfer_discovery_575k.json"


def test_campaign_enumerates_all_requested_cells_without_h3_fixation():
    campaign = load_campaign(CAMPAIGN)
    grid = listed_cells(campaign)
    assert len(grid) == 252
    assert len({row["name"] for row in grid}) == 252
    assert {row["H"] for row in grid} == {1, 2, 3}
    assert {row["J"] for row in grid} == {1, 2, 4, 6, 8, 10}
    assert [row["index"] for row in grid] == list(range(252))
    assert grid[0]["J"] == 10 and grid[-1]["J"] == 1
    assert campaign["seeds"] == [101, 102, 103]
    rho_pairs = {(arm["actor_rho"], arm["critic_rho"]) for name, arm in campaign["arms"].items()
                 if name.startswith("rho_")}
    assert rho_pairs == {(a, c) for a in (0., .5, 1.) for c in (0., .5, 1.)}


def test_cell_resolution_preserves_uniform_budget_and_replay_capacity():
    campaign = load_campaign(CAMPAIGN)
    base = dict(selector="base", algorithm_config=dict(alg_params=dict(inner_replay_capacity=3072)))
    original = deepcopy(base)
    for h, j in ((1, 1), (2, 8), (3, 10)):
        params = resolved_cell(base, campaign, horizon=h, rounds=j, arm="rho_a1_c1")["algorithm_config"]["alg_params"]
        assert params["inner_rollout_horizon"] == h and params["inner_rounds"] == j
        assert params["inner_first_action_rounds"] is None
        assert params["inner_rollouts_per_round"] == 128
        assert params["inner_critic_updates_per_round"] == 16
        assert params["inner_actor_updates_per_round"] == 4
        assert params["inner_batch_size"] == 256
        assert params["inner_replay_capacity"] == max(3072, h * j * 128)
        assert params["inner_actor_scope"] == params["inner_critic_scope"] == "action"
        assert params["compile"]
    assert base == original


def test_blends_use_fixed_prior_and_exact_endpoints_without_mutation():
    prior = {"w": torch.tensor([1., 3.]), "counter": torch.tensor(2)}
    donor = {"w": torch.tensor([5., -1.]), "counter": torch.tensor(2)}
    for rho, expected in ((0., [1., 3.]), (.5, [3., 1.]), (1., [5., -1.])):
        result = blend_state(prior, donor, rho)
        torch.testing.assert_close(result["w"], torch.tensor(expected))
        result["w"].add_(99.)
    torch.testing.assert_close(prior["w"], torch.tensor([1., 3.]))
    torch.testing.assert_close(donor["w"], torch.tensor([5., -1.]))


def test_interventions_reset_target_from_selected_online_except_full_state():
    campaign = load_campaign(CAMPAIGN)
    actor, critic = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
    engine = SimpleNamespace(cfg=SimpleNamespace(compile=False), _actor_base=actor, _critic_base=critic)
    donor = dict(modules={"actor": deepcopy(actor.state_dict()), "critic": deepcopy(critic.state_dict())},
                 replay={"opaque": "checked-by-engine"})
    for name, arm in campaign["arms"].items():
        options = arm_initialization(engine, arm, donor)
        if arm.get("full_state", False):
            assert options["learner_state"] is donor
            assert not {"actor", "critic", "target", "replay"}.intersection(options)
        else:
            assert options["target"] == "online"
        if name == "behavior_only":
            assert "actor" not in options and "critic" not in options
            assert options["collection_actor"] is donor["modules"]["actor"]
        if name in {"fresh_replay25", "joint_replay25"}:
            assert options["replay"] is donor["replay"] and options["replay_fraction"] == .25


def test_short_evaluations_must_be_explicitly_smoke_labeled():
    campaign = load_campaign(CAMPAIGN)
    args = parser().parse_args(["--checkpoint", "/tmp/not-loaded", "--horizon", "1", "--rounds", "1",
        "--arm", "rho_a0_c0", "--output-dir", "/tmp/not-written", "--max-steps", "3"])
    with pytest.raises(ValueError, match="smoke"):
        validate_args(args, campaign)
    args.smoke = True
    assert validate_args(args, campaign) == ([101, 102, 103], 3, 55)
    args.max_steps = None
    assert validate_args(args, campaign)[1] == 2


@pytest.mark.parametrize("arm_name", ["rho_a0_c0", "rho_a05_c05", "behavior_only", "fresh_replay25",
                                     "joint_replay25", "full_state_replay25", "joint_prior_anchors"])
def test_campaign_real_inner_sac_episode_preserves_backbone_and_carries_between_decisions(arm_name):
    from tests.test_ambi_inner_decoupling import _assert_tree_equal, _clone_tree
    from tests.test_ambi_root_local_sac import _model_from_params
    from tests.test_aux_critic_transfer import critic_params
    campaign = load_campaign(CAMPAIGN)
    wrapped = _model_from_params(critic_params("return", inner_critic_scope="action",
        inner_rounds=1, inner_rollout_horizon=2))
    try:
        frozen = _clone_tree(wrapped.agent.model.state_dict())
        records = []
        episode = evaluate_episode(wrapped, wrapped.env, campaign["arms"][arm_name], episode_seed=101,
            controller_seed=55, max_steps=3, on_step=records.append, smoke=True)
        assert episode["steps"] == episode["length"] == 3
        assert episode["episode_solver_seed"] == episode["solver_seed"] == 3818519826
        assert episode["return"] == episode["return_value"] == sum(row["reward"] for row in records)
        assert episode["smoke"] and episode["truncated_by_evaluator"]
        assert len(records) == 3 and all(np.isfinite(row["action"]).all() for row in records)
        _assert_tree_equal(frozen, wrapped.agent.model.state_dict())
        if "replay25" in arm_name:
            assert records[0]["metrics"]["inner_previous_replay_samples"] == 0
            assert records[1]["metrics"]["inner_previous_replay_samples"] > 0
        if arm_name == "full_state_replay25":
            assert records[0]["metrics"]["inner_transfer_full_state"] == 0
            assert records[1]["metrics"]["inner_transfer_full_state"] == 1
    finally:
        wrapped.close()


def test_summary_counts_episodes_not_decisions_as_samples():
    episodes = [dict(reward=value, steps=500, control_seconds=2.) for value in (1., 2., 3.)]
    result = summarize_episodes(episodes)
    assert result["episodes"] == 3 and result["mean_return"] == 2.
    assert result["se_return"] == pytest.approx(1. / np.sqrt(3.))
    assert result["total_steps"] == 1500


def test_invalid_full_state_fraction_and_unknown_mechanism_fail_closed():
    with pytest.raises(ValueError):
        validate_arm(dict(actor_rho=.5, critic_rho=1., full_state=True))
    with pytest.raises(ValueError):
        validate_arm(dict(actor_rho=1., critic_rho=1., replay_fraction=-.1))
    with pytest.raises(ValueError):
        validate_arm(dict(actor_rho=1., critic_rho=1., invented_mechanism=True))


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("metric_policy", ["legacy", "all_scalars"])
def test_output_bundle_is_exclusive_and_records_completion_or_failure(tmp_path, monkeypatch, fail, metric_policy):
    import json
    import evaluate_ambi_transfer_campaign as evaluator
    campaign = load_campaign(CAMPAIGN)
    output = tmp_path / "cell"
    args = parser().parse_args(["--campaign", str(CAMPAIGN), "--checkpoint", str(tmp_path / "checkpoint"),
        "--horizon", "1", "--rounds", "1", "--arm", "rho_a0_c0",
        "--output-dir", str(output), "--smoke", "--seeds", "101", "--no-compile",
        "--metric-policy", metric_policy])
    monkeypatch.setattr(evaluator, "load_preset_matrix", lambda path: {"checkpoint_contract": campaign["checkpoint_contract"]})
    monkeypatch.setattr(evaluator, "load_checkpoint_context", lambda *a, **k: SimpleNamespace(metadata={"checkpoint": {"step": 575000}}))
    monkeypatch.setattr(evaluator, "resolve_preset", lambda *a, **k: {"algorithm_config": {"alg_params": {}}, "selector": "test"})
    monkeypatch.setattr(evaluator, "_validate_checkpoint_contract", lambda *a: None)
    monkeypatch.setattr(evaluator, "_file_sha256", lambda path: "testhash")
    monkeypatch.setattr(evaluator, "source_identity", lambda: {"git_head": "test", "files": {}, "sha256": "test"})
    monkeypatch.setattr(evaluator, "_make_env", lambda *a: object())
    monkeypatch.setattr(evaluator, "_initialize_frozen_model", lambda *a, **k: (object(), {}))
    monkeypatch.setattr(evaluator, "_outer_state_digest", lambda *a: "frozen")
    monkeypatch.setattr(evaluator, "_close_resources", lambda *a: None)
    def episode(*a, **kwargs):
        assert kwargs["metric_policy"] == metric_policy
        kwargs["on_step"](dict(decision=0, cumulative_reward=2.))
        if fail:
            raise RuntimeError("controlled failure")
        return dict(seed=101, reward=3., steps=2, control_seconds=.01)
    monkeypatch.setattr(evaluator, "evaluate_episode", episode)
    if fail:
        with pytest.raises(RuntimeError, match="controlled failure"):
            evaluator.run(args)
        assert json.loads((output / "progress.json").read_text())["status"] == "failed"
        assert json.loads((output / "failure.json").read_text())["type"] == "RuntimeError"
        assert not (output / "results.json").exists()
    else:
        result = evaluator.run(args)
        assert result["complete"] and result["frozen_outer_verified"]
        assert json.loads((output / "progress.json").read_text())["status"] == "complete"
        assert json.loads((output / "results.json").read_text())["summary"]["episodes"] == 1
    with pytest.raises(FileExistsError):
        evaluator.run(args)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["metric_policy"] == metric_policy
    assert manifest["metric_coverage"] == dict(policy=metric_policy, computed_scalars_only=True,
                                               extra_solver_probes=False, per_update_traces=False)
    assert (output / "decisions-seed-101.jsonl").is_file()
