"""Full-episode transfer recording, resets, timing, and publication identity."""
import json
from pathlib import Path

import pytest

import evaluate_ambi_checkpoint as evaluator
from tests.test_ambi_benchmark_evaluation import events
from tests.test_ambi_root_local_sac import _tiny_component_model, _tiny_params
from utils.eval_series_data import _metrics, planner_identity


@pytest.fixture
def transfer_matrix(tmp_path):
    options = dict(aux_return_mode="sac", aux_return_ent_coef=0,
        inner_critic_source="aux_return", inner_horizon_critic_source="aux_return",
        inner_rounds=1, inner_first_action_rounds=3,
        inner_critic_updates_per_round=2, inner_actor_updates_per_round=1,
        inner_actor_scope="episode", inner_rebase_persistent=False,
        inner_rollouts_per_round=2, inner_rollout_horizon=1, inner_replay_capacity=12)
    model = _tiny_component_model(**options)
    checkpoint = tmp_path / "tiny.pt"
    model.agent.save(checkpoint)
    model.env.close()
    params = _tiny_params(**options)
    params.pop("inner_updates_per_round")
    config = dict(alg="AMBITDMPC2/AMBITDMPC2", env="Pendulum-v1", seed=13,
                  device="cpu", total_steps=10, alg_params=params)
    Path(str(checkpoint) + ".metadata.json").write_text(json.dumps({
        "schema_version": 1, "trial_run_params": config,
        "experiment_params": {"env_params": {"max_episode_steps": 3}},
        "checkpoint": {"kind": "periodic", "step": 10, "episode": 2,
                       "best_score": None, "best_window": 1}}))
    matrix = dict(schema_version=1, base_alg_config="checkpoint", study_protocol="actor-transfer-v1",
        evaluation=dict(seeds=[101, 102], controller_seed=55, max_steps=3,
                        togo_return_rollouts=2, actor_transfer_diagnostics=True,
                        default_presets=["transfer/warm"]),
        comparisons={"transfer": {"reference": "cold", "variants": {
            "cold": {"alg_params": {"inner_actor_scope": "action"}},
            "warm": {"alg_params": {"inner_actor_scope": "episode"}}}}})
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(matrix))
    return checkpoint, path


def test_full_episode_transfer_counts_resets_timing_and_seed_order(transfer_matrix, tmp_path):
    checkpoint, matrix = transfer_matrix
    returns = []
    for index, seeds in enumerate(([101, 102], [102, 101])):
        bundle = tmp_path / f"bundle-{index}"
        result = evaluator.evaluate_matrix(matrix, checkpoint, seeds=seeds, bundle_dir=bundle)["results"][0]
        assert result["outer_state_unchanged"]
        returns.append({ep["seed"]: ep["return"] for ep in result["episodes"]})
        decisions = [row for row in events(bundle) if row["phase"] == "decision"]
        assert [row["metrics"]["decision/inner_rounds"] for row in decisions] == [3, 1, 1] * 2
        assert [row["metrics"]["decision/inner_actor_transferred"] for row in decisions] == [0, 1, 1] * 2
        boundaries = [row for row in events(bundle) if row["phase"] == "transfer_probe"]
        assert {row["stage"] for row in boundaries} == {
            "initial", "before_first_actor_block", "after_first_actor_block", "post_round"}
        for ep in result["episodes"]:
            timing = ep["transfer_latency"]
            assert timing["first"]["control_seconds"]["count"] == 1
            assert timing["steady"]["control_seconds"]["count"] == 2
            for row in timing["samples"]:
                assert row["diagnostic_seconds"] > 0
                assert row["prediction_seconds"] == pytest.approx(row["control_seconds"] + row["diagnostic_seconds"])
            assert sum(row["control_seconds"] for row in timing["samples"]) == pytest.approx(ep["control_seconds"])
        metrics = _metrics(result["episodes"])
        assert metrics["runtime/first_decisions"] == 2
        assert metrics["runtime/steady_decisions"] == 4
        assert metrics["work/critic_updates"] == 20
        assert metrics["work/actor_updates"] == 10
    assert returns[0] == returns[1]


def test_transfer_scope_requires_named_episode_protocol(transfer_matrix, tmp_path):
    checkpoint, path = transfer_matrix
    matrix = json.loads(path.read_text())
    matrix["evaluation"]["actor_transfer_diagnostics"] = False
    path.write_text(json.dumps(matrix))
    with pytest.raises(ValueError, match="fresh action-local"):
        evaluator.evaluate_matrix(path, checkpoint, bundle_dir=tmp_path / "invalid")


def test_transfer_identity_includes_first_dose_and_scope():
    config = dict(inner_operator="sac", inner_rounds=1, inner_actor_scope="action")
    identity = lambda c: planner_identity(c, {}, "AMBITDMPC2/AMBITDMPC2", "tanh_mean")
    assert identity(config) == identity({**config, "inner_first_action_rounds": None})
    cold = identity({**config, "inner_first_action_rounds": 10})
    warm = identity({**config, "inner_first_action_rounds": 10, "inner_actor_scope": "episode"})
    assert cold != warm
    assert cold["semantics"]["evaluation_protocol"] == "actor-transfer-v1"


def test_togo_stage_boundaries_do_not_merge():
    rows = [dict(round_index=1, actor_updates=0, critic_updates=2,
                 stage=stage, metrics={"value": value})
            for stage, value in (("initial", 1), ("before_first_actor_block", 5))]
    assert len(evaluator._togo_round_summaries(rows)) == 2
