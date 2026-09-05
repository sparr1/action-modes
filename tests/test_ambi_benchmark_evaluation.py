"""Small real-model checks of the frozen benchmark workflow, without training."""

import gzip
import json
from pathlib import Path

import pytest

import evaluate_ambi_checkpoint as evaluator
from tests.test_ambi_root_local_sac import _tiny_component_model, _tiny_params
from utils.ambi_benchmark import read_json


@pytest.fixture
def checkpoint_matrix(tmp_path):
    model = _tiny_component_model(inner_rounds=2, inner_critic_updates_per_round=2,
                                  inner_actor_updates_per_round=1)
    checkpoint = tmp_path / "tiny.pt"
    model.agent.save(str(checkpoint))
    model.env.close()
    params = _tiny_params()
    params.pop("inner_updates_per_round")
    config = {"alg": "AMBITDMPC2/AMBITDMPC2", "env": "Pendulum-v1", "seed": 13,
              "device": "cpu", "total_steps": 10, "alg_params": params}
    sidecar = {"schema_version": 1, "trial_run_params": config,
               "experiment_params": {"env_params": {"max_episode_steps": 3}},
               "checkpoint": {"kind": "periodic", "step": 10, "episode": 2,
                              "best_score": None, "best_window": 1}}
    Path(str(checkpoint) + ".metadata.json").write_text(json.dumps(sidecar))
    matrix = {"schema_version": 1, "base_alg_config": "checkpoint",
              "evaluation": {"seeds": [101, 102], "controller_seed": 55, "max_steps": 3,
                             "default_presets": ["budget/prior"]},
              "shared_alg_params": {"inner_rounds": 2, "inner_rollouts_per_round": 2,
                                    "inner_critic_updates_per_round": 2,
                                    "inner_actor_updates_per_round": 1},
              "comparisons": {"budget": {"reference": "prior", "variants": {
                  "prior": {"alg_params": {"inner_operator": "none", "inner_rounds": None,
                            "inner_rollouts_per_round": None, "inner_critic_updates_per_round": None,
                            "inner_actor_updates_per_round": None, "inner_temperature_mode": "inherit_outer"}},
                  "sac": {"alg_params": {"inner_operator": "sac"}}
              }}}}
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix))
    return checkpoint, matrix_path


def events(bundle):
    manifest = read_json(bundle / "manifest.json")
    rows = []
    for run in manifest["runs"]:
        for path in run["trace_files"]:
            with gzip.open(bundle / path, "rt") as stream:
                rows.extend(json.loads(line) for line in stream)
    return rows


def test_baseline_bank_screen_and_episode_confirmation(checkpoint_matrix, tmp_path):
    checkpoint, matrix = checkpoint_matrix
    prior = tmp_path / "prior"
    bank = tmp_path / "bank.json"
    baseline = evaluator.evaluate_matrix(matrix, checkpoint, bundle_dir=prior, save_root_bank=bank)
    assert all(result["outer_state_unchanged"] for result in baseline["results"])
    bank_value = read_json(bank)
    assert bank_value["complete"]
    assert [root["root_id"] for root in bank_value["roots"]] == ["seed-101-decision-0", "seed-102-decision-0"]
    assert not any(row["phase"] == "update" for row in events(prior))

    screen = tmp_path / "screen"
    result = evaluator.evaluate_matrix(matrix, checkpoint, selectors=["budget/sac"], bundle_dir=screen,
                                       root_bank_path=bank, bank_only=True, bank_repetitions=2)
    assert result["results"][0]["episodes"] == []
    assert result["results"][0]["bank_solve_count"] == 4
    assert result["results"][0]["bank_metrics"]["probe_model_steps"]["mean"] == 96
    screen_manifest = read_json(screen / "manifest.json")
    assert screen_manifest["status"] == "complete"
    assert len(screen_manifest["runs"][0]["roots"]) == 4
    assert sum(row["phase"] == "probe" for row in events(screen)) == 12
    assert screen_manifest["metric_catalog"]["actor_loss"]["preferred_axis"] == "actor_updates"

    confirm = tmp_path / "confirm"
    evaluation = evaluator.evaluate_matrix(matrix, checkpoint, selectors=["budget/sac"], bundle_dir=confirm,
                                           reference_bundle=prior)
    manifest = read_json(confirm / "manifest.json")
    assert manifest["status"] == "complete"
    for episode, base in zip(manifest["runs"][0]["episodes"], baseline["results"][0]["episodes"]):
        assert episode["paired_return_delta"] == pytest.approx(episode["return"] - base["return"])
    assert evaluation["results"][0]["outer_state_unchanged"]
    from report_ambi_benchmark import load_bundles
    # Validate actual recorder catalogs and full trace rows against the viewer.
    load_bundles([prior, screen, confirm])


def test_episode_order_does_not_change_returns(checkpoint_matrix, tmp_path):
    checkpoint, matrix = checkpoint_matrix
    results = []
    for index, seeds in enumerate(([101, 102], [102, 101])):
        result = evaluator.evaluate_matrix(matrix, checkpoint, selectors=["budget/sac"], seeds=seeds,
                                           bundle_dir=tmp_path / str(index))
        results.append({episode["seed"]: episode["return"] for episode in result["results"][0]["episodes"]})
    assert results[0] == results[1]


def test_reference_mismatch_and_existing_bundle_fail_before_model_creation(checkpoint_matrix, tmp_path, monkeypatch):
    checkpoint, matrix = checkpoint_matrix
    prior = tmp_path / "prior"
    evaluator.evaluate_matrix(matrix, checkpoint, bundle_dir=prior)
    monkeypatch.setattr(evaluator, "_make_env", lambda *_: pytest.fail("invalid request constructed environment"))
    with pytest.raises(FileExistsError, match="already exists"):
        evaluator.evaluate_matrix(matrix, checkpoint, bundle_dir=prior)
    with pytest.raises(ValueError, match="protocol does not match"):
        evaluator.evaluate_matrix(matrix, checkpoint, bundle_dir=tmp_path / "bad",
                                   reference_bundle=prior, max_steps=2)
    assert not (tmp_path / "bad").exists()


def test_interrupted_episode_keeps_readable_partial_trace(checkpoint_matrix, tmp_path, monkeypatch):
    checkpoint, matrix = checkpoint_matrix
    original = evaluator._make_env
    def fail_on_second_step(resolved):
        env = original(resolved)
        step = env.step
        calls = 0
        def wrapped(action):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected environment failure")
            return step(action)
        env.step = wrapped
        return env
    monkeypatch.setattr(evaluator, "_make_env", fail_on_second_step)
    bundle = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="injected environment failure"):
        evaluator.evaluate_matrix(matrix, checkpoint, selectors=["budget/sac"], bundle_dir=bundle)
    manifest = read_json(bundle / "manifest.json")
    assert manifest["status"] == "failed"
    assert manifest["runs"][0]["episodes"][0]["status"] == "failed"
    assert manifest["runs"][0]["episodes"][0]["length"] == 1
    assert max(row["decision_index"] for row in events(bundle)) == 1
    from report_ambi_benchmark import load_bundles
    load_bundles([bundle])


def test_checkpoint_materialization_does_not_start_evaluation(checkpoint_matrix, tmp_path, monkeypatch):
    checkpoint, matrix = checkpoint_matrix
    monkeypatch.setattr(evaluator, "evaluate_matrix", lambda *a, **k: pytest.fail("materialization ran a workload"))
    assert evaluator.main(["--matrix", str(matrix), "--checkpoint", str(checkpoint),
                           "--preset", "budget/sac", "--materialize-dir", str(tmp_path / "configs")]) == 0
    assert (tmp_path / "configs/budget__sac.json").is_file()


def test_selected_prior_supplies_deltas_before_candidate_publication(checkpoint_matrix, tmp_path):
    checkpoint, matrix = checkpoint_matrix
    bundle = tmp_path / "paired"
    evaluator.evaluate_matrix(matrix, checkpoint, selectors=["budget/sac", "budget/prior"], bundle_dir=bundle)
    runs = read_json(bundle / "manifest.json")["runs"]
    assert [run["selector"] for run in runs] == ["budget/prior", "budget/sac"]
    assert runs[1]["result"]["paired_return_delta_vs_prior"]["count"] == 2
    assert all("paired_return_delta" in episode for episode in runs[1]["episodes"])
