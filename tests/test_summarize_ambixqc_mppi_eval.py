"""Campaign aggregation accepts only complete, paired, frozen, matching inputs."""

import copy
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import summarize_ambixqc_mppi_eval as campaign
from utils import ambi_benchmark as storage


SOURCE = "1" * 40
SEEDS = [101, 102]


def _controller():
    params = json.loads(campaign.MATRIX.read_text())["comparisons"]["controller"]["variants"]["mppi"]["evaluation_controller"]["params"]
    return {"type": "mppi", "settings": {**params, "effective_iterations": 8}, "protocol": {
        "algorithm": "tdmpc2_mppi_over_frozen_xqc", "action_rule": storage.MPPI_ACTION_RULE,
        "terminal_value_source": "online_xqc_twin_mean",
        "terminal_value_units": "normalized_xqc_soft_q_times_frozen_real_reward_scale",
        "terminal_value_semantics": "learned_soft_q_tail_without_entropy_correction",
        "reward_units": "raw_environment_reward", "reward_scale": 2.5, "discount": 0.99,
        "batchnorm_mode": "running", "warm_start": "shift_previous_mean_within_episode_reset_before_episode",
        "rng": "private_device_generator", "value_finite_guard": "torch.nan_to_num(nan=0)",
        "upstream_tdmpc2_commit": "8bbc14ebabdb32ea7ada5c801dc525d0dc73bafe",
    }}


@pytest.fixture
def outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "code_identity", lambda: {
        "commit": SOURCE, "dirty": False, "source_sha256": "b" * 64, "diff_sha256": "c" * 64,
        "runtime": {"python": "3.10", "torch": "2.3.1", "numpy": "1.26", "gymnasium": "1.0"},
    })
    manifest = {"source_run": campaign.SOURCE_RUN, "checkpoints": [
        {"step": 50000, "path": "/checkpoint/50000.pt", "sha256": "a" * 64, "metadata_sha256": "e" * 64},
        {"step": 100000, "path": "/checkpoint/100000.pt", "sha256": "d" * 64, "metadata_sha256": "f" * 64},
    ]}
    manifest_path = tmp_path / "checkpoints.json"
    storage.atomic_json(manifest_path, manifest)
    results_root = tmp_path / "results"
    config = {"alg": "AMBIXQC/AMBIXQC", "env": "DMControl-v0", "alg_params": {"inner_operator": "none", "obs": "state"}}
    for index, entry in enumerate(manifest["checkpoints"]):
        destination = results_root / "production" / f'step_{entry["step"]}'
        resolved = {"selector": "controller/prior", "algorithm_config": copy.deepcopy(config),
                    "environment": {"id": "DMControl-v0", "params": {"task": "humanoid-walk", "obs": "state"}}}
        bundle = storage.BenchmarkBundle(destination / "bundle", checkpoint={
            **entry, "metadata": {"checkpoint": {"step": entry["step"]}},
        }, protocol=storage.protocol_for(resolved, 12345, 2))
        for kind in ("prior", "mppi"):
            resolved["selector"] = f"controller/{kind}"
            resolved["algorithm_config"] = copy.deepcopy(config)
            if kind == "mppi":
                controller = _controller()
                controller["protocol"]["reward_scale"] += index
                resolved["algorithm_config"]["evaluation_controller"] = {"type": "mppi", "params": {
                    key: value for key, value in controller["settings"].items() if key != "effective_iterations"
                }}
            run = bundle.start_run(resolved, "episodes")
            if kind == "mppi":
                bundle.set_evaluation_controller(run, controller)
            for seed, prior, gain in ((101, 10., 3.), (102, 20., 5.)):
                value = prior + index + (gain if kind == "mppi" else 0)
                seconds = .2 if kind == "mppi" else .02
                events = [{"episode_id": f"seed-{seed}", "decision_index": decision, "event_index": 0,
                           "phase": "decision", "round_index": 0, "critic_updates": 0, "actor_updates": 0,
                           "temperature_updates": 0, "metrics": {"decision/reward": value / 2,
                           "decision/control_seconds": seconds / 2,
                           "decision/inner_model_steps": 12336 if kind == "mppi" else 0,
                           "decision/inner_mppi_iterations": 8 if kind == "mppi" else 0}}
                          for decision in range(2)]
                bundle.episode(run, {"seed": seed, "solver_seed": storage.solver_seed(12345, "episode", seed),
                                     "return": value, "length": 2, "control_seconds": seconds,
                                     "terminated": False, "truncated": True, "truncated_by_evaluator": True,
                                     "model_metrics": {}}, events)
                if kind == "mppi":
                    run["episodes"][-1]["paired_return_delta"] = gain
            bundle.finish_run(run, {"outer_state_unchanged": True, "outer_updates_before": entry["step"],
                                     "outer_updates_after": entry["step"], "controller": kind,
                                     "environment_seeds": SEEDS, "controller_seed": 12345, "seed_scheme": "sha256-v1",
                                     "action_rule": storage.MPPI_ACTION_RULE if kind == "mppi" else "tanh_mean",
                                     "deterministic_execution": kind == "prior"})
        bundle.finish()
        storage.atomic_json(destination / "provenance.json", {
            "source_run": campaign.SOURCE_RUN, "checkpoint": entry,
            "checkpoint_manifest_sha256": campaign.file_sha256(manifest_path),
            "matrix_sha256": campaign.file_sha256(campaign.MATRIX), "mode": "production",
            "seeds": SEEDS, "max_steps": 2, "controller_seed": 12345,
        })
    return manifest_path, results_root


def _summarize(outputs, **kwargs):
    return campaign.summarize(*outputs, expected_source_sha=SOURCE, seeds=SEEDS, max_steps=2, **kwargs)


def _change(outputs, change):
    path = outputs[1] / "production/step_100000/bundle/manifest.json"
    manifest = json.loads(path.read_text())
    change(manifest)
    storage.atomic_json(path, manifest, overwrite=True)


def test_paired_campaign_means_population_std_and_raw_reward_timing(outputs):
    summary = _summarize(outputs)
    assert summary["status"] == "complete"
    first, last = summary["rows"]
    assert first["prior"]["return_mean"] == 15
    assert first["prior"]["return_std"] == 5
    assert first["mppi"]["return_mean"] == 19
    assert first["mppi"]["return_std"] == 6
    assert first["paired"] == {"delta_mean": 4, "delta_std": 1, "deltas": [3, 5]}
    assert first["mppi"]["control_seconds_per_decision"] == .1
    assert last["mppi_reward_scale"] == 3.5  # Frozen scales legitimately differ by checkpoint.
    assert last["outer_state_unchanged"] is True
    rendered = campaign.render_campaign_html(summary)
    assert "100,000" in rendered and "MPPI minus prior" in rendered
    assert "raw environment rewards" in rendered and "J8" in rendered
    assert '<script src=' not in rendered and '<link ' not in rendered
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize("change,match", [
    (lambda m: m.update(status="failed"), "incomplete"),
    (lambda m: m["code"].update(commit="3" * 40), "source"),
    (lambda m: m["code"].update(dirty=True), "source"),
    (lambda m: m["code"]["runtime"].update(torch="other"), "inconsistent"),
    (lambda m: m["checkpoint"].update(sha256="1" * 64), "identity"),
    (lambda m: m["checkpoint"].update(metadata_sha256="1" * 64), "identity"),
    (lambda m: m["protocol"].update(controller_seed=55), "protocol"),
    (lambda m: m["runs"][0].update(status="failed"), "full outer state"),
    (lambda m: m["runs"][0]["result"].update(outer_state_unchanged=False), "full outer state"),
    (lambda m: m["runs"][0]["result"].update(outer_updates_after=100001), "full outer state"),
    (lambda m: m["runs"][0]["episodes"][0].update(seed=103), "seeds"),
    (lambda m: m["runs"][0]["episodes"][0].update(solver_seed=1), "solver seed"),
    (lambda m: m["runs"][0]["episodes"][0].update(status="failed"), "did not complete"),
    (lambda m: m["runs"][0]["episodes"][0].update(control_seconds=2), "decision timings"),
    (lambda m: m["runs"][1]["episodes"][0].update(paired_return_delta=999), "Paired return delta"),
    (lambda m: m["runs"][1]["result"].update(deterministic_execution=True), "determinism"),
    (lambda m: m["runs"][1]["evaluation_controller"]["settings"].update(num_samples=256), "authored"),
    (lambda m: m["runs"][1]["evaluation_controller"]["protocol"].update(reward_scale=None), "finite positive"),
])
def test_rejects_failed_incomplete_foreign_or_unpaired_data_even_allow_partial(outputs, change, match):
    _change(outputs, change)
    with pytest.raises(ValueError, match=match):
        _summarize(outputs, allow_partial=True)


def test_missing_output_requires_explicit_partial_scope(outputs):
    path = outputs[1] / "production/step_100000"
    path.rename(path.with_name("not-selected"))
    with pytest.raises(ValueError, match="Missing checkpoint outputs"):
        _summarize(outputs)
    summary = _summarize(outputs, allow_partial=True)
    assert summary["status"] == "partial" and summary["missing_steps"] == [100000]
    assert len(summary["rows"]) == 1
    path.mkdir()
    with pytest.raises(FileNotFoundError):
        _summarize(outputs, allow_partial=True)


@pytest.mark.parametrize("change,match", [
    (lambda row: row.update(critic_updates=1), "optimizer updates"),
    (lambda row: row["metrics"].update({"decision/inner_model_steps": 999}), "model-step count"),
    (lambda row: row["metrics"].update({"decision/inner_mppi_iterations": 6}), "iteration count"),
    (lambda row: row["metrics"].update({"decision/reward": 999}), "decision rewards"),
])
def test_validates_real_decision_cost_and_reward_data(outputs, change, match):
    path = outputs[1] / "production/step_100000/bundle/controller__mppi/seed-101.jsonl.gz"
    with gzip.open(path, "rt") as stream:
        rows = [json.loads(line) for line in stream]
    change(rows[0])
    with gzip.open(path, "wt") as stream:
        stream.writelines(json.dumps(row) + "\n" for row in rows)
    with pytest.raises(ValueError, match=match):
        _summarize(outputs)


def test_missing_decision_and_nonfinite_return_are_rejected(outputs):
    _change(outputs, lambda m: m["runs"][0].update(trace_files=[]))
    with pytest.raises(ValueError, match="Missing or duplicated"):
        _summarize(outputs)
    path = outputs[1] / "production/step_100000/bundle/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["runs"][1]["episodes"][0]["return"] = float("nan")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="nonfinite JSON"):
        _summarize(outputs)


def test_cli_atomic_outputs_and_publication_failure_preserve_validated_local_results(outputs, tmp_path, monkeypatch):
    summarize = campaign.summarize
    monkeypatch.setattr(campaign, "summarize", lambda *a, **k: summarize(*a, **k, seeds=SEEDS, max_steps=2))
    output, html = tmp_path / "summary.json", tmp_path / "summary.html"
    args = ["--manifest", str(outputs[0]), "--results-root", str(outputs[1]), "--expected-source-sha", SOURCE,
            "--output", str(output), "--html", str(html)]
    assert campaign.main(args) == 0
    previous = output.read_bytes()
    with pytest.raises(FileExistsError):
        campaign.main(args)
    assert output.read_bytes() == previous
    monkeypatch.setattr(campaign, "publish_campaign", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    with pytest.raises(RuntimeError, match="offline"):
        campaign.main([*args, "--overwrite", "--wandb"])
    assert json.loads(output.read_text())["status"] == "complete"
    assert "Frozen XQC prior versus MPPI" in html.read_text()
    assert not list(tmp_path.rglob("*.tmp"))


def test_wandb_campaign_uses_checkpoint_axis_and_native_scalar_curves(outputs, monkeypatch):
    events, metrics, finished = [], [], []
    remote = SimpleNamespace(url="https://wandb.ai/entity/project/runs/test", summary={},
                             define_metric=lambda *a, **k: metrics.append((a, k)),
                             log=events.append, finish=lambda **k: finished.append(k))
    monkeypatch.setattr("utils.wandb_utils.init_wandb", lambda *a, **k: remote)
    summary = _summarize(outputs)
    assert campaign.publish_campaign(summary, project="test", entity="test") == remote.url
    assert metrics == [(("checkpoint_step",), {}), *(( (group + "/*",), {"step_metric": "checkpoint_step"})
                                                         for group in ("prior", "mppi", "paired"))]
    assert [event["checkpoint_step"] for event in events] == [50000, 100000]
    assert events[0]["prior/return_mean"] == 15 and events[0]["mppi/return_mean"] == 19
    assert events[0]["paired/delta_mean"] == 4 and events[0]["paired/delta_std"] == 1
    assert "paired/deltas" not in events[0]
    assert finished == [{"exit_code": 0}]


def test_wandb_campaign_closes_failed_publication_without_masking_primary_error(outputs, monkeypatch):
    failure = RuntimeError("publish failed")
    finished = []
    def finish(**kwargs):
        finished.append(kwargs)
        raise OSError("close failed")
    remote = SimpleNamespace(summary={}, define_metric=lambda *a, **k: None, finish=finish,
                             log=lambda payload: (_ for _ in ()).throw(failure))
    monkeypatch.setattr("utils.wandb_utils.init_wandb", lambda *a, **k: remote)
    with pytest.raises(RuntimeError, match="publish failed") as caught:
        campaign.publish_campaign(_summarize(outputs), project="test", entity="test")
    assert caught.value is failure
    assert finished == [{"exit_code": 1}]
    assert any("close failed" in note for note in getattr(failure, "__notes__", []))
