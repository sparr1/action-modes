import copy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

import publish_tdmpc2_mppi_eval as publisher


SOURCE = "rwgao_b-brown-university/ambi/xq3zva9u"
CAMPAIGN = "prior-mppi-test"


def write_result(root, step=100000, shift=0):
    directory = root / f"step_{step}"
    directory.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[1]
    config_name = "tdmpc2_humanoid_walk_state_prior_only_checkpoint_bank_1p5m.json"
    metadata = {
        "checkpoint": {"step": step},
        "trial_run_params": json.loads((repository / "configs/dmcontrol/algs" / config_name).read_text()),
        "experiment_params": json.loads((repository / "configs/dmcontrol/experiments" / config_name).read_text()),
    }
    metadata_path = root / f"original_{step}.metadata.json"
    # Deliberately noncanonical formatting: copying must preserve the pinned byte hash.
    metadata_path.write_text(json.dumps(metadata, indent=4) + "\n\n")
    episodes = []
    for index, seed in enumerate(range(101, 106)):
        arms = {}
        for controller, gain in (("policy_prior_mean", 0), ("native_mppi", 2)):
            arms[controller] = {"return": index + gain + shift, "length": 500,
                                "steps": [{}] * 500, "terminated": False,
                                "truncated": True, "capped": False, "seconds": 1.5}
        episodes.append({"environment_seed": seed, "return_delta": 2, **arms})
    result = {
        "schema_version": 1, "algorithm": "TDMPC2/TDMPC2Baseline", "environment": "DMControl-v0",
        "checkpoint_sha256": f"{step:064x}", "checkpoint_metadata": {"step": step},
        "configuration_source": str(metadata_path), "planner": copy.deepcopy(publisher.PLANNER),
        "frozen_state": {"unchanged": True, "model_digest_before": "abc", "model_digest_after": "abc",
                         "num_updates_before": 100, "num_updates_after": 100},
        "protocol": {"controllers": ["policy_prior_mean", "native_mppi"], "environment_seed_first": 101,
                     "environment_seed_last": 105, "max_steps": 500, "controller_seed_base": 12345},
        "episodes": episodes,
        "summary": {"paired_episodes": 5, "policy_prior_return_mean": 2 + shift,
                    "native_mppi_return_mean": 4 + shift, "paired_return_delta_mean": 2},
    }
    path = directory / "paired.json"
    path.write_text(json.dumps(result))
    provenance = {"source_run": SOURCE, "campaign": CAMPAIGN, "code_sha": "a" * 40,
                  "code_tree": "b" * 40, "checkpoint_sha256": result["checkpoint_sha256"],
                  "metadata_sha256": publisher._hash(metadata_path)}
    (directory / "provenance.json").write_text(json.dumps(provenance))
    return path


def change(path, mutate):
    value = json.loads(path.read_text())
    mutate(value)
    path.write_text(json.dumps(value))


def test_finished_checkpoints_stage_to_same_two_explicit_runs(tmp_path, monkeypatch):
    import sys
    calls = []
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setitem(sys.modules, "utils.eval_series", SimpleNamespace(
        load_run=lambda path: {"run_id": Path(path).name},
        stage_result=lambda run_dir, path, **kwargs: calls.append((run_dir, path, kwargs))))
    mapping = {"policy_prior": str(tmp_path / "prior"), "native_mppi": str(tmp_path / "mppi")}
    later = write_result(tmp_path, 200000, shift=10)
    first = publisher.publish(later, source_run=SOURCE, campaign=CAMPAIGN, eval_run_map=mapping)
    earlier = write_result(tmp_path)
    second = publisher.publish(earlier, source_run=SOURCE, campaign=CAMPAIGN, eval_run_map=mapping)
    assert first["status"] == second["status"] == "queued"
    assert [call[0] for call in calls] == list(mapping.values()) * 2
    assert [call[2]["selector"] for call in calls] == list(mapping) * 2
    assert all(call[2]["format"] == "tdmpc2-paired" for call in calls)
    assert not (earlier.parent / ".publication.json").exists()


def test_publication_requires_explicit_selection_and_preserves_result_on_staging_failure(tmp_path, monkeypatch):
    import sys
    monkeypatch.delenv("WANDB_MODE", raising=False)
    path = write_result(tmp_path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="explicit New/Append"):
        publisher.publish(path, source_run=SOURCE, campaign=CAMPAIGN)
    def fail(*args, **kwargs):
        raise OSError("owner offline")
    monkeypatch.setitem(sys.modules, "utils.eval_series", SimpleNamespace(load_run=lambda p: {}, stage_result=fail))
    record = publisher.publish(path, source_run=SOURCE, campaign=CAMPAIGN,
                               eval_run_map={"policy_prior": "prior", "native_mppi": "mppi"})
    assert record["status"] == "failed"
    assert path.read_bytes() == before


def test_disabled_smoke_preserves_sidecar_hash_and_is_portable(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    path = write_result(tmp_path)
    record = publisher.publish(path, source_run=SOURCE, campaign=CAMPAIGN, wandb_module=object())
    assert record["status"] == "validated_upload_disabled"
    original = tmp_path / "original_100000.metadata.json"
    copied = path.parent / "checkpoint.metadata.json"
    assert original.read_bytes() == copied.read_bytes()
    original.unlink()
    assert publisher.publish(path, source_run=SOURCE, campaign=CAMPAIGN)["result_sha256"] == record["result_sha256"]


def test_cold_start_with_reused_prior_stages_only_new_mppi_curve(tmp_path, monkeypatch):
    import sys
    calls = []
    path = write_result(tmp_path)
    def cold(result):
        result["planner"]["warm_start"] = "none"
        result["prior_reference"] = {"reused": True, "sha256": "c" * 64}
        for episode in result["episodes"]:
            episode["native_mppi"]["steps"] = [{"planner": {"planner_warm_start_used": 0.0}}] * 500
    change(path, cold)
    monkeypatch.setitem(sys.modules, "utils.eval_series", SimpleNamespace(
        load_run=lambda p: {}, stage_result=lambda *a, **kw: calls.append((a, kw))))
    result = publisher.publish(path, source_run=SOURCE, campaign=CAMPAIGN,
                               eval_run_map={"native_mppi": "new-cold-curve"})
    assert result["status"] == "queued" and len(calls) == 1
    assert calls[0][1]["selector"] == "native_mppi"
    change(path, lambda d: d["episodes"][0]["native_mppi"]["steps"][0]["planner"].update(planner_warm_start_used=1.0))
    with pytest.raises(ValueError, match="cold-start evidence"):
        publisher.load_result(path, SOURCE, CAMPAIGN)


@pytest.mark.parametrize("mutate", [
    lambda d: d["frozen_state"].update(unchanged=False),
    lambda d: d["frozen_state"].update(num_updates_after=101),
    lambda d: d["checkpoint_metadata"].update(step=150000),
    lambda d: d.update(checkpoint_sha256="f" * 64),
    lambda d: d["planner"].update(effective_iterations=10),
    lambda d: d["summary"].update(policy_prior_return_mean=99),
    lambda d: d["episodes"][0].update(return_delta=3),
    lambda d: d["episodes"][0]["native_mppi"].update({"return": float("nan")}),
    lambda d: d["episodes"].pop(),
    lambda d: d["episodes"][0]["native_mppi"].update(truncated=False),
])
def test_rejects_invalid_or_incomplete_result(tmp_path, mutate):
    path = write_result(tmp_path)
    change(path, mutate)
    with pytest.raises(ValueError):
        publisher.load_result(path, SOURCE, CAMPAIGN)


@pytest.mark.parametrize("key,value", [("source_run", "foreign/project/run"), ("campaign", "foreign"),
                                        ("code_sha", "c" * 40)])
def test_rejects_foreign_source_or_code_in_sibling(tmp_path, key, value):
    write_result(tmp_path)
    later = write_result(tmp_path, 150000)
    change(later.parent / "provenance.json", lambda d: d.update({key: value}))
    with pytest.raises(ValueError):
        publisher.campaign_data(tmp_path, SOURCE, CAMPAIGN)


def test_rejects_different_controller_rng_protocol(tmp_path):
    write_result(tmp_path)
    later = write_result(tmp_path, 150000)
    change(later, lambda d: d["protocol"].update(controller_seed_base=999))
    with pytest.raises(ValueError, match="protocols"):
        publisher.campaign_data(tmp_path, SOURCE, CAMPAIGN)


def test_rejects_republication_of_replaced_result(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    path = write_result(tmp_path)
    publisher.publish(path, source_run=SOURCE, campaign=CAMPAIGN)
    change(path, lambda d: d.update(extra="changed result"))
    with pytest.raises(ValueError, match="different result"):
        publisher.publish(path, source_run=SOURCE, campaign=CAMPAIGN)


def test_complete_campaign_has_all_actual_points_through_1p5m(tmp_path):
    for index, step in enumerate(reversed(publisher.STEPS)):
        write_result(tmp_path, step, shift=step / 100000)
    loaded, curves = publisher.campaign_data(tmp_path, SOURCE, CAMPAIGN)
    assert len(loaded) == 29
    assert curves[0] == [2 + step / 100000 for step in publisher.STEPS]
    assert curves[2] == [2] * 29


def test_expected_grid_rejects_result_beyond_selected_maximum(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    later = write_result(tmp_path, 1150000)
    with pytest.raises(ValueError, match="maximum"):
        publisher.publish(later, source_run=SOURCE, campaign=CAMPAIGN, expected_max_step=400000)


@pytest.fixture
def source_revisions(tmp_path):
    """Real Git objects verify source identity, without relying on fake hashes."""
    repository = tmp_path / "repo"
    repository.mkdir()

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repository, stderr=subprocess.PIPE).decode().strip()

    git("init")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    for name in publisher.EVALUATION_SOURCE_PATHS:
        path = repository / name
        if "." not in path.name:
            path /= "example.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original source\n")

    def commit():
        git("add", ".")
        git("commit", "-m", "fixture")
        return git("rev-parse", "HEAD"), git("rev-parse", "HEAD^{tree}")

    original = commit()
    (repository / "publish_tdmpc2_mppi_eval.py").write_text("publication change\n")
    publication = commit()
    (repository / "RL/example.py").write_text("changed scientific source\n")
    scientific = commit()
    return repository, original, publication, scientific


@pytest.mark.parametrize("mutation", ["publication", "scientific", "forged_tree", "forged_fingerprint"])
def test_mixed_commits_require_git_verified_identical_scientific_source(tmp_path, source_revisions, monkeypatch, mutation):
    repository, original, publication, scientific = source_revisions
    fingerprint = publisher.evaluation_source_fingerprint
    monkeypatch.setattr(publisher, "evaluation_source_fingerprint",
                        lambda sha, tree: fingerprint(sha, tree, repository))
    first = write_result(tmp_path / "results", 400000)
    later = write_result(tmp_path / "results", 450000)
    change(first.parent / "provenance.json", lambda d: d.update(code_sha=original[0], code_tree=original[1]))
    selected = scientific if mutation == "scientific" else publication
    change(later.parent / "provenance.json", lambda d: d.update(
        code_sha=selected[0], code_tree="0" * 40 if mutation == "forged_tree" else selected[1],
        evaluation_source_sha256="f" * 64 if mutation == "forged_fingerprint"
        else fingerprint(*selected, repository)))
    if mutation == "publication":
        loaded, curves = publisher.campaign_data(tmp_path / "results", SOURCE, CAMPAIGN, expected_max_step=450000)
        assert len(loaded) == 2 and curves[0][-2:] == [2, 2]
        assert loaded[0]["provenance"]["code_sha"] == original[0]
        assert loaded[1]["provenance"]["code_sha"] == publication[0]
    else:
        with pytest.raises(ValueError):
            publisher.campaign_data(tmp_path / "results", SOURCE, CAMPAIGN)


def test_installed_wandb_table_preserves_null_gaps():
    import wandb
    table = wandb.Table(columns=["step", "lineKey", "lineVal"],
                        data=[[100000, "prior", 1], [150000, "prior", None], [200000, "prior", 3]])
    assert table.data[1] == [150000, "prior", None]
