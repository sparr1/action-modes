"""Pinned prior reuse cannot fabricate timing, evidence, or partial imports."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

import import_ambi_critic_bypass_prior as importer
from utils.ambi_benchmark import solver_seed


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def seal_campaign(value):
    value.pop("campaign_id", None)
    value["campaign_id"] = importer.fingerprint(value)
    return value


def make_campaign(*, smoke):
    arms = ["model_score", "learned_q", "actor_mean", "mppi_h3", "prior"]
    config = dict(protocol=importer.PROTOCOL, arms=arms, seeds=[101] if smoke else list(range(101, 121)),
        controller_seed=55, max_steps=4 if smoke else 500, smoke=smoke,
        checkpoint=dict(step=800000, sha256=importer.CHECKPOINT_SHA256, metadata_sha256=importer.METADATA_SHA256))
    campaign = dict(**config, config=deepcopy(config), source_commit="a"*40, source_tree="b"*40,
        config_sha256="c"*64, checkpoint_path="/pinned/800000",
        task_list=[dict(task_id=f"{arm}-seed-{seed}", arm=arm, env_seed=seed, controller_seed=55)
                   for seed in config["seeds"] for arm in arms])
    return seal_campaign(campaign)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    root, smoke, source_path = tmp_path/"production", tmp_path/"smoke", tmp_path/"old"/"manifest.json"
    campaign, smoke_campaign = make_campaign(smoke=False), make_campaign(smoke=True)
    env = dict(id="DMControl-v0", params=dict(task="humanoid-walk", obs="state", render_mode=None))
    resolved = dict(environment=env, algorithm_config=dict(alg_params=dict(obs="state")))
    episodes = [dict(seed=seed, solver_seed=solver_seed(55, "episode", seed), return_=100. + seed,
        length=500, terminated=False, truncated=True, capped=False, truncated_by_evaluator=False,
        episode_id=f"seed-{seed}", inner_metrics_mean={}) for seed in range(101, 106)]
    for row in episodes:
        row["return"] = row.pop("return_")
    reported = [{key: value for key, value in row.items() if key not in ("capped", "episode_id", "inner_metrics_mean")}
                for row in episodes]
    source = dict(status="complete", checkpoint=dict(path="/pinned/800000", sha256=importer.CHECKPOINT_SHA256,
        source_run=importer.SOURCE_RUN, metadata=dict(checkpoint=dict(step=800000))),
        protocol=dict(environment=env, env_wrapper=None, env_wrappers=[], observation="state", action_rule="tanh_mean",
                      max_steps=500, controller_seed=55, seed_scheme="sha256-v1"), code=dict(commit="d"*40),
        runs=[dict(status="complete", selector="reference/prior", episodes=episodes,
            resolved_config=dict(inner_operator="none", inner_actor_source="sac", inner_execution_action="mean", compile=False),
            result=dict(outer_state_unchanged=True, outer_updates_before=799999, outer_updates_after=799999,
                        episodes=reported))])
    dump(root/"campaign.json", campaign); dump(root/"resolved.json", resolved)
    dump(smoke/"campaign.json", smoke_campaign)
    for task in smoke_campaign["task_list"]:
        directory = smoke/"tasks"/task["task_id"]
        checks = dict(outer_state_unchanged=True, full_episode=True)
        if task["arm"] == "prior":
            checks["prior_execution_path_parity"] = True
        result = dict(**task, complete=True, campaign_id=smoke_campaign["campaign_id"], protocol=importer.PROTOCOL,
                      source_commit=campaign["source_commit"], checks=checks, episode_length=4)
        dump(directory/"result.json", result)
        dump(directory/"manifest.json", dict(status="complete", result_sha256=importer.sha256(directory/"result.json")))
    dump(source_path, source)
    monkeypatch.setattr(importer, "PRIOR_MANIFEST", source_path)
    monkeypatch.setattr(importer, "PRIOR_SHA256", importer.sha256(source_path))
    return root, source_path, smoke


def test_exact_five_seed_import_accepts_trace_metadata_and_preserves_missing_timing(inputs):
    from slurm.publish_ambi_critic_bypass import collect
    root, manifest, smoke = inputs
    result = importer.import_prior(root, manifest, smoke)
    assert result["imported"] == 5 and not result["comparable_timing_imported"]
    records = [importer.read(path) for path in sorted((root/"tasks").glob("*/result.json"))]
    assert [r["env_seed"] for r in records] == list(range(101, 106))
    assert [r["episode_return"] for r in records] == list(range(201, 206))
    for record in records:
        assert record["reused_reference"] and record["runtime"]["timing_comparable"] is False
        assert record["controller_times_s"] == record["selection_times_s"] == []
        assert record["controller_time_mean_s"] is record["controller_time_p95_s"] is None
        assert "complete_candidate_banks" not in record["checks"]
        assert record["reference_sha256"] == importer.sha256(manifest)
    snapshot = collect(root, importer.read(root/"campaign.json"))
    assert snapshot["complete"] == 5 and snapshot["pending"] == 95
    assert not list(root.glob(".prior-import-*"))


def test_existing_output_never_overwritten(inputs):
    root, manifest, smoke = inputs
    sentinel = root/"tasks"/"prior-seed-105"/"keep.txt"
    sentinel.parent.mkdir(parents=True); sentinel.write_text("keep")
    with pytest.raises(FileExistsError, match="already exists"):
        importer.import_prior(root, manifest, smoke)
    assert sentinel.read_text() == "keep"
    assert list((root/"tasks").iterdir()) == [sentinel.parent]


def test_source_hash_change_prevents_all_outputs(inputs):
    root, manifest, smoke = inputs
    manifest.write_text(manifest.read_text() + " ")
    with pytest.raises(ValueError, match="hash changed"):
        importer.import_prior(root, manifest, smoke)
    assert not (root/"tasks").exists()


@pytest.mark.parametrize("change", ["last_seed", "return", "environment", "checkpoint", "actor"])
def test_invalid_historical_science_rejected_before_any_output(inputs, monkeypatch, change):
    root, manifest, smoke = inputs
    source = importer.read(manifest)
    if change == "last_seed":
        source["runs"][0]["episodes"][-1]["seed"] = 106
    elif change == "return":
        source["runs"][0]["result"]["episodes"][-1]["return"] += 1.
    elif change == "environment":
        source["protocol"]["environment"]["params"]["task"] = "cheetah-run"
    elif change == "checkpoint":
        source["checkpoint"]["sha256"] = "0"*64
    else:
        source["runs"][0]["resolved_config"]["inner_execution_action"] = "policy_sample"
    dump(manifest, source)
    monkeypatch.setattr(importer, "PRIOR_SHA256", importer.sha256(manifest))
    with pytest.raises(ValueError):
        importer.import_prior(root, manifest, smoke)
    assert not (root/"tasks").exists()
    assert not list(root.glob(".prior-import-*"))


def test_prior_smoke_parity_is_required(inputs):
    root, manifest, smoke = inputs
    path = smoke/"tasks"/"prior-seed-101"/"result.json"
    value = importer.read(path); value["checks"].pop("prior_execution_path_parity")
    dump(path, value)
    dump(path.parent/"manifest.json", dict(status="complete", result_sha256=importer.sha256(path)))
    with pytest.raises(ValueError, match="parity"):
        importer.import_prior(root, manifest, smoke)
    assert not (root/"tasks").exists()


def test_smoke_fingerprint_and_source_are_required(inputs):
    root, manifest, smoke = inputs
    path = smoke/"tasks"/"learned_q-seed-101"/"result.json"
    value = importer.read(path); value["checks"]["new_check"] = True
    dump(path, value)
    with pytest.raises(ValueError, match="fingerprint"):
        importer.import_prior(root, manifest, smoke)
    assert not (root/"tasks").exists()


def test_staging_write_failure_leaves_no_partial_visible_results(inputs, monkeypatch):
    root, manifest, smoke = inputs
    original = importer.write
    def fail_later(path, value):
        if path.parent.name == "prior-seed-104":
            raise OSError("simulated disk write failure")
        original(path, value)
    monkeypatch.setattr(importer, "write", fail_later)
    with pytest.raises(OSError, match="simulated"):
        importer.import_prior(root, manifest, smoke)
    assert not (root/"tasks").exists()
    assert not list(root.glob(".prior-import-*"))
