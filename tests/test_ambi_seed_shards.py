import copy
import gzip
import json
import os
from pathlib import Path

import numpy as np
import pytest

from utils.ambi_benchmark import atomic_json, canonical_hash, read_json
from utils.ambi_diagnostic_series import (
    build_diagnostic_record, read_diagnostic_bundle, record_from_model_bundle,
    write_diagnostic_bundle,
)
from utils.ambi_real_calibration import SimulatorSnapshot
from utils.ambi_seed_shards import (
    RECEIPT, SEAL, _summary, merge_episode_bundles, merge_real_bundles, seal_episode_bundle,
)


def episode_bundle(tmp_path, name, seeds, *, probes=True):
    directory = tmp_path / name
    directory.mkdir()
    manifest = {"schema_version": 1, "evaluation_id": name, "status": "complete",
                "checkpoint": {"sha256": "a" * 64, "metadata": {"checkpoint": {"step": 200000}}},
                "code": {"source_sha256": "b" * 64, "commit": "science", "dirty": False},
                "protocol": {"environment": {"id": "analytic"}, "action_rule": "tanh_mean", "max_steps": 2,
                             "controller_seed": 55, "seed_scheme": "sha256-v1"},
                "metric_catalog": {"reward": {"definition": "reward", "unit": "reward",
                    "sampling_phase": "decision", "preferred_axis": "decision_index"}}, "runs": [],
                "elapsed_seconds": float(len(seeds))}
    for variant in ("prior", "inherited"):
        selector = "initialization/" + variant
        identifier = selector.replace("/", "__")
        config = {"alg": "AMBITDMPC2/AMBITDMPC2", "alg_params": {"inner_rounds": 1,
                  "inner_operator": "none" if variant == "prior" else "sac"}}
        episodes, trace_files, probe_rows = [], [], []
        for seed in seeds:
            value = seed + (10 if variant == "inherited" else 0)
            episode = {"seed": seed, "episode_id": f"seed-{seed}", "solver_seed": seed + 555,
                       "return": float(value), "length": 2, "terminated": False, "truncated": True,
                       "truncated_by_evaluator": False, "control_seconds": .5, "model_metrics": {"q": float(seed)}}
            if variant == "inherited":
                episode["paired_return_delta"] = 10.
            episodes.append(episode)
            events = [{"run_id": identifier, "episode_id": f"seed-{seed}", "decision_index": d,
                       "event_index": 0, "phase": "decision", "metrics": {"reward": value / 2}}
                      for d in range(2)]
            relative = f"{identifier}/seed-{seed}.jsonl.gz"
            (directory / identifier).mkdir(exist_ok=True)
            (directory / relative).write_bytes(gzip.compress(("\n".join(map(json.dumps, events)) + "\n").encode(), mtime=0))
            trace_files.append(relative)
            if probes:
                for d in range(2):
                    for j in (0, 1):
                        probe_rows.append({"episode_id": f"seed-{seed}", "root_id": f"seed-{seed}-decision-{d}",
                            "decision_index": d, "solver_repeat": 0, "rollout_repeat": 0, "round_index": j,
                            "actor_updates": j * 4, "critic_updates": j * 32, "metrics": {"return": float(seed + j)}})
        result = {"selector": selector, "comparison": "initialization", "variant": variant,
                  "reference_variant": "prior", "environment_seeds": seeds, "episodes": copy.deepcopy(episodes),
                  "outer_state_unchanged": True, "outer_updates_before": 10, "outer_updates_after": 10,
                  "resolved_config": config["alg_params"], "return": _summary(ep["return"] for ep in episodes),
                  "episode_length": _summary([2] * len(seeds)), "bank_solve_count": 0,
                  "model_metrics": {"q": _summary([float(seed) for seed in seeds for _ in range(2)])},
                  "bank_metrics": {}, "model_metric_availability": ["q"],
                  "nonfinite_model_metrics": {}, "nonfinite_trace_metrics": {},
                  "paired_return_delta_vs_reference": _summary([10. if variant == "inherited" else 0.] * len(seeds))}
        run = {"id": identifier, "selector": selector, "kind": "episodes", "status": "complete", "config": config,
               "config_hash": canonical_hash(config), "resolved_config": config["alg_params"], "episodes": episodes,
               "roots": [], "trace_files": trace_files, "result": result, "serialization_seconds": .2}
        if probes:
            run.update(togo_return_probe={"rollouts": 32}, togo_probe_rows=probe_rows)
        manifest["runs"].append(run)
    atomic_json(directory / "manifest.json", manifest)
    seal_episode_bundle(directory)
    return directory


def real_bundle(tmp_path, name, seeds, *, value_offset=0.):
    assets = tmp_path / (name + "-assets")
    assets.mkdir()
    root_protocol = {"version": 1, "seeds": seeds, "decisions": [0, 1], "max_steps": 2,
                     "controller_seed": 55, "action_rule": "tanh_mean", "science": "b" * 64}
    roots = []
    for seed in seeds:
        for decision in (0, 1):
            observation = np.asarray([seed, decision], np.float32)
            snapshot = SimulatorSnapshot.capture({"environment": {"observation": observation,
                "step_count": 2 * decision, "runtime": {"action_repeat": 2, "version": "analytic"}}})
            roots.append({"episode_id": f"seed-{seed}", "root_id": f"seed-{seed}-decision-{decision}",
                          "seed": seed, "decision_index": decision, "observation": observation.tolist(),
                          "dtype": "float32", "shape": [2], "snapshot": snapshot.to_dict()})
    bank = {"schema_version": 1, "kind": "humanoid_integration_state_bank", "complete": True,
            "checkpoint_sha256": "a" * 64, "protocol": root_protocol, "roots": roots,
            "episodes": [{"episode_id": f"seed-{seed}", "length": 2, "return": float(seed)} for seed in seeds]}
    bank["id"] = canonical_hash(bank)
    protocol = {"root_protocol": root_protocol, "root_bank_id": bank["id"], "solver_repetitions": 2,
                "rollout_repetitions": 2, "rounds": [0, 1], "model_probe_rollouts": 32,
                "prefix_action_rule": "mean", "tail_action_rule": "sampled", "action_rule": "mean_prefix_sampled_tail"}
    identity = {"checkpoint": {"sha256": "a" * 64, "step": 200000}, "setting": {"inner_rounds": 1},
                "code": {"source_sha256": "b" * 64}, "protocol": protocol, "scope": "common_prior_roots"}
    atomic_json(assets / "root-bank.json", bank)
    atomic_json(assets / "matrix.json", {"seeds": list(range(101, 121))})
    artifacts = {"root-bank.json": assets / "root-bank.json", "matrix.json": assets / "matrix.json"}
    rows = []
    for root in roots:
        noise, qseed = root["seed"] + root["decision_index"], root["seed"] + 123
        reference_result = {"model_rows": [{}, {}], "real": {"rows": [
            {"rollout_index": k, "mc_complete": True, "episode_cutoff_complete": True} for k in range(2)]}}
        reference = {"identity": {"checkpoint": "a" * 64, "bank_id": bank["id"], "root_id": root["root_id"],
                     "snapshot_hash": canonical_hash(root["snapshot"]), "noise_seed": noise, "q_pair_seed": qseed,
                     "prefix_action_rule": "mean"}, "complete": True, "result": reference_result,
                     "sha256": canonical_hash(reference_result)}
        name_ref = f"references/{root['root_id']}.json"
        atomic_json(assets / name_ref, reference)
        artifacts[name_ref] = assets / name_ref
        for solver in range(2):
            solve_rows = []
            for j in (0, 1):
                for rollout in range(2):
                    row = {"episode_id": root["episode_id"], "root_id": root["root_id"],
                           "solver_repeat": solver, "rollout_repeat": rollout, "round_index": j,
                           "actor_updates": 4 * j, "critic_updates": 32 * j, "mc_complete": True,
                           "truncated": False, "terminated": False, "policy_noise_seed": noise, "q_pair_seed": qseed,
                           "actor_sha256": canonical_hash([root["root_id"], solver, j]),
                           "metrics": {"gain": float(root["seed"] + j + solver + rollout + value_offset)}}
                    solve_rows.append(row)
            events = [{"phase": "probe", "round_index": j, "metrics": {"model_return": float(j)}} for j in (0, 1)]
            probes = [dict(episode_id=root["episode_id"], root_id=root["root_id"], solver_repeat=solver, **ev) for ev in events]
            name_solve = f"solves/{root['root_id']}-solver-{solver}.json"
            atomic_json(assets / name_solve, {"rows": solve_rows, "model_probes": probes, "trace_events": events})
            artifacts[name_solve] = assets / name_solve
            rows.extend(solve_rows)
    plan = {"roots": [{"episode_id": r["episode_id"], "root_id": r["root_id"]} for r in roots],
            "solver_repeats": 2, "rollout_repeats": 2, "rounds": [0, 1]}
    record = build_diagnostic_record(identity, rows, plan, attempt_label="mean-1", bootstrap_resamples=50, bootstrap_seed=5,
                                    timing={"outer_state_unchanged": True, "simulator_seconds": len(seeds), "total_elapsed_seconds": 2.})
    directory = write_diagnostic_bundle(tmp_path / name, record, artifact_files=artifacts)
    return directory


def test_episode_merge_preserves_prior_pairing_traces_and_model_rows(tmp_path):
    shards = [episode_bundle(tmp_path, "first", [101, 102]), episode_bundle(tmp_path, "last", [103])]
    merged = merge_episode_bundles(shards, tmp_path / "merged", expected_seeds=[101, 102, 103])
    manifest = read_json(merged / "manifest.json")
    for index, run in enumerate(manifest["runs"]):
        assert run["result"]["environment_seeds"] == [101, 102, 103]
        assert run["result"]["return"]["mean"] == 102 + 10 * index
        assert run["result"]["model_metrics"]["q"] == _summary([101., 101., 102., 102., 103., 103.])
        assert run["result"]["paired_return_delta_vs_reference"]["mean"] == 10 * index
        assert len(run["trace_files"]) == 3
        for shard in shards:
            original = read_json(shard / "manifest.json")["runs"][index]
            for name in original["trace_files"]:
                assert (merged / name).read_bytes() == (shard / name).read_bytes()
        diagnostic = record_from_model_bundle(merged, run["selector"], "mean-1", bootstrap_resamples=2)
        assert diagnostic["status"] == "complete" and len(diagnostic["rows"]) == 12
    assert manifest["runs"][1]["result"]["paired_return_delta_vs_prior"]["mean"] == 10.
    assert read_json(merged / SEAL)["files"]["shards/0/" + SEAL]


@pytest.mark.parametrize("which", ["episodes", "real"])
def test_shard_order_cannot_change_merged_measurements(tmp_path, which):
    factory = episode_bundle if which == "episodes" else real_bundle
    merge = merge_episode_bundles if which == "episodes" else merge_real_bundles
    shards = [factory(tmp_path, "first", [101, 102]), factory(tmp_path, "last", [103])]
    left = merge(shards, tmp_path / "left", expected_seeds=[101, 102, 103])
    right = merge(shards[::-1], tmp_path / "right", expected_seeds=[101, 102, 103])
    assert read_json(left / "manifest.json") == read_json(right / "manifest.json")
    with pytest.raises(FileExistsError):
        merge(shards, left, expected_seeds=[101, 102, 103])


@pytest.mark.parametrize("which", ["episodes", "real"])
@pytest.mark.parametrize("seeds,expected,match", [([102], [101, 102], "overlap"),
    ([103], [101, 102, 103, 104], "missing"), ([103], [101, 102], "unexpected")])
def test_merge_rejects_overlapping_missing_and_extra_seeds(tmp_path, which, seeds, expected, match):
    factory = episode_bundle if which == "episodes" else real_bundle
    merge = merge_episode_bundles if which == "episodes" else merge_real_bundles
    shards = [factory(tmp_path, "first", [101, 102]), factory(tmp_path, "last", seeds)]
    with pytest.raises(ValueError, match=match):
        merge(shards, tmp_path / "merged", expected_seeds=expected)
    assert not (tmp_path / "merged").exists()


@pytest.mark.parametrize("which", ["episodes", "real"])
def test_corrupt_source_artifacts_fail_before_output_creation(tmp_path, which):
    factory = episode_bundle if which == "episodes" else real_bundle
    merge = merge_episode_bundles if which == "episodes" else merge_real_bundles
    shard = factory(tmp_path, "source", [101])
    file = next(shard.glob("initialization*/*.gz")) if which == "episodes" else shard / "paired-rows.jsonl.gz"
    file.write_bytes(file.read_bytes() + b"bad")
    with pytest.raises(ValueError, match="checksum"):
        merge([shard], tmp_path / "merged", expected_seeds=[101])
    assert not (tmp_path / "merged").exists()


def test_real_merge_rebuilds_full_identity_and_pools_episode_clusters(tmp_path):
    shards = [real_bundle(tmp_path, "first", [101, 102]), real_bundle(tmp_path, "last", [103], value_offset=12)]
    directory = merge_real_bundles(shards, tmp_path / "merged", expected_seeds=[101, 102, 103])
    merged = read_diagnostic_bundle(directory)
    bank = read_json(directory / "artifacts/root-bank.json")
    assert bank["protocol"]["seeds"] == [101, 102, 103]
    assert bank["id"] == canonical_hash({k: v for k, v in bank.items() if k != "id"})
    assert merged["identity"]["protocol"]["root_bank_id"] == bank["id"]
    assert len(bank["roots"]) == 6 and len(merged["rows"]) == 48
    assert merged["summaries"][0]["metrics"]["gain"]["mean"] == 107.
    assert merged["summaries"][0]["metrics"]["gain"]["episodes"] == 3
    assert merged["series_id"] not in [read_diagnostic_bundle(shard)["series_id"] for shard in shards]
    rows = [row for shard in shards for row in read_diagnostic_bundle(shard)["rows"]]
    assert sorted(map(canonical_hash, rows)) == sorted(map(canonical_hash, merged["rows"]))
    for i, shard in enumerate(shards):
        source = read_json(shard / "manifest.json")
        for name in source["artifact_files"]:
            assert (directory / f"artifacts/shards/{i}" / name).read_bytes() == (shard / name).read_bytes()
    assert len(list((directory / "artifacts/solves").glob("*.json"))) == 12
    assert read_json(directory / RECEIPT)["expected_seeds"] == [101, 102, 103]


def rewrite_real(source, mutate, *, artifact_mutation=None):
    record = read_diagnostic_bundle(source)
    manifest = read_json(source / "manifest.json")
    artifacts = {name.removeprefix("artifacts/"): source / name for name in manifest["artifact_files"]}
    if artifact_mutation:
        artifact_mutation(artifacts)
    mutate(record)
    record["record_sha256"] = canonical_hash({k: v for k, v in record.items() if k != "record_sha256"})
    target = source.with_name(source.name + "-changed")
    write_diagnostic_bundle(target, record, artifact_files=artifacts)
    return target


@pytest.mark.parametrize("mutation,match", [
    (lambda r: r.update(status="incomplete"), "complete"),
    (lambda r: r["identity"]["protocol"].update(prefix_action_rule="sampled"), "incompatible"),
    (lambda r: r["identity"]["code"].update(source_sha256="c" * 64), "incompatible"),
    (lambda r: r.update(attempt_label="another"), "incompatible"),
    (lambda r: r["rows"].pop(), "coverage"),
])
def test_real_merge_rejects_incomplete_or_incompatible_records(tmp_path, mutation, match):
    shards = [real_bundle(tmp_path, "first", [101]), real_bundle(tmp_path, "last", [102])]
    shards[1] = rewrite_real(shards[1], mutation)
    with pytest.raises(ValueError, match=match):
        merge_real_bundles(shards, tmp_path / "merged", expected_seeds=[101, 102])


def test_real_merge_requires_all_solve_artifacts(tmp_path):
    shard = real_bundle(tmp_path, "first", [101])
    changed = rewrite_real(shard, lambda r: None, artifact_mutation=lambda files: files.pop(next(k for k in files if k.startswith("solves/"))))
    with pytest.raises(ValueError, match="Missing completed solve"):
        merge_real_bundles([changed], tmp_path / "merged", expected_seeds=[101])


def test_episode_seal_is_immutable_and_missing_trace_is_rejected(tmp_path):
    shard = episode_bundle(tmp_path, "first", [101])
    assert seal_episode_bundle(shard) == shard / SEAL
    file = next(shard.glob("initialization*/*.gz"))
    file.unlink()
    with pytest.raises(ValueError, match="checksum"):
        seal_episode_bundle(shard)
    with pytest.raises(ValueError, match="checksum"):
        merge_episode_bundles([shard], tmp_path / "merged", expected_seeds=[101])


@pytest.mark.skipif(os.environ.get("AMBI_RUN_REAL_DMCONTROL_TESTS") != "1", reason="opt-in real Humanoid runtime")
def test_actual_humanoid_episode_and_mean_prefix_shards_merge_and_normalize(tmp_path):
    """Exercise actual evaluator schemas and publication adapters, without W&B."""
    import evaluate_ambi_calibration as calibration
    from evaluate_ambi_checkpoint import evaluate_matrix
    from tests.test_ambi_calibration_cli import _humanoid_checkpoint
    from utils.eval_series_data import normalize_bundle
    checkpoint, matrix_path = _humanoid_checkpoint(tmp_path)
    matrix = read_json(matrix_path)
    matrix["source_run"] = "research/validation/tiny-mean-shards"
    matrix["evaluation"].update(seeds=[101, 108], max_steps=2, default_presets=["init/inherited"])
    prior = read_json(Path("configs/research/ambi_prior_refinement_h1_parallel.json"))["comparisons"]["initialization"]["variants"]["prior"]
    prior["alg_params"]["inner_steps_per_update"] = None
    prior["alg_params"]["inner_update_timing"] = "round"
    matrix["comparisons"]["init"]["reference"] = "prior"
    matrix["comparisons"]["init"]["variants"]["prior"] = prior
    matrix["shared_alg_params"] = {"inner_rollout_horizon": 1}
    matrix["real_calibration"].update(decisions=[0], solver_repetitions=1, rollout_repetitions=2,
                                      tail_steps=2, prefix_action_rule="mean")
    matrix_path.write_text(json.dumps(matrix))
    inventory = tmp_path / "checkpoint-inventory.json"
    import hashlib
    atomic_json(inventory, {"source_run": matrix["source_run"], "checkpoints": [
        {"sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "step": 10}]})
    episode_shards, real_shards = [], []
    for seed in (101, 108):
        episodes = tmp_path / f"episodes-{seed}"
        evaluate_matrix(matrix_path, checkpoint, selectors=["init/prior", "init/inherited"], seeds=[seed],
                        bundle_dir=episodes, checkpoint_inventory=inventory, stage_results=False, device="cpu")
        seal_episode_bundle(episodes)
        episode_shards.append(episodes)
        real = tmp_path / f"real-{seed}"
        calibration.run_calibration(matrix_path, checkpoint, preset="init/inherited", seeds=[seed],
                                    bundle_dir=real, attempt_label="tiny-shards", device="cpu")
        real_shards.append(real)
    episodes = merge_episode_bundles(episode_shards, tmp_path / "episodes", expected_seeds=[101, 108])
    normalized = normalize_bundle(episodes, checkpoint_inventory=inventory)
    assert len(normalized) == 2 and all(len(record["episodes"]) == 2 for record in normalized)
    manifest = read_json(episodes / "manifest.json")
    prior_values = {ep["seed"]: ep["return"] for ep in manifest["runs"][0]["episodes"]}
    for ep in manifest["runs"][1]["episodes"]:
        assert ep["paired_return_delta"] == ep["return"] - prior_values[ep["seed"]]
    model = record_from_model_bundle(episodes, "init/inherited", "tiny-shards", bootstrap_resamples=20)
    assert model["status"] == "complete" and len(model["rows"]) == 2 * 2 * 3
    real = merge_real_bundles(real_shards, tmp_path / "real", expected_seeds=[101, 108])
    restored = read_diagnostic_bundle(real)
    assert restored["status"] == "complete" and len(restored["rows"]) == 2 * 1 * 2 * 3
    assert restored["identity"]["protocol"]["prefix_action_rule"] == "mean"
