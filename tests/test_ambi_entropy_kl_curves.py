"""Scientific compatibility and immutable staging for checkpoint comparison curves."""
from copy import deepcopy
import statistics

import pytest

from slurm import ambi_entropy_kl_curves as curves
from utils import eval_series as series


ALPHAS = {400000: .013081684708595276, 500000: .023066775873303413}
GAINS = [1., -2., 3., 0., 5.]


def identity(mode, step=400000):
    planner = {"type": "prior" if mode == "prior" else "sac", "action_rule": "tanh_sample"}
    if mode != "prior":
        planner["settings"] = dict(inner_rounds=8, inner_rollout_horizon=3,
            inner_critic_updates_per_round=16, inner_actor_updates_per_round=4,
            inner_rollouts_per_round=128, inner_batch_size=256,
            inner_actor_lr=.0003, inner_critic_lr=.0003, inner_temperature_lr=.0003,
            inner_entropy_enabled=mode != "kl",
            inner_temperature_mode="inherit_outer" if mode == "kl" else "auto",
            inner_outer_policy_kl_coef=ALPHAS[step] if mode in ("kl", "both") else 0.)
    return dict(backbone="entity/ambi/mppi7530927x0", planner=planner,
        protocol=dict(environment_seeds=[101, 102, 103, 104, 105], max_steps=500,
                      controller_seed=55, seed_scheme="sha256-v1", action_rule="tanh_sample"),
        science=dict(version=1, algorithm="AMBITDMPC2/AMBITDMPC2", source_sha256="a" * 64))


def profile(tmp_path):
    return dict(root=str(tmp_path / "curves"), family="test-mppi-entropy-kl", project="test",
        entity="entity", owner="local", workspace_url="https://example.invalid/workspace",
        identities={mode: curves.canonical_identity(identity(mode), mode, ALPHAS[400000])
                    for mode in curves.METHODS}, sources=[])


def record(tmp_path, mode, step=400000, *, paired=True):
    artifact = tmp_path / f"{mode}-{step}.json"
    artifact.write_text("{}\n")
    episodes = []
    for index, seed in enumerate(range(101, 106)):
        gain = 0. if mode == "prior" else GAINS[index]
        episode = dict(seed=seed, solver_seed=seed * 17, length=500, terminated=False,
                       truncated=True, truncated_by_evaluator=False, return_=10. * index + gain)
        episode["return"] = episode.pop("return_")
        if paired:
            episode["paired_return_delta"] = gain
        episodes.append(episode)
    returns = [ep["return"] for ep in episodes]
    return dict(identity=identity(mode, step), checkpoint=dict(step=step, sha256=str(step)[0] * 64),
        episodes=episodes, metrics={"eval/return_mean": statistics.mean(returns),
            "eval/return_sample_std": statistics.stdev(returns), "eval/frozen_state_unchanged": True,
            "runtime/control_seconds": 10., "diagnostics/inner_entropy": 7.},
        artifact_files={"results.json": str(artifact)}, provenance={}, record_id=f"{mode}-{step}",
        source_result_path=str(artifact), label=mode)


def source_for(tmp_path, raw, *, published=True):
    registry = series.create_run(tmp_path / "raw", raw, "original", "test", "entity", "local")
    series.stage_record(registry["run_dir"], raw)
    path = tmp_path / "raw" / registry["run_id"] / "publication.json"
    journal = series._read(path)
    if published:
        journal["records"][raw["record_id"]]["status"] = "published"
        series._atomic_json(path, journal)
    mode, step = raw["label"], raw["checkpoint"]["step"]
    return dict(mode=mode, run_dir=registry["run_dir"], run_id=registry["run_id"],
        expected_identity_sha256=registry["identity_sha256"], expected_checkpoint_steps=[step],
        outer_alpha_by_step={str(step): ALPHAS[step]})


@pytest.mark.parametrize("mode", ["kl", "both"])
def test_checkpoint_beta_rule_preserves_every_other_identity_field(mode):
    old, new = identity(mode), identity(mode, 500000)
    snapshot = deepcopy(old)
    actual = curves.canonical_identity(old, mode, ALPHAS[400000])
    assert actual == curves.canonical_identity(new, mode, ALPHAS[500000])
    assert old == snapshot
    assert actual["planner"]["settings"]["inner_outer_policy_kl_coef_rule"] == curves.KL_RULE
    for alpha in (ALPHAS[500000], 0., -1., float("nan")):
        with pytest.raises(series.SeriesError, match="beta"):
            curves.canonical_identity(old, mode, alpha)


@pytest.mark.parametrize("field", ["backbone", "protocol", "science", "lr", "mode"])
def test_derive_rejects_other_scientific_interventions(tmp_path, field):
    p = profile(tmp_path)
    raw, prior = record(tmp_path, "both"), record(tmp_path, "prior")
    if field == "backbone": raw["identity"]["backbone"] = "entity/ambi/another-training-run"
    if field == "protocol": raw["identity"]["protocol"]["action_rule"] = "tanh_mean"
    if field == "science": raw["identity"]["science"]["source_sha256"] = "b" * 64
    if field == "lr": raw["identity"]["planner"]["settings"]["inner_temperature_lr"] = .03
    if field == "mode": raw["identity"]["planner"]["settings"]["inner_entropy_enabled"] = False
    source = dict(mode="both", outer_alpha_by_step={"400000": ALPHAS[400000]})
    with pytest.raises(series.SeriesError):
        curves.derive(p, source, raw, {}, prior, tmp_path / "derived")


def test_recomputes_true_paired_gains_for_legacy_unpaired_entropy_records(tmp_path):
    p = profile(tmp_path)
    raw, prior = record(tmp_path, "entropy", paired=False), record(tmp_path, "prior")
    snapshot = deepcopy(raw)
    raw["episodes"].reverse()
    derived = curves.derive(p, {"mode": "entropy"}, raw, {}, prior, tmp_path / "derived")
    assert derived["metrics"]["eval/paired_gain_mean"] == statistics.mean(GAINS)
    assert derived["metrics"]["eval/paired_gain_sample_std"] == statistics.stdev(GAINS)
    assert [ep["paired_return_delta"] for ep in derived["episodes"]] == GAINS
    assert sorted(raw["episodes"], key=lambda ep: ep["seed"]) == snapshot["episodes"]
    raw["episodes"][0]["paired_return_delta"] = 123.
    with pytest.raises(series.SeriesError, match="paired"):
        curves.derive(p, {"mode": "entropy"}, raw, {}, prior, tmp_path / "wrong-delta")


def test_prior_only_exposes_return_metrics_and_zero_paired_gain(tmp_path):
    prior = record(tmp_path, "prior")
    derived = curves.derive(profile(tmp_path), {"mode": "prior"}, prior, {}, prior, tmp_path / "derived")
    assert derived["metrics"]["eval/paired_gain_mean"] == 0.
    assert derived["metrics"]["eval/paired_gain_sample_std"] == 0.
    assert not any(key.startswith(("runtime/", "diagnostics/")) for key in derived["metrics"])


@pytest.mark.parametrize("field", ["backbone", "protocol", "science", "checkpoint", "solver_seed"])
def test_prior_must_match_identity_checkpoint_and_rng(tmp_path, field):
    raw, prior = record(tmp_path, "entropy"), record(tmp_path, "prior")
    if field == "backbone": prior["identity"]["backbone"] = "entity/ambi/wrong"
    if field == "protocol": prior["identity"]["protocol"]["action_rule"] = "tanh_mean"
    if field == "science": prior["identity"]["science"]["source_sha256"] = "b" * 64
    if field == "checkpoint": prior["checkpoint"]["sha256"] = "0" * 64
    if field == "solver_seed": prior["episodes"][0]["solver_seed"] += 1
    with pytest.raises(series.SeriesError):
        curves.derive(profile(tmp_path), {"mode": "entropy"}, raw, {}, prior, tmp_path / "derived")


@pytest.mark.parametrize("change", ["missing_seed", "short", "capped", "unfrozen", "wrong_summary"])
def test_incomplete_or_inconsistent_episodes_are_rejected(tmp_path, change):
    raw, prior = record(tmp_path, "entropy"), record(tmp_path, "prior")
    if change == "missing_seed": raw["episodes"].pop()
    if change == "short": raw["episodes"][0]["length"] = 499
    if change == "capped": raw["episodes"][0]["truncated_by_evaluator"] = True
    if change == "unfrozen": raw["metrics"]["eval/frozen_state_unchanged"] = False
    if change == "wrong_summary": raw["metrics"]["eval/return_mean"] += 1
    with pytest.raises(series.SeriesError):
        curves.derive(profile(tmp_path), {"mode": "entropy"}, raw, {}, prior, tmp_path / "derived")


def test_prepare_and_stage_keep_ids_and_reject_duplicate_attempts(tmp_path):
    p = profile(tmp_path)
    prior, raw = record(tmp_path, "prior"), record(tmp_path, "entropy")
    p["sources"] = [source_for(tmp_path, prior), source_for(tmp_path, raw)]
    catalog = curves.prepare(p)
    assert curves.prepare(p) == catalog
    assert len({c["run_id"] for c in catalog["curves"]}) == 4
    first = curves.stage(p, catalog, require_complete=True)
    assert curves.stage(p, catalog, require_complete=True) == first
    assert first["counts"] == {"prior": 1, "entropy": 1, "kl": 0, "both": 0}
    source_path = tmp_path / "curves" / "sources.json"
    sources = series._read(source_path)
    sources.append(deepcopy(sources[1]))
    series._atomic_json(source_path, sources)
    with pytest.raises(series.SeriesError, match="Multiple selected attempts"):
        curves.stage(p, catalog, require_complete=True)


def test_conflicting_checkpoint_cannot_replace_accepted_derived_point(tmp_path):
    p = profile(tmp_path)
    prior, raw = record(tmp_path, "prior"), record(tmp_path, "entropy")
    first = curves.derive(p, {"mode": "entropy"}, raw, {}, prior, tmp_path / "derived")
    registry = series.create_run(tmp_path / "out", first, "attempt", "test", "entity", "local")
    series.stage_record(registry["run_dir"], first)
    changed = deepcopy(first)
    changed["record_id"] = "different-attempt"
    with pytest.raises(series.SeriesError, match="different accepted result"):
        series.stage_record(registry["run_dir"], changed)
    raw["record_id"] = "other-raw-attempt"
    with pytest.raises(series.SeriesError, match="source changed"):
        curves.derive(p, {"mode": "entropy"}, raw, {}, prior, tmp_path / "derived")


def test_only_published_untampered_source_records_can_be_imported(tmp_path):
    raw = record(tmp_path, "entropy")
    source = source_for(tmp_path, raw, published=False)
    assert curves.read_source(source) == []
    p = profile(tmp_path)
    p["sources"] = [source]
    with pytest.raises(series.SeriesError, match="Missing published"):
        curves.stage(p, curves.prepare(p), require_complete=True)
    journal_path = tmp_path / "raw" / source["run_id"] / "publication.json"
    journal = series._read(journal_path)
    journal["records"][raw["record_id"]]["status"] = "published"
    series._atomic_json(journal_path, journal)
    assert curves.read_source(source)[0][0] == raw
    record_path = tmp_path / "raw" / source["run_id"] / "records" / (raw["record_id"] + ".json")
    tampered = deepcopy(raw)
    tampered["episodes"][0]["return"] += .01
    series._atomic_json(record_path, tampered)
    with pytest.raises(series.SeriesError, match="immutable journal"):
        curves.read_source(source)


def test_register_new_checkpoint_reuses_canonical_run_ids(tmp_path):
    p = profile(tmp_path)
    catalog = curves.prepare(p)
    raw = record(tmp_path, "both", 500000)
    source = source_for(tmp_path, raw)
    cell = {**source, "step": 500000, "outer_alpha": ALPHAS[500000]}
    assert curves.register_campaign(p, {"cells": [cell]}) == catalog
    assert curves.register_campaign(p, {"cells": [cell]}) == catalog
    registered = series._read(tmp_path / "curves" / "sources.json")
    assert len(registered) == 1
    assert registered[0]["expected_checkpoint_steps"] == [500000]
    assert registered[0]["outer_alpha_by_step"] == {"500000": ALPHAS[500000]}
    wrong = deepcopy(cell)
    wrong["outer_alpha"] = ALPHAS[400000]
    with pytest.raises(series.SeriesError, match="beta"):
        curves.register_campaign(p, {"cells": [wrong]})
    assert series._read(tmp_path / "curves" / "sources.json") == registered


@pytest.mark.parametrize("field", ["run_id", "expected_identity_sha256"])
def test_source_registry_pins_are_enforced(tmp_path, field):
    source = source_for(tmp_path, record(tmp_path, "entropy"))
    source[field] = "wrong"
    with pytest.raises(series.SeriesError, match="identity changed"):
        curves.read_source(source)


@pytest.mark.parametrize("change", ["unvalidated", "smoke", "manifest", "receipt"])
def test_legacy_prior_requires_exact_validated_manifest_and_receipt(tmp_path, change):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    manifest = bundle / "manifest.json"
    receipt = tmp_path / "receipt.json"
    series._atomic_json(manifest, {"status": "complete"})
    series._atomic_json(receipt, {"validated": change != "unvalidated", "smoke": change == "smoke"})
    source = dict(bundle=str(bundle), receipt=str(receipt),
                  manifest_sha256=series._file_digest(manifest), receipt_sha256=series._file_digest(receipt))
    if change == "manifest":
        series._atomic_json(manifest, {"status": "changed"})
    if change == "receipt":
        series._atomic_json(receipt, {"validated": True, "unexpected": "change"})
    with pytest.raises(series.SeriesError):
        curves.read_source(source)
