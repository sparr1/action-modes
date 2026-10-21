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
            inner_entropy_enabled=mode not in ("kl", "retrace_kl"),
            inner_temperature_mode="inherit_outer" if mode in ("kl", "retrace_kl") else "auto",
            inner_outer_policy_kl_coef=ALPHAS[step] if mode in curves.KL_METHODS else 0.)
    if mode in curves.RETRACE_METHODS:
        planner["settings"].update(inner_sac_return_estimator="retrace", inner_retrace_lambda=1.,
                                   inner_retrace_batch_trajectories=86)
    if mode == "mppi":
        planner = dict(type="mppi", action_rule="weighted_elite_with_noise",
            settings=dict(planning_horizon=3, terminal_critic="aux_return"))
    return dict(backbone="entity/ambi/mppi7530927x0", planner=planner,
        protocol=dict(environment_seeds=[101, 102, 103, 104, 105], max_steps=500,
                      controller_seed=55, seed_scheme="sha256-v1", action_rule="tanh_sample"),
        science=dict(version=1, algorithm="AMBITDMPC2/AMBITDMPC2", source_sha256="a" * 64))


def profile(tmp_path):
    return dict(root=str(tmp_path / "curves"), family="test-mppi-entropy-kl", project="test",
        entity="entity", owner="local", workspace_url="https://example.invalid/workspace",
        identities={mode: curves.canonical_identity(identity(mode), mode, ALPHAS[400000])
                    for mode in curves.METHODS if mode in curves.BASE_METHODS}, sources=[])


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


@pytest.mark.parametrize("mode", ["kl", "both", "retrace_kl", "retrace_both"])
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


def test_semantically_equal_numeric_config_uses_stable_catalog_encoding(tmp_path):
    p = profile(tmp_path)
    p["identities"]["entropy"]["planner"]["settings"]["grad_clip_norm"] = 20
    raw, prior = record(tmp_path, "entropy"), record(tmp_path, "prior")
    raw["identity"]["planner"]["settings"]["grad_clip_norm"] = 20.0
    derived = curves.derive(p, {"mode": "entropy"}, raw, {}, prior, tmp_path / "derived")
    catalog = curves.prepare(p)
    directory = next(c["run_dir"] for c in catalog["curves"] if c["mode"] == "entropy")
    assert series.stage_record(directory, derived)["status"] == "staged"
    assert series._json(derived["identity"]) == series._json(p["identities"]["entropy"])
    assert type(derived["provenance"]["source_identity"]["planner"]["settings"]["grad_clip_norm"]) is float


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


def test_optional_mppi_catalog_extension_keeps_existing_runs_and_sources(tmp_path):
    p = profile(tmp_path)
    p["sources"] = [source_for(tmp_path, record(tmp_path, "prior"))]
    original = curves.prepare(p)
    before_sources = series._read(tmp_path / "curves" / "sources.json")
    extended = deepcopy(p)
    extended["identities"]["mppi"] = identity("mppi")
    # Adding a controller can have a distinct audited scientific fingerprint.
    extended["identities"]["mppi"]["science"]["source_sha256"] = "b" * 64
    new = curves.prepare(extended)
    assert new["curves"][:4] == original["curves"]
    assert len(new["curves"]) == 5
    assert new["curves"][-1]["mode"] == "mppi"
    assert new["curves"][-1]["color"] == "#D97706"
    assert curves.prepare(extended) == new
    assert series._read(tmp_path / "curves" / "sources.json") == before_sources
    with pytest.raises(series.SeriesError, match="retain every existing"):
        curves.prepare(p)
    changed = deepcopy(extended)
    changed["identities"]["both"]["science"]["source_sha256"] = "c" * 64
    with pytest.raises(series.SeriesError, match="Incompatible append"):
        curves.prepare(changed)
    assert series._read(tmp_path / "curves" / "catalog.json") == new


def test_mppi_registers_pending_checkpoints_without_temperature_and_publishes_paired_gains(tmp_path):
    p = profile(tmp_path)
    p["identities"]["mppi"] = identity("mppi")
    prior, raw = record(tmp_path, "prior"), record(tmp_path, "mppi", paired=False)
    raw["metrics"].update({"diagnostics/inner_actor_entropy_mean": 0.,
        "diagnostics/inner_alpha_final_mean": 0., "diagnostics/inner_final_outer_policy_kl_mean": 0.,
        "work/temperature_updates": 0, "work/model_steps": 12336})
    p["sources"] = [source_for(tmp_path, prior)]
    source = source_for(tmp_path, raw, published=False)
    cell = {key: source[key] for key in ("mode", "run_dir", "run_id")}
    cell["step"] = 400000
    catalog = curves.register_campaign(p, {"cells": [cell]})
    assert curves.register_campaign(p, {"cells": [cell]}) == catalog
    state = curves.stage(p, catalog)
    assert state["missing"] == [("mppi", 400000)]
    assert state["counts"]["mppi"] == 0
    registered = series._read(tmp_path / "curves" / "sources.json")
    assert "outer_alpha_by_step" not in registered[-1]
    raw_journal = series._read(tmp_path / "raw" / source["run_id"] / "publication.json")
    raw_journal["records"][raw["record_id"]]["status"] = "published"
    series._atomic_json(tmp_path / "raw" / source["run_id"] / "publication.json", raw_journal)
    complete = curves.stage(p, catalog, require_complete=True)
    assert complete["counts"]["mppi"] == 1 and not complete["missing"]
    derived = curves.derive(p, {"mode": "mppi"}, raw, {}, prior, tmp_path / "derived")
    assert derived["metrics"]["eval/paired_gain_mean"] == statistics.mean(GAINS)
    assert derived["metrics"]["eval/paired_gain_sample_std"] == statistics.stdev(GAINS)
    assert derived["metrics"]["runtime/control_seconds"] == 10.
    assert derived["metrics"]["work/model_steps"] == 12336
    assert "work/temperature_updates" not in derived["metrics"]
    assert "diagnostics/inner_actor_entropy_mean" not in derived["metrics"]
    assert "diagnostics/inner_alpha_final_mean" not in derived["metrics"]
    assert "diagnostics/inner_final_outer_policy_kl_mean" not in derived["metrics"]
    assert derived["provenance"]["coefficient_rule"] is None


def test_retrace_extension_preserves_five_ids_and_publishes_four_distinct_paired_points(tmp_path):
    p = profile(tmp_path)
    p["identities"]["mppi"] = identity("mppi")
    p["sources"] = [source_for(tmp_path, record(tmp_path, "prior", step)) for step in ALPHAS]
    original = curves.prepare(p)
    original_sources = series._read(tmp_path / "curves" / "sources.json")
    cells, raws = [], []
    for mode in ("retrace_kl", "retrace_both"):
        for step, alpha in ALPHAS.items():
            raw = record(tmp_path, mode, step)
            raw["identity"]["science"]["source_sha256"] = "b" * 64
            raw["metrics"].update({"diagnostics/inner_actor_entropy_mean": -6.,
                "diagnostics/inner_retrace_trace_coefficient_mean_mean": .7,
                "diagnostics/inner_retrace_effective_trace_length_mean": 1.6,
                "diagnostics/inner_retrace_correction_abs_mean_mean": .3,
                "work/retrace_trajectory_draws": 27520000, "work/retrace_critic_rows": 82560000})
            source = source_for(tmp_path, raw)
            cells.append(dict(source, step=step, outer_alpha=alpha))
            p["identities"][mode] = curves.canonical_identity(raw["identity"], mode, alpha)
            raws.append(raw)
    updated = curves.register_campaign(p, {"cells": cells})
    assert updated["curves"][:5] == original["curves"]
    assert len(updated["curves"]) == 7
    assert len({row["color"] for row in updated["curves"]}) == 7
    assert series._read(tmp_path / "curves" / "sources.json")[:2] == original_sources
    assert curves.register_campaign(p, {"cells": cells}) == updated
    result = curves.stage(p, updated, require_complete=True)
    assert result["counts"]["retrace_kl"] == result["counts"]["retrace_both"] == 2
    assert not result["missing"]
    raw = raws[-1]
    derived = curves.derive(p, cells[-1], raw, {}, record(tmp_path, "prior", 500000), tmp_path / "derived")
    assert derived["metrics"]["configuration/inner_outer_policy_kl_coef"] == ALPHAS[500000]
    assert derived["metrics"]["diagnostics/inner_retrace_trace_coefficient_mean_mean"] == .7
    assert derived["metrics"]["work/retrace_critic_rows"] == 82560000
    assert derived["metrics"]["diagnostics/inner_actor_entropy_mean"] == -6.
    assert derived["metrics"]["eval/paired_gain_mean"] == statistics.mean(GAINS)
    assert derived["provenance"]["coefficient_rule"] == curves.KL_RULE


@pytest.mark.parametrize("mode", ["retrace_kl", "retrace_both"])
@pytest.mark.parametrize("change", ["estimator", "entropy", "lambda", "batch"])
def test_retrace_curves_reject_changed_estimator_entropy_or_trace_recipe(tmp_path, mode, change):
    p = profile(tmp_path)
    p["identities"][mode] = curves.canonical_identity(identity(mode), mode, ALPHAS[400000])
    raw = record(tmp_path, mode)
    settings = raw["identity"]["planner"]["settings"]
    if change == "estimator": settings["inner_sac_return_estimator"] = "one_step"
    elif change == "entropy": settings["inner_entropy_enabled"] = not settings["inner_entropy_enabled"]
    elif change == "lambda": settings["inner_retrace_lambda"] = .5
    elif change == "batch": settings["inner_retrace_batch_trajectories"] = 85
    with pytest.raises(series.SeriesError):
        curves.derive(p, dict(mode=mode, outer_alpha_by_step={"400000": ALPHAS[400000]}),
                      raw, {}, record(tmp_path, "prior"), tmp_path / "derived")
