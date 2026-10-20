"""Publish only complete, correctly paired objective comparisons; keep context read-only."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from slurm import ambi_entropy_kl_mppi500k_publish as pub
from utils import eval_series as series
from utils import eval_series_data as data

ALPHA = 0.023066775873303413


@pytest.fixture
def campaign(tmp_path):
    campaign = dict(campaign_id="entropy-kl-mppi500k-20261020", source_commit="a" * 40,
                    seeds=[101, 102, 103, 104, 105], controller_seed=55, max_steps=500,
                    execution_action="sample", cells=[])
    inventory = tmp_path / "inventory.json"
    series._atomic_json(inventory, {"pinned": True})
    for index, mode in enumerate(pub.METHODS):
        enabled = mode in ("entropy", "both")
        beta = ALPHA if mode in ("kl", "both") else 0.0
        settings = dict(inner_operator="none" if mode == "prior" else "sac", aux_return_mode="sac",
            inner_rollout_horizon=3, inner_rounds=8, inner_rollouts_per_round=128, inner_batch_size=256,
            inner_critic_updates_per_round=16, inner_actor_updates_per_round=4,
            inner_entropy_enabled=enabled, inner_temperature_mode="auto" if enabled else "inherit_outer",
            inner_temperature_initialization="inherit_outer", inner_temperature_lr=0.0003,
            inner_actor_lr=0.0003, inner_critic_lr=0.0003, inner_outer_policy_kl_coef=beta,
            inner_actor_source="sac", inner_horizon_actor_source="sac", inner_critic_source="aux_return",
            inner_horizon_critic_source="aux_return", inner_sac_critic_target="reward_only")
        source = "rwgao_b-brown-university/ambi/mppi7530927x0"
        selector = "entropy_kl_mppi500k/" + mode
        identity = dict(backbone=source, science={"version": 1, "algorithm": "AMBITDMPC2/AMBITDMPC2", "source_sha256": "b" * 64},
            planner=data.planner_identity(settings, {}, "AMBITDMPC2/AMBITDMPC2", "tanh_sample"),
            protocol=dict(action_rule="tanh_sample", environment_seeds=campaign["seeds"], max_steps=500, controller_seed=55))
        registry = series.create_run(tmp_path / "runs", dict(identity=identity, selector=selector, label=mode),
            campaign["campaign_id"], pub.PROJECT, pub.ENTITY, pub.OWNER)
        campaign["cells"].append(dict(index=index, mode=mode, checkpoint_id="mppi_aux_500k",
            cell_id="mppi_aux_500k__" + mode, selector=selector, source_run=source, step=500000,
            checkpoint_sha256="c" * 64, beta=beta, outer_alpha=ALPHA, entropy_enabled=enabled,
            expected_temperature_steps=32 if enabled else 0, temperature_lr=0.0003,
            inventory_path=str(inventory), inventory_sha256=series._file_digest(inventory),
            output_dir=str(tmp_path / "production" / mode), run_id=registry["run_id"], run_dir=registry["run_dir"]))
    return campaign


def completed(campaign, monkeypatch):
    """A realistic normalized record and a hash-bound receipt for each objective."""
    records, receipts = {}, {}
    for cell in campaign["cells"]:
        output = Path(cell["output_dir"])
        manifest = output / "bundle/manifest.json"
        prior = cell["mode"] == "prior"
        episodes = [dict(seed=seed, solver_seed=1000 + seed, length=500, truncated_by_evaluator=False,
                         **{"return": 5.0 if prior else 7.0}) for seed in campaign["seeds"]]
        results = dict(checkpoint_sha256=cell["checkpoint_sha256"], results=[dict(
            selector=cell["selector"], outer_state_unchanged=True, execution_action="sample", episodes=episodes)])
        series._atomic_json(output / "results.json", results)
        series._atomic_json(manifest, dict(status="complete", code={"commit": campaign["source_commit"], "dirty": False},
            runs=[dict(selector=cell["selector"], status="complete", trace_files=["trace.jsonl.gz"])]))
        trace = manifest.parent / "trace.jsonl.gz"
        trace.write_bytes(b"full trace stays on Oscar")
        receipt = dict(validated=True, smoke=False, source_commit=campaign["source_commit"],
            **{key: cell[key] for key in ("checkpoint_id", "cell_id", "selector", "mode", "beta", "outer_alpha",
                                         "entropy_enabled", "expected_temperature_steps", "temperature_lr")},
            manifest_sha256=series._file_digest(manifest), results_sha256=series._file_digest(output / "results.json"),
            trace_sha256={trace.name: series._file_digest(trace)},
            prior_receipt_sha256=None, prior_manifest_sha256=None)
        if not prior:
            prior_output = Path(campaign["cells"][0]["output_dir"])
            receipt.update(prior_receipt_sha256=series._file_digest(prior_output / "receipt.json"),
                           prior_manifest_sha256=series._file_digest(prior_output / "bundle/manifest.json"),
                           prior_results_sha256=series._file_digest(prior_output / "results.json"),
                           prior_trace_sha256=receipts["prior"]["trace_sha256"])
        series._atomic_json(output / "receipt.json", receipt)
        metrics = {metric: 7.0 for _, metric in pub.METRICS}
        metrics.update({"eval/episodes": 5, "eval/frozen_state_unchanged": True, "work/environment_decisions": 2500,
            "work/actor_updates": 0 if prior else 80000, "work/critic_updates": 0 if prior else 320000,
            "work/temperature_updates": 2500 * cell["expected_temperature_steps"],
            "eval/paired_gain_mean": 2.0, "eval/paired_gain_sample_std": 0.0})
        actual_alpha = ALPHA if cell["entropy_enabled"] else 0.0
        metrics.update({"diagnostics/inner_alpha_initial_mean": actual_alpha,
                        "diagnostics/inner_alpha_final_mean": actual_alpha,
                        "diagnostics/inner_alpha_delta_mean": 0.0})
        if prior:
            metrics = {key: value for key, value in metrics.items() if not key.startswith("diagnostics/")}
        normalized_episodes = [{**ep, **({} if prior else {"paired_return_delta": 2.0})} for ep in episodes]
        record = dict(identity=series.load_run(cell["run_dir"])["identity"],
            checkpoint={"step": 500000, "sha256": cell["checkpoint_sha256"]},
            metrics=metrics, episodes=normalized_episodes, record_id="paired-" + cell["mode"],
            label=cell["mode"], selector=cell["selector"], provenance={},
            artifact_files={"manifest.json": str(manifest), trace.name: str(trace)}, source_result_path=str(manifest))
        records[cell["mode"]], receipts[cell["mode"]] = record, receipt
    monkeypatch.setattr(data, "load_records", lambda path, **kwargs:
                        [deepcopy(records[Path(path).parent.parent.name])])
    return records, receipts


def test_stages_four_objectives_prior_self_gain_and_no_large_traces(campaign, monkeypatch):
    records, _ = completed(campaign, monkeypatch)
    statuses = pub.stage_completed(campaign)
    assert set(statuses.values()) == {"complete"}
    for cell in campaign["cells"]:
        record = series._read(Path(cell["run_dir"]) / "records" / ("paired-" + cell["mode"] + ".json"))
        assert "trace.jsonl.gz" not in record["artifact_files"]
        assert record["metrics"]["reference/return_mean"] == 5.0
        assert record["metrics"]["eval/paired_gain_mean"] == (0.0 if cell["mode"] == "prior" else 2.0)
        if cell["mode"] == "prior":
            assert not any(k.startswith("diagnostics/") for k in record["metrics"])
    monkeypatch.setattr(data, "load_records", lambda *a, **k: pytest.fail("repeat normalization"))
    assert pub.stage_completed(campaign) == statuses


@pytest.mark.parametrize("mode,work", [("prior", 1), ("kl", 80000), ("entropy", 0), ("both", 0)])
def test_objective_specific_temperature_work_is_required(campaign, monkeypatch, mode, work):
    records, _ = completed(campaign, monkeypatch)
    records[mode]["metrics"]["work/temperature_updates"] = work
    with pytest.raises(ValueError, match="optimizer work"):
        pub.stage_completed(campaign)


@pytest.mark.parametrize("mode,alpha", [("kl", ALPHA), ("entropy", 0.0), ("both", 0.0)])
def test_kl_only_uses_zero_alpha_and_entropy_arms_use_checkpoint_alpha(campaign, monkeypatch, mode, alpha):
    records, _ = completed(campaign, monkeypatch)
    records[mode]["metrics"]["diagnostics/inner_alpha_initial_mean"] = alpha
    with pytest.raises(ValueError, match="initial alpha"):
        pub.stage_completed(campaign)


def test_kl_only_cannot_silently_enable_temperature_mid_solve(campaign, monkeypatch):
    records, _ = completed(campaign, monkeypatch)
    records["kl"]["metrics"]["diagnostics/inner_alpha_final_mean"] = ALPHA
    with pytest.raises(ValueError, match="actual alpha at zero"):
        pub.stage_completed(campaign)


@pytest.mark.parametrize("key,value", [("smoke", True), ("validated", False), ("mode", "both"),
    ("source_commit", "d" * 40), ("beta", 1.0), ("outer_alpha", 1.0), ("expected_temperature_steps", 32)])
def test_receipt_must_match_source_and_objective(campaign, monkeypatch, key, value):
    _, receipts = completed(campaign, monkeypatch)
    cell = campaign["cells"][0]
    receipts["prior"][key] = value
    series._atomic_json(Path(cell["output_dir"]) / "receipt.json", receipts["prior"])
    with pytest.raises(ValueError, match="validated production"):
        pub.stage_completed(campaign)


def test_rejects_result_and_trace_mutation(campaign, monkeypatch):
    completed(campaign, monkeypatch)
    output = Path(campaign["cells"][0]["output_dir"])
    (output / "bundle/trace.jsonl.gz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="full trace changed"):
        pub.stage_completed(campaign)
    series._atomic_json(output / "results.json", {"changed": True})
    with pytest.raises(ValueError, match="result or manifest changed"):
        pub.stage_completed(campaign)


def test_rejects_wrong_paired_gain_or_seed(campaign, monkeypatch):
    records, _ = completed(campaign, monkeypatch)
    records["entropy"]["episodes"][0]["paired_return_delta"] = 1.0
    with pytest.raises(ValueError, match="Paired gains disagree"):
        pub.stage_completed(campaign)
    records["entropy"]["episodes"][0]["paired_return_delta"] = 2.0
    records["entropy"]["episodes"][0]["solver_seed"] = 7
    with pytest.raises(ValueError, match="Paired gains disagree"):
        pub.stage_completed(campaign)


def test_missing_paired_prior_pin_is_rejected(campaign, monkeypatch):
    _, receipts = completed(campaign, monkeypatch)
    receipts["entropy"]["prior_receipt_sha256"] = "wrong"
    series._atomic_json(Path(campaign["cells"][1]["output_dir"]) / "receipt.json", receipts["entropy"])
    with pytest.raises(ValueError, match="paired prior changed"):
        pub.stage_completed(campaign)


def test_complete_seed_coverage_and_frozen_state_are_required(campaign, monkeypatch):
    records, _ = completed(campaign, monkeypatch)
    records["prior"]["episodes"].pop()
    with pytest.raises(ValueError, match="all five"):
        pub.stage_completed(campaign)
    records["prior"]["episodes"].append(dict(seed=105, **{"return": 5.0}))
    records["prior"]["metrics"]["eval/frozen_state_unchanged"] = False
    with pytest.raises(ValueError, match="complete frozen"):
        pub.stage_completed(campaign)


def test_canonical_prior_and_default_actor_routes_match_registry(campaign):
    for cell in campaign["cells"]:
        registry = pub.validate_registry(campaign, cell)
        if cell["mode"] == "prior":
            assert registry["identity"]["planner"] == {"type": "prior", "action_rule": "tanh_sample"}
        else:
            assert "inner_actor_source" not in registry["identity"]["planner"]["settings"]


def test_actual_driver_profile_contract_and_hashes(campaign, tmp_path):
    from slurm import ambi_entropy_kl_mppi500k as driver
    profile = series._read(driver.REPO / driver.PROFILE)
    campaign.update(schema_version=1, profile_sha256=series._file_digest(driver.REPO / driver.PROFILE),
                    matrix_sha256=series._file_digest(driver.REPO / driver.MATRIX))
    prior = Path(campaign["cells"][0]["output_dir"])
    for cell in campaign["cells"]:
        cell.update(profile["checkpoints"][driver.CHECKPOINT_ID])
        cell.update(reference_output_dir=str(prior), reference_bundle=str(prior / "bundle"))
    path = tmp_path / "campaign.json"
    series._atomic_json(path, campaign)
    assert pub.load_campaign(path) == campaign
    for field in ("matrix_sha256", "profile_sha256"):
        changed = deepcopy(campaign)
        changed[field] = "0" * 64
        series._atomic_json(path, changed)
        with pytest.raises(ValueError, match="profile or matrix changed"):
            pub.load_campaign(path)
    changed = deepcopy(campaign)
    changed["cells"][2]["beta"] = 3 * ALPHA
    series._atomic_json(path, changed)
    with pytest.raises(ValueError, match="intervention"):
        pub.load_campaign(path)


def test_active_temperature_lr_cannot_be_omitted(campaign, monkeypatch):
    cell = campaign["cells"][1]
    registry = series.load_run(cell["run_dir"])
    registry["identity"]["planner"]["settings"].pop("inner_temperature_lr")
    monkeypatch.setattr(series, "load_run", lambda _: registry)
    with pytest.raises(ValueError, match="objective or update recipe"):
        pub.validate_registry(campaign, cell)


def test_wrong_prior_actor_and_mean_execution_are_rejected(campaign, monkeypatch):
    cell = campaign["cells"][0]
    registry = series.load_run(cell["run_dir"])
    registry["identity"]["planner"]["settings"] = {"inner_actor_source": "return_actor"}
    monkeypatch.setattr(series, "load_run", lambda _: registry)
    with pytest.raises(ValueError, match="inherited primary actor"):
        pub.validate_registry(campaign, cell)
    registry["identity"]["protocol"]["action_rule"] = "tanh_mean"
    with pytest.raises(ValueError, match="sampled episode protocol"):
        pub.validate_registry(campaign, cell)


def test_summary_artifact_is_bounded(campaign, monkeypatch):
    records, _ = completed(campaign, monkeypatch)
    big = Path(campaign["cells"][0]["output_dir"]) / "big.json"
    with big.open("wb") as stream:
        stream.truncate(16 * 1024 * 1024)
    records["prior"]["artifact_files"][big.name] = str(big)
    with pytest.raises(ValueError, match="16 MiB"):
        pub.stage_completed(campaign)


@pytest.mark.parametrize("require_complete", [False, True])
def test_pending_publication_touches_only_four_new_runs(campaign, tmp_path, monkeypatch, require_complete):
    sessions = []
    class Publisher:
        def __init__(self, run_dir, **kwargs):
            self.run = SimpleNamespace(name=None, config={}, summary={}, define_metric=lambda *a, **k: None)
            sessions.append((run_dir, self.run))
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def publish_pending(self): return {}
    monkeypatch.setattr(series, "Publisher", Publisher)
    result = pub.publish_once(campaign, tmp_path / "pub", workspace_verifier=lambda *a: {"url": "verified"},
                              require_complete=require_complete)
    assert {r for r, _ in sessions} == {c["run_dir"] for c in campaign["cells"]}
    assert len(sessions) == 4 and not result["all_complete"]
    for _, run in sessions:
        assert run.summary["progress/checkpoints_complete"] == 0
        assert run.summary["progress/failed_cells"] == int(require_complete)
        assert not any(k.startswith("eval/") for k in run.summary)


def test_missing_layout_fails_before_remote_mutation(campaign, tmp_path, monkeypatch):
    monkeypatch.setattr(series, "Publisher", lambda *a, **k: pytest.fail("mutation before layout"))
    with pytest.raises(ValueError, match="native layout"):
        pub.publish_once(campaign, tmp_path / "pub")


def test_unique_readable_colors_and_correct_beta1_context(campaign):
    rows = pub.EXISTING + pub.curve_rows(campaign)
    assert len({r["name"] for r in rows}) == len(rows) == 7
    assert len({r["color"] for r in rows}) == 7
    assert pub.EXISTING[2]["run_id"] == "44c306e7c69c4625a1b1b7dce156fde3"
    assert all(row["line_mark"] == "points" for row in pub.curve_rows(campaign))
    assert pub.EXISTING[0]["line_mark"] == "solid"


@pytest.mark.parametrize("expanded_server_colors", [False, True])
def test_native_layout_idempotence_and_readback(campaign, tmp_path, monkeypatch, expanded_server_colors):
    ws = pytest.importorskip("wandb_workspaces.workspaces")
    wr = pytest.importorskip("wandb_workspaces.reports.v2")
    if pub.importlib.metadata.version("wandb") != "0.28.1" or pub.importlib.metadata.version("wandb-workspaces") != "0.4.5":
        pytest.skip("Layout validation needs its separate pinned SDK runtime")
    saved = []
    monkeypatch.setattr(ws.Workspace, "save", lambda self: saved.append(self))
    monkeypatch.setattr(ws.Workspace, "url", property(lambda self: f"https://wandb.ai/{pub.ENTITY}/{pub.PROJECT}?nw=testview"))
    def read_back(cls, url):
        if not expanded_server_colors:
            return saved[-1]
        restored = deepcopy(saved[-1])
        for section in restored.sections:
            for panel in section.panels:
                if isinstance(panel, wr.LinePlot):
                    panel.line_colors = {key: {"color": pub._line_color(value),
                        "transparentColor": pub._line_color(value) + "19"}
                        for key, value in panel.line_colors.items()}
        return restored
    monkeypatch.setattr(ws.Workspace, "from_url", classmethod(read_back))
    path = tmp_path / "workspace.json"
    first = pub.install_workspace(campaign, path)
    saved[-1].sections.append(ws.Section(name="User notes", panels=[]))
    second = pub.install_workspace(campaign, path)
    assert first["url"] == second["url"] and second["api_verified"]
    assert second["panel_count"] == pub.PANEL_COUNT and len(saved[-1].sections) == 4
    plots = [p for s in saved[-1].sections for p in s.panels if isinstance(p, wr.LinePlot)]
    assert len(plots) == len(pub.METRICS) and all(not p.aggregate for p in plots)
    for plot in plots:
        metric = getattr(plot.y[0], "name", plot.y[0])
        for row in second["runs"]:
            assert plot.line_marks[row["run_id"] + ":" + metric] == row["line_mark"]
            assert plot.line_colors[row["run_id"] + ":" + metric] == row["color"]


def test_final_cli_marks_missing_results_failed(campaign, tmp_path, monkeypatch):
    monkeypatch.setattr(pub, "load_campaign", lambda path: campaign)
    monkeypatch.setattr(pub, "publish_once", lambda *a, **k: {"all_complete": False})
    root = tmp_path / "pub"
    monkeypatch.setattr(pub.sys, "argv", ["publish", "publish", "--campaign", str(tmp_path / "campaign.json"),
        "--root", str(root), "--once", "--require-complete"])
    with pytest.raises(SystemExit, match="marked failed"):
        pub.main()
    assert "marked failed" in series._read(root / "publication-failure.json")["message"]
