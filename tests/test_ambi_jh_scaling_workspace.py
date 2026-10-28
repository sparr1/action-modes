"""Dedicated comparison ownership, pending state and real SDK serialization."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from slurm import ambi_jh_scaling_workspace as publisher


def make_run(variant, identity=None):
    spec = publisher.VARIANTS[variant]
    params = dict(
        inner_operator="sac", inner_rollout_horizon=spec["horizon"],
        inner_rounds=spec["rounds"], inner_solve_interval=spec["interval"],
        train_unroll_horizon=3, outer_planning_horizon=3,
        inner_rollouts_per_round=128, inner_critic_updates_per_round=16,
        inner_actor_updates_per_round=4, inner_batch_size=256,
        inner_replay_capacity=128 * spec["horizon"] * spec["rounds"],
        aux_return_mode="off", outer_critic_target="reward_only",
        inner_sac_critic_target="reward_only", ent_coef="auto_1.0",
        target_entropy=-10.5, inner_temperature_mode="auto",
        inner_target_entropy="inherit_outer", inner_actor_adaptation="clone",
        inner_critic_adaptation="clone", inner_hold_encoder="solve_snapshot",
        inner_actor_initialization="prior", inner_critic_initialization="prior",
        inner_execution_action="policy_sample", inner_behavior_action="policy_sample",
        wandb_group=publisher.GROUP,
    )
    return SimpleNamespace(id=identity or (publisher.BASELINE if variant == "baseline" else variant),
        state="running", config={"config": params, "run_params": {"seed": 55, "total_steps": 1000000}},
        summary={"env_step": 123000, "episode/return": 42.0})


class Api:
    def __init__(self, variants=()):
        self.baseline = make_run("baseline")
        self.candidates = [make_run(variant) for variant in variants]

    def run(self, path):
        assert path == f"{publisher.ENTITY}/{publisher.PROJECT}/{publisher.BASELINE}"
        return self.baseline

    def runs(self, project, filters, per_page):
        assert project == f"{publisher.ENTITY}/{publisher.PROJECT}"
        assert filters == {"group": publisher.GROUP} and per_page == 4
        return iter(self.candidates)


def test_pending_is_explicit_and_not_zero():
    rows = publisher.discover_rows(Api())
    assert len(rows) == 4 and rows[0]["run_id"] == publisher.BASELINE
    assert all(row["run_id"] is None and row["decisions"] is None and row["last_return"] is None
               and row["state"] == "pending W&B startup" for row in rows[1:])
    text = publisher._markdown(rows, "now")
    assert text.count("pending W&B startup") == 3
    assert "not K1 checkpoint evaluations" in text
    assert "Model training unroll and\nouter planning horizons remain 3" in text
    assert "160/3 to 160/4 to 160/5" in text


def test_exact_variants_and_launch_id_validation():
    api = Api(list(publisher.VARIANTS)[1:])
    expected = {variant: variant for variant in list(publisher.VARIANTS)[1:]}
    rows = publisher.discover_rows(api, expected_ids=expected)
    assert len({row["run_id"] for row in rows}) == len({row["color"] for row in rows}) == 4
    expected["h4_j10_k4"] = "wrong-run"
    with pytest.raises(ValueError, match="launch identity"):
        publisher.discover_rows(api, expected_ids=expected)


@pytest.mark.parametrize("field,value", [
    ("train_unroll_horizon", 4), ("outer_planning_horizon", 4),
    ("inner_replay_capacity", 3840), ("inner_solve_interval", 1),
    ("inner_sac_critic_target", "soft"), ("target_entropy", -21),
    ("inner_execution_action", "mean"), ("wandb_group", "another-campaign"),
])
def test_rejects_scientific_or_campaign_mismatch(field, value):
    api = Api(["h4_j10_k4"])
    api.candidates[0].config["config"][field] = value
    with pytest.raises(ValueError):
        publisher.discover_rows(api)


def test_rejects_duplicate_variants_and_wrong_baseline():
    api = Api(["h3_j12_k3", "h3_j12_k3"])
    with pytest.raises(ValueError, match="Duplicate"):
        publisher.discover_rows(api)
    api = Api()
    api.baseline.id = "not-the-authorized-baseline"
    with pytest.raises(ValueError, match="Reference run ID"):
        publisher.discover_rows(api)


@pytest.fixture
def sdk_backend(monkeypatch):
    # Exercise the actual pinned SDK's frontend/backend model round trip, with
    # only save/from_url/url stubbed to prohibit external calls.
    ws = pytest.importorskip("wandb_workspaces.workspaces")
    import wandb_workspaces.reports.v2 as wr
    from wandb_workspaces import expr
    saved = {}
    calls = []

    def url(work):
        return f"https://forge.coreweave.com/wandb/{work.entity}/{work.project}?nw={work._internal_name}"

    def save(work):
        if not work._internal_name:
            work._internal_name = f"test-view-{len(saved)}"
        saved[work.url] = work._to_model()
        calls.append(work.url)

    def from_url(value):
        return ws.Workspace._from_model(deepcopy(saved[value]))

    monkeypatch.setattr(ws.Workspace, "url", property(url))
    monkeypatch.setattr(ws.Workspace, "save", save)
    monkeypatch.setattr(ws.Workspace, "from_url", staticmethod(from_url))
    return (ws, wr, expr), saved, calls


def test_install_roundtrip_idempotence_pending_then_four_runs(tmp_path, sdk_backend):
    backend, saved, calls = sdk_backend
    path = tmp_path / "workspace.json"
    first = publisher.install(path, api=Api(), backend=backend)
    assert first["api_verified"] and not first["browser_verified"]
    assert first["panel_count"] == 20 and len(saved) == 1
    work = backend[0].Workspace.from_url(first["url"])
    work.sections.append(backend[0].Section(name="Unrelated saved section", panels=[]))
    work.save()
    old_url = first["url"]
    second = publisher.install(path, api=Api(list(publisher.VARIANTS)[1:]), backend=backend)
    assert second["url"] == old_url and len(saved) == 1
    assert all(row["run_id"] for row in second["rows"])
    work = backend[0].Workspace.from_url(second["url"])
    assert any(section.name == "Unrelated saved section" for section in work.sections)
    assert work.settings.max_runs == 4 and not work.runset_settings.groupby
    plots = [panel for section in work.sections for panel in section.panels
             if isinstance(panel, backend[1].LinePlot)]
    assert len(plots) == 19
    assert all(not panel.aggregate and panel.smoothing_type == "none"
               and panel.max_runs_to_show == 4 and len(panel.line_titles) == 4 for panel in plots)
    count = len(calls)
    with pytest.raises(ValueError, match="disappeared or changed"):
        publisher.install(path, api=Api(), backend=backend)
    assert len(calls) == count


def test_receipt_ownership_and_ambiguous_create_never_mutate(tmp_path, sdk_backend):
    backend, _, calls = sdk_backend
    path = tmp_path / "workspace.json"
    path.write_text(json.dumps({"entity": "unrelated"}))
    with pytest.raises(ValueError, match="another comparison"):
        publisher.install(path, api=Api(), backend=backend)
    path.write_text(json.dumps(dict(entity=publisher.ENTITY, project=publisher.PROJECT,
        group=publisher.GROUP, baseline=publisher.BASELINE, creation_intent=True)))
    with pytest.raises(RuntimeError, match="unknown outcome"):
        publisher.install(path, api=Api(), backend=backend)
    assert not calls


def test_fail_closed_if_readback_changes_run_filter(tmp_path, sdk_backend, monkeypatch):
    backend, _, _ = sdk_backend
    ws, _, _ = backend
    original = ws.Workspace.from_url

    def wrong_filter(url):
        work = original(url)
        work.runset_settings.groupby = [backend[2].Config("seed")]
        return work

    monkeypatch.setattr(ws.Workspace, "from_url", staticmethod(wrong_filter))
    path = tmp_path / "workspace.json"
    with pytest.raises(RuntimeError, match="independent run selection"):
        publisher.install(path, api=Api(), backend=backend)
    receipt = json.loads(path.read_text())
    assert not receipt["api_verified"] and receipt["url"]
    assert receipt["error_type"] == "RuntimeError"


@pytest.mark.parametrize("key,value", [("seed", 56), ("total_steps", 500000)])
def test_wrong_seed_or_budget_is_rejected(key, value):
    api = Api(["h5_j10_k5"])
    api.candidates[0].config["run_params"][key] = value
    with pytest.raises(ValueError):
        publisher.discover_rows(api)


def test_persisted_expected_launch_ids_cannot_change(tmp_path, sdk_backend):
    backend, _, calls = sdk_backend
    path = tmp_path / "workspace.json"
    expected = {variant: variant for variant in list(publisher.VARIANTS)[1:]}
    publisher.install(path, api=Api(), backend=backend, expected_ids=expected)
    count = len(calls)
    expected["h4_j10_k4"] = "different"
    with pytest.raises(ValueError, match="launch identities changed"):
        publisher.install(path, api=Api(), backend=backend, expected_ids=expected)
    assert len(calls) == count
