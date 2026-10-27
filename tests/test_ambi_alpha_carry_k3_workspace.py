"""Protect exact comparison identity, idempotence, and visible setup failures."""
from copy import deepcopy
import json
from types import SimpleNamespace as Object

import pytest

from slurm import ambi_alpha_carry_k3_workspace as layout


class Metric:
    def __init__(self, name):
        self.name = name

    def isin(self, values):
        return Object(to_model=lambda: Object(model_dump=lambda **kwargs:
                      {"op": "IN", "key": {"name": self.name}, "value": values}))

    def __eq__(self, value):
        return Object(to_model=lambda: Object(model_dump=lambda **kwargs:
                      {"op": "=", "key": {"name": self.name}, "value": value}))


def either(*items):
    return Object(to_model=lambda: Object(model_dump=lambda **kwargs:
                  {"op": "OR", "filters": [item.to_model().model_dump() for item in items]}))


class LinePlot(Object):
    pass


def backend(corrupt=None):
    state = {"created": 0, "saved": {}}

    class Workspace(Object):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            if kwargs:
                state["created"] += 1
                self.url = f"https://wandb.ai/{layout.ENTITY}/{layout.PROJECT}?nw=test{state['created']}"

        def save(self):
            state["saved"][self.url] = deepcopy(self)

        @classmethod
        def from_url(cls, url):
            work = deepcopy(state["saved"][url])
            if corrupt:
                corrupt(work)
            return work

    ws = Object(Workspace=Workspace, Section=Object, WorkspaceSettings=Object,
                RunsetSettings=lambda **kwargs: Object(query="", **kwargs), RunSettings=Object)
    wr = Object(LinePlot=LinePlot, MarkdownPanel=Object, Layout=Object)
    return (ws, wr, Object(Metric=Metric, Or=either)), state


def api(pending=False, j10_pending=True, duplicate=None):
    def runs(path, filters, per_page):
        assert path == f"{layout.ENTITY}/{layout.PROJECT}"
        expected = {"$or": [{"name": layout.SOURCE_RUN}, {"group": layout.GROUP}, {"group": layout.J10_GROUP}]}
        assert filters == expected
        assert per_page == 4
        cells = [(layout.SOURCE_RUN, None)]
        if not pending:
            cells.append(("newk3run", layout.GROUP))
        if not j10_pending:
            cells.append(("j10run", layout.J10_GROUP))
        if duplicate is not None:
            cells.append(("duplicate", duplicate))
        return [Object(id=run_id, group=group, state="running", summary={"env_step": 500})
                for run_id, group in cells]
    return Object(runs=runs)


def test_pending_then_registered_uses_same_view_and_preserves_user_sections(tmp_path):
    sdk, state = backend()
    path = tmp_path / "receipt.json"
    receipt = layout.install(path, "newk3run", api=api(pending=True), backend=sdk)
    assert receipt["rows"][1]["state"] == "pending registration"
    assert receipt["rows"][1]["decisions"] is None
    url = receipt["url"]
    state["saved"][url].sections.append(Object(name="User notes", panels=[Object(text="Preserve")]))
    refreshed = layout.install(path, "newk3run", api=api(), backend=sdk)
    assert refreshed["api_verified"] and refreshed["url"] == url
    assert refreshed["rows"][1]["decisions"] == 500
    assert state["created"] == 1
    assert state["saved"][url].sections[-1].panels[0].text == "Preserve"
    plots = [panel for section in state["saved"][url].sections for panel in section.panels
             if isinstance(panel, LinePlot)]
    assert len(plots) == len(layout.METRICS)
    assert all(len(plot.line_titles) == 2 and not plot.aggregate and plot.max_runs_to_show == 3 for plot in plots)
    assert refreshed["rows"][2]["state"] == "pending startup"


@pytest.mark.parametrize("explicit_bind", [False, True])
def test_group_pending_view_binds_unique_id_without_recreating(tmp_path, explicit_bind):
    sdk, state = backend()
    path = tmp_path / "receipt.json"
    pending = layout.install(path, api=api(pending=True), backend=sdk)
    assert pending["api_verified"] and pending["run_id"] is None
    assert pending["rows"][1]["state"] == "pending startup"
    saved = state["saved"][pending["url"]]
    assert set(saved.runset_settings.run_settings) == {layout.SOURCE_RUN}
    assert "awaiting registration" in saved.sections[0].panels[0].markdown
    assert saved.runset_settings.filters.to_model().model_dump()["op"] == "OR"
    result = layout.install(path, "newk3run" if explicit_bind else None,
                            j10_run_id="j10run" if explicit_bind else None,
                            api=api(j10_pending=False), backend=sdk)
    assert result["url"] == pending["url"] and state["created"] == 1
    assert result["run_id"] == "newk3run" and result["api_verified"]
    assert result["j10_run_id"] == "j10run"
    assert state["saved"][result["url"]].runset_settings.filters[0].to_model().model_dump()["op"] == "IN"
    # Once bound, later calls without --run-id retain exact identity.
    assert layout.install(path, api=api(j10_pending=False), backend=sdk)["run_id"] == "newk3run"


@pytest.mark.parametrize("duplicate_group", [layout.GROUP, layout.J10_GROUP])
def test_group_discovery_rejects_ambiguous_k3_runs(tmp_path, duplicate_group):
    path = tmp_path / "receipt.json"
    with pytest.raises(layout.WorkspaceError, match="Multiple K3"):
        layout.install(path, api=api(j10_pending=False, duplicate=duplicate_group), backend=backend()[0])
    assert not json.loads(path.read_text())["api_verified"]


def test_legacy_receipt_migrates_existing_view_with_pending_j10(tmp_path):
    sdk, state = backend()
    old = sdk[0].Workspace(entity=layout.ENTITY, project=layout.PROJECT, name=layout.LEGACY_NAME,
                           sections=[Object(name="User notes", panels=[Object(text="Keep")])])
    old.save()
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(dict(entity=layout.ENTITY, project=layout.PROJECT,
        source_run_id=layout.SOURCE_RUN, group=layout.GROUP, name=layout.LEGACY_NAME,
        run_id=None, url=old.url)))
    receipt = layout.install(path, api=api(), backend=sdk)
    assert receipt["api_verified"] and receipt["url"] == old.url and state["created"] == 1
    assert receipt["name"] == layout.NAME and receipt["run_id"] == "newk3run"
    assert receipt["j10_run_id"] is None
    saved = state["saved"][old.url]
    assert saved.sections[0].panels[0].text == "Keep"
    protocol = saved.sections[1].panels[0].markdown
    assert "K=3, J=10 | pending startup" in protocol
    refreshed = layout.install(path, api=api(j10_pending=False), backend=sdk)
    assert refreshed["url"] == old.url and refreshed["j10_run_id"] == "j10run"
    plots = [panel for section in state["saved"][old.url].sections for panel in section.panels
             if isinstance(panel, LinePlot)]
    assert len(plots) == len(layout.METRICS)
    assert all(len(panel.line_titles) == 3 and len(set(panel.line_colors.values())) == 3 for panel in plots)
    assert "K=3 J=8" in plots[0].line_titles["newk3run:episode/return"]
    assert "K=3 J=10" in plots[0].line_titles["j10run:episode/return"]


def test_bound_id_cannot_silently_select_a_different_group_member(tmp_path):
    with pytest.raises(layout.WorkspaceError, match="differs from its bound ID"):
        layout.install(tmp_path / "receipt.json", "wrongrun", api=api(), backend=backend()[0])


@pytest.mark.parametrize("failure", ["axis", "filter", "color"])
def test_server_corruption_fails_and_retains_view_identity(tmp_path, failure):
    def corrupt(work):
        if failure == "axis":
            work.sections[1].panels[0].x = "Step"
        elif failure == "filter":
            work.runset_settings.filters = [Metric("ID").isin(["wrong"])]
        else:
            work.runset_settings.run_settings["newk3run"].color = "#FFFFFF"
    sdk, state = backend(corrupt)
    path = tmp_path / "receipt.json"
    with pytest.raises(layout.WorkspaceError):
        layout.install(path, "newk3run", api=api(), backend=sdk)
    receipt = json.loads(path.read_text())
    assert receipt["url"] and not receipt["api_verified"]
    assert receipt["error_type"] == "WorkspaceError" and state["created"] == 1


def test_receipt_for_different_variant_cannot_be_replaced(tmp_path):
    path = tmp_path / "receipt.json"
    original = '{"run_id":"another"}'
    path.write_text(original)
    with pytest.raises(layout.WorkspaceError, match="another comparison"):
        layout.install(path, "newk3run", api=api(), backend=backend()[0])
    assert path.read_text() == original


@pytest.mark.parametrize("pending", [False, True])
def test_pinned_sdk_serialization_without_network(tmp_path, monkeypatch, pending):
    ws = pytest.importorskip("wandb_workspaces.workspaces")
    if (layout.importlib.metadata.version("wandb") != "0.28.1"
            or layout.importlib.metadata.version("wandb-workspaces") != "0.4.5"):
        pytest.skip("Requires the isolated pinned workspace runtime")
    state = {}
    url = f"https://wandb.ai/{layout.ENTITY}/{layout.PROJECT}?nw=mockview"
    monkeypatch.setattr(ws.Workspace, "url", property(lambda self: url))
    monkeypatch.setattr(ws.Workspace, "save", lambda self: state.update(work=deepcopy(self)))
    monkeypatch.setattr(ws.Workspace, "from_url", classmethod(lambda cls, value: deepcopy(state["work"])))
    receipt = layout.install(tmp_path / "receipt.json", None if pending else "newk3run",
                             api=api(pending=pending, j10_pending=pending))
    assert receipt["api_verified"]
    wire = state["work"]._to_model().model_dump_json()
    assert layout.SOURCE_RUN in wire
    assert (layout.GROUP if pending else "newk3run") in wire
    assert (layout.J10_GROUP if pending else "j10run") in wire
    assert layout.DIAGNOSTIC_STEP in wire and "solid" in wire
