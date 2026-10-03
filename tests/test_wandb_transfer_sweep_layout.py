"""Offline schema, preservation and readback checks for transfer sweep panels."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from tests.test_wandb_results_layout import Service, sample_spec
from utils import wandb_results_layout as actor_layout
from utils import wandb_transfer_sweep_layout as layout


def install(tmp_path, service, **kwargs):
    return layout.ensure_transfer_sweep_results_layout(
        SimpleNamespace(_service_api=service), entity="entity", project="project",
        receipt_dir=tmp_path, **kwargs,
    )


def test_patch_preserves_actor_sections_filters_and_all_unrelated_fields():
    original = actor_layout.patch_actor_transfer_spec(sample_spec())
    original["section"]["runSets"].append({"filters": {"run": "keep-existing-selection"}})
    before = deepcopy(original)
    actor_sections = actor_layout.actor_transfer_sections()
    patched = layout.patch_transfer_sweep_spec(original)
    assert original == before
    assert layout.patch_transfer_sweep_spec(patched) == patched
    assert layout._without_owned(patched) == original
    assert actor_layout.actor_transfer_sections() == actor_sections
    assert patched["section"]["runSets"] == original["section"]["runSets"]
    assert actor_layout._bank(patched)["sections"][4:7] == actor_sections
    assert not set(layout.OWNED_SECTION_IDS) & set(actor_layout.OWNED_SECTION_IDS)


def test_visible_sections_have_exact_publisher_keys_and_unambiguous_pending_text():
    sections = layout.transfer_sweep_sections()
    assert tuple(section["__id__"] for section in sections) == layout.OWNED_SECTION_IDS
    assert all(section["isOpen"] and not section["isPanelsAuto"] for section in sections)
    panels = [panel for section in sections for panel in section["panels"]]
    assert len(panels) == 13
    assert all(not panel["isAuto"] for panel in panels)
    ids = {panel["__id__"] for panel in panels}
    assert len(ids) == len(panels)
    assert not ids & {panel["__id__"] for section in actor_layout.actor_transfer_sections()
                      for panel in section["panels"]}
    charts = [panel for panel in panels if panel["viewType"] == "Vega2"]
    assert len(charts) == 8
    chart_keys = []
    for panel in charts:
        config = panel["config"]
        assert config["panelDefId"] == "wandb/lineseries/v0"
        assert config["fieldSettings"] == {"step": "step", "lineKey": "lineKey", "lineVal": "lineVal"}
        chart_keys.append(config["userQuery"]["queryFields"][0]["fields"][0]["args"][0]["value"])
        assert config["stringSettings"]["xname"] in {
            "J rounds / solve", "Controller seconds / real decision",
        }
    assert chart_keys == [key + "_table" for key in layout.CHART_KEYS]
    assert set(layout.CHART_KEYS) == {
        f"transfer_sweep/{source}_i{interval}_return_vs_{axis}"
        for source in ("soft", "return") for interval in (1, 3) for axis in ("j", "compute")
    }
    media_keys = [panel["config"]["mediaKeys"][0] for panel in panels
                  if panel["viewType"] == "Media Browser"]
    assert tuple(media_keys) == layout.TABLE_KEYS
    assert set(media_keys) == {
        "transfer_sweep/settings", "transfer_sweep/results",
        "transfer_sweep/paired_effects", "transfer_sweep/episodes",
    }
    intro = panels[0]["config"]["value"]
    for text in ("575K", "H3", "J1/J8", "25 progress rows", "24 solver settings",
                 "frozen-prior reference", "fresh / actor-only / critic-only", "hold3",
                 "Pending values are null, never zero", "Partial episodes show progress only",
                 "completed five-seed means", "500-decision", "101–105"):
        assert text in intro


def test_install_preserves_all_other_views_and_repeated_install_does_not_write(tmp_path):
    service = Service()
    service.views[0]["spec"] = json.dumps(actor_layout.patch_actor_transfer_spec(sample_spec()))
    before = deepcopy(service.views)
    receipt = install(tmp_path, service, run_id="specific-transfer-overview")
    assert receipt["status"] == "verified" and receipt["changed"]
    assert receipt["layout_version"] == layout.LAYOUT_VERSION
    assert receipt["expected_table_keys"] == list(layout.TABLE_KEYS)
    assert receipt["expected_chart_keys"] == list(layout.CHART_KEYS)
    assert receipt["url"] == "https://wandb.ai/entity/project/runs/specific-transfer-overview?nw=nwuserrwgao_b"
    assert receipt["workspace_url"] == "https://wandb.ai/entity/project/workspace?nw=nwuserrwgao_b"
    assert service.writes == 1 and service.views[1:] == before[1:]
    assert layout._without_owned(json.loads(service.views[0]["spec"])) == json.loads(before[0]["spec"])
    mutation = service.mutations[0]
    assert mutation["id"] == "personal" and mutation["type"] == "project-view"
    assert not {"entityName", "projectName", "projectId", "parentId", "userId"} & mutation.keys()
    again = install(tmp_path, service, run_id="specific-transfer-overview")
    assert again["status"] == "verified" and again["changed"] is False
    assert service.writes == 1
    assert json.loads((tmp_path / "results-layout-receipt.json").read_text()) == again
    assert (tmp_path / "results-layout-intent.json").is_file()
    assert list(tmp_path.glob("before-*.json")) and list(tmp_path.glob("after-*.json"))


def normalized_spec():
    spec = layout.patch_transfer_sweep_spec(sample_spec())
    for section in actor_layout._bank(spec)["sections"][:4]:
        section.pop("type")
        section["flowConfig"] = {"columnsPerPage": section["flowConfig"]["columnsPerPage"]}
        for panel in section["panels"]:
            panel.pop("layout")
    return spec


def test_ui_omitted_defaults_are_accepted_without_mutating_or_rewriting(tmp_path):
    spec = normalized_spec()
    before = deepcopy(spec)
    assert layout._installed(spec) and spec == before
    service = Service()
    service.views[0]["spec"] = json.dumps(spec)
    before_views = deepcopy(service.views)
    receipt = install(tmp_path, service)
    assert receipt["status"] == "verified" and receipt["changed"] is False
    assert service.writes == 0 and service.views == before_views


@pytest.mark.parametrize("corruption", [
    "section_id", "panel_id", "missing_panel", "section_order", "hidden", "auto",
    "query", "table", "intro", "flow_null", "flow_changed", "layout_changed",
])
def test_installed_check_rejects_changed_content_and_visibility(corruption):
    spec = normalized_spec()
    sections = actor_layout._bank(spec)["sections"]
    chart = sections[1]["panels"][0]
    if corruption == "section_id":
        sections[1]["__id__"] = "not-ours"
    elif corruption == "panel_id":
        chart["__id__"] = "wrong-panel"
    elif corruption == "missing_panel":
        sections[1]["panels"].pop()
    elif corruption == "section_order":
        sections[1], sections[2] = sections[2], sections[1]
    elif corruption == "hidden":
        sections[1]["isOpen"] = False
    elif corruption == "auto":
        chart["isAuto"] = True
    elif corruption == "query":
        chart["config"]["userQuery"]["queryFields"][0]["fields"][0]["args"][0]["value"] = "wrong_table"
    elif corruption == "table":
        sections[2]["panels"][0]["config"]["mediaKeys"] = ["wrong/results"]
    elif corruption == "intro":
        sections[0]["panels"][0]["config"]["value"] = "Treat pending as zero."
    elif corruption == "flow_null":
        sections[1]["flowConfig"] = None
    elif corruption == "flow_changed":
        sections[1]["flowConfig"]["columnsPerPage"] = 3
    elif corruption == "layout_changed":
        chart["layout"] = {"x": 0, "y": 0, "w": 1, "h": 6}
    assert not layout._installed(spec)


def test_concurrent_edit_before_mutation_is_preserved(tmp_path):
    service = Service(concurrent=True)
    with pytest.raises(layout.ResultsLayoutError, match="changed during preparation"):
        install(tmp_path, service)
    assert service.writes == 0


def test_lost_success_response_is_reconciled_without_duplicate_write(tmp_path):
    service = Service(timeout=True)
    receipt = install(tmp_path, service)
    assert receipt["status"] == "verified" and receipt["uncertain_response_reconciled"]
    assert install(tmp_path, service)["changed"] is False
    assert service.writes == 1


def test_unapplied_write_leaves_uncertain_receipt_then_retries_without_duplicate_sections(tmp_path):
    service = Service(timeout=True, apply=False)
    with pytest.raises(layout.ResultsLayoutError, match="could not be verified"):
        install(tmp_path, service)
    receipt = json.loads((tmp_path / "results-layout-receipt.json").read_text())
    assert receipt["status"] == "uncertain" and receipt["mutation_error_type"] == "TimeoutError"
    service.timeout = False
    service.apply = True
    assert install(tmp_path, service)["status"] == "verified"
    sections = actor_layout._bank(json.loads(service.views[0]["spec"]))["sections"]
    assert len({section["__id__"] for section in sections}) == len(sections)


def test_readback_detects_unrelated_view_changes(tmp_path):
    class AlterOtherView(Service):
        def execute_graphql(self, query, variables):
            response = super().execute_graphql(query, variables)
            if "mutation " in query:
                self.views[2]["displayName"] = "Concurrent external edit"
            return response
    with pytest.raises(layout.ResultsLayoutError, match="could not be verified"):
        install(tmp_path, AlterOtherView())
    receipt = json.loads((tmp_path / "results-layout-receipt.json").read_text())
    assert receipt["status"] == "uncertain"


def test_missing_or_nonpersonal_view_never_creates_a_workspace(tmp_path):
    service = Service()
    service.views[0]["name"] = "shared-view"
    with pytest.raises(layout.ResultsLayoutError, match="Expected one"):
        install(tmp_path, service)
    assert service.writes == 0
    with pytest.raises(layout.ResultsLayoutError, match="personal project view"):
        install(tmp_path, service, view_name="shared-view")
    assert service.writes == 0
