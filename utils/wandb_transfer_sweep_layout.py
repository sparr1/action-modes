"""Install the 575k transfer sweep's panels in an existing personal W&B view.

Transport, view selection, panel serialization and receipts reuse the maintained
results-layout helper. This module owns a distinct section/panel namespace, so
historical actor-transfer panels, filters and unrelated workspace content stay
unchanged. It never edits a run or its history.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from utils import wandb_results_layout as base


LAYOUT_VERSION = "transfer-sweep-results-v1"
DEFAULT_VIEW_NAME = base.DEFAULT_VIEW_NAME
ResultsLayoutError = base.ResultsLayoutError
OWNED_SECTION_IDS = tuple(
    f"ambi-{LAYOUT_VERSION}-{name}" for name in ("progress", "curves", "results", "episodes")
)
TABLE_KEYS = (
    "transfer_sweep/settings", "transfer_sweep/results",
    "transfer_sweep/paired_effects", "transfer_sweep/episodes",
)
CHART_KEYS = tuple(
    f"transfer_sweep/{source}_i{interval}_return_vs_{axis}"
    for axis in ("j", "compute") for interval in (1, 3) for source in ("soft", "return")
)


def _panel(identifier, kind, config, *, width=12, height=6):
    panel = base._panel(identifier, kind, config, width=width, height=height)
    panel["__id__"] = f"ambi-{LAYOUT_VERSION}-{identifier}"
    return panel


def _chart(key, title, xname):
    panel = base._chart(key, title, xname)
    panel["__id__"] = f"ambi-{LAYOUT_VERSION}-{key.replace('/', '-')}"
    panel["layout"]["w"] = 12
    return panel


def transfer_sweep_sections():
    """Static queries display progress before any five-seed result is complete."""
    intro = (
        "### 575K backbone · H3 warm-start comparison\n\n"
        "**25 progress rows: 24 solver settings plus one frozen-prior reference.** "
        "The sweep crosses J1/J8, fresh / actor-only / critic-only transfer, "
        "soft/soft or return/return critics, and solving every decision or every "
        "three decisions (hold3). Held steps use the fixed feedback policy on "
        "fresh observations. Each setting uses five paired environment seeds "
        "(101–105), controller seed55 and full 500-decision mean-action episodes. "
        "**Pending values are null, never zero. Partial episodes show progress only; "
        "result tables and curves include only completed five-seed means.** "
        "Returns are raw undiscounted episode rewards. Critic-only carries the "
        "online critic and resets the actor; its target is copied from that online "
        "critic at each solve. Actor-only retains the actor and resets the critic. "
        "Replay, optimizers and temperature reset at each solve; the backbone and "
        "terminal critic remain frozen. Compare gains against matching fresh "
        "settings and the frozen prior in the paired-effects table. Controller "
        "time excludes measured diagnostic probes; uncertainty is reported in "
        "the tables rather than implied by line styles."
    )
    progress = [
        _panel("intro", "Markdown Panel", {"value": intro}, width=24, height=5),
        _panel("settings", "Media Browser", {
            "chartTitle": "25 conditions · settings and episode progress",
            "mediaKeys": [TABLE_KEYS[0]],
        }, width=24, height=9),
    ]
    curves = []
    for axis in ("j", "compute"):
        for interval in (1, 3):
            cadence = "every decision" if interval == 1 else "hold3 feedback"
            for source in ("soft", "return"):
                critic = "Soft / soft" if source == "soft" else "Return / return"
                key = f"transfer_sweep/{source}_i{interval}_return_vs_{axis}"
                curves.append(_chart(
                    key,
                    f"{critic} · {cadence} · return vs " + ("J" if axis == "j" else "control time"),
                    "J rounds / solve" if axis == "j" else "Controller seconds / real decision",
                ))
    results = [
        _panel("results", "Media Browser", {
            "chartTitle": "Completed five-seed episode returns and timing",
            "mediaKeys": [TABLE_KEYS[1]],
        }, height=8),
        _panel("paired-effects", "Media Browser", {
            "chartTitle": "Paired gains over matched fresh settings and the frozen prior",
            "mediaKeys": [TABLE_KEYS[2]],
        }, height=8),
    ]
    episodes = [_panel("episodes", "Media Browser", {
        "chartTitle": "Per-seed episode outcomes · partial coverage is progress only",
        "mediaKeys": [TABLE_KEYS[3]],
    }, width=24, height=8)]
    sections = []
    for identifier, name, panels, columns, rows in zip(
        OWNED_SECTION_IDS,
        ("Transfer sweep | progress", "Transfer sweep | return and compute",
         "Transfer sweep | completed comparisons", "Transfer sweep | per-seed episodes"),
        (progress, curves, results, episodes), (1, 2, 2, 1), (2, 4, 1, 1),
    ):
        sections.append({
            "__id__": identifier, "name": name, "isOpen": True, "type": "flow",
            "flowConfig": {
                "snapToColumns": True, "columnsPerPage": columns, "rowsPerPage": rows,
                "gutterWidth": 16, "boxWidth": 560, "boxHeight": 320,
            },
            "sorted": 0, "pinned": True, "isPanelsAuto": False, "panels": panels,
        })
    return sections


def patch_transfer_sweep_spec(spec):
    """Replace only this sweep's stable sections, leaving the caller's spec intact."""
    proposed = deepcopy(spec)
    bank = base._bank(proposed)
    other_sections = [section for section in bank["sections"]
                      if section.get("__id__") not in OWNED_SECTION_IDS]
    bank["sections"] = transfer_sweep_sections() + other_sections
    return proposed


def _without_owned(spec):
    result = deepcopy(spec)
    bank = base._bank(result)
    bank["sections"] = [section for section in bank["sections"]
                        if section.get("__id__") not in OWNED_SECTION_IDS]
    return result


def _installed(spec):
    """Accept the same UI-omitted defaults as the existing results installer."""
    actual = deepcopy([section for section in base._bank(spec)["sections"]
                       if section.get("__id__") in OWNED_SECTION_IDS])
    expected = transfer_sweep_sections()
    if len(actual) != len(expected):
        return False
    for section, wanted in zip(actual, expected):
        section.setdefault("type", wanted["type"])
        flow = section.setdefault("flowConfig", {})
        if not isinstance(flow, dict):
            return False
        for key, value in wanted["flowConfig"].items():
            flow.setdefault(key, value)
        panels = section.get("panels")
        if not isinstance(panels, list) or len(panels) != len(wanted["panels"]):
            return False
        for panel, wanted_panel in zip(panels, wanted["panels"]):
            if not isinstance(panel, dict):
                return False
            panel.setdefault("layout", wanted_panel["layout"])
    return actual == expected


def ensure_transfer_sweep_results_layout(api, *, entity, project, receipt_dir,
                                         view_name=DEFAULT_VIEW_NAME, run_id=None):
    """Idempotently install and read back the selected personal view, with receipts.

    The maintained helper's two-read concurrency and uncertain-write reconciliation
    contract is retained. There is no atomic compare-and-swap: a concurrent edit
    after the second read can still be overwritten. Browser rendering must be
    verified separately after the saved workspace schema passes readback.
    """
    root = Path(receipt_dir)
    root.mkdir(parents=True, exist_ok=True)
    views = base._views(api, entity, project)
    view = base._selected(views, view_name)
    before = base._spec(view)
    proposed = patch_transfer_sweep_spec(before)
    fingerprint = base._hash(proposed)
    assert _without_owned(before) == _without_owned(proposed)
    workspace_url = f"https://wandb.ai/{entity}/{project}/workspace?nw={view_name[3:-2]}"
    url = f"https://wandb.ai/{entity}/{project}/runs/{run_id}?nw={view_name[3:-2]}" if run_id else None
    receipt = dict(
        schema_version=1, layout_version=LAYOUT_VERSION, entity=entity, project=project,
        view_id=view["id"], view_name=view_name, view_type="project-view",
        layout_scope="selected personal project workspace and its run pages",
        url=url, workspace_url=workspace_url, run_id=run_id,
        owned_section_ids=list(OWNED_SECTION_IDS), expected_chart_keys=list(CHART_KEYS),
        expected_table_keys=list(TABLE_KEYS), before_sha256=base._hash(before),
        proposed_sha256=fingerprint,
        verification="saved workspace schema read back; browser rendering must be checked separately",
    )
    base._write(root / ("before-" + base._hash(before)[:16] + ".json"), views)
    base._write(root / ("proposed-" + fingerprint[:16] + ".json"), proposed)
    if _installed(before):
        receipt.update(status="verified", changed=False, after_sha256=base._hash(before))
        base._write(root / "results-layout-receipt.json", receipt)
        return receipt
    current_views = base._views(api, entity, project)
    if base._selected(current_views, view_name) != view:
        raise ResultsLayoutError("Personal workspace changed during preparation; retry from its fresh state.")
    base._write(root / "results-layout-intent.json", {**receipt, "status": "prepared"})
    mutation_error = None
    try:
        base._execute(api, base._MUTATION, {
            "id": view["id"], "type": view["type"], "name": view_name,
            "displayName": view["displayName"], "spec": json.dumps(proposed, separators=(",", ":")),
        })
    except Exception as exc:
        mutation_error = type(exc).__name__
    try:
        after_views = base._views(api, entity, project)
        after_view = base._selected(after_views, view_name)
        after = base._spec(after_view)
        base._write(root / ("after-" + base._hash(after)[:16] + ".json"), after_views)
        old_others = {item["id"]: item for item in views if item["id"] != view["id"]}
        new_others = {item["id"]: item for item in after_views if item["id"] != view["id"]}
        if (after != proposed
                or {key: value for key, value in after_view.items() if key != "spec"}
                != {key: value for key, value in view.items() if key != "spec"}
                or any(new_others.get(key) != value for key, value in old_others.items())):
            raise ResultsLayoutError("Workspace readback differs; preserve receipts and inspect concurrent edits before retrying.")
        receipt.update(
            status="verified", changed=True, after_sha256=base._hash(after),
            uncertain_response_reconciled=mutation_error is not None,
            preserved_existing_views=len(old_others),
        )
        base._write(root / "results-layout-receipt.json", receipt)
        return receipt
    except Exception as exc:
        base._write(root / "results-layout-receipt.json", {
            **receipt, "status": "uncertain", "mutation_error_type": mutation_error,
            "verification_error_type": type(exc).__name__,
        })
        raise ResultsLayoutError(
            "Results-layout write could not be verified; inspect the saved before/after receipts. Evaluation data are unaffected."
        ) from exc
