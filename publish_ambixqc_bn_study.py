"""Publication helpers for the explicitly launched, staged XQC BN study.

GPU workers never publish. The Oscar CPU coordinator owns all curve registries;
this module installs a dedicated workspace and verifies immutable result receipt.
"""
from __future__ import annotations

import json
from pathlib import Path

from utils.eval_series import Publisher, create_run, load_run, validate_identity, validate_record
from utils.ambi_benchmark import atomic_json

CAMPAIGN = "ambixqc-bn-study-20260929"
ENTITY, PROJECT = "rwgao_b-brown-university", "ambi-inner-bench"
OWNER = "oscar-rgao48"
VIEW_NAME = "nw-xqc-bn290926-v"
VIEW_URL = f"https://wandb.ai/{ENTITY}/{PROJECT}/workspace?nw=xqc-bn290926"
VIEW_QUERY = '''query View($entityName:String,$name:String){project(name:$name,entityName:$entityName){allViews(viewType:"project-view"){edges{node{id name type displayName spec}}}}}'''
VIEW_MUTATION = '''mutation UpdateCharts($id:ID,$entityName:String,$projectName:String,$type:String,$name:String,$displayName:String,$spec:String){upsertView(input:{id:$id,entityName:$entityName,projectName:$projectName,type:$type,name:$name,displayName:$displayName,spec:$spec,createdUsing:WANDB_SDK}){view{id name} inserted}}'''


def install_workspace(api, spec_path):
    """Touch this dedicated view only, preserving panel identities on retries."""
    from wandb_gql import gql
    proposed = json.loads(Path(spec_path).read_text())
    if proposed.get("campaign") != CAMPAIGN or proposed.get("name") != VIEW_NAME:
        raise ValueError("Workspace proposal belongs to a different study")
    spec = proposed["spec"]
    variables = {"entityName": ENTITY, "name": PROJECT}
    response = api.client.execute(gql(VIEW_QUERY), variable_values=variables)
    nodes = [e["node"] for e in response["project"]["allViews"]["edges"]
             if e["node"]["name"] == VIEW_NAME]
    if len(nodes) > 1:
        raise ValueError("Ambiguous dedicated workspace; refusing to replace views")
    if nodes:
        old = json.loads(nodes[0]["spec"]) if isinstance(nodes[0]["spec"], str) else nodes[0]["spec"]
        old_sections = {s["name"]: s for s in old["section"]["panelBankConfig"]["sections"]}
        for section in spec["section"]["panelBankConfig"]["sections"]:
            previous = old_sections.get(section["name"])
            if previous:
                for panel, saved in zip(section["panels"], previous["panels"]):
                    if panel["viewType"] == saved["viewType"]:
                        panel["__id__"] = saved["__id__"]
    api.client.execute(gql(VIEW_MUTATION), variable_values={
        "id": nodes[0]["id"] if nodes else None, "entityName": ENTITY,
        "projectName": PROJECT, "type": "project-view", "name": VIEW_NAME,
        "displayName": "AMBI-XQC 475k · BN and adaptation study",
        "spec": json.dumps(spec, separators=(",", ":")),
    })
    check = api.client.execute(gql(VIEW_QUERY), variable_values=variables)
    node, = [e["node"] for e in check["project"]["allViews"]["edges"]
             if e["node"]["name"] == VIEW_NAME]
    saved = json.loads(node["spec"]) if isinstance(node["spec"], str) else node["spec"]
    if saved != spec:
        raise RuntimeError("Dedicated workspace readback differs from installed panels")
    return {"url": VIEW_URL, "view_id": node["id"], "verified": True}


def allocate_curve(root, spec, stage, selector):
    """Idempotently recover an allocation from its exact scientific identity."""
    root = Path(root)
    attempt = CAMPAIGN
    candidates = [load_run(p.parent) for p in (root / "registries").glob("*/run.json")]
    candidates = [r for r in candidates if r["identity"] == spec["identity"]
                  and r["attempt_label"] == attempt and r["owner"] == OWNER]
    if len(candidates) > 1:
        raise ValueError("Multiple registry allocations match this study condition")
    registry = candidates[0] if candidates else create_run(
        root / "registries", spec, attempt_label=attempt,
        project=PROJECT, entity=ENTITY, owner=OWNER)
    validate_identity(registry, spec["identity"])
    # Initialize a visible pending curve before GPU evaluation. No fake history
    # row or zero-valued return is inserted. SDK resume ownership stays intact.
    with Publisher(registry["run_dir"], owner=OWNER) as pub:
        pub.run.config.update({"campaign_id": CAMPAIGN, "study_stage": stage,
                               "study_selector": selector}, allow_val_change=True)
        pub.run.summary["study/expected_episodes"] = 5
        entries = json.loads((Path(registry["run_dir"]) / "publication.json").read_text())["records"]
        pub.run.summary["study/status"] = (
            "complete" if len(entries) == 1 and all(e["status"] == "published" for e in entries.values())
            else "publishing" if entries else "pending"
        )
    return registry


def _validated_record(run_dir, selector, checkpoint_sha, source_sha):
    """Check the staged scientific result before any history upload occurs."""
    registry = load_run(run_dir)
    index = json.loads((Path(run_dir) / "publication.json").read_text())
    if len(index["records"]) != 1:
        raise RuntimeError("Completed condition must stage exactly one checkpoint result")
    rid, entry = next(iter(index["records"].items()))
    record = validate_record(json.loads((Path(run_dir) / "records" / (rid + ".json")).read_text()),
                             registry["identity"])
    actual = record.get("selector") or record.get("provenance", {}).get("selector")
    code = record["provenance"].get("code", {})
    episodes = record["episodes"]
    if (entry["checkpoint_sha256"] != checkpoint_sha or actual != selector
            or record["checkpoint"] != {"step": 475000, "sha256": checkpoint_sha}
            or code.get("commit") != source_sha or code.get("dirty") is not False
            or sorted(ep.get("seed", -1) for ep in episodes) != [101, 102, 103, 104, 105]
            or any(ep.get("length") != 500 or ep.get("capped") for ep in episodes)):
        raise ValueError("Publisher received a different checkpoint, source, controller or episode protocol")
    return registry, rid, entry


def publish_curve(run_dir, selector, checkpoint_sha, *, source_sha, publisher_factory=Publisher):
    with publisher_factory(run_dir, owner=OWNER) as pub:
        # Loading immutable pointers performs local validation/staging only.
        # Reject mismatched records before publish_pending can emit a row.
        pub._load_incoming()
        _validated_record(run_dir, selector, checkpoint_sha, source_sha)
        progress = pub.publish_pending()
        if progress["accepted"] != 1:
            raise RuntimeError("Completed condition must stage exactly one checkpoint result")
        pub.run.summary["study/status"] = "publishing"
    registry, rid, entry = _validated_record(run_dir, selector, checkpoint_sha, source_sha)
    if entry["status"] != "published":
        raise RuntimeError("W&B has not acknowledged the complete condition")
    receipt = {"selector": selector, "checkpoint_sha256": checkpoint_sha,
               "source_sha": source_sha, "run_id": registry["run_id"], "record_id": rid,
               "record_sha256": entry["record_sha256"], "accepted": 1, "published": 1}
    # A second, summary-only session marks completion after acknowledged history.
    with publisher_factory(run_dir, owner=OWNER) as pub:
        pub.run.summary["study/status"] = "complete"
    atomic_json(Path(run_dir) / "bn-study-publication-verified.json", receipt, overwrite=True)
    return receipt


def update_progress(api, run_id, **values):
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    for key, value in values.items():
        run.summary["study/" + key] = value
    run.summary.update()
