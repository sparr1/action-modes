"""The CPU watcher binds only the three immutable training identities."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from slurm import watch_ambi_jh_scaling_workspace as watcher
from utils.resume_identity import scientific_trial_parameters


COMMIT = "a" * 40
IDS = {variant: str(index) * 32 for index, variant in enumerate(watcher.LINEAGES, 1)}
REPO = Path(__file__).resolve().parents[1]


def trial(variant):
    value = json.loads((REPO / "configs/dmcontrol/algs" /
                       f"ambi_full_{variant}_reward_only.json").read_text())
    value.update(name=f"ambi_full_{variant}_reward_only", total_steps=1000000)
    return scientific_trial_parameters(value)


def lineage(root, variant):
    science = {"trial_run_params": trial(variant), "source": {"commit": COMMIT, "dirty": False}}
    identity = {"schema_version": 1, "scientific": science,
                "fingerprint": hashlib.sha256(json.dumps(
                    science, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()}
    value = {"schema_version": 1, "metadata": {"schema_version": 1, "total_steps": 1000000,
             "scientific_identity": identity, "initial_wandb_run_id": IDS[variant]}}
    path = root / "lineages" / watcher.LINEAGES[variant] / "LINEAGE.json"
    watcher.publisher._write(path, value)
    return path


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path.resolve()
    workspace = root / "publication/workspace.json"
    receipt = {key: getattr(watcher.publisher, key.upper()) for key in
               ("entity", "project", "group", "baseline", "name")}
    receipt.update(url=f"https://wandb.ai/{receipt['entity']}/{receipt['project']}?nw=owned-view",
                   rows=[], expected_ids={}, api_verified=True)
    watcher.publisher._write(workspace, receipt)
    jobs = root / "production-jobs.json"
    watcher.publisher._write(jobs, {variant: 100 + index for index, variant in enumerate(IDS)})
    return root, workspace, jobs


class Api:
    def __init__(self, visible):
        self.visible = visible

    def run(self, path):
        config = trial("h3_j12_k3")
        config["alg_params"].update(inner_rounds=10, inner_replay_capacity=3840,
            aux_return_mode="off", inner_actor_initialization="prior", inner_critic_initialization="prior")
        return SimpleNamespace(id=watcher.publisher.BASELINE, config={"run_params": config,
            "alg_params": config["alg_params"]}, summary={}, state="running")

    def runs(self, *args, **kwargs):
        for variant in self.visible:
            config = trial(variant)
            config["alg_params"].update(aux_return_mode="off", inner_actor_initialization="prior",
                                        inner_critic_initialization="prior")
            yield SimpleNamespace(id=IDS[variant], config={"run_params": config,
                "alg_params": config["alg_params"]}, summary={}, state="running")


def running(jobs):
    return {job: {"state": "RUNNING", "exit_code": "0:0"} for job in jobs.values()}


def test_wait_partial_publish_then_complete_without_creating_runs(prepared):
    root, workspace, jobs = prepared
    now, installs = [0], []

    def sleep(seconds):
        now[0] += seconds
        for variant in IDS:
            lineage(root, variant)

    def api():
        return Api(list(IDS)[:1] if now[0] < 2 else IDS)

    def install(path, *, api, expected_ids):
        assert path == workspace and expected_ids == IDS
        result = watcher.owned_workspace(path)
        result.update(rows=watcher.publisher.discover_rows(api, expected_ids=expected_ids),
                      expected_ids=expected_ids, api_verified=True)
        watcher.publisher._write(path, result)
        installs.append([row["run_id"] for row in result["rows"] if row["run_id"]])
        return result

    result = watcher.watch(root, workspace, jobs, COMMIT, timeout_seconds=5, poll_seconds=1,
        api_factory=api, installer=install, read_states=running, clock=lambda: now[0], sleep=sleep)
    assert result["state"] == "complete" and result["visible_ids"] == IDS
    assert [len(row) for row in installs] == [2, 4]
    assert not result["browser_verified"]


@pytest.mark.parametrize("mutation", ["dirty", "commit", "fingerprint", "run_id", "seed", "group", "name"])
def test_rejects_wrong_lineage_before_publication(prepared, mutation):
    root, workspace, jobs = prepared
    variant = next(iter(IDS))
    path = lineage(root, variant)
    value = json.loads(path.read_text())
    meta = value["metadata"]
    science = meta["scientific_identity"]["scientific"]
    if mutation in {"dirty", "commit"}:
        science["source"][mutation] = True if mutation == "dirty" else "b" * 40
    elif mutation == "run_id":
        meta["initial_wandb_run_id"] = "not-a-genesis-id"
    elif mutation == "seed":
        science["trial_run_params"]["seed"] = 56
    elif mutation == "group":
        science["trial_run_params"]["alg_params"]["wandb_group"] = "other"
    elif mutation == "name":
        science["trial_run_params"]["name"] = "other"
    meta["scientific_identity"]["fingerprint"] = hashlib.sha256(json.dumps(
        science, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if mutation == "fingerprint":
        meta["scientific_identity"]["fingerprint"] = "wrong"
    watcher.publisher._write(path, value)
    with pytest.raises(ValueError):
        watcher.watch(root, workspace, jobs, COMMIT, read_states=running)
    assert json.loads((root / "publication/workspace-watch-failure.json").read_text())["state"] == "failed"


def test_failed_gpu_job_never_calls_api_or_installer(prepared):
    root, workspace, jobs = prepared
    with pytest.raises(ValueError, match="Production job failed"):
        watcher.watch(root, workspace, jobs, COMMIT,
            read_states=lambda jobs: {"100": {"state": "FAILED", "exit_code": "1:0"}})
    result = json.loads((root / "publication/workspace-watch-failure.json").read_text())
    assert result["failed_jobs"]["100"]["state"] == "FAILED"


def test_timeout_missing_lineages_is_explicit_failure(prepared):
    root, workspace, jobs = prepared
    now = [0]
    with pytest.raises(TimeoutError):
        watcher.watch(root, workspace, jobs, COMMIT, timeout_seconds=2, poll_seconds=1,
            read_states=running, clock=lambda: now[0], sleep=lambda seconds: now.__setitem__(0, now[0]+seconds))
    result = json.loads((root / "publication/workspace-watch-status.json").read_text())
    assert result["state"] == "failed" and result["expected_ids"] == {}


def test_ownership_mismatch_cannot_create_new_workspace(prepared):
    root, workspace, jobs = prepared
    receipt = json.loads(workspace.read_text())
    receipt["url"] = "https://wandb.ai/someone/else?nw=owned-view"
    watcher.publisher._write(workspace, receipt)
    with pytest.raises(ValueError, match="named workspace"):
        watcher.watch(root, workspace, jobs, COMMIT)


def test_duplicate_jobs_rejected(prepared):
    root, workspace, jobs = prepared
    watcher.publisher._write(jobs, dict.fromkeys(IDS, "123"))
    with pytest.raises(ValueError, match="distinct Slurm"):
        watcher.read_jobs(jobs)


def test_lineage_ids_match_exactly_and_reject_duplicates(prepared):
    root, _, _ = prepared
    paths = [lineage(root, variant) for variant in IDS]
    assert watcher.lineage_ids(root, COMMIT) == IDS
    value = json.loads(paths[1].read_text())
    value["metadata"]["initial_wandb_run_id"] = next(iter(IDS.values()))
    watcher.publisher._write(paths[1], value)
    with pytest.raises(ValueError, match="same W&B"):
        watcher.lineage_ids(root, COMMIT)
