import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from slurm.boom_likelihood_pilot_oscar import GPU_TESTS, MANIFEST, parity_cases, select_plan
from slurm import boom_likelihood_pilot_oscar as pilot
from tests.test_ambi_root_local_sac import _build_cfg

ROOT = Path(__file__).resolve().parents[1]


def test_pilot_preserves_backbone_and_enables_exact_likelihood_diagnostics():
    manifest = json.loads(MANIFEST.read_text())
    assert len(manifest["configs"]) == 1 and manifest["trials"] == 1
    assert manifest["overrides_alg"]["total_steps"] == 400000
    target = json.loads((ROOT / "configs/dmcontrol/algs" / (manifest["configs"][0] + ".json")).read_text())
    source = json.loads((ROOT / "configs/dmcontrol/algs/ambi_posttanh_prior_outer_adapt_boom_x0p01.json").read_text())
    params = target["alg_params"]
    for key, value in source["alg_params"].items():
        if not key.startswith("wandb"):
            assert params[key] == value, key
    cfg = _build_cfg(**params)
    assert cfg.outer_boom_coef == .0042 and cfg.outer_boom_q_temperature == "scale"
    assert cfg.outer_boom_alignment == "replay_likelihood" and cfg.store_behavior_pre_tanh_action
    assert cfg.outer_boom_diagnostics and cfg.critic_geometry_diagnostics
    assert cfg.inner_cuda_graph and cfg.compile and cfg.compile_strict
    assert cfg.critic_ensemble_backend == "vmap"
    assert target["total_steps"] == 400000 and target["checkpoint_every"] == 25000


@pytest.mark.parametrize("invalid", ["skipped", "failure", "error", "missing", "wrong_name", None])
def test_gpu_gate_requires_all_actual_cuda_cases(tmp_path, invalid):
    root = ET.Element("testsuite")
    for node in GPU_TESTS:
        ET.SubElement(root, "testcase", name=node.split("::")[-1])
    if invalid == "missing":
        root.remove(root[0])
    elif invalid == "wrong_name":
        root[0].set("name", "different_test")
    elif invalid:
        ET.SubElement(root[0], invalid)
    path = tmp_path / "tests.xml"
    ET.ElementTree(root).write(path)
    if invalid:
        with pytest.raises(AssertionError):
            parity_cases(path)
    else:
        assert len(parity_cases(path)) == 4


def test_resource_plan_uses_400k_budget_and_measured_cpu_tradeoff():
    def row(cpus, seconds):
        return {"index": 0, "physical_cpus": cpus, "measured": {"seconds_per_decision": seconds}, "checkpoint_bytes": 100}
    plan = select_plan([row(1, .2), row(2, .196)])
    assert plan["cpus"] == 1 and plan["wall_hours"] == 29
    assert select_plan([row(1, .2), row(2, .17)])["cpus"] == 2


@pytest.mark.parametrize("mismatch", [None, "commit", "gate", "dependency"])
def test_release_only_resizes_the_bound_dependent_production_job(tmp_path, monkeypatch, mismatch):
    state = {"commit": "abc", "runs": [{"index": 0, "gate_job": "41", "job": "42"}]}
    if mismatch == "commit":
        state["commit"] = "other"
    if mismatch == "gate":
        state["runs"][0]["gate_job"] = "40"
    (tmp_path / "submission.json").write_text(json.dumps(state))
    monkeypatch.setenv("SLURM_JOB_ID", "41")
    calls = []
    monkeypatch.setattr(pilot.base, "slurm_fields", lambda job: {"Dependency": "afterok:99(unfulfilled)" if mismatch == "dependency" else "afterok:41(unfulfilled)"})
    monkeypatch.setattr(pilot.base, "resize_pending", lambda job, plan: calls.append((job, plan)) or {"CPUs/Task": "1"})
    receipt = {"passed": True, "binding": {"commit": "abc"}, "plan": {"cpus": 1}}
    if mismatch:
        with pytest.raises(AssertionError):
            pilot.release_production(receipt, tmp_path)
        assert not calls
    else:
        pilot.release_production(receipt, tmp_path)
        assert calls == [(42, {"cpus": 1})]
        assert json.loads((tmp_path / "allocation.json").read_text())["production_job"] == 42
