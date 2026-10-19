"""Launcher provenance/resource contracts without checkpoint computation."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from slurm import ambi_gradient_probe_campaign as campaign


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    rows = []
    for backbone, source in campaign.SOURCES.items():
        checkpoint, metadata = tmp_path / backbone, tmp_path / (backbone + ".metadata.json")
        checkpoint.write_bytes(backbone.encode())
        campaign.write_new(metadata, dict(checkpoint=dict(step=400000),
            trial_run_params=dict(alg_params=dict(inner_operator="mppi" if backbone == "mppi_aux" else "sac",
                                                 aux_return_mode="sac"))))
        rows.append(dict(backbone_id=backbone, source_run=source, step=400000,
                         checkpoint=str(checkpoint), metadata=str(metadata),
                         checkpoint_sha256=campaign.sha256(checkpoint), metadata_sha256=campaign.sha256(metadata)))
    inventory, root = tmp_path / "inventory.json", tmp_path / "outputs"
    campaign.write_new(inventory, dict(checkpoints=rows))
    monkeypatch.setattr(campaign, "verify_source", lambda expected: expected)
    data = campaign.prepare(inventory, root, "a" * 40)
    return root, data, inventory


def make_bundle(bundle, data, spec):
    """Minimal valid probe output, using the actual canonical checksum schema."""
    target = data["checkpoints"][spec["target"]]
    source = data["checkpoints"][spec["source"] or spec["target"]]
    roots = [dict(seed=seed, decision_index=decision, root_id=f"seed-{seed}-decision-{decision}")
             for seed in spec["seeds"] for decision in spec["decisions"]]
    bank = dict(complete=True, checkpoint_sha256=source["checkpoint_sha256"], roots=roots)
    bank["id"] = campaign.canonical_hash(bank)
    if spec["root_bank"]:
        bank = campaign.read(spec["root_bank"])
    campaign.write_new(bundle / "root-bank.json", bank)
    records = []
    for i, root in enumerate(roots):
        row = dict(root_id=root["root_id"], work=dict(simulator_decisions=8))
        path = f"roots/{i:04d}.json"
        campaign.write_new(bundle / path, row)
        records.append(dict(path=path, sha256=campaign.canonical_hash(row)))
    protocol = {k: spec[k] for k in ("horizon", "tail_steps", "rollouts", "gradient_rollouts", "epsilons")}
    protocol.update(roots=len(roots), seed=55, q_reduction="mean_all", entropy_bonus=False)
    result = dict(kind="ambi_inherited_gradient_probe", complete=True, outer_state_unchanged=True,
                  checkpoint=dict(sha256=target["checkpoint_sha256"], step=400000,
                                  metadata_sha256=target["metadata_sha256"]),
                  root_bank_id=bank["id"], root_bank_checkpoint_sha256=source["checkpoint_sha256"],
                  protocol=protocol, root_records=records, branch_simulator_decisions=8 * len(roots))
    result["sha256"] = campaign.canonical_hash(result)
    campaign.write_new(bundle / "result.json", result)


def test_prepare_pins_two_inputs_and_refuses_overwrite(prepared):
    root, data, inventory = prepared
    assert data["inventory_sha256"] == campaign.sha256(inventory)
    assert data["seeds"] == [101, 102] and data["decisions"] == [0, 100]
    assert set(data["checkpoints"]) == {"mppi_aux", "sac_aux"}
    with pytest.raises(FileExistsError):
        campaign.prepare(inventory, root, "a" * 40)


@pytest.mark.parametrize("change", ["checkpoint", "metadata", "step", "source_run"])
def test_checkpoint_pins_detect_mutation(prepared, change):
    _, data, _ = prepared
    row = dict(data["checkpoints"]["mppi_aux"])
    if change in ("checkpoint", "metadata"):
        Path(row[change]).write_text("changed")
    else:
        row[change] = 150000 if change == "step" else "wrong-run"
    with pytest.raises((ValueError, RuntimeError)):
        campaign.verify_checkpoint(row)


def test_metadata_semantics_are_validated_after_hash(prepared):
    _, data, _ = prepared
    row = dict(data["checkpoints"]["sac_aux"])
    metadata = campaign.read(row["metadata"])
    metadata["trial_run_params"]["alg_params"]["aux_return_mode"] = "off"
    Path(row["metadata"]).write_text(json.dumps(metadata))
    row["metadata_sha256"] = campaign.sha256(row["metadata"])
    with pytest.raises(ValueError, match="semantics"):
        campaign.verify_checkpoint(row)


def test_prepare_rejects_duplicate_backbones(prepared, tmp_path):
    _, data, _ = prepared
    path = tmp_path / "duplicate.json"
    campaign.write_new(path, dict(checkpoints=[data["checkpoints"]["mppi_aux"]] * 2))
    with pytest.raises(ValueError, match="exactly one"):
        campaign.prepare(path, tmp_path / "other", "a" * 40)


def test_stage_commands_preserve_coverage_and_pairing(prepared):
    root, data, _ = prepared
    bank = campaign.stage_spec(root, "banks", 1)
    command = campaign.probe_command(data, bank, "/provided/python")
    assert command[0] == "/provided/python"
    assert bank["purpose"] == "root_bank_smoke_not_production"
    assert [bank[k] for k in ("horizon", "tail_steps", "rollouts")] == [1, 1, 1]
    assert command[-6:] == ["--seeds", "101", "102", "--decisions", "0", "100"]
    for index, (target, source) in enumerate(campaign.PAIRS):
        spec = campaign.stage_spec(root, "production", index, "cpu", 2)
        command = campaign.probe_command(data, spec, "/provided/python")
        assert (spec["target"], spec["source"]) == (target, source)
        assert "--seeds" not in command and "--decisions" not in command
        assert command[-2:] == ["--root-bank", str(root / "banks" / source / "bundle/root-bank.json")]
        assert spec["tail_steps"] == 1000 and spec["rollouts"] == 4


def test_benchmark_protocol_and_production_resource_selection(prepared):
    root, _, _ = prepared
    for i, (_, device, threads) in enumerate(campaign.BENCHMARKS):
        spec = campaign.stage_spec(root, "benchmark", i)
        assert (spec["device"], spec["torch_threads"]) == (device, threads)
        assert spec["target"] == "mppi_aux" and spec["seeds"] == [101] and spec["decisions"] == [0]
        assert (spec["horizon"], spec["tail_steps"], spec["rollouts"], spec["gradient_rollouts"]) == (3, 100, 4, 32)
        assert spec["epsilons"] == [.03, .1] and spec["random_directions"] == 1
    with pytest.raises(ValueError, match="fixed"):
        campaign.stage_spec(root, "benchmark", 1, torch_threads=1)
    with pytest.raises(ValueError, match="selected from"):
        campaign.stage_spec(root, "production", 0)
    with pytest.raises(ValueError, match="index"):
        campaign.stage_spec(root, "banks", -1)


def test_result_checksums_coverage_and_records(prepared):
    root, data, _ = prepared
    spec = campaign.stage_spec(root, "banks", 0)
    bundle = Path(spec["output"]) / "bundle"
    make_bundle(bundle, data, spec)
    assert campaign.verify_result(bundle, data, spec)["roots"] == 4
    (bundle / "roots/0000.json").write_text("{}")
    with pytest.raises(ValueError, match="Changed root record"):
        campaign.verify_result(bundle, data, spec)


def test_source_verification_rejects_dirty_or_wrong_commit(monkeypatch):
    # Restore the real checker here; no Git repository or subprocess is altered.
    monkeypatch.setattr(campaign.subprocess, "check_output",
                        lambda command, **kwargs: "dirty" if command[-1] == "--porcelain" else "a" * 40)
    with pytest.raises(RuntimeError, match="clean source"):
        campaign.verify_source("a" * 40)
    with pytest.raises(RuntimeError, match="expected Git"):
        campaign.verify_source("b" * 40)


def test_compute_resource_gate_and_cuda_probe(monkeypatch, tmp_path):
    spec = campaign.stage_spec(tmp_path, "benchmark", 1)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="srun"):
        campaign.require_compute(spec, "/provided/python")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_STEP_ID", "0")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    with pytest.raises(RuntimeError, match="allocation"):
        campaign.require_compute(spec, "/provided/python")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    assert campaign.require_compute(spec, "/provided/python") == 2
    calls = []
    monkeypatch.setattr(campaign.subprocess, "run", lambda command, **kwargs: calls.append(command))
    campaign.require_compute(campaign.stage_spec(tmp_path, "benchmark", 2), "/provided/python")
    assert calls[0][0] == "/provided/python" and "torch.cuda.is_available()" in calls[0][2]


@pytest.mark.parametrize("success", [True, False])
def test_runner_writes_immutable_validated_or_failure_receipt(prepared, monkeypatch, success):
    root, data, _ = prepared
    spec = campaign.stage_spec(root, "banks", 0)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(campaign, "require_compute", lambda *args: 1)
    def fake_run(command, **kwargs):
        if not success:
            raise RuntimeError("synthetic child failure")
        make_bundle(Path(spec["output"]) / "bundle", data, spec)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(campaign.subprocess, "run", fake_run)
    args = (root / "campaign.json", "banks", 0, "a" * 40, "/provided/python")
    if success:
        assert campaign.run(*args)["validated"] is True
    else:
        with pytest.raises(RuntimeError, match="synthetic child"):
            campaign.run(*args)
    receipt = campaign.read(Path(spec["output"]) / "receipt.json")
    assert receipt["validated"] is success and receipt["elapsed_seconds"] >= 0
    assert receipt["child_user_seconds"] >= 0 and receipt["peak_child_rss_bytes"] >= 0
    with pytest.raises(FileExistsError):
        campaign.run(*args)
