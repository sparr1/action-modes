"""Launcher provenance/resource contracts without checkpoint computation."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from slurm import ambi_gradient_probe_campaign as campaign

EARLY_PROFILE = campaign.REPO / "configs/research/ambi_gradient_probe_early150k.json"


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


@pytest.fixture
def early_prepared(prepared, tmp_path):
    _, legacy, _ = prepared
    rows = list(copy.deepcopy(legacy["checkpoints"]).values())
    for row in rows:
        row["step"] = 150000
        metadata = campaign.read(row["metadata"])
        metadata["checkpoint"]["step"] = 150000
        Path(row["metadata"]).write_text(json.dumps(metadata))
        row["metadata_sha256"] = campaign.sha256(row["metadata"])
    inventory = tmp_path / "early-inventory.json"
    campaign.write_new(inventory, dict(checkpoints=rows))
    root = tmp_path / "early-outputs"
    return root, campaign.prepare(inventory, root, "a" * 40, EARLY_PROFILE), inventory


def make_bundle(bundle, data, spec):
    """Minimal valid probe output, using the actual canonical checksum schema."""
    target = data["checkpoints"][spec["target"]]
    source = data["checkpoints"][spec["source"] or spec["target"]]
    roots = [dict(seed=seed, decision_index=decision, root_id=f"seed-{seed}-decision-{decision}")
             for seed in spec["seeds"] for decision in spec["decisions"]]
    bank = dict(complete=True, checkpoint_sha256=source["checkpoint_sha256"], roots=roots,
                protocol=dict(seeds=spec["seeds"], decisions=spec["decisions"],
                              controller_seed=spec["controller_seed"], action_rule="tanh_mean",
                              max_steps=max(spec["decisions"]) + 1))
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
    candidates = 1 + 2 * len(spec["epsilons"]) * ((2 if spec["horizon"] == 1 else 3) + spec["random_directions"])
    protocol.update(roots=len(roots), seed=spec["controller_seed"], q_reduction=spec["q_reduction"], entropy_bonus=False,
                    candidate_actions_per_root_upper_bound=candidates,
                    branch_simulator_decisions_upper_bound=len(roots) * candidates * spec["rollouts"] * (spec["horizon"] + spec["tail_steps"]))
    result = dict(kind="ambi_inherited_gradient_probe", complete=True, outer_state_unchanged=True,
                  checkpoint=dict(sha256=target["checkpoint_sha256"], step=target["step"],
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
    assert data["schema"] == "ambi-gradient-probe-campaign-v1" and "protocol" not in data
    assert campaign.campaign_protocol(data) == campaign.DEFAULT_PROTOCOL
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


def test_completed_legacy_receipt_spec_remains_readable(prepared):
    root, data, _ = prepared
    spec = campaign.stage_spec(root, "banks", 0)
    bundle = Path(spec["output"]) / "bundle"
    make_bundle(bundle, data, spec)
    for key in ("checkpoint_step", "controller_seed", "q_reduction"):
        spec.pop(key)
    assert campaign.verify_result(bundle, data, spec)["roots"] == 4
    command = campaign.probe_command(data, spec, "/provided/python")
    assert command[command.index("--seed") + 1] == "55"
    spec["checkpoint_step"] = 150000
    with pytest.raises(ValueError, match="pinned campaign"):
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


def test_early_profile_is_pinned_and_uses_complete_coverage(early_prepared):
    root, data, _ = early_prepared
    protocol = campaign.campaign_protocol(data)
    assert data["schema"] == "ambi-gradient-probe-campaign-v2"
    assert data["protocol_sha256"] == campaign.canonical_hash(campaign.read(EARLY_PROFILE))
    assert protocol["checkpoint_step"] == 150000 and protocol["controller_seed"] == 55
    assert protocol["q_reduction"] == "mean_all"
    for i in range(2):
        spec = campaign.stage_spec(root, "banks", i, protocol=protocol)
        assert spec["seeds"] == [101, 102, 103, 104] and spec["decisions"] == [0, 10, 25, 50]
        assert [spec[k] for k in ("horizon", "tail_steps", "rollouts", "gradient_rollouts")] == [1, 1, 1, 1]
        command = campaign.probe_command(data, spec, "/provided/python")
        assert command[-10:] == ["--seeds", "101", "102", "103", "104", "--decisions", "0", "10", "25", "50"]
    for i, (target, source) in enumerate(campaign.PAIRS):
        spec = campaign.stage_spec(root, "production", i, "cuda", 1, protocol=protocol)
        assert (spec["target"], spec["source"]) == (target, source)
        assert [spec[k] for k in ("checkpoint_step", "horizon", "tail_steps", "rollouts", "gradient_rollouts")] == [150000, 3, 1500, 8, 32]
        assert spec["epsilons"] == [.03, .1] and spec["random_directions"] == 1
        command = campaign.probe_command(data, spec, "/provided/python")
        assert "--seeds" not in command and "--decisions" not in command
        for option, expected in (("--tail-steps", "1500"), ("--rollouts", "8"), ("--gradient-rollouts", "32"),
                                 ("--q-reduction", "mean_all"), ("--seed", "55")):
            assert command[command.index(option) + 1] == expected
        assert 16 * 17 * spec["rollouts"] * (spec["horizon"] + spec["tail_steps"]) == 3270528
    # Mutating a returned value cannot silently change the stored campaign.
    protocol["production"]["seeds"].append(999)
    assert data["protocol"]["production"]["seeds"] == [101, 102, 103, 104]


def test_early_benchmark_matches_production_dose_at_two_contacts(early_prepared):
    root, data, _ = early_prepared
    protocol = campaign.campaign_protocol(data)
    for i, (_, device, threads) in enumerate(campaign.BENCHMARKS):
        spec = campaign.stage_spec(root, "benchmark", i, protocol=protocol)
        assert (spec["device"], spec["torch_threads"]) == (device, threads)
        assert spec["target"] == "mppi_aux" and spec["seeds"] == [101] and spec["decisions"] == [0, 50]
        assert (spec["horizon"], spec["tail_steps"], spec["rollouts"], spec["gradient_rollouts"]) == (3, 100, 8, 32)
        command = campaign.probe_command(data, spec, "/provided/python")
        assert command[-5:] == ["--seeds", "101", "--decisions", "0", "50"]


@pytest.mark.parametrize("path,value", [
    (("checkpoint_step",), 150000.0), (("controller_seed",), -1),
    (("production", "rollouts"), True), (("production", "gradient_rollouts"), 0),
    (("production", "tail_steps"), 0), (("production", "horizon"), 0),
    (("production", "random_directions"), -1), (("production", "epsilons"), [.03, float("nan")]),
    (("production", "seeds"), [101, 101]), (("production", "decisions"), [0, 500]),
    (("benchmark", "decisions"), [100]), (("q_reduction",), "automatic"),
])
def test_invalid_profiles_are_rejected(path, value):
    profile = campaign.read(EARLY_PROFILE)
    target = profile if len(path) == 1 else profile[path[0]]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        campaign.validate_protocol(profile)


def test_protocol_keys_are_exact_and_duplicate_json_is_rejected(tmp_path):
    profile = campaign.read(EARLY_PROFILE)
    profile["production"]["rollout"] = profile["production"].pop("rollouts")
    with pytest.raises(ValueError, match="protocol keys"):
        campaign.validate_protocol(profile)
    path = tmp_path / "duplicate.json"
    path.write_text('{"checkpoint_step":150000,"checkpoint_step":400000}')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        campaign.read(path)


def test_prepare_and_metadata_must_match_selected_step(prepared, early_prepared, tmp_path):
    _, _, legacy_inventory = prepared
    with pytest.raises(ValueError, match="150000"):
        campaign.prepare(legacy_inventory, tmp_path / "wrong-step", "a" * 40, EARLY_PROFILE)
    _, data, _ = early_prepared
    row = dict(data["checkpoints"]["sac_aux"])
    campaign.verify_checkpoint(row, 150000)
    with pytest.raises(ValueError, match="400000"):
        campaign.verify_checkpoint(row)
    metadata = campaign.read(row["metadata"])
    metadata["checkpoint"]["step"] = 400000
    Path(row["metadata"]).write_text(json.dumps(metadata))
    row["metadata_sha256"] = campaign.sha256(row["metadata"])
    with pytest.raises(ValueError, match="metadata"):
        campaign.verify_checkpoint(row, 150000)


def test_protocol_tamper_and_legacy_override_are_rejected(prepared, early_prepared):
    _, early, _ = early_prepared
    edited = copy.deepcopy(early)
    edited["protocol"]["production"]["rollouts"] = 4
    # Even an updated outer checksum does not validate an unpinned profile change.
    edited["sha256"] = campaign.canonical_hash({k: v for k, v in edited.items() if k != "sha256"})
    with pytest.raises(ValueError, match="protocol hash"):
        campaign.campaign_protocol(edited)
    edited = copy.deepcopy(early)
    edited["decisions"] = [0, 100]
    with pytest.raises(ValueError, match="coverage"):
        campaign.campaign_protocol(edited)
    _, legacy, _ = prepared
    edited = copy.deepcopy(legacy)
    edited["seeds"] = [101]
    with pytest.raises(ValueError, match="legacy"):
        campaign.campaign_protocol(edited)


def test_profile_mismatch_cannot_change_command(early_prepared):
    root, data, _ = early_prepared
    legacy_spec = campaign.stage_spec(root, "production", 0, "cpu", 1)
    with pytest.raises(ValueError, match="pinned campaign"):
        campaign.probe_command(data, legacy_spec, "/provided/python")
    spec = campaign.stage_spec(root, "production", 0, "cpu", 1, protocol=campaign.campaign_protocol(data))
    spec["random_directions"] = 0
    with pytest.raises(ValueError, match="pinned campaign"):
        campaign.probe_command(data, spec, "/provided/python")


@pytest.mark.parametrize("change", ["step", "coverage", "direction_count", "bank_protocol"])
def test_early_result_rejects_rehashed_wrong_science(early_prepared, change):
    root, data, _ = early_prepared
    spec = campaign.stage_spec(root, "banks", 0, protocol=campaign.campaign_protocol(data))
    bundle = Path(spec["output"]) / "bundle"
    make_bundle(bundle, data, spec)
    assert campaign.verify_result(bundle, data, spec)["roots"] == 16
    result = campaign.read(bundle / "result.json")
    if change == "step":
        result["checkpoint"]["step"] = 400000
    elif change == "direction_count":
        result["protocol"]["candidate_actions_per_root_upper_bound"] += 2
    else:
        bank = campaign.read(bundle / "root-bank.json")
        if change == "coverage":
            bank["roots"].pop()
        else:
            bank["protocol"]["controller_seed"] = 7
        bank["id"] = campaign.canonical_hash({k: v for k, v in bank.items() if k != "id"})
        (bundle / "root-bank.json").write_text(json.dumps(bank))
        result["root_bank_id"] = bank["id"]
    result["sha256"] = campaign.canonical_hash({k: v for k, v in result.items() if k != "sha256"})
    (bundle / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="identity|coverage|direction count|collection protocol"):
        campaign.verify_result(bundle, data, spec)


def test_early_runner_derives_bank_and_production_protocol(early_prepared, monkeypatch):
    root, data, _ = early_prepared
    profile = campaign.campaign_protocol(data)
    bank_spec = campaign.stage_spec(root, "banks", 0, protocol=profile)
    make_bundle(Path(bank_spec["output"]) / "bundle", data, bank_spec)
    campaign.write_new(Path(bank_spec["output"]) / "receipt.json", dict(
        validated=True, validation=campaign.verify_result(Path(bank_spec["output"]) / "bundle", data, bank_spec)))
    production = campaign.stage_spec(root, "production", 1, "cpu", 1, protocol=profile)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(campaign, "require_compute", lambda *args: 1)
    def fake_run(command, **kwargs):
        assert command[command.index("--rollouts") + 1] == "8"
        assert command[command.index("--root-bank") + 1] == str(Path(bank_spec["output"]) / "bundle/root-bank.json")
        make_bundle(Path(production["output"]) / "bundle", data, production)
    monkeypatch.setattr(campaign.subprocess, "run", fake_run)
    receipt = campaign.run(root / "campaign.json", "production", 1, "a" * 40, "/provided/python", "cpu", 1)
    assert receipt["validated"] and receipt["validation"]["roots"] == 16
    assert receipt["spec"]["checkpoint_step"] == 150000
