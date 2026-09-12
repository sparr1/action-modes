"""Pin the round comparison, seed pairing and publication boundary."""
import copy
import inspect
import json
from pathlib import Path

import pytest

from slurm import ambi_round_scaling_campaign as launch


@pytest.fixture
def campaign(tmp_path):
    cells = []
    for j in (1, 2, 4):
        for seed in (55, 56, 57):
            cell_id = f"j{j}-c{seed}"
            cell_root = tmp_path / "production" / cell_id
            cells.append(dict(cell_id=cell_id, rounds=j, controller_seed=seed,
                matrix=f"configs/research/round_scaling_j{j}_h1_200k.json", selector=launch.SELECTOR,
                seeds=launch.SEEDS, seed_shards=[launch.SEEDS[i:i + 20 // j] for i in range(0, 20, 20 // j)],
                reused=j == 4 and seed == 55, bundle=str(cell_root / "bundle"),
                model_series=str(cell_root / "model-series"), identity_spec=str(tmp_path / f"{cell_id}-spec.json"),
                eval_run_dir=str(tmp_path / "registry" / cell_id)))
    data = dict(schema_version=1, attempt_label="test-rounds", output_root=str(tmp_path),
                inventory=str(tmp_path / "inventory.json"), cells=cells)
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(data))
    return path


def test_tasks_have_equal_nominal_work_and_preserve_every_seed(campaign):
    data = launch.load_campaign(campaign)
    tasks = launch.task_cells(data)
    assert len(tasks) == 17
    assert tasks[:4] == [(7, i) for i in range(4)]
    assert tasks[4:8] == [(8, i) for i in range(4)]
    assert tasks[8:14] == [(c, s) for c in (3, 4, 5) for s in (0, 1)]
    assert tasks[14:] == [(c, 0) for c in (0, 1, 2)]
    seen = set()
    for index, shard in tasks:
        cell = data["cells"][index]
        assert cell["rounds"] * len(cell["seed_shards"][shard]) == 20
        for seed in cell["seed_shards"][shard]:
            key = cell["cell_id"], seed
            assert key not in seen
            seen.add(key)
    assert len(seen) == 160
    assert not any(key[0] == "j4-c55" for key in seen)


@pytest.mark.parametrize("mutation", ["seed", "reused", "selector", "shards", "output", "identity"])
def test_reject_campaign_drift_before_mutation(campaign, mutation):
    data = json.loads(campaign.read_text())
    cell = data["cells"][0]
    if mutation == "seed":
        cell["controller_seed"] = 58
    elif mutation == "reused":
        cell["reused"] = True
    elif mutation == "selector":
        cell["selector"] = "initialization/prior"
    elif mutation == "shards":
        cell["seed_shards"][0].pop()
    elif mutation == "output":
        cell["bundle"] = str(campaign.parent / "another-cell/bundle")
    else:
        cell["cell_id"] = "j1-c54"
    campaign.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        launch.load_campaign(campaign)
    assert not (campaign.parent / "production").exists()


@pytest.mark.parametrize("index", [-1, 17, True, 1.5])
def test_invalid_task_index(campaign, index):
    with pytest.raises(ValueError, match="Invalid task"):
        launch._selected(launch.load_campaign(campaign), index, "task")


def _mock_runtime(monkeypatch, tmp_path):
    calls = []
    def fake_run(args, **kwargs):
        args = list(map(str, args))
        calls.append(args)
        if args[:2] == ["merge_ambi_seed_shards.py", "episodes"]:
            Path(args[args.index("--output") + 1]).mkdir(parents=True)
    monkeypatch.setattr(launch, "_run", fake_run)
    monkeypatch.setattr(launch, "load_inventory", lambda *a: [dict(step=200000, sha256="a" * 64)])
    monkeypatch.setattr(launch, "verify_checkpoint", lambda row: tmp_path / "checkpoint")
    return calls


@pytest.mark.parametrize("task,j,controller,seed", [(0, 4, 56, 101), (9, 2, 55, 111), (16, 1, 57, 101)])
def test_smoke_preserves_training_and_controller_seed(campaign, monkeypatch, task, j, controller, seed):
    calls = _mock_runtime(monkeypatch, campaign.parent)
    launch.run_worker(campaign, task, smoke=True)
    command = next(c for c in calls if c[0] == "evaluate_ambi_checkpoint.py")
    assert command[command.index("--controller-seed") + 1] == str(controller)
    assert command[command.index("--seeds") + 1] == str(seed)
    assert command[command.index("--max-steps") + 1] == "2"
    assert f"round_scaling_j{j}_h1_200k.json" in command[command.index("--matrix") + 1]
    assert "initialization/prior" in command and "initialization/inherited" in command
    assert not any(c[0] == "eval_series.py" for c in calls)
    assert calls[-1][1] == "seal-episodes"
    data = launch.load_campaign(campaign)
    index, shard = launch.task_cells(data)[task]
    receipt = json.loads((launch._shard_path(data, data["cells"][index], shard, True) / "worker-completion.json").read_text())
    assert receipt["seeds"] == [seed]
    assert receipt["controller_seed"] == controller and receipt["rounds"] == j
    assert receipt["campaign_sha256"] == launch._hash(campaign)


def test_metadata_emits_full_panel_without_evaluation(campaign, monkeypatch):
    calls = _mock_runtime(monkeypatch, campaign.parent)
    commands = launch.metadata(campaign)
    assert len(commands) == 9 and calls == []
    for item, cell in zip(commands, launch.load_campaign(campaign)["cells"]):
        command = item["command"]
        assert "--eval-series-spec-dir" in command and "--bundle-dir" in command
        assert command[command.index("--bundle-dir") + 1].endswith("/unused-bundle")
        assert command[command.index("--controller-seed") + 1] == str(cell["controller_seed"])
        assert command[command.index("--seeds") + 1:command.index("--eval-series-spec-dir")] == list(map(str, launch.SEEDS))


def test_production_merge_rejects_missing_shards_before_outputs(campaign, monkeypatch):
    calls = _mock_runtime(monkeypatch, campaign.parent)
    with pytest.raises(ValueError, match="every seed shard"):
        launch.merge_cell(campaign, 3, shard_indices=[0])
    assert calls == [] and not (campaign.parent / "production").exists()


def test_merge_rejects_receipt_from_different_campaign(campaign, monkeypatch):
    calls = _mock_runtime(monkeypatch, campaign.parent)
    launch.run_worker(campaign, 14, smoke=True)
    calls.clear()
    data = json.loads(campaign.read_text())
    data["attempt_label"] = "different-attempt"
    campaign.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="receipt differs"):
        launch.merge_cell(campaign, 0, smoke=True)
    assert calls == [] and not (campaign.parent / "smoke").exists()


def test_smoke_merge_validates_complete_selected_shards_and_never_stages(campaign, monkeypatch):
    calls = _mock_runtime(monkeypatch, campaign.parent)
    launch.run_worker(campaign, 8, smoke=True)
    launch.run_worker(campaign, 9, smoke=True)
    validations = []
    monkeypatch.setattr(launch, "_validate_identity", lambda *args: validations.append(args))
    calls.clear()
    launch.merge_cell(campaign, 3, smoke=True)
    assert len(validations) == 1 and validations[0][-2:] == ([101, 111], True)
    assert calls[0][:2] == ["merge_ambi_seed_shards.py", "episodes"]
    assert calls[0][-3:] == ["--seeds", "101", "111"]
    assert calls[1][:2] == ["evaluate_ambi_calibration.py", "export-model"]
    assert not any(c[0] in ("report_ambi_benchmark.py", "eval_series.py") for c in calls)
    receipt = json.loads((campaign.parent / "smoke/j2-c55/merge-completion.json").read_text())
    assert receipt["seeds"] == [101, 111] and receipt["staged"] is False


def test_identity_validation_requires_exact_controller_and_protocol(campaign, monkeypatch):
    from utils import eval_series_data
    _mock_runtime(monkeypatch, campaign.parent)
    data = launch.load_campaign(campaign)
    cell = data["cells"][0]
    identity = dict(protocol=dict(environment_seeds=launch.SEEDS, max_steps=500, controller_seed=55),
                    planner={"rounds": 1}, science={"sha": "fixed"})
    Path(cell["identity_spec"]).write_text(json.dumps({"identity": identity}))
    record = dict(selector=launch.SELECTOR, identity=copy.deepcopy(identity), checkpoint={"sha256": "a" * 64})
    monkeypatch.setattr(eval_series_data, "normalize_bundle", lambda *a, **k: [record])
    launch._validate_identity(data, cell, "bundle", launch.SEEDS, False)
    record["identity"]["protocol"]["controller_seed"] = 56
    with pytest.raises(ValueError, match="science/protocol"):
        launch._validate_identity(data, cell, "bundle", launch.SEEDS, False)


def test_reused_bundle_is_pinned_and_cannot_be_smoke(campaign, monkeypatch):
    calls = _mock_runtime(monkeypatch, campaign.parent)
    with pytest.raises(ValueError, match="cannot become a smoke"):
        launch.merge_cell(campaign, 6, smoke=True)
    assert calls == []
    data = json.loads(campaign.read_text())
    bundle = Path(data["cells"][6]["bundle"])
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="pinned manifest/seal"):
        launch.merge_cell(campaign, 6)


def test_launchers_keep_locked_runtime_and_worker_publication_boundary():
    body = inspect.getsource(launch.run_worker)
    assert "eval_series.py" not in body and "report_ambi_benchmark.py" not in body
    for name in ("oscar", "merge_oscar"):
        script = (launch.ROOT / f"slurm/run_ambi_round_scaling_{name}.sbatch").read_text()
        assert "git status --porcelain" in script and "EXPECTED_ACTION_MODES_SHA" in script
        assert "AMBI_ROUND_SCALING_CAMPAIGN" in script and "WANDB_MODE=offline" in script


@pytest.mark.parametrize("index", [0, 6])
def test_publication_requires_verified_merge_and_reuses_existing_run(campaign, monkeypatch, index):
    calls = _mock_runtime(monkeypatch, campaign.parent)
    cell = launch.load_campaign(campaign)["cells"][index]
    directory = campaign.parent / "production" / cell["cell_id"]
    directory.mkdir(parents=True)
    receipt = dict(status="complete", cell_id=cell["cell_id"],
                   campaign_sha256=launch._hash(campaign), seeds=cell["seeds"], staged=True)
    (directory / "merge-completion.json").write_text(json.dumps(receipt))
    launch.publish_cell(campaign, index)
    if index == 6:
        assert calls == [] and not (directory / "publication-completion.json").exists()
    else:
        assert calls[0][:3] == ["eval_series.py", "publish", cell["eval_run_dir"]]
        assert calls[1][:4] == ["evaluate_ambi_calibration.py", "publish", "--bundle", cell["model_series"]]
        assert json.loads((directory / "publication-completion.json").read_text())["status"] == "complete"
    calls.clear()
    receipt["campaign_sha256"] = "invalid"
    (directory / "merge-completion.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="verified merge"):
        launch.publish_cell(campaign, index)
    assert calls == []
