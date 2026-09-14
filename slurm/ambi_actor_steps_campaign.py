"""Vary actor optimization in the unchanged J1/H1 closed-loop refinement learner.

The immutable manifest owns nine new cells and three verified historical
A4 cells. Workers and mergers stay offline; the complete-panel reporter alone
publishes a new comparison. Critic fitting stays fixed at 32 updates before the actor phase.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slurm.ambi_takeoff_campaign import _hash, _run, load_campaign as load_matrix, load_inventory, verify_checkpoint
from utils.ambi_research import load_preset_matrix

ROOT = Path(__file__).resolve().parents[1]
SELECTOR = "initialization/inherited"
SEEDS = list(range(101, 121))
CONTROLLERS = [55, 56, 57]
ACTOR_UPDATES = [1, 2, 4, 8]
CELL_IDS = [f"a{count}-s{seed}" for count in ACTOR_UPDATES for seed in CONTROLLERS]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _absolute(value, name):
    _require(isinstance(value, str) and Path(value).is_absolute(), f"{name} must be absolute")
    return Path(value)


def load_campaign(path):
    """Reject learner, identity, coverage and output drift before mutations."""
    data = json.loads(Path(path).read_text())
    _require(data.get("schema_version") == 1, "Unsupported campaign schema")
    _require(isinstance(data.get("attempt_label"), str) and data["attempt_label"].strip(),
             "Choose an explicit attempt label")
    root = _absolute(data.get("output_root"), "output_root")
    _absolute(data.get("inventory"), "inventory")
    _require(data.get("actor_updates") == ACTOR_UPDATES and data.get("critic_updates") == 32 and data.get("rounds") == 1,
             "Campaign must preserve the declared actor-step sweep at J1")
    _require(data.get("seeds") == SEEDS and data.get("controller_seeds") == CONTROLLERS,
             "Campaign requires environment seeds101-120 and controller seeds55-57")
    cells = data.get("cells", [])
    _require([cell.get("cell_id") for cell in cells] == CELL_IDS, "Campaign must contain the ordered 12 cells")
    destinations = set()
    for cell in cells:
        count, j, seed = cell.get("actor_updates"), cell.get("rounds"), cell.get("controller_seed")
        _require(type(count) is int and count in ACTOR_UPDATES and type(j) is int and j == 1
                 and cell.get("critic_updates") == 32 and type(seed) is int and seed in CONTROLLERS
                 and cell["cell_id"] == f"a{count}-s{seed}", "Cell coordinates differ from its identity")
        _require(cell.get("selector") == SELECTOR and cell.get("seeds") == SEEDS,
                 "Each cell requires the inherited selector and full 20-seed panel")
        width = 5
        expected_shards = [SEEDS[i:i + width] for i in range(0, 20, width)]
        _require(cell.get("seed_shards") == expected_shards, "Shard partition differs from the five-seed schedule")
        _require(cell.get("reused") is (count == 4), "Only the three completed A4 cells are reused")
        matrix_path = Path(cell["matrix"])
        if not matrix_path.is_absolute():
            matrix_path = ROOT / matrix_path
        matrix_path, steps, selector = load_matrix(matrix_path)
        config = load_preset_matrix(matrix_path)
        reference = load_preset_matrix(ROOT / "configs/research/round_scaling_j1_h1_200k.json")
        _require(steps == (200000,) and selector == SELECTOR, "Only the inherited 200k backbone is selected")
        expected = copy.deepcopy(reference["shared_alg_params"])
        expected["inner_actor_updates_per_round"] = count
        _require(config["shared_alg_params"] == expected, "Only actor update count may change the learner")
        _require(config["evaluation"] == reference["evaluation"], "Episode and model-probe protocol must remain unchanged")
        _require(config["checkpoint_contract"] == reference["checkpoint_contract"], "Checkpoint contract changed")
        _require(config["source_run"] == reference["source_run"] and config["base_alg_config"] == "checkpoint",
                 "Checkpoint source changed")
        _require(set(config["comparisons"]) == set(reference["comparisons"]), "Comparison selectors changed")
        for name, comparison in reference["comparisons"].items():
            actual = config["comparisons"][name]
            _require(actual["reference"] == comparison["reference"]
                     and set(actual["variants"]) == set(comparison["variants"]), "Comparison reference changed")
            _require(all(actual["variants"][key]["alg_params"] == variant["alg_params"]
                         for key, variant in comparison["variants"].items()), "Variant learner overrides changed")
        for key in ("bundle", "model_series"):
            target = _absolute(cell.get(key), key)
            _require(str(target) not in destinations, "Cells cannot share result output paths")
            destinations.add(str(target))
            if not cell["reused"]:
                expected_path = root / "production" / cell["cell_id"] / ("bundle" if key == "bundle" else "model-series")
                _require(target == expected_path, "New cell outputs must use their unique campaign directory")
        expected_spec = root / "specs" / cell["cell_id"] / "initialization__inherited.json"
        _require(_absolute(cell.get("identity_spec"), "identity_spec") == expected_spec,
                 "Each cell needs its own metadata-only identity spec")
        if cell["reused"]:
            _require(all(isinstance(cell.get(key), str) and re.fullmatch(r"[0-9a-f]{64}", cell[key])
                         for key in ("bundle_manifest_sha256", "bundle_seal_sha256")),
                     "Historical A4 cells require pinned manifest and seal file SHA256s")
        cell["matrix"] = str(matrix_path)
    return data


def task_cells(data):
    """Longest actor fits first; each cell has four five-seed shards."""
    return [(index, shard) for index, cell in sorted(enumerate(data["cells"]),
            key=lambda item: (-item[1]["actor_updates"], item[1]["controller_seed"]))
            if not cell["reused"] for shard in range(len(cell["seed_shards"]))]


def _selected(data, index, kind):
    values = task_cells(data) if kind == "task" else data["cells"]
    _require(type(index) is int and 0 <= index < len(values), f"Invalid {kind} index")
    return values[index]


def _common(data, cell, checkpoint):
    return ["evaluate_ambi_checkpoint.py", "--matrix", cell["matrix"], "--checkpoint", checkpoint,
            "--preset", "initialization/prior", "--preset", SELECTOR,
            "--controller-seed", cell["controller_seed"], "--checkpoint-inventory", data["inventory"]]


def metadata(campaign, execute=False):
    data = load_campaign(campaign)
    row = load_inventory(data["inventory"], (200000,))[0]
    checkpoint = verify_checkpoint(row)
    commands = []
    for cell in data["cells"]:
        directory = Path(data["output_root"]) / "specs" / cell["cell_id"]
        arguments = [*_common(data, cell, checkpoint), "--seeds", *cell["seeds"],
                     "--eval-series-spec-dir", directory, "--bundle-dir", directory / "unused-bundle"]
        commands.append(dict(cell_id=cell["cell_id"], command=[sys.executable, *map(str, arguments)],
                             identity_spec=str(directory / "initialization__inherited.json")))
        if execute:
            _run(arguments)
    print(json.dumps(commands, indent=2), flush=True)
    return commands


def _shard_path(data, cell, shard, smoke):
    return Path(data["output_root"]) / ("smoke-shards" if smoke else "shards") / cell["cell_id"] / f"shard_{shard}"


def _receipt(data, path, cell, shard, row, smoke):
    return dict(status="complete", cell_id=cell["cell_id"], actor_updates=cell["actor_updates"], critic_updates=cell["critic_updates"], rounds=cell["rounds"],
                controller_seed=cell["controller_seed"], shard_index=shard,
                seeds=cell["seed_shards"][shard][:1] if smoke else cell["seed_shards"][shard],
                smoke=smoke, attempt_label=data["attempt_label"], campaign_sha256=_hash(path),
                checkpoint_sha256=row["sha256"], matrix_sha256=_hash(cell["matrix"]), selector=SELECTOR)


def run_worker(campaign, task_index, smoke=False):
    data = load_campaign(campaign)
    index, shard = _selected(data, task_index, "task")
    cell = data["cells"][index]
    row = load_inventory(data["inventory"], (200000,))[0]
    checkpoint = verify_checkpoint(row)
    receipt = _receipt(data, campaign, cell, shard, row, smoke)
    output = _shard_path(data, cell, shard, smoke)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    _run(["-c", "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; "
          "print({'device':torch.cuda.get_device_name(0),'torch':torch.__version__})"])
    if smoke:
        _run(["-m", "pytest", "-q", "tests/test_ambi_actor_steps_campaign.py",
              "tests/test_ambi_finite_horizon.py",
              "tests/test_ambi_togo_trace.py",
              "tests/test_ambi_inner_decoupling.py::test_cuda_act_preserves_all_global_rng_streams_and_outer_state"])
    arguments = [*_common(data, cell, checkpoint), "--device", "cuda", "--bundle-dir", output / "bundle",
                 "--output", output / "results.json", "--seeds", *receipt["seeds"]]
    if smoke:
        arguments += ["--max-steps", "2"]
    _run(arguments)
    _run(["merge_ambi_seed_shards.py", "seal-episodes", "--bundle", output / "bundle"])
    receipt.update(task_index=task_index, worker_elapsed_seconds=time.perf_counter() - started)
    (output / "worker-completion.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


def _validate_identity(data, cell, bundle, expected_seeds, smoke):
    from utils.eval_series_data import normalize_bundle
    records = normalize_bundle(bundle, checkpoint_inventory=data["inventory"])
    selected = [record for record in records if record["selector"] == SELECTOR]
    _require(len(selected) == 1, "Missing inherited publication record")
    record = selected[0]
    spec = json.loads(_absolute(cell.get("identity_spec"), "identity_spec").read_text())
    expected = copy.deepcopy(spec["identity"])
    if smoke:
        expected["protocol"]["environment_seeds"] = expected_seeds
        expected["protocol"]["max_steps"] = 2
    _require(record["identity"] == expected, "Completed cell differs from the approved science/protocol identity")
    _require(record["checkpoint"]["sha256"] == load_inventory(data["inventory"], (200000,))[0]["sha256"],
             "Completed cell checkpoint differs from inventory")
    return record


def merge_cell(campaign, cell_index, smoke=False, shard_indices=None):
    data = load_campaign(campaign)
    cell = _selected(data, cell_index, "cell")
    row = load_inventory(data["inventory"], (200000,))[0]
    started = time.perf_counter()
    receipt_dir = Path(data["output_root"]) / ("smoke" if smoke else "production") / cell["cell_id"]
    _require(not (receipt_dir / "merge-completion.json").exists(), "Cell is already complete")
    if cell["reused"]:
        _require(not smoke and shard_indices is None, "Reused production cell cannot become a smoke result")
        from utils.ambi_seed_shards import _read_episode, SEAL
        bundle = Path(cell["bundle"])
        for name, key in (("manifest.json", "bundle_manifest_sha256"), (SEAL, "bundle_seal_sha256")):
            _require(_hash(bundle / name) == cell.get(key), "Reused bundle differs from pinned manifest/seal")
        _, seeds, _ = _read_episode(bundle)
        _require(seeds == cell["seeds"], "Reused result has an incomplete seed panel")
        _validate_identity(data, cell, bundle, seeds, False)
        # Exercise the current streaming reporter on real historical traces
        # before declaring that historical cell complete for comparison.
        from slurm.ambi_actor_steps_report import load_cell
        validated = load_cell(cell)
        historical_validation = {
            "episode_count": len(validated["rows"]),
            "critic_training_points": sum(len(row["critic_training_steps"]) for row in validated["rows"]),
            "actor_training_points": sum(len(row["actor_training_steps"]) for row in validated["rows"]),
            "trace_source_count": len(validated["identity"]["training_trace_sources"]),
        }
        receipt_dir.mkdir(parents=True, exist_ok=True)
        receipt = dict(status="complete", cell_id=cell["cell_id"], actor_updates=cell["actor_updates"], critic_updates=cell["critic_updates"], rounds=cell["rounds"], reused=True, bundle=str(bundle),
                       campaign_sha256=_hash(campaign), seeds=seeds, staged=False,
                       historical_validation=historical_validation, elapsed_seconds=time.perf_counter() - started)
    else:
        all_indices = list(range(len(cell["seed_shards"])))
        indices = all_indices if shard_indices is None else shard_indices
        _require(indices and len(indices) == len(set(indices))
                 and all(type(i) is int and i in all_indices for i in indices)
                 and indices == sorted(indices)
                 and (smoke or indices == all_indices), "Production must merge every seed shard exactly once")
        expected_seeds = [s for i in indices for s in (cell["seed_shards"][i][:1] if smoke else cell["seed_shards"][i])]
        sources = [_shard_path(data, cell, i, smoke) for i in indices]
        for index, source in zip(indices, sources):
            receipt = json.loads((source / "worker-completion.json").read_text())
            expected = _receipt(data, campaign, cell, index, row, smoke)
            expected["task_index"] = task_cells(data).index((cell_index, index))
            _require(all(receipt.get(key) == value for key, value in expected.items()),
                     f"Worker receipt differs from campaign: {source}")
        bundle = receipt_dir / "bundle" if smoke else Path(cell["bundle"])
        model_series = receipt_dir / "model-series" if smoke else Path(cell["model_series"])
        _run(["merge_ambi_seed_shards.py", "episodes", "--sources", *[s / "bundle" for s in sources],
              "--output", bundle, "--seeds", *expected_seeds])
        _validate_identity(data, cell, bundle, expected_seeds, smoke)
        _run(["evaluate_ambi_calibration.py", "export-model", "--bundle", bundle, "--selector", SELECTOR,
              "--attempt-label", f"{data['attempt_label']}-{cell['cell_id']}", "--output", model_series])
        receipt = dict(status="complete", cell_id=cell["cell_id"], actor_updates=cell["actor_updates"], critic_updates=cell["critic_updates"], rounds=cell["rounds"], reused=False, smoke=smoke,
                       source_shards=indices, seeds=expected_seeds, campaign_sha256=_hash(campaign),
                       elapsed_seconds=time.perf_counter() - started, staged=False)
    (receipt_dir / "merge-completion.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    worker = commands.add_parser("worker")
    worker.add_argument("--task-index", type=int, required=True)
    merge = commands.add_parser("merge")
    merge.add_argument("--cell-index", type=int, required=True)
    merge.add_argument("--shard-indices", nargs="+", type=int)
    meta = commands.add_parser("metadata")
    meta.add_argument("--execute", action="store_true")
    for command in (worker, merge, meta):
        command.add_argument("--campaign", type=Path, required=True)
    for command in (worker, merge):
        command.add_argument("--smoke", action="store_true")
    args = vars(parser.parse_args(argv))
    {"worker": run_worker, "merge": merge_cell, "metadata": metadata}[args.pop("command")](**args)


if __name__ == "__main__":
    main()
