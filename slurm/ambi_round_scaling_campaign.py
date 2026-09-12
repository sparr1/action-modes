"""Run a paired J1/J2/J4 episode panel with independent controller seeds.

The immutable campaign manifest assigns every shard and publication identity.
Workers never publish partial seed panels. The previously completed J4/c55 cell
is validated and referenced in place, without repeating its evaluation.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slurm.ambi_takeoff_campaign import _hash, _run, load_campaign as load_matrix, load_inventory, verify_checkpoint
from utils.ambi_research import load_preset_matrix

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_MATRIX = ROOT / "configs/research/ambi_prior_mean_prefix_h1_20seeds.json"
SELECTOR = "initialization/inherited"
SEEDS = list(range(101, 121))
CELL_IDS = [f"j{j}-c{seed}" for j in (1, 2, 4) for seed in (55, 56, 57)]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _absolute(value, name):
    _require(isinstance(value, str) and Path(value).is_absolute(), f"{name} must be absolute")
    return Path(value)


def load_campaign(path):
    """Validate the one-axis protocol before any worker or merger mutation."""
    path = Path(path)
    data = json.loads(path.read_text())
    _require(data.get("schema_version") == 1, "Unsupported campaign schema")
    _require(isinstance(data.get("attempt_label"), str) and data["attempt_label"].strip(),
             "Choose an explicit attempt label")
    root = _absolute(data.get("output_root"), "output_root")
    _absolute(data.get("inventory"), "inventory")
    cells = data.get("cells", [])
    _require([cell.get("cell_id") for cell in cells] == CELL_IDS, "Campaign must contain the ordered nine cells")
    reference = load_preset_matrix(REFERENCE_MATRIX)
    for cell in cells:
        j, seed = cell.get("rounds"), cell.get("controller_seed")
        _require(type(j) is int and j in (1, 2, 4) and type(seed) is int and seed in (55, 56, 57)
                 and cell["cell_id"] == f"j{j}-c{seed}", "Cell coordinates differ from its identity")
        _require(cell.get("selector") == SELECTOR and cell.get("seeds") == SEEDS,
                 "Each cell requires the inherited selector and full 20-seed panel")
        expected_shards = [SEEDS[i:i + 20 // j] for i in range(0, 20, 20 // j)]
        _require(cell.get("seed_shards") == expected_shards, "Shard partition differs from the weighted schedule")
        _require(cell.get("reused") is (j == 4 and seed == 55), "Only the completed J4/c55 cell is reused")
        matrix_path = Path(cell["matrix"])
        if not matrix_path.is_absolute():
            matrix_path = ROOT / matrix_path
        matrix_path, steps, selector = load_matrix(matrix_path)
        config = load_preset_matrix(matrix_path)
        _require(steps == (200000,) and selector == SELECTOR, "Only the inherited 200k backbone is selected")
        expected = copy.deepcopy(reference["shared_alg_params"])
        expected["inner_rounds"] = j
        _require(config["shared_alg_params"] == expected, "Only inner_rounds may change the learner")
        _require(config["evaluation"] == reference["evaluation"], "Episode and probe protocol must remain unchanged")
        contract = copy.deepcopy(reference["checkpoint_contract"])
        contract["checkpoints"] = [row for row in contract["checkpoints"] if row["step"] == 200000]
        _require(config["checkpoint_contract"] == contract, "Checkpoint contract changed")
        _require(set(config["comparisons"]) == set(reference["comparisons"]), "Comparison selectors changed")
        for name, comparison in reference["comparisons"].items():
            actual = config["comparisons"][name]
            _require(actual["reference"] == comparison["reference"]
                     and set(actual["variants"]) == set(comparison["variants"]), "Comparison reference changed")
            _require(all(actual["variants"][key]["alg_params"] == variant["alg_params"]
                         for key, variant in comparison["variants"].items()), "Variant learner overrides changed")
        for key in ("bundle", "model_series"):
            target = _absolute(cell.get(key), key)
            if not cell["reused"]:
                expected_path = root / "production" / cell["cell_id"] / ("bundle" if key == "bundle" else "model-series")
                _require(target == expected_path, "New cell outputs must use their unique campaign directory")
        cell["matrix"] = str(matrix_path)
    return data


def task_cells(data):
    """Longest cells first; equal nominal J × seeds work per GPU task."""
    return [(index, shard) for index, cell in sorted(enumerate(data["cells"]),
            key=lambda item: (-item[1]["rounds"], item[1]["controller_seed"]))
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
    return dict(status="complete", cell_id=cell["cell_id"], rounds=cell["rounds"],
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
        _run(["-m", "pytest", "-q", "tests/test_ambi_round_scaling_campaign.py",
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
        _run(["eval_series.py", "append", _absolute(cell.get("eval_run_dir"), "eval_run_dir"),
              "--spec", cell["identity_spec"]])
        receipt_dir.mkdir(parents=True, exist_ok=True)
        receipt = dict(status="complete", cell_id=cell["cell_id"], reused=True, bundle=str(bundle),
                       campaign_sha256=_hash(campaign), seeds=seeds, staged=False,
                       elapsed_seconds=time.perf_counter() - started)
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
        if not smoke:
            _run(["eval_series.py", "append", _absolute(cell.get("eval_run_dir"), "eval_run_dir"),
                  bundle, "--selector", SELECTOR, "--checkpoint-inventory", data["inventory"]])
        receipt = dict(status="complete", cell_id=cell["cell_id"], reused=False, smoke=smoke,
                       source_shards=indices, seeds=expected_seeds, campaign_sha256=_hash(campaign),
                       elapsed_seconds=time.perf_counter() - started, staged=not smoke)
    (receipt_dir / "merge-completion.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


def publish_cell(campaign, cell_index):
    """Publish only a fully validated cell through its existing owners."""
    data = load_campaign(campaign)
    cell = _selected(data, cell_index, "cell")
    directory = Path(data["output_root"]) / "production" / cell["cell_id"]
    receipt = json.loads((directory / "merge-completion.json").read_text())
    _require(receipt.get("status") == "complete" and receipt.get("cell_id") == cell["cell_id"]
             and receipt.get("campaign_sha256") == _hash(campaign)
             and receipt.get("seeds") == cell["seeds"], "Cell has no complete verified merge")
    if cell["reused"]:
        print(json.dumps(dict(status="reused", cell_id=cell["cell_id"])), flush=True)
        return
    _require(receipt.get("staged") is True, "Cell was not staged for native publication")
    _run(["eval_series.py", "publish", cell["eval_run_dir"], "--owner", "oscar-rgao48"])
    _run(["evaluate_ambi_calibration.py", "publish", "--bundle", cell["model_series"],
          "--mode", "online", "--entity", data.get("entity", "rwgao_b-brown-university"),
          "--project", data.get("project", "ambi-inner-bench")])
    record = dict(status="complete", cell_id=cell["cell_id"], campaign_sha256=_hash(campaign))
    (directory / "publication-completion.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


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
    publish = commands.add_parser("publish")
    publish.add_argument("--cell-index", type=int, required=True)
    for command in (worker, merge, meta, publish):
        command.add_argument("--campaign", type=Path, required=True)
    for command in (worker, merge):
        command.add_argument("--smoke", action="store_true")
    args = vars(parser.parse_args(argv))
    {"worker": run_worker, "merge": merge_cell, "metadata": metadata,
     "publish": publish_cell}[args.pop("command")](**args)


if __name__ == "__main__":
    main()
