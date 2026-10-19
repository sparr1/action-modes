"""Pinned common-root gradient probes; legacy 400k defaults, no training/publication."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
SOURCES = {
    "mppi_aux": "rwgao_b-brown-university/ambi/mppi7530927x0",
    "sac_aux": "rwgao_b-brown-university/ambi/full7391933x4",
}
BENCHMARKS = (("cpu1", "cpu", 1), ("cpu2", "cpu", 2), ("cuda1", "cuda", 1))
PAIRS = (("mppi_aux", "mppi_aux"), ("sac_aux", "mppi_aux"),
         ("sac_aux", "sac_aux"), ("mppi_aux", "sac_aux"))
DEFAULT_PROTOCOL = dict(
    schema="ambi-gradient-probe-protocol-v1", checkpoint_step=400000,
    controller_seed=55, q_reduction="mean_all",
    production=dict(seeds=[101, 102], decisions=[0, 100], horizon=3, tail_steps=1000,
                    rollouts=4, gradient_rollouts=32, epsilons=[.03, .1], random_directions=1),
    benchmark=dict(seeds=[101], decisions=[0], tail_steps=100),
)


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_protocol(protocol):
    """Require a complete, explicit profile; never fill a misspelled/missing key."""
    def keys(value, expected, name):
        if not isinstance(value, dict) or set(value) != set(expected):
            raise ValueError(f"Unexpected or missing {name} protocol keys")

    def integer(value, name, lower=1, upper=None):
        if (type(value) is not int or value < lower or (upper is not None and value >= upper)):
            raise ValueError(f"Invalid protocol {name}")

    def coverage(values, name, upper):
        if not isinstance(values, list) or not values:
            raise ValueError(f"Invalid protocol {name}")
        for value in values:
            integer(value, name, 0, upper)
        if values != sorted(set(values)):
            raise ValueError(f"Protocol {name} must be unique and sorted")

    keys(protocol, DEFAULT_PROTOCOL, "top-level")
    if protocol["schema"] != DEFAULT_PROTOCOL["schema"]:
        raise ValueError("Unknown gradient probe protocol schema")
    integer(protocol["checkpoint_step"], "checkpoint_step")
    integer(protocol["controller_seed"], "controller_seed", 0, 2**32)
    if protocol["q_reduction"] not in ("mean_all", "mean_pair", "min_all", "min_pair"):
        raise ValueError("Invalid protocol q_reduction")
    for section in ("production", "benchmark"):
        values = protocol[section]
        keys(values, DEFAULT_PROTOCOL[section], section)
        coverage(values["seeds"], section + ".seeds", 2**32)
        coverage(values["decisions"], section + ".decisions", 500)
        integer(values["tail_steps"], section + ".tail_steps")
    production, benchmark = protocol["production"], protocol["benchmark"]
    for key in ("horizon", "rollouts", "gradient_rollouts"):
        integer(production[key], "production." + key)
    integer(production["random_directions"], "production.random_directions", 0)
    epsilons = production["epsilons"]
    if (not isinstance(epsilons, list) or not epsilons
            or any(type(x) not in (float, int) or not math.isfinite(x) or x <= 0 for x in epsilons)
            or len(epsilons) != len(set(epsilons))):
        raise ValueError("Invalid protocol production.epsilons")
    if any(not set(benchmark[key]).issubset(production[key]) for key in ("seeds", "decisions")):
        raise ValueError("Benchmark roots must be a subset of production coverage")
    return copy.deepcopy(protocol)


def campaign_protocol(campaign):
    """Old, already-published campaigns retain exactly the original 400k recipe."""
    if "protocol" not in campaign:
        if (campaign.get("schema") != "ambi-gradient-probe-campaign-v1"
                or "protocol_sha256" in campaign
                or campaign["seeds"] != DEFAULT_PROTOCOL["production"]["seeds"]
                or campaign["decisions"] != DEFAULT_PROTOCOL["production"]["decisions"]
                or campaign["controller_seed"] != DEFAULT_PROTOCOL["controller_seed"]):
            raise ValueError("Changed legacy 400k campaign protocol")
        return copy.deepcopy(DEFAULT_PROTOCOL)
    protocol = validate_protocol(campaign["protocol"])
    if (campaign.get("schema") != "ambi-gradient-probe-campaign-v2"
            or campaign.get("protocol_sha256") != canonical_hash(protocol)
            or campaign["seeds"] != protocol["production"]["seeds"]
            or campaign["decisions"] != protocol["production"]["decisions"]
            or campaign["controller_seed"] != protocol["controller_seed"]):
        raise ValueError("Campaign protocol hash or coverage changed")
    return protocol


def verify_source(expected):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()
    actual = git("rev-parse", "HEAD")
    if actual != expected or git("status", "--porcelain"):
        raise RuntimeError("Requires the expected Git commit and a clean source checkout")
    return actual


def verify_checkpoint(row, checkpoint_step=400000):
    if row["backbone_id"] not in SOURCES or row["source_run"] != SOURCES[row["backbone_id"]]:
        raise ValueError("Unexpected backbone/source run")
    if type(row["step"]) is not int or row["step"] != checkpoint_step:
        raise ValueError(f"This campaign selects only step {checkpoint_step} checkpoints")
    for key in ("checkpoint", "metadata"):
        if not Path(row[key]).is_absolute() or sha256(row[key]) != row[key + "_sha256"]:
            raise RuntimeError(f"Changed or nonabsolute {key}: {row[key]}")
    metadata = read(row["metadata"])
    params = metadata["trial_run_params"]["alg_params"]
    operator = "mppi" if row["backbone_id"] == "mppi_aux" else "sac"
    if (type(metadata["checkpoint"]["step"]) is not int
            or metadata["checkpoint"]["step"] != checkpoint_step or params["inner_operator"] != operator
            or params["aux_return_mode"] != "sac" or params.get("obs", "state") != "state"
            or params.get("episodic", False) or params.get("critic_value_mode", "single") != "single"):
        raise ValueError("Unexpected checkpoint metadata/auxiliary reward semantics")


def prepare(inventory_path, root, expected_sha, protocol_path=None):
    verify_source(expected_sha)
    protocol = validate_protocol(read(protocol_path)) if protocol_path is not None else copy.deepcopy(DEFAULT_PROTOCOL)
    root = Path(root).resolve()
    if root == REPO or REPO in root.parents:
        raise ValueError("Campaign outputs must be outside the source checkout")
    rows = read(inventory_path)["checkpoints"]
    if len(rows) != 2 or {r["backbone_id"] for r in rows} != set(SOURCES):
        raise ValueError("Inventory must contain exactly one selected checkpoint per backbone")
    for row in rows:
        verify_checkpoint(row, protocol["checkpoint_step"])
    campaign = dict(schema="ambi-gradient-probe-campaign-v1", source_commit=expected_sha,
                    inventory_sha256=sha256(inventory_path), checkpoints={r["backbone_id"]: r for r in rows},
                    seeds=protocol["production"]["seeds"], decisions=protocol["production"]["decisions"],
                    controller_seed=protocol["controller_seed"])
    if protocol_path is not None:
        campaign.update(schema="ambi-gradient-probe-campaign-v2", protocol=protocol,
                        protocol_sha256=canonical_hash(protocol))
    campaign["sha256"] = canonical_hash(campaign)
    write_new(root / "campaign.json", campaign)
    return campaign


def stage_spec(root, stage, index, device=None, torch_threads=None, *, protocol=None):
    protocol = validate_protocol(DEFAULT_PROTOCOL if protocol is None else protocol)
    production, benchmark = protocol["production"], protocol["benchmark"]
    horizon, tail, rollouts, gradients, epsilons, random = (
        production[key] for key in ("horizon", "tail_steps", "rollouts", "gradient_rollouts", "epsilons", "random_directions"))
    seeds, decisions = production["seeds"], production["decisions"]
    root = Path(root)
    count = {"banks": 2, "benchmark": 3, "production": 4}.get(stage, 0)
    if not 0 <= index < count:
        raise ValueError("Unknown stage or cell index")
    if stage == "benchmark":
        name, selected_device, selected_threads = BENCHMARKS[index]
        if device not in (None, selected_device) or torch_threads not in (None, selected_threads):
            raise ValueError("Benchmark device/thread settings are fixed")
        device, torch_threads = selected_device, selected_threads
        target, source = "mppi_aux", None
        tail = benchmark["tail_steps"]
        seeds, decisions = benchmark["seeds"], benchmark["decisions"]
    elif stage == "banks":
        name = target = list(SOURCES)[index]
        source = None
        device = "cpu" if device is None else device
        torch_threads = 1 if torch_threads is None else torch_threads
        horizon, tail, rollouts, gradients, epsilons, random = 1, 1, 1, 1, [.03], 0
    else:
        if device is None or torch_threads is None:
            raise ValueError("Production requires device and torch_threads selected from the benchmark")
        target, source = PAIRS[index]
        name = f"cell-{index:02d}"
    if device not in ("cpu", "cuda") or type(torch_threads) is not int or torch_threads < 1:
        raise ValueError("Select cpu/cuda and a positive torch thread count")
    return dict(stage=stage, index=index, target=target, source=source, device=device,
                checkpoint_step=protocol["checkpoint_step"], controller_seed=protocol["controller_seed"],
                q_reduction=protocol["q_reduction"],
                purpose={"banks": "root_bank_smoke_not_production", "benchmark": "timing_smoke_not_production",
                         "production": "inherited_gradient_diagnostic"}[stage],
                torch_threads=torch_threads, horizon=horizon, tail_steps=tail, rollouts=rollouts,
                gradient_rollouts=gradients, epsilons=epsilons, random_directions=random,
                seeds=seeds, decisions=decisions, output=str(root / stage / name),
                root_bank=None if source is None else str(root / "banks" / source / "bundle/root-bank.json"))


def verify_spec(campaign, spec):
    protocol = campaign_protocol(campaign)
    if "protocol" not in campaign:
        # Completed v1 receipts predate these explicit fields. Only their
        # known legacy defaults may be filled; conflicting saved values fail.
        spec = {**{key: protocol[key] for key in ("checkpoint_step", "controller_seed", "q_reduction")}, **spec}
    expected = stage_spec(Path(spec["output"]).parent.parent, spec["stage"], spec["index"],
                          spec["device"], spec["torch_threads"], protocol=protocol)
    if spec != expected:
        raise ValueError("Stage specification differs from the pinned campaign protocol")
    return protocol


def probe_command(campaign, spec, python):
    protocol = verify_spec(campaign, spec)
    row = campaign["checkpoints"][spec["target"]]
    command = [str(python), str(REPO / "probe_ambi_gradients.py"), "--checkpoint", row["checkpoint"],
               "--metadata", row["metadata"], "--output", str(Path(spec["output"]) / "bundle"),
               "--device", spec["device"], "--torch-threads", str(spec["torch_threads"]),
               "--seed", str(protocol["controller_seed"]), "--q-reduction", protocol["q_reduction"]]
    for key in ("horizon", "tail_steps", "rollouts", "gradient_rollouts", "random_directions"):
        command += ["--" + key.replace("_", "-"), str(spec[key])]
    command += ["--epsilons", *map(str, spec["epsilons"])]
    if spec["root_bank"]:
        command += ["--root-bank", spec["root_bank"]]
    else:
        command += ["--seeds", *map(str, spec["seeds"]), "--decisions", *map(str, spec["decisions"])]
    return command


def verify_result(bundle, campaign, spec):
    configured = verify_spec(campaign, spec)
    bundle = Path(bundle)
    result, bank = read(bundle / "result.json"), read(bundle / "root-bank.json")
    row = campaign["checkpoints"][spec["target"]]
    source = campaign["checkpoints"][spec["source"] or spec["target"]]
    for data, key in ((result, "sha256"), (bank, "id")):
        if data.get(key) != canonical_hash({k: v for k, v in data.items() if k != key}):
            raise ValueError("Result or root bank checksum mismatch")
    expected = {(seed, decision) for seed in spec["seeds"] for decision in spec["decisions"]}
    coverage = [(r["seed"], r["decision_index"]) for r in bank["roots"]]
    if len(coverage) != len(expected) or set(coverage) != expected:
        raise ValueError("Incomplete root coverage")
    if any(bank["protocol"].get(key) != value for key, value in dict(
            seeds=spec["seeds"], decisions=spec["decisions"], controller_seed=configured["controller_seed"],
            action_rule="tanh_mean", max_steps=max(spec["decisions"]) + 1).items()):
        raise ValueError("Changed root-bank collection protocol")
    if (result.get("kind") != "ambi_inherited_gradient_probe" or not result.get("complete")
            or not result.get("outer_state_unchanged") or not bank.get("complete")
            or result["checkpoint"] != dict(sha256=row["checkpoint_sha256"], step=configured["checkpoint_step"],
                                            metadata_sha256=row["metadata_sha256"])
            or result["root_bank_id"] != bank["id"]
            or result["root_bank_checkpoint_sha256"] != source["checkpoint_sha256"]
            or bank["checkpoint_sha256"] != source["checkpoint_sha256"]):
        raise ValueError("Incomplete or mismatched probe identity")
    if spec["root_bank"] and bank != read(spec["root_bank"]):
        raise ValueError("Production did not preserve the shared root bank")
    protocol = result["protocol"]
    for key in ("horizon", "tail_steps", "rollouts", "gradient_rollouts", "epsilons"):
        if protocol[key] != spec[key]:
            raise ValueError(f"Changed probe protocol: {key}")
    if (protocol["roots"] != len(expected) or protocol["seed"] != configured["controller_seed"]
            or protocol["q_reduction"] != configured["q_reduction"] or protocol["entropy_bonus"] is not False):
        raise ValueError("Changed root/noise/value protocol")
    candidates = 1 + 2 * len(spec["epsilons"]) * ((2 if spec["horizon"] == 1 else 3) + spec["random_directions"])
    if (protocol["candidate_actions_per_root_upper_bound"] != candidates
            or protocol["branch_simulator_decisions_upper_bound"] !=
            len(expected) * candidates * spec["rollouts"] * (spec["horizon"] + spec["tail_steps"])):
        raise ValueError("Changed direction count or simulator workload")
    if len(result["root_records"]) != len(expected):
        raise ValueError("Missing root records")
    decisions, root_ids = 0, set()
    for record in result["root_records"]:
        path = (bundle / record["path"]).resolve()
        if bundle.resolve() not in path.parents:
            raise ValueError("Root record escapes the bundle")
        data = read(path)
        if canonical_hash(data) != record["sha256"]:
            raise ValueError("Changed root record")
        root_ids.add(data["root_id"])
        decisions += data["work"]["simulator_decisions"]
    if (root_ids != {r["root_id"] for r in bank["roots"]}
            or decisions != result["branch_simulator_decisions"]):
        raise ValueError("Root records or simulator counts disagree")
    return dict(result_sha256=sha256(bundle / "result.json"), root_bank_sha256=sha256(bundle / "root-bank.json"),
                roots=len(expected), branch_simulator_decisions=decisions)


def require_compute(spec, python):
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_STEP_ID"):
        raise RuntimeError("Run through srun inside a Slurm compute allocation")
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    if cpus < spec["torch_threads"]:
        raise RuntimeError("CPU allocation is smaller than the requested thread count")
    if spec["device"] == "cuda":
        subprocess.run([str(python), "-c", "import torch; assert torch.cuda.is_available() and torch.cuda.device_count() > 0, 'CUDA unavailable'"], check=True)
    return cpus


def run(campaign_path, stage, index, expected_sha, python, device=None, torch_threads=None):
    campaign_path = Path(campaign_path).resolve()
    campaign = read(campaign_path)
    if (campaign["source_commit"] != expected_sha or campaign["sha256"] !=
            canonical_hash({k: v for k, v in campaign.items() if k != "sha256"})):
        raise ValueError("Campaign source or checksum changed")
    verify_source(expected_sha)
    protocol = campaign_protocol(campaign)
    spec = stage_spec(campaign_path.parent, stage, index, device, torch_threads, protocol=protocol)
    verify_checkpoint(campaign["checkpoints"][spec["target"]], protocol["checkpoint_step"])
    if spec["source"]:
        bank_spec = stage_spec(campaign_path.parent, "banks", list(SOURCES).index(spec["source"]), protocol=protocol)
        bank_output = Path(bank_spec["output"])
        validation = verify_result(bank_output / "bundle", campaign, bank_spec)
        bank_receipt = read(bank_output / "receipt.json")
        if not bank_receipt.get("validated") or bank_receipt.get("validation") != validation:
            raise ValueError("Shared bank lacks a valid completed receipt")
    cpus = require_compute(spec, python)
    output = Path(spec["output"])
    output.mkdir(parents=True, exist_ok=False)
    command = probe_command(campaign, spec, python)
    write_new(output / "started.json", dict(spec=spec, command=command, source_commit=expected_sha,
        job_id=os.environ["SLURM_JOB_ID"], cpus=cpus))
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.perf_counter()
    receipt = dict(stage=stage, index=index, source_commit=expected_sha, spec=spec, cpus=cpus, validated=False)
    try:
        with (output / "probe.log").open("x") as log:
            environment = {**os.environ, **{key: str(spec["torch_threads"]) for key in
                ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}
            subprocess.run(command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        verify_source(expected_sha)
        receipt["validation"] = verify_result(output / "bundle", campaign, spec)
        receipt["validated"] = True
    except BaseException as exc:
        receipt["error"] = str(exc)
        raise
    finally:
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        receipt.update(elapsed_seconds=time.perf_counter() - started,
            child_user_seconds=after.ru_utime - before.ru_utime,
            child_system_seconds=after.ru_stime - before.ru_stime,
            peak_child_rss_bytes=int(after.ru_maxrss * (1 if sys.platform == "darwin" else 1024)),
            rss_scope="maximum_child_process_in_runner", job_id=os.environ["SLURM_JOB_ID"])
        write_new(output / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--inventory", type=Path, required=True)
    prepare_parser.add_argument("--root", type=Path, required=True)
    prepare_parser.add_argument("--expected-sha", required=True)
    prepare_parser.add_argument("--protocol", type=Path, help="Complete pinned protocol JSON; omission preserves the legacy 400k pilot.")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--campaign", type=Path, required=True)
    run_parser.add_argument("--stage", choices=("banks", "benchmark", "production"), required=True)
    run_parser.add_argument("--index", type=int, required=True)
    run_parser.add_argument("--expected-sha", required=True)
    run_parser.add_argument("--python", type=Path, required=True)
    run_parser.add_argument("--device", choices=("cpu", "cuda"))
    run_parser.add_argument("--torch-threads", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.inventory, args.root, args.expected_sha, args.protocol)
    else:
        result = run(args.campaign, args.stage, args.index, args.expected_sha, args.python,
                     args.device, args.torch_threads)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
