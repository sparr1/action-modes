"""Run old/old/new default H2 smokes on one GPU, without publishing or changing a gate."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

OLD_SOURCE = "10852a8cb6501f53b800cf50a774e732fac1be90"
CHECKPOINT_SHA = "00d9347f00eb7719f55d4e3ef7abff8c24d7f2e214c923cdeba7b8e34afa1368"
LOCK_SHA = "f123ba99aadde092401c0e912dbeb88994f00ae420680c69c18003965485efe6"
SELECTOR = "controller/return_return_running_h2_j2_g6_actor_high"
SETTINGS = {"inner_operator": "xqc", "inner_rounds": 2, "inner_rollout_horizon": 2,
    "inner_rollouts_per_round": 256, "inner_updates_per_round": 6, "inner_batch_size": 256,
    "inner_replay_capacity": 1024, "inner_replay_sampling": "with_replacement", "inner_update_timing": "round",
    "inner_policy_delay": 3, "inner_actor_lr": 5e-5, "inner_critic_lr": 5e-5, "inner_temperature_lr": 5e-5,
    "inner_terminal_bootstrap": "outer", "inner_reward_normalization": "frozen_real_scale",
    "inner_diagnostics_every": 1, "inner_actor_bn_mode": "running", "inner_critic_bn_mode": "running",
    "inner_critic_source": "aux_return", "inner_horizon_critic_source": "aux_return", "inner_critic_target": "reward_only"}


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def bind(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def check_binding(value):
    path = Path(value["path"])
    if not path.is_absolute() or bind(path) != value:
        raise ValueError("Changed diagnostic input binding")
    return path


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def checkout(root, sha):
    root = Path(root).resolve(strict=True)
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    if (not re.fullmatch(r"[0-9a-f]{40}", sha) or git("rev-parse", "HEAD") != sha
            or git("status", "--porcelain=v1", "--untracked-files=all")
            or git("remote", "get-url", "origin") != "git@github.com:sparr1/action-modes.git"
            or bind(root / "environments/dmcontrol/uv.lock")["sha256"] != LOCK_SHA):
        raise ValueError("Require pinned clean GitHub checkout and locked runtime")
    return {"root": str(root), "commit": sha, "tree": git("rev-parse", "HEAD^{tree}")}


def inputs(plan_path):
    plan = read(plan_path)
    if (plan.get("schema") != "ambixqc-target-bn-study-v1" or plan.get("stage") != "stage11"
            or plan.get("plan_sha256") != digest({k: v for k, v in plan.items() if k != "plan_sha256"})
            or plan.get("checkpoint_sha256") != CHECKPOINT_SHA or plan.get("source_sha") == OLD_SOURCE
            or plan.get("source_sha") != plan.get("execution", {}).get("commit")
            or plan.get("smoke_seeds") != [101, 102] or plan.get("smoke_max_steps") != 3
            or plan.get("controller_seed") != 12345 or len(plan.get("reused", [])) != 1):
        raise ValueError("Require the signed Stage11 plan and exact short comparison protocol")
    baseline = plan["reused"][0]["condition"]
    if baseline.get("selector") != SELECTOR or baseline.get("settings") != SETTINGS:
        raise ValueError("Diagnostic must preserve the exact H2/J2/G6 baseline")
    previous = read(check_binding(plan["parent"]["stage10"]["plan"]))
    if previous.get("source_sha") != OLD_SOURCE or previous.get("execution", {}).get("commit") != OLD_SOURCE:
        raise ValueError("Historical execution source is not original10852a8")
    check_binding(plan["inputs"]["manifest"])
    return plan, previous["execution"]


CHILD = r'''
import importlib,json,os,platform,subprocess,sys
from pathlib import Path
request=json.loads(Path(sys.argv[1]).read_text())
root=Path(request['execution']['root']); os.chdir(root);sys.path.insert(0,str(root))
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==request['execution']['commit']
assert not subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=all'],text=True).strip()
import torch
assert sys.version_info[:2]==(3,10) and torch.__version__.split('+')[0]=='2.3.1'
assert torch.cuda.is_available() and torch.cuda.device_count()==1
from evaluate_ambi_checkpoint import evaluate_matrix
from run_ambixqc_inner_475k_screen import select_checkpoint
for name in ('evaluate_ambi_checkpoint','RL.AMBIXQC','utils.ambi_benchmark'):
 module=importlib.import_module(name)
 assert Path(module.__file__).resolve().is_relative_to(root)
out=Path(request['output'])
runtime={'python':sys.version,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),
 'hostname':platform.node(),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
 'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
 'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark,
 'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
 'torch_threads':torch.get_num_threads(),'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG')}
(out/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\n')
row=select_checkpoint(request['manifest'],checkpoint_root=request['checkpoint_root'])
payload=evaluate_matrix(request['matrix'],row['path'],selectors=[request['selector']],seeds=[101,102],
 controller_seed=12345,max_steps=3,device='cuda',bundle_dir=out/'bundle',
 checkpoint_inventory=request['manifest'],source_run=row['source_run'],stage_results=False)
cfg=payload['results'][0]['resolved_config']
runtime.update(compile=cfg.get('compile'),compile_strict=cfg.get('compile_strict'),
 deterministic_algorithms_after=torch.are_deterministic_algorithms_enabled())
(out/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\n')
(out/'results.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==request['execution']['commit']
assert not subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=all'],text=True).strip()
'''


def events(directory, sha):
    directory = Path(directory)
    manifest = read(directory / "bundle/manifest.json")
    if (manifest.get("status") != "complete" or manifest.get("code", {}).get("commit") != sha
            or manifest.get("code", {}).get("dirty") is not False
            or manifest.get("checkpoint", {}).get("sha256") != CHECKPOINT_SHA
            or len(manifest.get("runs", [])) != 1):
        raise ValueError("Diagnostic evaluated an unexpected source or checkpoint")
    run = manifest["runs"][0]
    result = run["result"]
    if (run.get("selector") != SELECTOR or result.get("outer_state_unchanged") is not True
            or run.get("actual_optimizer_steps") != {"critic": 72, "actor": 24, "temperature": 24}
            or [ep["seed"] for ep in run["episodes"]] != [101, 102]
            or any(ep["length"] != 3 for ep in run["episodes"])
            or any(result["resolved_config"].get(k) != v for k, v in SETTINGS.items())
            or result["resolved_config"].get("inner_critic_target_bn_mode", "batch_no_update") != "batch_no_update"):
        raise ValueError("Diagnostic changed default recipe, seeds, counts or frozen state")
    traces = {}
    for relative in run["trace_files"]:
        path = (directory / "bundle" / relative).resolve()
        if not path.is_relative_to((directory / "bundle").resolve()):
            raise ValueError("Trace path escapes bundle")
        with gzip.open(path, "rt") as stream:
            for line in stream:
                event = json.loads(line)
                key = (event["episode_id"], event["decision_index"])
                if key in traces or event["nonfinite"] or event["metrics"].get("decision/inner_compile_fallback") != 0:
                    raise ValueError("Invalid diagnostic trace")
                traces[key] = event
    if set(traces) != {(f"seed-{seed}", step) for seed in (101, 102) for step in range(3)}:
        raise ValueError("Diagnostic decisions are missing or misaligned")
    return run, traces


def exact_metric(key):
    tokens = set(key.split("/")[-1].split("_"))
    return bool(tokens & {"count", "steps", "updates", "rows", "slots", "size", "capacity",
                         "rollouts", "evaluations", "fallback", "compiled", "flag"}) or key.startswith("decision/inner_reward_")


def compare(left, right):
    """Keep the failed gate's tolerances; report every residual without accepting drift."""
    left_run, left_events = left
    right_run, right_events = right
    differences = []
    def add(a, b, key, decision=None):
        exact = exact_metric(key)
        finite = isinstance(a, (float, int)) and isinstance(b, (float, int)) and math.isfinite(a) and math.isfinite(b)
        passed = finite and (a == b if exact else math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-6))
        differences.append({"key": key, "decision": decision, "left": a, "right": b,
                            "absolute_difference": abs(a-b) if finite else None, "exact": exact, "passed": passed})
    if set(left_events) != set(right_events):
        raise ValueError("Comparison decisions differ")
    for old, new in zip(left_run["episodes"], right_run["episodes"]):
        if old["seed"] != new["seed"]:
            raise ValueError("Comparison episode seeds differ")
        add(old["return"], new["return"], "episode_return:" + str(old["seed"]))
    for identity in sorted(left_events):
        old, new = left_events[identity], right_events[identity]
        for key in ("phase", "event_index", "round_index", "critic_updates", "actor_updates", "temperature_updates"):
            if old[key] != new[key]:
                raise ValueError("Comparison discrete trace fields differ")
        def metrics(event):
            return {k: v for k, v in event["metrics"].items()
                    if "_seconds" not in k and k != "decision/inner_critic_target_bn_running"}
        old_values, new_values = metrics(old), metrics(new)
        if old_values.keys() != new_values.keys():
            raise ValueError("Comparison scientific metric schemas differ")
        for key in sorted(old_values):
            add(old_values[key], new_values[key], key, list(identity))
    return {"rtol": 1e-5, "atol": 1e-6, "passed": all(row["passed"] for row in differences),
            "first_decisions_passed": all(row["passed"] for row in differences if row["decision"] and row["decision"][1] == 0),
            "failures": [row for row in differences if not row["passed"]], "comparisons": differences}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID") or sys.version_info[:2] != (3, 10):
        raise ValueError("Run this diagnostic in a GPU scheduler allocation with locked Python3.10")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        raise ValueError("Preserve the campaign's preconfigured CUDA workspace")
    plan, old = inputs(args.plan)
    new = plan["execution"]
    old, new = checkout(old["root"], OLD_SOURCE), checkout(new["root"], plan["source_sha"])
    tooling = Path(__file__).resolve().parent
    tooling_sha = subprocess.check_output(["git", "-C", str(tooling), "rev-parse", "HEAD"], text=True).strip()
    tooling = checkout(tooling, tooling_sha)
    if bind(Path(sys.prefix).parent / "uv.lock")["sha256"] != LOCK_SHA:
        raise ValueError("Diagnostic interpreter lock differs")
    output = args.output_root.resolve()
    if any(output.is_relative_to(Path(item["root"])) for item in (old, new, tooling)):
        raise ValueError("Diagnostic outputs must be outside source checkouts")
    output.mkdir(parents=True, exist_ok=False)
    matrix = {"schema_version": 1, "base_alg_config": "checkpoint", "shared_alg_params": {},
        "evaluation": {"controller_seed": 12345, "seeds": [101, 102], "max_steps": 3, "default_presets": []},
        "comparisons": {"controller": {"reference": "prior", "variants": {
            SELECTOR.split("/")[1]: {"description": "Default compatibility diagnostic only", "alg_params": SETTINGS},
            "prior": {"description": "Not executed", "alg_params": {"inner_operator": "none"}}}}}}
    write(output / "matrix.json", matrix)
    write(output / "provenance.json", {"plan": bind(args.plan), "old_execution": old,
        "new_execution": new, "tooling": tooling, "job": os.environ["SLURM_JOB_ID"], "publication": False})
    measured, runtimes = {}, {}
    try:
        for name, execution in (("old1", old), ("old2", old), ("new", new)):
            phase = output / name; phase.mkdir()
            request = {"execution": execution, "matrix": str(output / "matrix.json"),
                "manifest": str(check_binding(plan["inputs"]["manifest"])), "checkpoint_root": plan["checkpoint_root"],
                "selector": SELECTOR, "output": str(phase)}
            write(phase / "request.json", request)
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", WANDB_MODE="disabled", WANDB_DISABLED="true")
            env.pop("PYTHONPATH", None)
            # Fresh caches/processes avoid inherited compiler/model state. All
            # runs share this exact allocation, GPU visibility and thread budget.
            for key in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH"):
                cache = phase / key.lower(); cache.mkdir(); env[key] = str(cache)
            with (phase / "stdout.log").open("x") as stdout, (phase / "stderr.log").open("x") as stderr:
                subprocess.run([sys.executable, "-c", CHILD, str(phase / "request.json")],
                    cwd=execution["root"], env=env, stdout=stdout, stderr=stderr, check=True)
            runtimes[name] = read(phase / "runtime.json")
            measured[name] = events(phase, execution["commit"])
        if not (runtimes["old1"] == runtimes["old2"] == runtimes["new"]):
            raise ValueError("Fresh processes did not share the same GPU/runtime/numerical settings")
        report = {"schema": "ambixqc-default-parity-diagnostic-v1", "runtime": runtimes,
            "old_old": compare(measured["old1"], measured["old2"]),
            "old1_new": compare(measured["old1"], measured["new"]),
            "old2_new": compare(measured["old2"], measured["new"]),
            "publication": False, "production_gate_changed": False}
        write(output / "report.json", report)
        print(json.dumps({"output": str(output), **{key: {"passed": report[key]["passed"],
            "first_decisions_passed": report[key]["first_decisions_passed"], "failures": len(report[key]["failures"])}
            for key in ("old_old", "old1_new", "old2_new")}}, indent=2))
    except Exception as error:
        write(output / "FAILED.json", {"type": type(error).__name__, "message": str(error)})
        raise


if __name__ == "__main__":
    main()
