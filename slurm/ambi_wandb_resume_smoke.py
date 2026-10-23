"""Online, CPU-only validation of the pinned SDK's bundled core and exact resume.

Creates one explicitly tagged validation run; never reads production lineages.
All local artifacts must be outside the source checkout.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import uuid


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(Path(path).parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run(args):
    output = args.output.resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("Canary output must be outside the source checkout.")
    output.mkdir(parents=True, exist_ok=False)
    # Set this before importing W&B: this is the supported spelling in 0.17.4.
    os.environ["WANDB__REQUIRE_CORE"] = "true"
    os.environ["WANDB_MODE"] = "online"
    os.environ["WANDB_INIT_TIMEOUT"] = "45"
    os.environ["WANDB_HTTP_TIMEOUT"] = "20"
    os.environ["WANDB__SERVICE_WAIT"] = "30"
    os.environ["WANDB_DISABLE_CODE"] = "true"
    for key in ("WANDB_RUN_ID", "WANDB_RESUME", "WANDB_SWEEP_ID"):
        os.environ.pop(key, None)
    for key, name in (("WANDB_DIR", "wandb"), ("WANDB_CACHE_DIR", "cache"),
                      ("WANDB_DATA_DIR", "data"), ("WANDB_ARTIFACT_DIR", "artifacts")):
        directory = output / name
        directory.mkdir()
        os.environ[key] = str(directory)

    started = time.monotonic()
    result = {
        "schema": "ambi-wandb-resume-smoke-v1", "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": uuid.uuid4().hex, "entity": args.entity, "project": args.project,
        "backend": "bundled-wandb-core", "checks": {},
    }
    write_new(output / "started.json", result)
    active = None

    def deadline(_signum, _frame):
        raise TimeoutError("W&B resume canary exceeded its total time budget.")

    previous_handler = signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.timeout_seconds)
    try:
        import wandb
        from utils.wandb_resume import WandbResumeContext, fetch_remote_history, validate_remote_prefix
        from utils.wandb_utils import init_wandb, log_wandb, finish_wandb

        if wandb.__version__ != "0.17.4":
            raise RuntimeError("This canary must use the locked wandb 0.17.4 runtime.")
        core = Path(wandb.util.get_core_path())
        if not core.is_file() or not os.access(core, os.X_OK):
            raise RuntimeError("The locked SDK's bundled core binary is unavailable.")
        result["sdk_version"] = wandb.__version__
        result["checks"]["bundled_core_available"] = True
        params = {
            "wandb": True, "wandb_mode": "online", "wandb_entity": args.entity,
            "wandb_project": args.project, "wandb_group": "ambi-sdk-validation",
            "wandb_tags": ["validation-only", "wandb-core", "exact-resume-smoke"],
            "wandb_run_name": f"SDK validation | core resume | {result['run_id'][:8]}",
        }
        config = {"purpose": "SDK connectivity and exact resume validation",
                  "scientific_training": False, "sdk_backend": "core"}
        first_dir, second_dir = output / "first", output / "resumed"
        first_dir.mkdir()
        second_dir.mkdir()
        context = WandbResumeContext.new(run_id=result["run_id"], directory=first_dir)
        active = init_wandb(params, default_project=args.project, run_name=None,
                            config=config, resume_context=context, wandb_module=wandb)
        if active.raw_run.id != result["run_id"]:
            raise RuntimeError("The new SDK run does not match the prepared run identity.")
        log_wandb(active, {"train/canary_value": 1.0}, step=0)
        log_wandb(active, {"train/canary_value": 2.0}, step=1)
        prefix = active.checkpoint_state()
        write_new(output / "prefix-checkpoint.json", prefix)
        active.publish_committed(prefix)
        finish_wandb(active)
        active = None
        rows = fetch_remote_history(wandb, entity=args.entity, project=args.project,
                                    run_id=result["run_id"])
        if validate_remote_prefix(context.buffer.events, rows) != 2:
            raise RuntimeError("First committed prefix is incomplete.")
        write_new(output / "prefix-history.json", rows)
        result["checks"]["new_run_two_committed_events"] = True

        context = WandbResumeContext.resume(
            run_id=result["run_id"],
            checkpoint_state=json.loads((output / "prefix-checkpoint.json").read_text()),
            remote_rows=rows, directory=second_dir,
        )
        active = init_wandb(params, default_project=args.project, run_name=None,
                            config=config, resume_context=context, wandb_module=wandb)
        if active.raw_run.id != result["run_id"] or not active.raw_run.resumed:
            raise RuntimeError("resume='must' did not resume the prepared physical run.")
        # A repeated environment step distinguishes the event axis from env_step.
        log_wandb(active, {"train/canary_value": 3.0}, step=1)
        complete = active.checkpoint_state()
        write_new(output / "final-checkpoint.json", complete)
        active.publish_committed(complete)
        finish_wandb(active)
        active = None
        rows = fetch_remote_history(wandb, entity=args.entity, project=args.project,
                                    run_id=result["run_id"])
        if validate_remote_prefix(context.buffer.events, rows) != 3:
            raise RuntimeError("Resumed committed history is incomplete.")
        write_new(output / "final-history.json", rows)
        result["checks"].update(resume_must_same_run=True, exact_three_event_history=True,
                                  repeated_environment_step_preserved=True)
        result["status"] = "passed"
        result["run_url"] = f"https://wandb.ai/{args.entity}/{args.project}/runs/{result['run_id']}"
    except BaseException as error:
        result["status"] = "failed"
        result["error_type"] = type(error).__name__
        # Stack locations aid diagnosis without recording credentials, headers,
        # environment values or arbitrary SDK exception payloads.
        result["traceback"] = traceback.format_list(traceback.extract_tb(error.__traceback__))
        result["cause_types"] = []
        cause = error.__cause__
        while cause is not None:
            result["cause_types"].append(type(cause).__name__)
            cause = cause.__cause__
        if active is not None:
            signal.alarm(10)
            try:
                active.abort()
            except BaseException as cleanup_error:
                result["cleanup_error_type"] = type(cleanup_error).__name__
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        result["elapsed_seconds"] = time.monotonic() - started
        write_new(output / "result.json", result)
    print(json.dumps({key: result[key] for key in ("status", "run_id", "elapsed_seconds")}), flush=True)
    return 0 if result["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default="rwgao_b-brown-university")
    parser.add_argument("--project", default="ambi")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=360)
    args = parser.parse_args()
    if not 30 <= args.timeout_seconds <= 600:
        parser.error("--timeout-seconds must be between 30 and 600")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
