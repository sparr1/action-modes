"""Capture warm-history actors or evaluate matched replanning interventions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.warm_actor_capture import capture_episode, evaluate_replanning_root, DEFAULT_DECISIONS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("--checkpoint", type=Path, required=True)
    capture.add_argument("--matrix", type=Path, required=True)
    capture.add_argument("--selector", required=True)
    capture.add_argument("--seed", type=int, required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--decisions", nargs="+", type=int, default=DEFAULT_DECISIONS)
    capture.add_argument("--max-steps", type=int, default=500)
    capture.add_argument("--controller-seed", type=int)
    capture.add_argument("--device")
    capture.add_argument("--metadata", type=Path)
    replan = subparsers.add_parser("replan")
    replan.add_argument("--root", type=Path, required=True)
    replan.add_argument("--output", type=Path, required=True)
    replan.add_argument("--checkpoint", type=Path)
    replan.add_argument("--matrix", type=Path)
    replan.add_argument("--device")
    replan.add_argument("--metadata", type=Path)
    args = parser.parse_args(argv)
    if args.command == "capture":
        result = capture_episode(checkpoint=args.checkpoint, matrix_path=args.matrix,
                                 selector=args.selector, seed=args.seed, output=args.output,
                                 decisions=args.decisions, max_steps=args.max_steps,
                                 controller_seed=args.controller_seed, device=args.device,
                                 metadata_path=args.metadata)
    else:
        result = evaluate_replanning_root(root_path=args.root, output=args.output,
                                          checkpoint=args.checkpoint, matrix_path=args.matrix,
                                          device=args.device, metadata_path=args.metadata)
    print(json.dumps({"status": result["status"], "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
