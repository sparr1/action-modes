"""Standalone ablations of identical-budget teacher arithmetic and concurrency.

Each invocation measures the production full-refresh path in one fresh process.
Feature overrides are private to this benchmark; they never create checkpoints.
"""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from RL.tdmpc2_core.batched_teacher import BatchedTeacherCore
from tests.benchmark_ambi_reanalyze import benchmark


FEATURES = {
    "reference": frozenset(),
    "heads": frozenset({"selective_q"}),
    "heads-cache": frozenset({"selective_q", "actor_cache"}),
    "all": frozenset({"selective_q", "actor_cache", "fused_updates"}),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", choices=FEATURES, default="all")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--roots", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--profile", choices=("tiny", "reference"), default="reference")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--matmul-precision", choices=("high", "highest"), default="high")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.workers, args.width, args.roots, args.repeats, args.warmup) < 1:
        parser.error("All counts must be positive.")
    if args.workers > 1 and args.device != "cuda":
        parser.error("Concurrent workers require CUDA.")
    args.compile = not args.eager
    args.full_refresh = True
    args.batched_backend = "batched_cuda_graph" if args.device == "cuda" else "batched"
    args.reference_batched = args.features == "reference"
    args.batched_workers = args.workers
    args.parallel_roots = [args.width]
    args.threads = 1
    args.graph_only = False
    args.reuse_graph_pool = False
    args.compile_cache_limit = 128
    initialize = BatchedTeacherCore.__init__

    def with_features(self, *positional, **keywords):
        keywords["optimization_features"] = FEATURES[args.features]
        initialize(self, *positional, **keywords)

    with patch.object(BatchedTeacherCore, "__init__", with_features):
        result = benchmark(args)
    result.update(optimization_features=sorted(FEATURES[args.features]),
                  feature_set=args.features, workers=args.workers,
                  checkpointing=False, wandb=False)
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
