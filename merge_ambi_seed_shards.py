#!/usr/bin/env python3
"""Consolidate completed seed shards before episode/diagnostic publication."""
import argparse
import json

from utils.ambi_seed_shards import merge_episode_bundles, merge_real_bundles, seal_episode_bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    seal = commands.add_parser("seal-episodes", help="Hash a completed worker episode bundle")
    seal.add_argument("--bundle", required=True)
    for name in ("episodes", "real"):
        merge = commands.add_parser(name)
        merge.add_argument("--sources", nargs="+", required=True)
        merge.add_argument("--output", required=True)
        merge.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "seal-episodes":
        output = seal_episode_bundle(args.bundle)
    else:
        merge = merge_episode_bundles if args.command == "episodes" else merge_real_bundles
        output = merge(args.sources, args.output, expected_seeds=args.seeds)
    print(json.dumps({"status": "complete", "output": str(output)}))


if __name__ == "__main__":
    main()
