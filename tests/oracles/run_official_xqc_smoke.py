#!/usr/bin/env python3
"""Run the pinned official smoke and assert its completed learner updates."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


OFFICIAL_COMMIT = "9a6832bb742ef01bbe9f1e06153a9338e612dae5"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--expected-updates", type=int, required=True)
    args, hydra_args = parser.parse_known_args()

    repo = args.official_repo.resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != OFFICIAL_COMMIT:
        raise SystemExit(
            f"official checkout must be {OFFICIAL_COMMIT}, found {commit}"
        )
    if subprocess.check_output(
        [
            "git",
            "-C",
            str(repo),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        text=True,
    ):
        raise SystemExit("official checkout must be clean")
    if args.expected_updates < 1:
        raise SystemExit("--expected-updates must be positive")

    sys.path.insert(0, str(repo))
    sys.argv = [sys.argv[0], *hydra_args]

    from xqc.agents import XQCLearner
    from train_parallel import main as official_main

    completed_updates = 0
    original_update = XQCLearner.update

    def counted_update(self, batch, num_updates=1, time_to_intervene=False):
        nonlocal completed_updates
        result = original_update(
            self,
            batch,
            num_updates=num_updates,
            time_to_intervene=time_to_intervene,
        )
        completed_updates += int(num_updates)
        return result

    XQCLearner.update = counted_update
    try:
        official_main()
    finally:
        XQCLearner.update = original_update

    if completed_updates != args.expected_updates:
        raise SystemExit(
            "official XQC smoke completed "
            f"{completed_updates} learner updates; expected {args.expected_updates}"
        )
    print(f"Official XQC learner updates: {completed_updates}")


if __name__ == "__main__":
    main()
