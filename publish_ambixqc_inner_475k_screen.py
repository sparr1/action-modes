"""Publish one screen condition; an empty final scan can never report success."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from eval_series import scheduler_done
import run_ambixqc_inner_475k_screen as campaign
from utils.eval_series import Publisher, load_run


def verify_record(run_dir, index, *, require_published=False):
    """Validate any accepted record before success, including its exact selector."""
    records = campaign.banks.read_json(Path(run_dir) / "publication.json")["records"]
    if not records:
        return False
    if len(records) != 1:
        raise ValueError("Screen curve must accept exactly one checkpoint result.")
    rid, entry = next(iter(records.items()))
    record = campaign.banks.read_json(Path(run_dir) / "records" / (rid + ".json"))
    selector = record.get("selector") or record.get("provenance", {}).get("selector")
    if (entry.get("checkpoint_step") != campaign.STEP
            or entry.get("checkpoint_sha256") != campaign.CHECKPOINT_SHA
            or selector != campaign.selector_for(index)):
        raise ValueError("Publisher accepted the wrong checkpoint or controller.")
    if require_published and entry.get("status") != "published":
        raise RuntimeError("Expected result was not acknowledged; retain journal for reconciliation.")
    return True


def publish(run_dir, index, jobs, *, owner="oscar-rgao48", poll_seconds=15,
            visibility_seconds=180, publisher_factory=Publisher, scheduler=scheduler_done,
            monotonic=time.monotonic, sleep=time.sleep):
    campaign.selector_for(index)
    if not jobs or poll_seconds <= 0 or visibility_seconds <= 0:
        raise ValueError("Provide compute jobs and positive polling/visibility intervals.")
    registry = load_run(run_dir)
    identity = registry["identity"]
    # planner_identity deliberately removes these default soft-XQC fields.
    # Decode precisely those defaults; do not treat omitted nondefault science
    # settings as permission to accept a different curve.
    settings = {"inner_critic_source": "xqc", "inner_horizon_critic_source": "xqc",
                "inner_critic_target": "entropy_augmented", **identity["planner"]["settings"]}
    expected = campaign.expected_settings(index)
    required = ("inner_operator", "inner_actor_bn_mode", "inner_rounds", "inner_actor_lr",
                "inner_critic_lr", "inner_critic_source", "inner_critic_target",
                "inner_horizon_critic_source", "inner_reward_normalization",
                "inner_replay_capacity", "inner_rollout_horizon", "inner_rollouts_per_round",
                "inner_batch_size", "inner_updates_per_round", "inner_terminal_bootstrap")
    if (identity.get("backbone") != campaign.banks.SOURCE_RUNS[campaign.CELL]
            or identity["planner"].get("type") != "xqc"
            or identity["planner"].get("action_rule") != "tanh_mean"
            or any(settings.get(key) != expected[key] for key in required)):
        raise ValueError("Publisher registry does not match this screen condition.")
    terminal_at, observed_jobs, last = None, set(), None
    # Preserve the stock single-owner lock, immutable result acceptance and
    # uncertain-write reconciliation. Repeated scans never resend a queued row
    # already sent in this session; Publisher.finish verifies remote receipt.
    with publisher_factory(run_dir, owner=owner) as publisher:
        while True:
            progress = publisher.publish_pending()
            accepted = verify_record(run_dir, index)
            if progress != last:
                print(json.dumps(progress, sort_keys=True), flush=True)
                last = progress
            if scheduler(jobs, observed_jobs):
                if accepted:
                    break
                if terminal_at is None:
                    terminal_at = monotonic()
                if monotonic() - terminal_at >= visibility_seconds:
                    raise TimeoutError("Compute ended without the expected published result pointer; "
                                       "preserve completed GPU bundles and repair publication without reevaluation.")
            sleep(poll_seconds)
    if not verify_record(run_dir, index, require_published=True):
        raise RuntimeError("Publisher finished without its expected result.")
    receipt = {"run_id": registry["run_id"], "selector": campaign.selector_for(index),
               "checkpoint_sha256": campaign.CHECKPOINT_SHA, "checkpoint_step": campaign.STEP,
               "accepted": 1, "published": 1}
    from utils.ambi_benchmark import atomic_json
    atomic_json(Path(run_dir) / "screen-publication-verified.json", receipt)
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--jobs", nargs="+", required=True)
    p.add_argument("--owner", default="oscar-rgao48")
    p.add_argument("--poll-seconds", type=float, default=15)
    p.add_argument("--visibility-seconds", type=float, default=180)
    args = p.parse_args(argv)
    print(json.dumps(publish(args.run_dir, args.index, args.jobs, owner=args.owner,
                             poll_seconds=args.poll_seconds, visibility_seconds=args.visibility_seconds)))


if __name__ == "__main__":
    main()
