"""Run one full-episode inner-SAC transfer discovery cell with paired seeds."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np
import torch

from evaluate_ambi_checkpoint import (
    _close_resources, _file_sha256, _initialize_frozen_model, _make_env,
    _outer_state_digest, _validate_checkpoint_contract,
)
from utils.ambi_benchmark import atomic_json
from utils.ambi_research import load_preset_matrix, resolve_preset
from utils.checkpoint_context import load_checkpoint_context
from utils.transfer_campaign import (
    PROTOCOL, cells, evaluate_episode, load_campaign, resolved_cell, summarize_episodes,
)
from utils.transfer_campaign_diagnostics import (
    CampaignDiagnostics, diagnostic_settings, verify_episode_diagnostics,
    verify_observational_isolation,
)
from utils.spectral_campaign import (
    SpectralCampaignDiagnostics, spectral_settings, validate_probe_settings,
    verify_spectral_diagnostics,
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", type=Path, default=Path("configs/research/ambi_transfer_discovery_575k.json"))
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--metadata", type=Path)
    p.add_argument("--horizon", type=int)
    p.add_argument("--rounds", type=int)
    p.add_argument("--arm")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--controller-seed", type=int)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--smoke", action="store_true", help="Two decisions by default; explicitly --max-steps may extend the smoke.")
    p.add_argument("--no-compile", action="store_true", help="Explicit eager override, recorded in the manifest.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--list-cells", action="store_true", help="Print JSON cells in high-J-first scheduling order without loading a checkpoint.")
    return p


def listed_cells(campaign):
    grid = cells(campaign)
    grid.sort(key=lambda row: (-row["rounds"], row["horizon"], list(campaign["arms"]).index(row["arm"])))
    return [dict(index=index, name=row["cell_id"], H=row["horizon"], J=row["rounds"], arm=row["arm"])
            for index, row in enumerate(grid)]


def source_identity():
    root = Path(__file__).resolve().parent
    files = [Path(__file__), root / "utils/transfer_campaign.py",
        root / "RL/tdmpc2_core/inner_improvement.py", root / "RL/tdmpc2_core/inner_trace.py",
        root / "RL/tdmpc2_core/common/inner_utils.py",
        root / "utils/transfer_campaign_diagnostics.py",
        root / "utils/transfer_diagnostic_metrics.py", root / "utils/transfer_diagnostics.py",
        root / "utils/spectral_transfer.py", root / "utils/spectral_transfer_probes.py",
        root / "utils/spectral_campaign.py"]
    digests = {str(path.relative_to(root)): _file_sha256(path) for path in files}
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("Git is required to record source identity.")
    head = subprocess.run([git, "-C", str(root), "rev-parse", "HEAD"], close_fds=False,
        capture_output=True, text=True, check=True).stdout.strip()
    return dict(git_head=head, files=digests,
        sha256=hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest())


def validate_args(args, campaign):
    for name in ("checkpoint", "horizon", "rounds", "arm"):
        if getattr(args, name) is None:
            raise ValueError(f"--{name.replace('_', '-')} is required unless --list-cells is used.")
    if not args.dry_run and args.output_dir is None:
        raise ValueError("--output-dir is required for evaluation.")
    seeds = campaign["seeds"] if args.seeds is None else args.seeds
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("Episode seeds must be unique and nonnegative.")
    steps = args.max_steps if args.max_steps is not None else 2 if args.smoke else campaign["max_steps"]
    if steps < 1 or steps > campaign["max_steps"]:
        raise ValueError("max-steps must be positive and no larger than the campaign episode horizon.")
    if not args.smoke and steps != campaign["max_steps"]:
        raise ValueError("Short runs must be explicitly labeled --smoke.")
    controller_seed = campaign["controller_seed"] if args.controller_seed is None else args.controller_seed
    if controller_seed < 0:
        raise ValueError("Controller seed must be nonnegative.")
    return list(seeds), steps, controller_seed


def run(args):
    campaign = load_campaign(args.campaign)
    diagnostics_config = diagnostic_settings(campaign.get("diagnostics"))
    spectral_config = spectral_settings(campaign.get("spectral_diagnostics"))
    spectral_probe = validate_probe_settings(campaign.get("spectral_probe"))
    if args.list_cells:
        return listed_cells(campaign)
    seeds, max_steps, controller_seed = validate_args(args, campaign)
    if not args.dry_run and args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite discovery output: {args.output_dir}")
    matrix_path = Path(campaign["base_matrix_path"])
    matrix = load_preset_matrix(matrix_path)
    if matrix.get("checkpoint_contract") != campaign["checkpoint_contract"]:
        raise ValueError("Discovery and base-matrix checkpoint contracts disagree.")
    context = load_checkpoint_context(args.checkpoint, metadata_path=args.metadata)
    base = resolve_preset(matrix_path, campaign["base_preset"], matrix=matrix, checkpoint_context=context)
    resolved = resolved_cell(base, campaign, horizon=args.horizon, rounds=args.rounds,
                             arm=args.arm, no_compile=args.no_compile)
    _validate_checkpoint_contract(matrix, args.checkpoint, context, [resolved])
    arm = campaign["arms"][args.arm]
    cell_id = f"h{args.horizon}_j{args.rounds}_{args.arm}"
    manifest = dict(schema_version=1, protocol=PROTOCOL, cell_id=cell_id, arm=args.arm,
        arm_definition=arm, horizon=args.horizon, rounds=args.rounds, seeds=seeds,
        controller_seed=controller_seed, max_steps=max_steps, smoke=bool(args.smoke),
        diagnostics=diagnostics_config,
        spectral_diagnostics=spectral_config, spectral_probe=spectral_probe,
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=_file_sha256(args.checkpoint),
        checkpoint_step=context.metadata["checkpoint"]["step"],
        campaign=str(args.campaign.resolve()), campaign_sha256=_file_sha256(args.campaign),
        base_matrix_sha256=_file_sha256(matrix_path), resolved=resolved, source=source_identity(),
        device=args.device, compile=resolved["algorithm_config"]["alg_params"]["compile"],
        runtime=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__),
        semantics=dict(scope="successive_real_decisions_within_episode", objective="return_return",
            action="adapted_actor_mean", solve_interval=1, first_action_rounds="selected_J",
            seed_scheme="solver_seed(controller_seed, 'episode', seed); persistent private streams",
            reuse="No historical return/timing is silently substituted.",
            timing="Prediction wall time plus donor construction/export; sampled diagnostic snapshots and probes measured separately and excluded from control_seconds.",
            diagnostics="Fixed prior-continuation model targets shared by prior/donor/initial/final stages. Disposable fixed-target fits; no learner updates or real-return ground-truth claim.",
            selection="Three development seeds; exploratory mechanism screen, not confirmatory.",
            compute="Fixed J,C,A,N per cell; imagination transitions scale with H; no equal-compute claim.",
            replay="Previous solve only, fixed requested minibatch fraction; no real replay.",
            anchors="Actor KL(current||frozen prior); critic mean over tensors of MSE/(mean prior squared+1e-6)."))
    if campaign.get("family") == "spectral_transfer":
        metadata_path = (args.metadata or Path(str(args.checkpoint) + ".metadata.json")).resolve()
        manifest.update(metadata_path=str(metadata_path), metadata_sha256=_file_sha256(metadata_path))
        manifest["semantics"].update(
            spectral="Layerwise donor-minus-frozen-prior matrix deltas; nonmatrix parameters reset. Dense SAC after initialization. Rank clamps to each matrix dimension.",
            spectral_scoring="Prior-root bank, frozen-prior anchor. Actor: negative frozen-prior Q at deterministic mean, no entropy. Critic: per-head decoded MSE versus fixed prior model returns.",
            spectral_evaluation="Independent heldout stream; same bank and labels at prior/donor/initial/post-J. Model surrogates, not environment ground truth.",
            timing="Controller includes selection probes, SVD/covariance/gradient filtering, initialization and donor export. Heldout diagnostic probes and snapshot time excluded and reported separately.")
    if args.dry_run:
        return manifest
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "manifest.json", manifest)
    started = time.perf_counter()
    episodes = []
    env = model = None
    def progress(status, **extra):
        atomic_json(output / "progress.json", dict(protocol=PROTOCOL, cell_id=cell_id,
            status=status, completed_episodes=len(episodes), total_episodes=len(seeds),
            elapsed_seconds=time.perf_counter() - started, smoke=bool(args.smoke), **extra), overwrite=True)
    progress("initializing")
    try:
        env = _make_env(resolved)
        model, runtime_config = _initialize_frozen_model(resolved, env, args.checkpoint,
                                                        controller_seed, device=args.device)
        frozen = _outer_state_digest(model)
        atomic_json(output / "runtime.json", dict(config=runtime_config, outer_digest=frozen))
        for episode_index, episode_seed in enumerate(seeds):
            diagnostics = (CampaignDiagnostics(diagnostics_config, episode_seed=episode_seed,
                controller_seed=controller_seed, smoke=args.smoke) if diagnostics_config else None)
            if spectral_config:
                diagnostics = SpectralCampaignDiagnostics(spectral_config, basic_settings=diagnostics_config,
                    episode_seed=episode_seed, controller_seed=controller_seed, smoke=args.smoke)
            reference_rows, observed_rows = [], []
            reference_state = None
            if args.smoke and diagnostics is not None:
                progress("verifying_diagnostic_isolation", seed=episode_seed, episode_index=episode_index)
                evaluate_episode(model, env, arm, episode_seed=episode_seed,
                    controller_seed=controller_seed, max_steps=max_steps,
                    on_step=reference_rows.append, smoke=True, spectral_probe=spectral_probe)
                reference_state = dict(learner=model.agent.inner_engine.export_diagnostic_state(),
                    rng=deepcopy(model.agent.inner_engine.rng.training_state_dict()))
            with (output / f"decisions-seed-{episode_seed}.jsonl").open("x") as stream:
                def on_step(row):
                    if reference_state is not None:
                        observed_rows.append(row)
                    stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
                    stream.flush()
                    progress("running", seed=episode_seed, episode_index=episode_index,
                        decision=row["decision"], episode_return=row["cumulative_reward"],
                        completed_decisions=sum(ep["steps"] for ep in episodes) + row["decision"] + 1,
                        total_decisions=max_steps * len(seeds))
                episode = evaluate_episode(model, env, arm, episode_seed=episode_seed,
                    controller_seed=controller_seed, max_steps=max_steps, on_step=on_step, smoke=args.smoke,
                    spectral_probe=spectral_probe,
                    **({"diagnostics": diagnostics} if diagnostics is not None else {}))
            verify_episode_diagnostics(episode, diagnostics_config, smoke=args.smoke)
            verify_spectral_diagnostics(episode, spectral_config, smoke=args.smoke)
            if reference_state is not None:
                observed_state = dict(learner=model.agent.inner_engine.export_diagnostic_state(),
                    rng=deepcopy(model.agent.inner_engine.rng.training_state_dict()))
                verify_observational_isolation(reference_rows, observed_rows, reference_state, observed_state)
                episode["diagnostic_isolation_verified"] = True
            if _outer_state_digest(model) != frozen:
                raise RuntimeError("Frozen outer model or optimizer changed during discovery evaluation.")
            atomic_json(output / f"episode-seed-{episode_seed}.json", episode)
            episodes.append(episode)
            progress("running", seed=episode_seed, completed_decisions=sum(ep["steps"] for ep in episodes),
                     total_decisions=max_steps * len(seeds))
        result = dict(protocol=PROTOCOL, cell_id=cell_id, complete=True, smoke=bool(args.smoke),
            horizon=args.horizon, rounds=args.rounds, arm=args.arm, seeds=seeds,
            episodes=episodes, summary=summarize_episodes(episodes), source=manifest["source"],
            checkpoint_sha256=manifest["checkpoint_sha256"], frozen_outer_verified=True,
            total_seconds=time.perf_counter() - started)
        atomic_json(output / "results.json", result)
        progress("complete", completed_decisions=sum(ep["steps"] for ep in episodes),
                 total_decisions=max_steps * len(seeds), summary=result["summary"])
        return result
    except BaseException as exc:
        failure = dict(type=type(exc).__name__, message=str(exc), completed_episodes=len(episodes),
                       elapsed_seconds=time.perf_counter() - started)
        atomic_json(output / "failure.json", failure)
        progress("failed", error=failure)
        raise
    finally:
        _close_resources(model, env)


def main(argv=None):
    result = run(parser().parse_args(argv))
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
