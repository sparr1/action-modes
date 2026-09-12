"""Measure MPPI and inner-SAC action boundaries on identical frozen prior roots.

This is an offline distribution diagnostic, not an episode-return comparison.
It observes the unchanged AMBI MPPI helper and separately reports its optimized
mean and the boundary probability of selecting an elite by its fitted weight.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gzip
import html
import json
from pathlib import Path
import random
import shutil
import statistics
import time
from unittest.mock import patch

import numpy as np
import torch

from RL.tdmpc2_core import mppi

DEFAULT_MATRIX = Path(__file__).parent / "configs/research/ambi_prior_mean_prefix_h1_20seeds.json"
METRICS = ("exact_boundary_fraction", "near_boundary_fraction", "mean_absolute_action")


def boundary_statistics(actions, weights=None):
    """Fractions are per action component; optional weights mix candidate rows."""
    if actions.ndim == 2:
        actions = actions.unsqueeze(0)
    if actions.ndim != 3 or not actions.numel() or not torch.isfinite(actions).all():
        raise ValueError("Actions must be finite [horizon, candidates, components] values")
    magnitude = actions.abs()
    if bool((magnitude > 1).any()):
        raise ValueError("Actions must already be bounded by [-1, 1]")
    values = [(magnitude == 1).float(), (magnitude >= .99).float(), magnitude]
    if weights is None:
        result = [value.mean() for value in values]
    else:
        weights = torch.as_tensor(weights, device=actions.device, dtype=actions.dtype).reshape(-1)
        if (weights.numel() != actions.shape[1] or not torch.isfinite(weights).all()
                or bool((weights < 0).any()) or not torch.isclose(weights.sum(), weights.new_tensor(1.))):
            raise ValueError("Candidate weights must be finite nonnegative probabilities")
        # Categorical selection uses weights proportionally. Normalize in
        # double precision so fitted float32 summation cannot report >100%.
        probabilities = weights.double() / weights.double().sum()
        result = [(value.double().mean(dim=(0, 2)) * probabilities).sum() for value in values]
    return {**dict(zip(METRICS, torch.stack(result).cpu().tolist())),
            "component_count": actions.numel(), "candidate_count": actions.shape[1]}


@torch.no_grad()
def population_statistics(actions, values, *, num_elites, num_pi_trajs, temperature, min_std, max_std):
    """Reproduce the helper's fit after observation, consuming no randomness."""
    values = values.nan_to_num(0.0)
    indices = torch.topk(values.squeeze(-1), num_elites, dim=0).indices
    elite_values, elites = values.index_select(0, indices), actions.index_select(1, indices)
    weights = torch.exp(temperature * (elite_values - elite_values.max(dim=0).values))
    weights = weights / weights.sum(dim=0).clamp_min(1e-9)
    mean = (weights.unsqueeze(0) * elites).sum(dim=1)
    variance = (weights.unsqueeze(0) * (elites - mean.unsqueeze(1)).square()).sum(dim=1)
    std = variance.sqrt().clamp(min_std, max_std)
    stats = {
        "candidates_all": boundary_statistics(actions),
        "elites_unweighted": boundary_statistics(elites),
        "weighted_elite_selection": boundary_statistics(elites, weights),
        "optimized_mean": boundary_statistics(mean.clamp(-1, 1).unsqueeze(1)),
    }
    if num_pi_trajs:
        stats["candidates_prior"] = boundary_statistics(actions[:, :num_pi_trajs])
    if num_pi_trajs < actions.shape[1]:
        stats["candidates_proposal"] = boundary_statistics(actions[:, num_pi_trajs:])
    raw = dict(actions=actions.cpu().tolist(), values=values.cpu().tolist(),
               elite_indices=indices.cpu().tolist(), elite_weights=weights.cpu().tolist(),
               optimized_mean=mean.cpu().tolist(), proposal_std=std.cpu().tolist())
    return stats, raw, mean


@torch.no_grad()
def observe_mppi(root_z, **kwargs):
    """Observe the exact solve; the scoped patch also restores after exceptions."""
    if kwargs.get("eval_mode", True) is not True:
        raise ValueError("This diagnostic observes optimized-mean evaluation only")
    kwargs["eval_mode"] = True
    populations = []
    original = mppi._estimate_value

    def observe(*args, **options):
        value = original(*args, **options)
        populations.append((args[1].detach().clone(), value.detach().clone()))
        return value

    with patch.object(mppi, "_estimate_value", observe):
        result = mppi.mppi_plan(root_z, **kwargs)
    records = []
    for iteration, (actions, values) in enumerate(populations, start=1):
        stats, raw, mean = population_statistics(actions, values, **{
            key: kwargs[key] for key in ("num_elites", "num_pi_trajs", "temperature", "min_std", "max_std")})
        records.append(dict(iteration=iteration, statistics=stats, population=raw))
    if (len(records) != kwargs["iterations"] or not torch.equal(result.next_mean, mean)
            or not torch.equal(result.action, mean[0].clamp(-1, 1))):
        raise RuntimeError("Observed MPPI fit differs from the exact executed helper")
    return result, records


@contextmanager
def preserve_runtime(model):
    """Restore module modes and global RNG streams, including failure cleanup."""
    modes = [(module, module.training) for module in model.modules()]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        model.eval()
        yield
    finally:
        for module, training in modes:
            module.training = training
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


@torch.no_grad()
def actor_statistics(model, root_z, policy, bounds, noise):
    with preserve_runtime(model):
        samples, _ = model.pi(root_z.expand(noise.shape[0], -1), policy=policy, noise=noise, **bounds)
        mean, _ = model.pi(root_z, policy=policy, noise=torch.zeros_like(noise[:1]), **bounds)
    return {"policy_samples": boundary_statistics(samples), "policy_mean": boundary_statistics(mean)}


def aggregate_rows(rows, resamples=2000):
    """Average solvers within roots, roots within episodes, then episodes equally."""
    groups = {}
    for row in rows:
        key = row["operator"], row["iteration"], row["distribution"]
        groups.setdefault(key, {}).setdefault(row["episode_id"], {}).setdefault(row["root_id"], []).append(row)
    output = []
    for (operator, iteration, distribution), episodes in sorted(groups.items()):
        values = {metric: [statistics.fmean(statistics.fmean(row[metric] for row in repeats)
                                           for repeats in roots.values()) for _, roots in sorted(episodes.items())]
                  for metric in METRICS}
        rng = np.random.default_rng(0)
        indices = rng.integers(0, len(episodes), size=(resamples, len(episodes)))
        summary = {}
        for metric, numbers in values.items():
            boot = np.asarray(numbers)[indices].mean(axis=1)
            summary[metric] = dict(mean=statistics.fmean(numbers), episode_values=numbers,
                episode_std=statistics.stdev(numbers) if len(numbers) > 1 else None,
                ci95=np.quantile(boot, [.025, .975]).tolist() if len(numbers) > 1 else None)
        output.append(dict(operator=operator, iteration=iteration, distribution=distribution,
                           episodes=len(episodes), metrics=summary))
    return output


def write_html(path, record):
    table = []
    for row in record["summary"]:
        metrics = row["metrics"]
        table.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in (
            row["operator"], row["iteration"], row["distribution"],
            f"{100 * metrics[METRICS[0]]['mean']:.2f}%",
            f"{100 * metrics[METRICS[1]]['mean']:.2f}%",
            f"{metrics[METRICS[2]]['mean']:.4f}")) + "</tr>")
    text = """<!doctype html><meta charset="utf-8"><title>MPPI and SAC action boundaries</title>
<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;color:#172232}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px;text-align:left;border-bottom:1px solid #ddd}
th{background:#edf2f7}code{background:#edf2f7}</style><h1>MPPI and SAC action boundaries</h1>
<p>Shared prior roots, frozen 200k backbone. Fractions count action components exactly at ±1 or within 0.01 of a boundary.
Results average solver repetitions within root, roots within episode, then episodes equally.</p>
<p><b>Weighted elite selection</b> is the expected saturation of sampling an elite with its fitted probability.
<b>Optimized mean</b> is the deterministic action returned by this AMBI helper in evaluation mode.
The official TD-MPC2 evaluation samples a weighted elite; these are different action rules.
This diagnostic does not compare real episode returns.</p>
<p><a href="results.json">Complete summaries, episode-cluster intervals and provenance</a> ·
<a href="measurements.jsonl.gz">Per-root measurements</a> · <a href="mppi-populations.jsonl.gz">Observed MPPI populations</a></p>
<table><thead><tr><th>Operator</th><th>Round / iteration</th><th>Distribution</th><th>Exactly ±1</th><th>Near ±1</th><th>Mean |action|</th></tr></thead><tbody>"""
    path.write_text(text + "".join(table) + "</tbody></table>")


def run(checkpoint, root_bank, output_dir, *, matrix=DEFAULT_MATRIX, device="cuda", smoke=False):
    from evaluate_ambi_calibration import _bank_protocol, _science_identity, _synchronize, load_root_bank
    from evaluate_ambi_checkpoint import (_close_resources, _file_sha256, _initialize_frozen_model, _jsonable, _make_env,
        _outer_state_digest, _validate_checkpoint_contract, _validate_frozen_selection)
    from RL.tdmpc2_core.inner_trace import InnerActionTrace
    from utils.ambi_benchmark import atomic_json, solver_seed
    from utils.ambi_research import load_preset_matrix, resolve_preset
    from utils.checkpoint_context import load_checkpoint_context

    checkpoint, root_bank, matrix = map(lambda p: Path(p).resolve(), (checkpoint, root_bank, matrix))
    output = Path(output_dir).resolve()
    work = output.with_name(output.name + ".work")
    if output.exists() or work.exists():
        raise FileExistsError("Choose a new output directory; measurements are immutable")
    config = load_preset_matrix(matrix)
    context = load_checkpoint_context(checkpoint)
    resolved = resolve_preset(matrix, "initialization/inherited", config, checkpoint_context=context)
    _validate_frozen_selection(config, [resolved])
    _validate_checkpoint_contract(config, checkpoint, context, [resolved])
    if context.metadata["checkpoint"]["step"] != 200000:
        raise ValueError("This companion selects the 200k checkpoint")
    params = resolved["algorithm_config"]["alg_params"]
    expected = dict(inner_rounds=4, inner_rollout_horizon=1, inner_rollouts_per_round=128,
                    inner_critic_updates_per_round=32, inner_actor_updates_per_round=4,
                    inner_batch_size=256, inner_temperature=0., inner_actor_initialization="prior",
                    inner_critic_initialization="prior", inner_actor_initial_std=None)
    if any(params.get(key) != value for key, value in expected.items()):
        raise ValueError("The paired SAC diagnostic requires the unchanged inherited J4 reference")
    science = _science_identity()
    seeds = list(range(101, 121))
    protocol = _bank_protocol(resolved, seeds, 55, dict(max_steps=500, decisions=[0, 100, 200, 300, 400]), science)
    bank = load_root_bank(root_bank, _file_sha256(checkpoint), protocol)
    roots, repetitions = (bank["roots"][:1], 1) if smoke else (bank["roots"], 3)
    env = model = error = None
    rows = []
    started = time.perf_counter()
    timing = dict(mppi_seconds=0., inner_sac_seconds=0., policy_measurement_seconds=0.)
    work.mkdir(parents=True)
    try:
        env = _make_env(resolved)
        model, _ = _initialize_frozen_model(resolved, env, checkpoint, 55, device=device)
        before = _outer_state_digest(model)
        settings = dict(horizon=1, iterations=4, num_samples=128, num_elites=16, num_pi_trajs=6,
                        temperature=.5, min_std=.05, max_std=2., discount=float(model.agent.discount),
                        q_reduction=model.cfg.mppi_terminal_q_reduction, eval_mode=True)
        with gzip.open(work / "mppi-populations.jsonl.gz", "wt") as populations, preserve_runtime(model.agent.model):
            for root in roots:
                observation = np.asarray(root["observation"], np.float32)
                with torch.no_grad():
                    z = model.agent.model.encode(torch.as_tensor(observation, device=model.agent.device)[None]).detach()
                noise_seed = solver_seed(55, "saturation-action-noise", root["root_id"])
                generator = torch.Generator(device=z.device).manual_seed(noise_seed)
                noise = torch.randn((1024, model.cfg.action_dim), device=z.device, dtype=z.dtype, generator=generator)
                for repeat in range(repetitions):
                    coordinate = dict(episode_id=root["episode_id"], root_id=root["root_id"],
                                      seed=root["seed"], decision_index=root["decision_index"], solver_repeat=repeat)
                    plan_seed = solver_seed(55, "mppi-saturation-solve", root["root_id"], repeat)
                    generator = torch.Generator(device=z.device).manual_seed(plan_seed)
                    tick = time.perf_counter()
                    result, observations = observe_mppi(z, model=model.agent.model, generator=generator, **settings)
                    _synchronize(model)
                    timing["mppi_seconds"] += time.perf_counter() - tick
                    for measured in observations:
                        populations.write(json.dumps(dict(**coordinate, solver_seed=plan_seed, **measured["population"],
                                                           iteration=measured["iteration"]), allow_nan=False) + "\n")
                        for distribution, metrics in measured["statistics"].items():
                            rows.append(dict(**coordinate, operator="mppi", iteration=measured["iteration"],
                                             solver_seed=plan_seed, distribution=distribution, **metrics))
                    solve_seed = solver_seed(55, "real-calibration-solve", root["root_id"], repeat)
                    model.agent.inner_engine.reset_for_evaluation(solve_seed, reuse_action_pool=True)
                    trace = InnerActionTrace(probes=False, capture_actors=True, actor_rounds=[0, 1, 2, 4])
                    tick = time.perf_counter()
                    model.predict(observation, deterministic=True, episode_start=True, trace=trace)
                    _synchronize(model)
                    timing["inner_sac_seconds"] += time.perf_counter() - tick
                    if [snapshot.round_index for snapshot in trace.actor_snapshots] != [0, 1, 2, 4]:
                        raise RuntimeError("Missing requested SAC actor snapshots")
                    tick = time.perf_counter()
                    for snapshot in trace.actor_snapshots:
                        policy = snapshot.make_policy(model.agent.device)
                        measured = actor_statistics(model.agent.model, z, policy, snapshot.policy_bounds, noise)
                        for distribution, metrics in measured.items():
                            rows.append(dict(**coordinate, operator="sac", iteration=snapshot.round_index,
                                actor_updates=snapshot.actor_updates, critic_updates=snapshot.critic_updates,
                                solver_seed=solve_seed, policy_noise_seed=noise_seed, actor_sha256=snapshot.sha256,
                                distribution=distribution, **metrics))
                        del policy
                    _synchronize(model)
                    timing["policy_measurement_seconds"] += time.perf_counter() - tick
                print(json.dumps(dict(root_id=root["root_id"], completed_measurements=len(rows))), flush=True)
        unchanged = before == _outer_state_digest(model)
        if not unchanged:
            raise RuntimeError("Frozen outer weights or optimizer state changed")
        with gzip.open(work / "measurements.jsonl.gz", "wt") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        summary = aggregate_rows(rows)
        record = dict(schema_version=1, kind="mppi_sac_action_saturation", status="complete", smoke=smoke,
            checkpoint=str(checkpoint), checkpoint_sha256=_file_sha256(checkpoint), root_bank=str(root_bank),
            root_bank_id=bank["id"], root_bank_sha256=_file_sha256(root_bank), matrix=str(matrix), matrix_sha256=_file_sha256(matrix),
            science=science, diagnostic_source_sha256=_file_sha256(__file__), root_protocol=protocol,
            outer_state_unchanged=unchanged,
            mppi=settings, roots=len(roots), solver_repetitions=repetitions, sac_policy_samples=1024,
            sac_rounds=[0, 1, 2, 4], sac_resolved_config=_jsonable(vars(model.cfg)), aggregation="solver_within_root_within_episode_equal_episodes",
            bootstrap_resamples=2000, bootstrap_seed=0, summary=summary, timing=timing,
            elapsed_seconds=time.perf_counter() - started, units="fraction_of_action_components",
            action_rules=dict(optimized_mean="AMBI mppi_plan eval_mode=True",
                weighted_elite_selection="Expected component statistics under categorical elite weights; no sampled selection added",
                official_tdmpc2="Evaluation samples a weighted elite, not the optimized mean"))
        atomic_json(work / "results.json", record)
        write_html(work / "report.html", record)
        for source, name in ((matrix, "matrix.json"), (root_bank, "root-bank.json"),
                             (context.source, "checkpoint.metadata.json")):
            shutil.copyfile(source, work / name)
        if (_file_sha256(work / "matrix.json") != record["matrix_sha256"]
                or _file_sha256(work / "root-bank.json") != record["root_bank_sha256"]):
            raise RuntimeError("Source artifacts changed while recording the diagnostic")
        atomic_json(work / "checksums.json", {path.name: _file_sha256(path) for path in sorted(work.iterdir())})
        work.rename(output)
        return record
    except BaseException as exc:
        error = exc
        raise
    finally:
        _close_resources(model, env, primary_error=error)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root-bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    run(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
