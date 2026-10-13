"""Aggregate frozen PAVE gradient measurements without treating them as returns."""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import combinations

import numpy as np


def distribution(values):
    values = list(values)
    finite = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return {
        "count": len(values), "finite_count": len(finite),
        "median": float(np.median(finite)) if finite else None,
        "p90": float(np.quantile(finite, .9)) if finite else None,
        "min": min(finite) if finite else None,
        "max": max(finite) if finite else None,
    }


def summarize_batches(batches):
    """Keep each checkpoint and critic separate; never average their ratios away."""
    if not batches:
        raise ValueError("At least one probe batch is required.")
    candidates = defaultdict(list)
    for batch in batches:
        for row in batch["candidates"]:
            candidates[(row["variant"], row["scope"], row["scale"])].append(row)
    summaries = []
    for (variant, scope, scale), rows in sorted(candidates.items()):
        summaries.append({
            "variant": variant, "scope": scope, "scale": scale,
            "weights": rows[0]["weights"],
            "critics": {
                name: {metric: distribution(row["critics"][name][metric] for row in rows)
                       for metric in ("pave_gradient_norm", "ratio_to_td", "cosine_with_td", "cancellation_ratio",
                                      "combined_critic_gradient_norm")}
                for name in rows[0]["critics"]
            },
            **{metric: distribution(row[metric] for row in rows)
               for metric in ("full_gradient_norm", "clip_multiplier",
                              "clip_multiplier_relative_to_baseline")},
        })
    critics = {}
    for name in batches[0]["critics"]:
        rows = [batch["critics"][name] for batch in batches]
        order = ("td", "mpr", "vfc", "curvature")
        cosines = defaultdict(list)
        for row in rows:
            for i, j in combinations(range(4), 2):
                gram = row["gradient_gram"]
                a, b, dot = gram[i][i], gram[j][j], gram[i][j]
                value = (dot / math.sqrt(a * b)
                         if a is not None and b is not None and dot is not None and a > 0 and b > 0 else None)
                cosines[f"{order[i]}__{order[j]}"].append(max(-1., min(1., value)) if value is not None else None)
        critics[name] = {
            "td_coefficient": rows[0]["td_coefficient"],
            **{metric: distribution(row[metric] for row in rows)
               for metric in ("td_gradient_norm", "action_gradient_rms",
                              "curvature_violation_fraction")},
            **{group: {term: distribution(row[group][term] for row in rows)
                       for term in ("mpr", "vfc", "curvature")}
               for group in ("raw_losses", "weighted_term_gradient_norms")},
            "term_gradient_ratios_to_td": {
                term: distribution(row["weighted_term_gradient_norms"][term] / row["td_gradient_norm"]
                                   if row["td_norm_reliable"] and row["weighted_term_gradient_norms"][term] is not None
                                   else None for row in rows)
                for term in ("mpr", "vfc", "curvature")
            },
            "gradient_cosines": {pair: distribution(values) for pair, values in cosines.items()},
        }
    return {
        "batch_count": len(batches), "critics": critics, "candidates": summaries,
        "baseline_gradient_norm": distribution(b["baseline"]["full_gradient_norm"] for b in batches),
        "baseline_clip_multiplier": distribution(b["baseline"]["clip_multiplier"] for b in batches),
        "timing": {metric: distribution(b["timing"][metric] for b in batches)
                   for metric in ("baseline_seconds", "pave_seconds")},
        "cuda_memory": {
            metric: distribution((b.get("cuda_memory") or {}).get(metric) for b in batches)
            for metric in ("baseline_peak_allocated_bytes", "pave_peak_allocated_bytes")
        },
    }


def recommend_scales(checkpoints):
    """A conservative, explicitly heuristic common-scale screen across checkpoints.

    Require the reference ratio to fall inside the proposed 10--30% band for
    every selected critic at every supplied checkpoint. Fail closed on missing
    gradients. Stronger curvature is evaluated at that SAME k to retain the
    curvature-only comparison, never independently normalized into equivalence.
    """
    rules = dict(median_ratio_min=.1, median_ratio_max=.3, p90_ratio_max=1., p90_individual_term_ratio_max=1.,
                 minimum_relative_clip_multiplier=.9)
    groups = defaultdict(list)
    for index, checkpoint in enumerate(checkpoints):
        for row in checkpoint["summary"]["candidates"]:
            if row["variant"] == "reference":
                groups[(row["scope"], row["scale"])].append((index, row))
    screened = defaultdict(list)
    for (scope, scale), rows in sorted(groups.items()):
        failures, medians = [], []
        if len(rows) != len(checkpoints):
            failures.append("scope_not_available_at_every_checkpoint")
        for i, row in rows:
            for name, metrics in row["critics"].items():
                ratio = metrics["ratio_to_td"]
                if ratio["finite_count"] != ratio["count"] or ratio["median"] is None:
                    failures.append(f"checkpoint_{i}/{name}:unreliable_gradient_ratio")
                    continue
                medians.append(ratio["median"])
                if not .1 <= ratio["median"] <= .3:
                    failures.append(f"checkpoint_{i}/{name}:median_outside_band")
                if ratio["p90"] > 1.:
                    failures.append(f"checkpoint_{i}/{name}:large_p90_ratio")
                terms = checkpoints[i]["summary"].get("critics", {}).get(name, {}).get("term_gradient_ratios_to_td", {})
                for term in ("mpr", "vfc", "curvature"):
                    metric = terms.get(term)
                    if metric is None or metric["finite_count"] != metric["count"] or metric["p90"] is None:
                        failures.append(f"checkpoint_{i}/{name}/{term}:unreliable_term_ratio")
                    elif scale * metric["p90"] > 1.:
                        failures.append(f"checkpoint_{i}/{name}/{term}:large_individual_term")
            clip = row["clip_multiplier_relative_to_baseline"]
            if clip["finite_count"] != clip["count"] or clip["min"] is None or clip["min"] < .9:
                failures.append(f"checkpoint_{i}:additional_global_clipping")
        score = max((abs(x - .2) for x in medians), default=math.inf)
        screened[scope].append(dict(scale=scale, eligible=not failures, reasons=failures,
                                    distance_from_target=score if math.isfinite(score) else None))
    selections = {}
    for scope, candidates in screened.items():
        eligible = [row for row in candidates if row["eligible"]]
        chosen = min(eligible, key=lambda row: (row["distance_from_target"], row["scale"])) if eligible else None
        scale = chosen["scale"] if chosen else None
        stronger = []
        if scale is not None:
            for index, checkpoint in enumerate(checkpoints):
                match = next((row for row in checkpoint["summary"]["candidates"]
                              if row["variant"] == "stronger_curvature"
                              and row["scope"] == scope and row["scale"] == scale), None)
                if match is not None:
                    stronger.append({"checkpoint_index": index, **match})
        selections[scope] = {
            "scale": scale,
            "reference_weights": dict(zip(("mpr", "vfc", "curvature"),
                                          [scale * x for x in (.1, .1, .01)])) if scale is not None else None,
            "status": "candidate_for_training_smoke" if chosen else "no_common_scale_in_tested_grid",
            "screen": candidates, "stronger_curvature_at_same_scale": stronger,
        }
    return {
        "kind": "gradient_scale_heuristic_not_return_optimization", "rules": rules,
        "selections": selections,
        "limitations": [
            "Gradient norm ratios do not equal Adam parameter update ratios.",
            "A mature checkpoint alone cannot calibrate learning from initialization.",
            "A stochastic prior bank is a proxy for historical training replay.",
            "Random directional curvature violations do not certify a negative definite Hessian.",
            "The thresholds are engineering starting points, not validated optimal PAVE weights.",
        ],
    }


def _number(value):
    return "unavailable" if value is None else f"{value:.4g}"


def render_report(result):
    lines = ["# Frozen PAVE gradient probe", "",
             "No optimizer or target updates were performed. Ratios compare PAVE with the saved, weighted TD objective.",
             "Timing covers diagnostic forwards and separate gradient measurements; it is not full-update training overhead.", ""]
    for index, checkpoint in enumerate(result["checkpoints"]):
        summary = checkpoint["summary"]
        lines.extend([f"## Checkpoint {index}: {checkpoint['checkpoint']['path']}", "",
                      f"Source: `{checkpoint['bank']['provenance']['source_kind']}`. "
                      f"Batches: {summary['batch_count']}. SHA256: `{checkpoint['checkpoint']['sha256']}`.", "",
                      "| Critic | TD norm median | MPR/TD norm | VFC/TD norm | Curvature/TD norm | Curvature violations |",
                      "|---|---:|---:|---:|---:|---:|"])
        for name, row in summary["critics"].items():
            values = [row["td_gradient_norm"]["median"],
                      *[row["term_gradient_ratios_to_td"][t]["median"] for t in ("mpr", "vfc", "curvature")],
                      row["curvature_violation_fraction"]["median"]]
            lines.append(f"| {name} | " + " | ".join(map(_number, values)) + " |")
        lines.extend(["", "Term ratios above use the reference coefficients (0.1, 0.1, 0.01) at k=1.", "",
                      "| Variant | Scope | k | Critic | PAVE/TD median | P90 | Cosine median | Relative clip min |",
                      "|---|---|---:|---|---:|---:|---:|---:|"])
        for row in summary["candidates"]:
            for name, metrics in row["critics"].items():
                values = [metrics["ratio_to_td"]["median"], metrics["ratio_to_td"]["p90"],
                          metrics["cosine_with_td"]["median"], row["clip_multiplier_relative_to_baseline"]["min"]]
                lines.append(f"| {row['variant']} | {row['scope']} | {row['scale']:g} | {name} | "
                             + " | ".join(map(_number, values)) + " |")
        timing = summary["timing"]
        lines.extend(["", f"Median diagnostic time: ordinary losses {_number(timing['baseline_seconds']['median'])} s; "
                      f"PAVE {_number(timing['pave_seconds']['median'])} s.", ""])
    lines.extend(["## Common-scale screening", "",
                  "Reference ratio: (0.1, 0.1, 0.01). Every checkpoint/critic must have median ratio 0.1–0.3, "
                  "combined and individual-term P90 ≤ 1, and relative global clip multiplier ≥ 0.9. These are proposed heuristics.", ""])
    for scope, row in result["recommendations"]["selections"].items():
        lines.append(f"- {scope}: " + (f"k={row['scale']:g}; candidate for a training smoke."
                                      if row["scale"] is not None else "no common scale passed in the tested grid."))
    lines.extend(["", "The stronger-curvature variant is reported at the same k; its increased gradient is part of the comparison.", ""])
    lines.extend(f"- {text}" for text in result["recommendations"]["limitations"])
    return "\n".join(lines) + "\n"
