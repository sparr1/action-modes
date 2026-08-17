import csv
import json
from copy import deepcopy

import pytest

import plot_ambi_latency as plotter


def _conditions():
    return {
        (2, 32, 3, 4): "center_j2_n32_g4",
        (2, 32, 3, 0): "g_sweep_g0",
        (2, 32, 3, 2): "g_sweep_g2",
        (2, 32, 3, 8): "g_sweep_g8",
        (2, 32, 3, 16): "g_sweep_g16",
        (2, 8, 3, 4): "n_sweep_n8",
        (2, 16, 3, 4): "n_sweep_n16",
        (2, 64, 3, 4): "n_sweep_n64",
        (2, 128, 3, 4): "n_sweep_n128",
        (1, 32, 3, 4): "natural_j1",
        (4, 32, 3, 4): "natural_j4",
        (8, 32, 3, 4): "natural_j8",
        (1, 64, 3, 8): "matched_j1_n64_g8",
        (4, 16, 3, 2): "matched_j4_n16_g2",
        (8, 8, 3, 1): "matched_j8_n8_g1",
    }


def _stats(value, count=7):
    return {
        "count": count,
        "mean": value * 1.04,
        "std": value * 0.08,
        "p50": value,
        "p90": value * 1.12,
        "p95": value * 1.18,
        "min": value * 0.78,
        "max": value * 1.32,
    }


def _zero_stats(count=7):
    return {
        "count": count,
        "mean": 0.0,
        "std": 0.0,
        "p50": 0.0,
        "p90": 0.0,
        "p95": 0.0,
        "min": 0.0,
        "max": 0.0,
    }


def _cell(name, J, N, H, G, *, process_factor, count=7):
    latency = 0.002 + 0.00005 * (J * N * H) + 0.0004 * (J * G)
    latency *= process_factor
    phase_fractions = {
        "inner_setup_seconds": 0.10,
        "inner_rollout_seconds": 0.22,
        "inner_update_seconds": 0.38 if G else 0.0,
        "inner_execution_seconds": 0.05,
    }
    phases = {
        "inner_action_seconds": _stats(latency * 0.94, count),
        **{
            phase: (_stats(latency * fraction, count) if fraction else _zero_stats(count))
            for phase, fraction in phase_fractions.items()
        },
    }
    return {
        "name": name,
        "J": J,
        "N": N,
        "H": H,
        "G": G,
        "resolved_device": "cuda:0",
        "compile_enabled": True,
        "resolved_work": {
            "rollout_paths": J * N,
            "imagined_transitions": J * N * H,
            "update_slots": J * G,
            "inner_batch_size": 64,
            "replay_capacity": J * N * H,
        },
        "cold_call": {
            "wall_seconds": latency * 2,
            "phase_seconds": {},
            "work_counters": {},
            # A cold-only fallback is valid and must not reject steady-state data.
            "compile_fallbacks": {"inner_compile_actor_fallback": 1.0},
        },
        "warmup_calls": 3,
        "measurements": {
            "count": count,
            "wall_seconds": _stats(latency, count),
            "phase_seconds": phases,
            "work_counters": {"inner_model_steps": _stats(float(J * N * H), count)},
            "compile_fallbacks": {
                "inner_compile_actor_fallback": {
                    **_zero_stats(count),
                    "any": False,
                    "all": False,
                }
            },
            # Raw calls exist in real outputs, but are intentionally not used as
            # independent points by the synthesizer.
            "samples": {"wall_seconds": [latency] * count},
        },
        "validation": {
            "passed": True,
            "expected_counters": {"passed": True},
            "compile_fallbacks": {"passed": True},
        },
        "outer_state_unchanged": True,
    }


def _document(cell, *, selected_cell):
    return {
        "schema_version": 1,
        "benchmark": "ambi-inner-latency",
        "metadata": {
            "timestamp_utc": "2026-08-17T00:00:00Z",
            "git": {"commit": "abc123", "branch": "bench", "dirty": False},
            "hardware": {
                "hostname": "ignored-per-process-host",
                "python": "3.10.14",
                "torch": "2.7.0",
                "cuda_available": True,
                "cuda_version": "12.8",
                "cudnn_version": 90701,
                "requested_device": "cuda",
                "cuda_device": {
                    "index": 0,
                    "name": "NVIDIA L40S",
                    "total_memory_bytes": 1_000_000,
                    "capability": [8, 9],
                },
            },
            "checkpoint": {"path": "/checkpoints/frozen.pt", "size_bytes": 1234},
            "config_path": "/repo/configs/research/ambi_latency_benchmark.json",
            "algorithm_config_path": "/repo/configs/dmcontrol/algs/ambi.json",
            "process_id": 999,
            "benchmark_config_metadata": {"center": "center_j2_n32_g4"},
        },
        "settings": {
            "device": "cuda",
            "effective_device": "cuda",
            "cold_calls": 1,
            "warmup_calls": 3,
            "measured_calls": 7,
            "observation_bank_size": 4,
            "environment_seed": 55,
            "controller_seed": 55,
            "action_mode": "training",
            "include_samples": True,
            "device_override": "cuda",
            "H": 3,
            "B": 64,
            "blocks": 3,
            "block_order_seed": 20260817,
            "selected_cells": [selected_cell],
            "fresh_model_per_cell": True,
            "process_isolation": True,
            "collect_diagnostics": False,
            "wandb": False,
            "environment_reset_calls_per_cell": 4,
            "environment_step_calls": 0,
            "outer_update_calls": 0,
            "wandb_enabled": False,
            "observation_bank_sha256": "same-bank",
            "timing_contract": {
                "wall_seconds": "synchronized",
                "phase_seconds": "CUDA events",
                "cold_call": "fresh process",
            },
        },
        "cells": [cell],
        "counter_formulas": {"inner_model_steps": "J*N*H"},
        "validation": {"passed": True, "failed_cells": []},
    }


def _write_matrix(tmp_path, factors=(1.0, 4.0, 2.0)):
    paths = []
    documents = []
    for block, factor in enumerate(factors):
        block_dir = tmp_path / f"block_{block}"
        block_dir.mkdir(parents=True)
        for cell_index, ((J, N, H, G), name) in enumerate(_conditions().items()):
            cell = _cell(name, J, N, H, G, process_factor=factor)
            document = _document(cell, selected_cell=f"{J},{N},{G}")
            document["metadata"]["process_id"] = 1000 + block * 100 + cell_index
            document["metadata"]["timestamp_utc"] = (
                f"2026-08-17T00:{block:02d}:{cell_index:02d}Z"
            )
            path = block_dir / f"{name}.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            paths.append(path)
            documents.append((path, document))
        (block_dir / "COMPLETE").touch()
    return paths, documents


def test_aggregates_process_files_and_infers_all_views(tmp_path):
    paths, _ = _write_matrix(tmp_path)

    assert set(plotter._expand_inputs([str(tmp_path)])) == set(paths)

    dataset = plotter.build_dataset(paths, bootstrap_samples=200, bootstrap_seed=9)

    center = plotter.CellKey(J=2, N=32, H=3, G=4)
    unscaled = 0.002 + 0.00005 * 192 + 0.0004 * 8
    summary = dataset.summaries[center]
    assert dataset.reference == center
    assert summary.blocks == 3
    # This is the median of three process-level p50s (1x, 4x, 2x), not an
    # expansion or weighting of the seven raw action calls in each file.
    assert summary.wall["p50"].center == pytest.approx(unscaled * 2)
    assert len(summary.wall["p50"].values) == 3
    assert [key.G for key in dataset.views["G"]] == [0, 2, 4, 8, 16]
    assert [key.N for key in dataset.views["N"]] == [8, 16, 32, 64, 128]
    assert [key.J for key in dataset.views["J"]] == [1, 2, 4, 8]
    assert [(key.J, key.N, key.G) for key in dataset.views["matched_J"]] == [
        (1, 64, 8),
        (2, 32, 4),
        (4, 16, 2),
        (8, 8, 1),
    ]


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda doc: doc["metadata"]["checkpoint"].update(size_bytes=99), "metadata/settings differ"),
        (lambda doc: doc["settings"].update(action_mode="evaluation"), "action_mode"),
        (lambda doc: doc["metadata"]["hardware"]["cuda_device"].update(name="A6000"), "metadata/settings differ"),
        (lambda doc: doc["metadata"]["hardware"].update(torch="2.8.0"), "metadata/settings differ"),
        (lambda doc: doc["metadata"]["hardware"].update(cuda_version="12.9"), "metadata/settings differ"),
        (lambda doc: doc["metadata"]["hardware"].update(cudnn_version=99999), "metadata/settings differ"),
        (lambda doc: doc["metadata"]["git"].update(dirty=True), "git.dirty"),
        (lambda doc: doc["settings"].update(H=4), "metadata/settings differ"),
        (lambda doc: doc["settings"].update(B=128), "metadata/settings differ"),
        (lambda doc: doc["settings"].update(blocks=4), "metadata/settings differ"),
        (lambda doc: doc["settings"].update(block_order_seed=9), "metadata/settings differ"),
        (lambda doc: doc["settings"].update(process_isolation=False), "process_isolation"),
        (lambda doc: doc["validation"].update(passed=False), "validation.passed"),
        (lambda doc: doc["cells"][0]["resolved_work"].update(imagined_transitions=191), "J/N/H/G imply"),
        (lambda doc: doc["cells"][0].update(outer_state_unchanged=False), "outer_state_unchanged"),
        (lambda doc: doc["cells"][0]["validation"].update(passed=False), "validation.passed"),
        (
            lambda doc: doc["cells"][0].pop("validation"),
            r"cells\[0\].validation must be an object",
        ),
        (
            lambda doc: doc["cells"][0]["measurements"]["compile_fallbacks"]["inner_compile_actor_fallback"].update(max=1.0),
            "compile fallback",
        ),
    ],
)
def test_rejects_unfair_or_invalid_inputs(tmp_path, mutation, match):
    paths, documents = _write_matrix(tmp_path)
    path, document = documents[-1]
    mutation(document)
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(plotter.ValidationError, match=match):
        plotter.build_dataset(paths, bootstrap_samples=0)


def test_rejects_duplicate_and_nonfinite_json_keys(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version": 1, "schema_version": 1, "benchmark": "ambi-inner-latency"}',
        encoding="utf-8",
    )
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value": NaN}', encoding="utf-8")

    with pytest.raises(plotter.ValidationError, match="duplicate JSON key"):
        plotter.load_blocks([duplicate])
    with pytest.raises(plotter.ValidationError, match="non-finite JSON number"):
        plotter.load_blocks([nonfinite])


def test_rejects_partial_or_incomplete_process_matrix(tmp_path):
    paths, _ = _write_matrix(tmp_path)

    with pytest.raises(plotter.ValidationError, match="exact 15-cell design"):
        plotter.build_dataset(paths[:-1], bootstrap_samples=0)

    only_two_blocks = [path for path in paths if path.parent.name != "block_2"]
    with pytest.raises(plotter.ValidationError, match="block identities"):
        plotter.build_dataset(only_two_blocks, bootstrap_samples=0)

    without_g0 = [path for path in paths if path.stem != "g_sweep_g0"]
    with pytest.raises(plotter.ValidationError, match="exact 15-cell design"):
        plotter.build_dataset(without_g0, bootstrap_samples=0)


def test_requires_one_cell_file_and_unique_process_identity(tmp_path):
    paths, documents = _write_matrix(tmp_path)
    path, document = documents[-1]
    document["cells"].append(deepcopy(document["cells"][0]))
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(plotter.ValidationError, match="exactly one process-isolated cell"):
        plotter.build_dataset(paths, bootstrap_samples=0)

    paths, documents = _write_matrix(tmp_path / "second")
    first_identity = documents[0][1]["metadata"]
    path, document = documents[-1]
    for key in ("timestamp_utc", "process_id"):
        document["metadata"][key] = first_identity[key]
    document["metadata"]["hardware"]["hostname"] = first_identity["hardware"]["hostname"]
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(plotter.ValidationError, match="duplicate process artifact identity"):
        plotter.build_dataset(paths, bootstrap_samples=0)


def test_directory_inputs_require_complete_markers_but_explicit_files_do_not(tmp_path):
    paths, _ = _write_matrix(tmp_path)
    marker = tmp_path / "block_2" / "COMPLETE"
    marker.unlink()

    with pytest.raises(plotter.ValidationError, match="lack COMPLETE"):
        plotter._expand_inputs([str(tmp_path)])
    assert set(plotter._expand_inputs([str(path) for path in paths])) == set(paths)


def test_paired_phase_stack_is_exactly_additive(tmp_path):
    paths, documents = _write_matrix(tmp_path, factors=(1.0, 1.0, 1.0))
    center_documents = [
        (path, document)
        for path, document in documents
        if document["cells"][0]["name"] == "center_j2_n32_g4"
    ]
    fractions = ((0.9, 0.0), (0.0, 0.9), (0.5, 0.5))
    for (path, document), (setup, rollout) in zip(center_documents, fractions):
        measurements = document["cells"][0]["measurements"]
        latency = measurements["wall_seconds"]["p50"]
        phases = measurements["phase_seconds"]
        phases["inner_setup_seconds"] = _stats(latency * setup)
        phases["inner_rollout_seconds"] = _stats(latency * rollout)
        phases["inner_update_seconds"] = _zero_stats()
        phases["inner_execution_seconds"] = _zero_stats()
        path.write_text(json.dumps(document), encoding="utf-8")

    dataset = plotter.build_dataset(paths, bootstrap_samples=0)
    summary = dataset.summaries[plotter.CellKey(J=2, N=32, H=3, G=4)]

    assert sum(summary.phase_stack_seconds.values()) == pytest.approx(
        summary.wall["mean"].center
    )
    assert summary.phase_representative_block in {0, 1, 2}


def test_writes_csv_and_all_headless_plot_formats(tmp_path):
    pytest.importorskip("matplotlib")
    paths, _ = _write_matrix(tmp_path)
    dataset = plotter.build_dataset(paths, bootstrap_samples=20)
    output_dir = tmp_path / "plots"

    outputs = plotter.render_outputs(dataset, output_dir, prefix="synthetic", tail="p90")

    assert [path.name for path in outputs] == [
        "synthetic_scaling.png",
        "synthetic_scaling.pdf",
        "synthetic_phases.png",
        "synthetic_phases.pdf",
        "synthetic_summary.csv",
    ]
    assert all(path.stat().st_size > 100 for path in outputs)
    with outputs[-1].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 15
    center_row = next(row for row in rows if row["name"] == "center_j2_n32_g4")
    assert center_row["process_blocks"] == "3"
    assert center_row["views"] == "G;N;J;matched_J"
    assert float(center_row["phase_unattributed_paired_ms"]) >= 0
