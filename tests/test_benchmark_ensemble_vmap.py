"""The head-vectorization timing CLI preserves pairing and records honest scope."""

import copy
from contextlib import contextmanager
import json
import subprocess
from types import SimpleNamespace

import pytest

import benchmark_ensemble_vmap as benchmark
def _args(tmp_path, **overrides):
    args = benchmark.build_parser().parse_args([
        "--checkpoint", str(tmp_path / "model"), "--bank", str(tmp_path / "bank.npz"),
        "--output-dir", str(tmp_path / "output"),
    ])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _row(args, case, repeat, source):
    overrides = benchmark.case_overrides(case)
    config = {**overrides, "batch_size": 256, "train_unroll_horizon": 3, "num_q": 5}
    values = [10. * (list(benchmark.CASES).index(case) + 1) * (repeat + 1)] * args.updates
    with benchmark._backend_context(case) as backend:
        backend = dict(backend)
    return {
        "schema": "ambi-ensemble-vmap-complete-update-timing-case", "version": 1,
        "case": case, "repeat": repeat, "source_identity": source, "backend": backend,
        "overrides": overrides, "resolved_cfg": config,
        "resolved_cfg_sha256": benchmark.timing._fingerprint(config),
        "checkpoint": {"sha256": "checkpoint"}, "bank": {"sha256": "bank"},
        "runtime": {"gpu_uuid": "one_gpu", "compile_strict": True},
        "repeat_seed": (args.seed + 1_000_003 * repeat) % 2**32,
        "batch_seeds": benchmark.timing.batch_seeds(args.seed, repeat, args.warmup_updates + args.updates),
        "start_num_updates": 575000, "end_num_updates": 575000 + args.warmup_updates + args.updates,
        "warmup_updates": args.warmup_updates, "measured_updates": args.updates,
        "num_updates_delta": args.warmup_updates + args.updates,
        "aux_num_updates_delta": args.warmup_updates + args.updates,
        "compile_fallback": False, "parameters_finite_before_and_after": True,
        "all_update_metrics_finite": True, "pave_rows": 768 if benchmark.CASES[case]["pave_enabled"] else 0,
        "timings_ms": values, "summary": benchmark.timing.summarize_ms(values),
        "memory": {"peak_allocated_bytes": 2**20 * (repeat + 1)},
    }


def test_four_crossed_cases_have_balanced_rotation_and_only_pave_weights_differ(tmp_path):
    orders = [benchmark.case_order(repeat) for repeat in range(4)]
    assert all(set(order) == set(benchmark.CASES) for order in orders)
    assert all(len({order[position] for order in orders}) == 4 for position in range(4))
    for pave in (False, True):
        suffix = "pave" if pave else "no_pave"
        controls = benchmark.case_overrides("loop_" + suffix)
        assert controls == benchmark.case_overrides("vmap_" + suffix)
        assert controls["outer_pave_batch_size"] is None
        assert controls["compile"] and controls["compile_strict"]
        assert controls["aux_return_pave_mpr_coef"] == (.005 if pave else 0.)
        assert controls["aux_return_pave_vfc_coef"] == (.005 if pave else 0.)
        assert controls["aux_return_pave_curvature_coef"] == (.0005 if pave else 0.)
        assert controls["outer_pave_obs_noise_std"] == .01
        assert controls["outer_pave_curvature_margin"] == 1.
        assert all(controls[f"outer_pave_{term}_coef"] == 0 for term in ("mpr", "vfc", "curvature"))
    args = _args(tmp_path)
    assert (args.repeats, args.warmup_updates, args.updates) == (4, 10, 100)
    with pytest.raises(ValueError, match="Unknown"):
        benchmark.case_overrides("unknown")


def test_source_identity_includes_original_and_new_timing_and_vmap_implementation():
    source = benchmark._source()
    assert {"benchmark_pave.py", "benchmark_ensemble_vmap.py", "utils/ensemble_vmap_benchmark.py"} <= source["files"].keys()
    assert source["sha256"] == benchmark.timing._fingerprint(source["files"])


def test_pairing_validation_and_summaries(tmp_path):
    args = _args(tmp_path, repeats=2, updates=3, warmup_updates=1)
    rows = [_row(args, case, repeat, {}) for repeat in range(args.repeats) for case in benchmark.case_order(repeat)]
    benchmark.validate_results(rows, args, {})
    summary, comparisons = benchmark.summarize_cases(rows)
    assert summary["loop_no_pave"]["count"] == 6
    assert summary["loop_no_pave"]["median_ms"] == 15.
    assert summary["vmap_pave"]["max_peak_allocated_bytes"] == 2 * 2**20
    assert comparisons["vmap_vs_loop_without_pave"]["paired_median_ratios"] == [2., 2.]
    assert comparisons["vmap_vs_loop_with_pave"]["median_paired_ratio"] == pytest.approx(4 / 3)
    assert comparisons["pave_overhead_loop"]["median_paired_ratio"] == 3.
    assert comparisons["pave_overhead_vmap"]["median_paired_ratio"] == 2.


@pytest.mark.parametrize("corruption", [
    lambda rows: rows.pop(),
    lambda rows: rows.append(copy.deepcopy(rows[0])),
    lambda rows: rows[0]["batch_seeds"].__setitem__(0, -1),
    lambda rows: rows[0].__setitem__("repeat_seed", -1),
    lambda rows: rows[0].__setitem__("start_num_updates", -1),
    lambda rows: rows[0].__setitem__("checkpoint", {"sha256": "different"}),
    lambda rows: rows[0].__setitem__("runtime", {"gpu_uuid": "other_gpu"}),
    lambda rows: rows[0].__setitem__("source_identity", {"different": True}),
    lambda rows: rows[0]["resolved_cfg"].__setitem__("num_q", 2),
    lambda rows: rows[0]["overrides"].__setitem__("outer_pave_batch_size", 64),
    lambda rows: rows[0]["resolved_cfg"].__setitem__("compile_strict", False),
    lambda rows: rows[0].__setitem__("compile_fallback", True),
    lambda rows: rows[0].__setitem__("parameters_finite_before_and_after", False),
    lambda rows: rows[0].__setitem__("aux_num_updates_delta", 0),
    lambda rows: rows[0].__setitem__("end_num_updates", 0),
    lambda rows: rows[0].__setitem__("measured_updates", 1),
    lambda rows: rows[0].__setitem__("pave_rows", 64),
    lambda rows: rows[0]["timings_ms"].__setitem__(0, float("nan")),
    lambda rows: rows[0]["summary"].__setitem__("median_ms", 999.),
    lambda rows: rows[0]["backend"].__setitem__("view_fusion", True),
    lambda rows: rows[1]["backend"].__setitem__("context_seams_installed", False),
    lambda rows: rows[1]["backend"].__setitem__("parameter_stacking", "persistent"),
])
def test_invalid_or_unpaired_comparisons_are_rejected(tmp_path, corruption):
    args = _args(tmp_path, repeats=1)
    rows = [_row(args, case, 0, {}) for case in benchmark.CASES]
    corruption(rows)
    with pytest.raises(ValueError):
        benchmark.validate_results(rows, args, {})


def _mock_children(monkeypatch, args, *, fail_at=None):
    commands = []
    monkeypatch.setattr(benchmark.timing, "_record", lambda _: {})
    monkeypatch.setattr(benchmark, "_source", lambda: {"sha256": "source"})

    def run(command, **kwargs):
        commands.append(command)
        assert kwargs["check"] and kwargs["timeout"] > 0
        assert kwargs["stderr"] is subprocess.STDOUT
        if fail_at == len(commands):
            raise subprocess.CalledProcessError(1, command)
        worker = benchmark.build_parser().parse_args(command[2:])
        worker.output_dir.mkdir()
        row = _row(args, worker.worker_case, worker.worker_repeat, {"sha256": "source"})
        benchmark.timing._save(worker.output_dir / "result.json", row)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(benchmark.subprocess, "run", run)
    return commands


def test_parent_launches_fresh_workers_and_publishes_verified_report(tmp_path, monkeypatch):
    args = _args(tmp_path)
    commands = _mock_children(monkeypatch, args)
    result = benchmark.run_benchmark(args)
    assert result["complete"] and len(commands) == 16
    workers = [benchmark.build_parser().parse_args(command[2:]) for command in commands]
    assert [(worker.worker_repeat, worker.worker_case) for worker in workers] == [
        (repeat, case) for repeat in range(4) for case in benchmark.case_order(repeat)]
    assert len({worker.output_dir for worker in workers}) == 16
    assert all(worker.seed == args.seed and worker.checkpoint == args.checkpoint and worker.bank == args.bank
               for worker in workers)
    assert (args.output_dir / "results.json").exists()
    report = (args.output_dir / "report.md").read_text()
    assert "not end-to-end training throughput" in report
    assert "distribution-equivalent" in report and "not independent trials" in report
    assert result["protocol"]["vmap_randomness"] == "different"
    assert result["protocol"]["view_fusion"] is False
    with pytest.raises(FileExistsError):
        benchmark.run_benchmark(args)


def test_failure_preserves_completed_worker_results_and_partial_aggregate(tmp_path, monkeypatch):
    args = _args(tmp_path)
    _mock_children(monkeypatch, args, fail_at=3)
    with pytest.raises(subprocess.CalledProcessError):
        benchmark.run_benchmark(args)
    partial = json.loads((args.output_dir / "partial-results.json").read_text())
    failure = json.loads((args.output_dir / "failure.json").read_text())
    assert not partial["complete"] and len(partial["results"]) == 2
    assert failure["completed_cases"] == 2
    assert not (args.output_dir / "results.json").exists()
    assert len(list(args.output_dir.glob("repeat-*/*result.json"))) == 2


@pytest.mark.parametrize("extra", [["--device", "cpu"], ["--updates", "0"],
                                  ["--repeats", "0"], ["--warmup-updates", "0"],
                                  ["--timeout-seconds", "nan"]])
def test_cli_rejects_invalid_protocol_before_output(tmp_path, extra):
    assert benchmark.main(["--checkpoint", "missing.pt", "--bank", "missing.npz",
                           "--output-dir", str(tmp_path / "output"), *extra]) == 1
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("fail_during_timing", [False, True])
def test_worker_backend_context_covers_construction_timing_and_cleanup(tmp_path, monkeypatch, fail_during_timing):
    args = _args(tmp_path, worker_case="vmap_pave", worker_repeat=0, updates=1, warmup_updates=1)
    events = []
    active = False

    @contextmanager
    def context(_):
        nonlocal active
        active = True
        events.append("enter")
        try:
            yield {"test_backend": True}
        finally:
            events.append("exit")
            active = False

    def construct(*_):
        assert active
        events.append("construct")
        agent = SimpleNamespace(cfg=SimpleNamespace(batch_size=256, train_unroll_horizon=3), device="cuda")
        return SimpleNamespace(agent=agent, cfg=agent.cfg), {}

    def measure(*_, **__):
        assert active
        events.append("measure")
        if fail_during_timing:
            raise RuntimeError("timing failed")
        return {"timings_ms": [1.]}

    def close(*_, **__):
        assert active
        events.append("close")

    record = {"resolved": {"algorithm_config": {"alg_params": {}}},
              "path": args.checkpoint, "bank_path": args.bank,
              "context": SimpleNamespace(source=tmp_path / "metadata")}
    monkeypatch.setattr(benchmark, "_backend_context", context)
    monkeypatch.setattr(benchmark, "_source", lambda: {})
    monkeypatch.setattr(benchmark.timing, "_record", lambda _: record)
    monkeypatch.setattr(benchmark.timing, "_file_sha256", lambda _: "hash")
    monkeypatch.setattr(benchmark.timing, "_make_env", lambda _: object())
    monkeypatch.setattr(benchmark.timing, "_initialize_probe_model", construct)
    monkeypatch.setattr(benchmark.timing, "_close_resources", close)
    monkeypatch.setattr(benchmark.timing, "load_probe_bank", lambda _: type("Bank", (), {"__len__": lambda _: 5000, "provenance": {}})())
    monkeypatch.setattr(benchmark.timing, "_validate_bank", lambda *_: None)
    monkeypatch.setattr(benchmark.timing, "sample_probe_windows", lambda *_, **__: ())
    monkeypatch.setattr(benchmark.timing, "time_updates", measure)
    monkeypatch.setattr(benchmark.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(benchmark.torch.cuda, "manual_seed_all", lambda _: None)
    monkeypatch.setattr(benchmark.torch.cuda, "get_device_properties", lambda _: SimpleNamespace(
        name="test GPU", total_memory=1000, major=1, minor=0))
    if fail_during_timing:
        with pytest.raises(RuntimeError, match="timing failed"):
            benchmark.run_worker(args)
        assert json.loads((args.output_dir / "failure.json").read_text())["complete"] is False
        assert not (args.output_dir / "result.json").exists()
    else:
        benchmark.run_worker(args)
    assert events == ["enter", "construct", "measure", "close", "exit"]
