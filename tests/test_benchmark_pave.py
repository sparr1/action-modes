"""Complete-update timing has paired inputs and excludes validation from clocks."""

import math

import pytest
import torch

import benchmark_pave as benchmark
from tests.test_ambi_root_local_sac import _tiny_model
from utils.pave_identity import PAVE_COEFFICIENT_KEYS


def test_cases_and_rotation_are_balanced_and_auxiliary_only():
    orders = [benchmark.case_order(index) for index in range(4)]
    assert all(set(order) == set(benchmark.CASES) for order in orders)
    assert all(len({order[position] for order in orders}) == 4 for position in range(4))
    for case, cap in benchmark.CASES.items():
        controls = benchmark.case_overrides(case)
        assert controls["compile"] and controls["compile_strict"]
        assert not controls["outer_policy_diagnostics"]
        assert not controls["value_equivalence_diagnostics"]
        assert controls["outer_pave_batch_size"] == cap
        assert all(controls[key] == 0 for key in PAVE_COEFFICIENT_KEYS if key.startswith("outer_"))
        assert controls["aux_return_pave_mpr_coef"] == (0 if case == "baseline" else .005)
        assert controls["aux_return_pave_vfc_coef"] == (0 if case == "baseline" else .005)
        assert controls["aux_return_pave_curvature_coef"] == (0 if case == "baseline" else .0005)
    assert benchmark.batch_seeds(12345, 0, 110) == benchmark.batch_seeds(12345, 0, 110)
    assert set(benchmark.batch_seeds(12345, 0, 110)).isdisjoint(benchmark.batch_seeds(12345, 1, 110))


def test_tiny_complete_updates_include_optimizers_and_keep_checks_outside_timing(monkeypatch):
    model = _tiny_model(aux_return_mode="return_actor", ent_coef="auto_0.1",
                        aux_return_ent_coef="auto_0.1", inner_rounds=1,
                        inner_updates_per_round=1, aux_return_pave_mpr_coef=.005,
                        aux_return_pave_vfc_coef=.005, aux_return_pave_curvature_coef=.0005)
    try:
        agent = model.agent
        h, b = agent.cfg.train_unroll_horizon, agent.cfg.batch_size
        generator = torch.Generator().manual_seed(9)
        batches = [(torch.randn(h + 1, b, 3, generator=generator),
                    torch.randn(h, b, 1, generator=generator).tanh(),
                    torch.randn(h, b, 1, generator=generator), torch.zeros(h, b, 1)) for _ in range(3)]
        events = []
        update = agent.update
        check_metrics = benchmark._check_metrics
        check_parameters = benchmark._check_parameters

        def measured_update(buffer):
            events.append("update")
            return update(buffer)

        def metrics(info):
            events.append("metrics")
            return check_metrics(info)

        def parameters(value):
            events.append("parameters")
            return check_parameters(value)

        def clock():
            events.append("clock")
            return len(events) / 1000.

        monkeypatch.setattr(agent, "update", measured_update)
        monkeypatch.setattr(benchmark, "_check_metrics", metrics)
        monkeypatch.setattr(benchmark, "_check_parameters", parameters)
        result = benchmark.time_updates(agent, batches, warmup_updates=1,
                                        synchronize=lambda: events.append("sync"),
                                        require_compilation=False, clock=clock)
        assert events.count("update") == 3
        clocks = [index for index, event in enumerate(events) if event == "clock"]
        assert len(clocks) == 4
        for start, stop in zip(clocks[::2], clocks[1::2]):
            assert events[start + 1:stop] == ["update", "sync"]
        assert result["num_updates_delta"] == result["aux_num_updates_delta"] == 3
        assert result["summary"]["count"] == 2
        assert result["memory"] is None
        for optimizer in (agent.optim, agent.pi_optim, agent.ent_coef_optim,
                          agent.aux_return.pi_optim, agent.aux_return.ent_coef_optim):
            assert optimizer.state
            assert all(state["step"].item() == 3 for state in optimizer.state.values())
    finally:
        model.env.close()
        model.close()


def test_summary_uses_per_repeat_paired_baseline():
    rows = []
    for repeat, baseline in enumerate((10., 20.)):
        for index, case in enumerate(benchmark.CASES):
            values = [baseline * (index + 1)] * 3
            rows.append({"case": case, "repeat": repeat, "timings_ms": values,
                         "summary": benchmark.summarize_ms(values)})
    result = benchmark.summarize_cases(rows)
    assert result["aux_256"]["paired_median_ratios_to_baseline"] == [3., 3.]
    assert result["aux_256"]["median_paired_ratio_to_baseline"] == 3.
    assert result["baseline"]["count"] == 6
    assert result["baseline"]["median_ms"] == 15.


@pytest.mark.parametrize("values", [[], [math.nan], [0.], [-1.]])
def test_invalid_timing_samples_rejected(values):
    with pytest.raises(ValueError):
        benchmark.summarize_ms(values)


def test_resident_batch_preserves_optional_behavior_call_contract():
    values = tuple(torch.zeros(1) for _ in range(4))
    buffer = benchmark._ResidentBatch(values)
    assert buffer.sample() == (*values, None)
    assert buffer.sample(include_behavior_policy=True) == (*values, None, None, None, None)


def test_cli_rejects_cpu_before_output(tmp_path):
    output = tmp_path / "result"
    assert benchmark.main(["--checkpoint", "missing.pt", "--bank", "missing.npz",
                           "--output-dir", str(output), "--device", "cpu"]) == 1
    assert not output.exists()
