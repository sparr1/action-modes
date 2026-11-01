"""The MPR throughput comparison may only change cadence, speed and smoke controls."""
import json
from types import SimpleNamespace

import pytest

from tests.benchmark_pave_graph import ROOT, training_recipe


@pytest.mark.parametrize("interval", [1, 3])
def test_source_mpr_recipe_preserves_scientific_workload(interval):
    path = ROOT / "configs/research/ambi_mpr_hydra_speed.json"
    source = json.loads(path.read_text())
    _, manifest, run, _ = training_recipe(SimpleNamespace(
        source_config=path, interval=interval, term="mpr", graph=True,
        warmup=12, decisions=192,
    ))
    assert run["seed"] == source["seed"] == 55
    assert manifest["env_params"] == {"task": "humanoid-walk", "obs": "state", "render_mode": None}
    assert run["total_steps"] == 705
    expected = {**source["alg_params"], "seed_steps": 500, "pretrain_steps": 2,
                "buffer_size": 4096, "inner_solve_interval": interval,
                "inner_cuda_graph": True, "outer_pave_cuda_graph": True}
    assert run["alg_params"] == expected
    assert not expected["wandb"] and expected["wandb_mode"] == "disabled"
    assert expected["outer_pave_mpr_coef"] == .004
    assert expected["outer_pave_batch_size"] is None
