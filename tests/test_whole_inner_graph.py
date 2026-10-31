"""Guard unsupported protocols, then gate CUDA capture on numerical state parity."""

from copy import copy
import os
from types import SimpleNamespace

import pytest
import torch

from RL.tdmpc2_core.whole_inner_graph import WholeInnerGraph, validate_graph_workload
from tests.benchmark_whole_inner_graph import benchmark, make_agent


@pytest.fixture
def engine():
    return make_agent(device="cpu", tiny=True, compile_enabled=False).inner_engine


@pytest.mark.parametrize("key,value", [
    ("episodic", True), ("inner_critic_scope", "episode"),
    ("inner_critic_bernoulli_p", .5), ("inner_outer_replay_fraction", .25),
    ("inner_actor_initialization", "random"), ("inner_operator", "td3"),
])
def test_graph_rejects_unsupported_protocol_before_mutation(engine, key, value):
    engine.cfg = copy(engine.cfg)
    setattr(engine.cfg, key, value)
    rng = engine.rng
    with pytest.raises(ValueError, match=key):
        WholeInnerGraph(engine, torch.zeros(1, engine.cfg.latent_dim))
    assert engine.rng is rng
    assert engine.state.actor is None


def test_cpu_rejection_and_benchmark_protocol(engine):
    validate_graph_workload(engine)
    with pytest.raises(ValueError, match="CUDA"):
        WholeInnerGraph(engine, torch.zeros(1, engine.cfg.latent_dim))
    assert engine.cfg.wandb is False
    assert engine.cfg.inner_critic_updates_per_action == 4
    assert engine.cfg.inner_actor_updates_per_action == 2


@pytest.mark.skipif(os.environ.get("AMBI_RUN_INNER_GRAPH_CUDA") != "1" or not torch.cuda.is_available(),
                    reason="Requires the explicit scheduled CUDA gate")
@pytest.mark.parametrize("eager", [True, False])
def test_whole_graph_matches_uncaptured_control_on_fresh_roots_and_changed_priors(tmp_path, eager):
    torch._dynamo.reset()
    benchmark(SimpleNamespace(tiny=True, eager=eager, threads=1, warmup=4,
                              steps=2, repeats=1, output=tmp_path / "result.json"))
