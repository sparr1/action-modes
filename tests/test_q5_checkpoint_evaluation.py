"""Frozen consecutive evaluation keeps the saved Q5 ensemble and reductions."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

from slurm.dog_stand_mrsq import REDUCTIONS
from slurm import graph_checkpoint_campaign as campaign
from utils.ambi_research import load_preset_matrix, resolve_preset


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs/research/ambi_checkpoint_graph_eval_consecutive.json"


def test_saved_q5_controller_and_prior_preserve_architecture_and_reductions():
    saved = json.loads((ROOT / "configs/dmcontrol/algs/ambi_humanoid_walk_mrsq_q5_seed55.json").read_text())
    context = SimpleNamespace(
        source=Path("metadata.json"),
        trial_run_params={**saved, "env": "DMControl-v0"},
        experiment_params={"env_params": {"task": "humanoid-walk", "obs": "state", "render_mode": None}},
    )
    matrix = load_preset_matrix(MATRIX)
    for selector in campaign.SELECTORS:
        result = resolve_preset(MATRIX, selector, matrix=matrix, checkpoint_context=context)
        params = result["algorithm_config"]["alg_params"]
        assert params["num_q"] == params["q_pair_size"] == 5
        for key, value in REDUCTIONS.items():
            assert params[key] == value
        if selector == "controller/graph":
            for key, value in saved["alg_params"].items():
                if key not in matrix["shared_alg_params"]:
                    assert params[key] == value
        assert result["evaluation"]["episode_sampling"] == "consecutive"
        assert result["evaluation"]["num_episodes"] == 10
        assert result["evaluation"]["environment_seed"] == 55
        assert result["evaluation"]["controller_seed"] == 55
        assert result["evaluation"]["max_steps"] == 500


def test_new_hydra_owner_cannot_replace_existing_planned_owner(tmp_path, monkeypatch):
    args = SimpleNamespace(
        campaign=tmp_path, source_run="source", training_root=Path("training"), config_name="q5",
        max_step=1_000_000, checkpoint_every=50_000, descending=False,
        episode_sampling="consecutive", owner="hydra-rgao48",
    )
    settings = campaign.evaluation_settings("consecutive")
    previous = dict(source_run="source", training_root="training", config_name="q5", planned=True,
                    checkpoints=[{"step": n} for n in campaign.checkpoint_steps(1_000_000)],
                    matrix_sha256=settings["matrix_sha256"])
    monkeypatch.setattr(campaign, "git", lambda *args: "")
    monkeypatch.setattr(campaign, "load_campaign", lambda path: previous)
    assert campaign.campaign_owner(previous) == "oscar-rgao48"
    assert campaign.campaign_owner({"owner": "hydra-rgao48"}) == "hydra-rgao48"
    with pytest.raises(ValueError, match="differs from the queued"):
        campaign.prepare(args)


@pytest.mark.parametrize("option,expected", [([], "consecutive"),
                                            (["--episode-sampling", "independent_seeds"], "independent_seeds")])
def test_campaign_cli_defaults_to_consecutive_with_explicit_legacy_opt_in(monkeypatch, option, expected):
    captured = []
    monkeypatch.setattr(campaign, "plan", captured.append)
    monkeypatch.setattr(sys, "argv", ["campaign", "plan", "--campaign", "campaign", "--training-root", "training",
                                    "--source-run", "source", "--config-name", "q5", "--max-step", "1000000",
                                    "--registry", "registry", *option])
    campaign.main()
    assert captured[0].episode_sampling == expected
    assert captured[0].owner == "oscar-rgao48"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires scheduled CUDA GPU")
def test_cuda_q5_humanoid_mean_graph_parity(monkeypatch):
    """Exercise all five heads, Humanoid dimensions, mean actions, RNG and frozen state."""
    from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
    from tests import test_frozen_graph_evaluation as checks
    original = checks.make_agent

    def make_agent(entropy, scope, update, *, device, compiled):
        cfg = deepcopy(original(entropy, scope, update).cfg)
        cfg.action_dim = 21
        cfg.action_dims = [21]
        cfg.obs_shape = {"state": (67,)}
        cfg.obs_shapes = [{"state": (67,)}]
        cfg.ent_coef = cfg.inner_temperature = .0021
        cfg.target_entropy = cfg.inner_target_entropy = -21.
        cfg.num_q = cfg.q_pair_size = 5
        for key, value in REDUCTIONS.items():
            setattr(cfg, key, value)
        cfg.device = device
        cfg.compile = cfg.compile_strict = compiled
        agent = AMBITDMPC2Agent(cfg)
        agent.model.eval()
        agent.actor_loss_scale.fill_(4.)
        return agent

    monkeypatch.setattr(checks, "make_agent", make_agent)
    checks.test_cuda_mean_graph_matches_control_and_resets_episode_rng("policy_sample")
