"""The retired temporal recipe remains readable but cannot resume exactly."""

import copy
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

import main as training_main
import render_checkpoint as renderer
import utils.resume_identity as resume_identity
from RL.AMBITDMPC2 import AMBITDMPC2
from RL.TDMPC2 import TDMPC2Baseline
from tests.test_tdmpc2_horizon_normalization import _tiny_network_params
from utils.resume_training import (
    ResumeIncompatibilityError,
    TrainingResumeSession,
)


@pytest.mark.parametrize("algorithm,wrapper", [
    ("AMBITDMPC2/AMBITDMPC2", AMBITDMPC2),
    ("TDMPC2/TDMPC2Baseline", TDMPC2Baseline),
])
def test_historical_temporal_sidecar_loads_weights_for_evaluation(
    tmp_path, algorithm, wrapper
):
    params = {
        **_tiny_network_params(),
        "train_unroll_horizon": 6,
        "outer_planning_horizon": 2,
        "rho": 0.7,
        "wandb": False,
    }
    if wrapper is AMBITDMPC2:
        params["inner_operator"] = "none"
    run_params = {
        "name": "HistoricalTemporal",
        "alg": algorithm,
        "env": "Pendulum-v1",
        "device": "cpu",
        "seed": 3,
        "total_steps": 12,
        "alg_params": params,
    }
    experiment_params = {"env_params": {"max_episode_steps": 5}}
    source_env = gym.make("Pendulum-v1", max_episode_steps=5)
    restored_env = gym.make("Pendulum-v1", max_episode_steps=5)
    source = restored = None
    try:
        source = wrapper(
            "HistoricalTemporal", source_env, params, run_params, experiment_params
        )
        # Distinguish saved weights from fresh deterministic initialization.
        with torch.no_grad():
            next(source.agent.model.parameters()).add_(0.125)
        checkpoint = Path(source.save(str(tmp_path), "historical.pt"))
        source.flush_checkpoints()
        sidecar = Path(f"{checkpoint}.metadata.json")
        historical = json.loads(sidecar.read_text())
        historical["trial_run_params"]["alg_params"].update({
            "temporal_loss_normalization": "reference_weighted_mean",
            "temporal_loss_reference_horizon": 3,
        })
        historical["trial_run_params"]["resolved_runtime"] = {
            "temporal_loss": {
                "temporal_loss_normalization": "reference_weighted_mean",
                "temporal_loss_reference_horizon": 3,
                "rho": 0.7,
            }
        }
        sidecar.write_text(json.dumps(historical))
        saved_checkpoint = checkpoint.read_bytes()
        saved_sidecar = sidecar.read_bytes()

        context = renderer.resolve_render_context(checkpoint)
        backend = renderer._backend_for(algorithm)
        evaluation_params, evaluation_experiment = renderer._prepare_run_params(
            context, backend=backend, device="cpu", controller_seed=3
        )
        with pytest.warns(FutureWarning, match="Legacy temporal fields"):
            restored = renderer._initialize_model(
                checkpoint, evaluation_params, evaluation_experiment,
                restored_env, backend,
            )

        assert restored.cfg.rho == 0.7  # Explicit historical rho stays configurable.
        assert restored.cfg.temporal_loss_normalization == "divide_horizon"
        assert not hasattr(restored.cfg, "temporal_loss_reference_horizon")
        assert source.agent.model.state_dict().keys() == restored.agent.model.state_dict().keys()
        for name, expected in source.agent.model.state_dict().items():
            torch.testing.assert_close(
                restored.agent.model.state_dict()[name], expected, rtol=0, atol=0
            )
        observation, _ = restored_env.reset(seed=3)
        action, _ = restored.predict(observation, deterministic=True)
        assert np.isfinite(action).all()
        metadata = training_main._resolved_runtime_metadata(
            restored, trial_run_params=evaluation_params
        )
        assert metadata["temporal_loss"] == {
            "temporal_loss_normalization": "divide_horizon", "rho": 0.7,
        }
        assert "temporal_loss_reference_horizon" not in json.dumps(metadata)
        assert context.metadata == historical
        assert sidecar.read_bytes() == saved_sidecar
        assert checkpoint.read_bytes() == saved_checkpoint
    finally:
        for model in (source, restored):
            if model is not None:
                model._checkpoint_writer.shutdown()
                model.close()
        source_env.close()
        restored_env.close()


@pytest.mark.parametrize("algorithm", [
    "AMBITDMPC2/AMBITDMPC2", "TDMPC2/TDMPC2Baseline",
])
@pytest.mark.parametrize("changed", ["source", "temporal_rule", "rho"])
def test_exact_resume_rejects_temporal_migration(
    tmp_path, monkeypatch, algorithm, changed
):
    # Exercise the actual lineage preflight independently for source and
    # configuration changes, before loading any trainer/checkpoint tensors.
    source = {"commit": "historical", "dirty": False}
    monkeypatch.setattr(resume_identity, "source_identity", lambda _root: dict(source))
    monkeypatch.setattr(resume_identity, "dependency_identity", lambda: {"python": "test"})
    historical = {
        "alg": algorithm,
        "seed": 3,
        "alg_params": {
            "train_unroll_horizon": 6,
            "rho": 0.7,
            "temporal_loss_normalization": "reference_weighted_mean",
            "temporal_loss_reference_horizon": 3,
        },
    }

    def identity(params):
        return resume_identity.lineage_identity(
            trial_run_params=params, experiment_params={}, repo_root=tmp_path
        )

    old_identity = identity(historical)
    lineage = tmp_path / "old-lineage"
    options = dict(total_steps=12, checkpoint_minutes=5, drain_after_seconds=None)
    original = TrainingResumeSession.open(
        lineage, mode="new", scientific_identity=old_identity, **options
    )
    original.close()
    current = copy.deepcopy(historical)
    if changed == "source":
        source["commit"] = "canonical-temporal-weighting"
    elif changed == "temporal_rule":
        current["alg_params"].pop("temporal_loss_reference_horizon")
        current["alg_params"].pop("temporal_loss_normalization")
    else:
        current["alg_params"]["rho"] = 0.5
    new_identity = identity(current)
    assert old_identity["fingerprint"] != new_identity["fingerprint"]
    with pytest.raises(ResumeIncompatibilityError, match="differs from the lineage"):
        TrainingResumeSession.open(
            lineage, mode="required", scientific_identity=new_identity, **options
        )
    # The same changed contract is valid as an intentional new lineage.
    transferred = TrainingResumeSession.open(
        tmp_path / "new-lineage", mode="new",
        scientific_identity=new_identity, **options,
    )
    transferred.close()
