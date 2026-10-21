"""Retrace publication pins its implementation without splitting older curves."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest

from utils import eval_series_data as data


ROOT = Path(__file__).resolve().parents[1]
ALGORITHM = "AMBITDMPC2/AMBITDMPC2"


def _head():
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def _config():
    return dict(inner_operator="sac", inner_sac_return_estimator="retrace",
                inner_retrace_lambda=1., inner_rollout_horizon=3, inner_batch_size=256,
                inner_retrace_batch_trajectories=86, inner_entropy_enabled=True,
                inner_outer_policy_kl_coef=.023066775873303413)


def test_retrace_mixin_is_hashed_only_for_the_active_estimator(monkeypatch):
    commit = _head()
    data.scientific_identity.cache_clear()
    baseline = {controller: data.scientific_identity(ALGORITHM, controller, commit)
                for controller in ("prior", "sac", "mppi")}
    retrace = data.scientific_identity(ALGORITHM, "sac", commit, return_estimator="retrace")
    assert retrace != baseline["sac"]
    original = data._git_output
    seen = []

    def changed(root, *args):
        output = original(root, *args)
        if args[0] == "show":
            seen.append(args[1])
            if args[1].endswith(":RL/tdmpc2_core/inner_retrace.py"):
                output += b"\n_RETRACE_FINGERPRINT_TEST = 1\n"
        return output

    monkeypatch.setattr(data, "_git_output", changed)
    data.scientific_identity.cache_clear()
    try:
        for controller, expected in baseline.items():
            assert data.scientific_identity(ALGORITHM, controller, commit) == expected
        assert not any(path.endswith(":RL/tdmpc2_core/inner_retrace.py") for path in seen)
        assert data.scientific_identity(ALGORITHM, "sac", commit,
                                        return_estimator="retrace") != retrace
        assert any(path.endswith(":RL/tdmpc2_core/inner_retrace.py") for path in seen)
        assert any(path.endswith(":RL/tdmpc2_core/common/retrace.py") for path in seen)
    finally:
        data.scientific_identity.cache_clear()


def test_planner_identity_retains_lambda_and_resolved_trajectory_batch():
    config = _config()
    identity = data.planner_identity(config, {}, ALGORITHM, "tanh_sample")
    settings = identity["settings"]
    assert settings["inner_sac_return_estimator"] == "retrace"
    assert settings["inner_retrace_lambda"] == 1.
    assert settings["inner_retrace_batch_trajectories"] == 86
    implicit = deepcopy(config)
    implicit.pop("inner_retrace_batch_trajectories")
    assert data.planner_identity(implicit, {}, ALGORITHM, "tanh_sample") == identity
    for key, value in (("inner_retrace_lambda", .5), ("inner_retrace_batch_trajectories", 85)):
        changed = {**config, key: value}
        assert data.planner_identity(changed, {}, ALGORITHM, "tanh_sample") != identity
    one_step = {**config, "inner_sac_return_estimator": "one_step"}
    absent = {key: value for key, value in one_step.items()
              if key not in {"inner_sac_return_estimator", "inner_retrace_lambda",
                             "inner_retrace_batch_trajectories"}}
    assert data.planner_identity(one_step, {}, ALGORITHM, "tanh_sample") == data.planner_identity(
        absent, {}, ALGORITHM, "tanh_sample")


def test_preflight_and_bundle_normalization_use_the_same_retrace_science(tmp_path, monkeypatch):
    config = _config()
    checkpoint = dict(sha256="a" * 64, source_run="entity/project/source", source_run_verified=True,
                      metadata={"checkpoint": {"step": 500000}})
    protocol = dict(environment={"id": "test"}, action_rule="tanh_sample",
                    controller_seed=55, seed_scheme="sha256-v1", max_steps=500)
    resolved = {"algorithm_config": {"alg": ALGORITHM, "alg_params": config}}
    code = {"commit": _head(), "dirty": False}
    path = tmp_path / "manifest.json"
    monkeypatch.setattr(data, "resolved_checkpoint_config", lambda *args, **kwargs: config)
    preflight = data.identity_for_ambi_checkpoint(checkpoint, resolved, protocol, [101], code, path=path)
    episode = dict(seed=101, length=500, terminated=False, truncated=True,
                   truncated_by_evaluator=False, model_metrics={})
    episode["return"] = 200.
    result = dict(resolved_config=config, environment_seeds=[101], episodes=[episode],
                  outer_state_unchanged=True, outer_updates_before=100, outer_updates_after=100)
    manifest = dict(schema_version=1, status="complete", checkpoint=checkpoint, protocol=protocol,
                    code=code, runs=[dict(config=resolved["algorithm_config"], status="complete",
                                         selector="retrace/both", kind="episodes", episodes=[episode],
                                         result=result, trace_files=[])])
    path.write_text(json.dumps(manifest))
    record, = data.normalize_bundle(path)
    assert record["identity"] == preflight
    assert preflight["science"] == data.scientific_identity(
        ALGORITHM, "sac", code["commit"], return_estimator="retrace")
    assert preflight["science"] != data.scientific_identity(ALGORITHM, "sac", code["commit"])


def test_retrace_work_totals_are_exact_and_absent_for_old_records():
    episodes = []
    for seed in (101, 102):
        episode = dict(seed=seed, length=500, model_metrics={
            "inner_retrace_trajectory_draws": 128 * 86,
            "inner_retrace_critic_rows": 128 * 86 * 3,
            "inner_retrace_effective_trace_length": 2.,
        })
        episode["return"] = float(seed)
        episodes.append(episode)
    metrics = data._metrics(episodes)
    assert metrics["work/retrace_trajectory_draws"] == 2 * 500 * 128 * 86
    assert metrics["work/retrace_critic_rows"] == 2 * 500 * 128 * 86 * 3
    assert metrics["diagnostics/inner_retrace_effective_trace_length_mean"] == 2.
    for episode in episodes:
        episode["model_metrics"] = {}
    assert not any("retrace" in key for key in data._metrics(episodes))


@pytest.mark.parametrize("controller", ["prior", "mppi"])
def test_inactive_controllers_cannot_claim_retrace_science(controller):
    with pytest.raises(ValueError, match="AMBI SAC"):
        data.scientific_identity(ALGORITHM, controller, _head(), return_estimator="retrace")
