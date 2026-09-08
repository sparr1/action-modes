"""New backbone evaluation preserves the established controller and pairing."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tests.test_ambi_inner_benchmark_launcher import launch_env
from tests.test_checkpoint_research_configs import checkpoint_context, _build_cfg
from utils.ambi_research import load_preset_matrix, normalize_selectors, resolve_preset
from utils.eval_series import backbone_display_label, concise_curve_label, evaluation_run_name


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "slurm/run_ambi_h5h10_step_rounds_oscar.sbatch"
SOURCES = {5: "rwgao_b-brown-university/ambi/8vlf8z3w",
           10: "rwgao_b-brown-university/ambi/gky2rxlf"}


@pytest.mark.parametrize("horizon", [5, 10])
def test_new_matrices_preserve_original_science_and_saved_outer_config(checkpoint_context, horizon):
    originals = {kind: load_preset_matrix(ROOT / "configs/research" / name) for kind, name in (
        ("inner", "ambi_humanoid_inner_step_rounds.json"),
        ("prior", "ambi_humanoid_inner_benchmark.json"))}
    params = checkpoint_context.trial_run_params["alg_params"]
    params.pop("horizon", None)
    params.update(train_unroll_horizon=horizon, outer_planning_horizon=3,
                  temporal_loss_normalization="reference_weighted_mean",
                  temporal_loss_reference_horizon=3)
    before = deepcopy(checkpoint_context)
    for kind, filename in (("inner", f"ambi_humanoid_h{horizon}_inner_step_rounds.json"),
                           ("prior", f"ambi_humanoid_h{horizon}_prior_benchmark.json")):
        path = ROOT / "configs/research" / filename
        matrix = load_preset_matrix(path)
        assert matrix["source_run"] == SOURCES[horizon]
        assert matrix["base_alg_config"] == "checkpoint"
        assert matrix["shared_alg_params"] == originals[kind]["shared_alg_params"]
        evaluation = deepcopy(matrix["evaluation"])
        expected_evaluation = deepcopy(originals[kind]["evaluation"])
        if kind == "prior":
            expected_evaluation["default_presets"] = ["named_run/prior"]
        assert evaluation == expected_evaluation
        selectors = [f"step_rounds/j{j}" for j in (1, 3, 6)] if kind == "inner" else ["named_run/prior"]
        assert normalize_selectors(matrix) == selectors
        for selector in selectors:
            comparison, variant = selector.split("/")
            assert matrix["comparisons"][comparison]["variants"][variant].get("alg_params", {}) == originals[kind]["comparisons"][comparison]["variants"][variant].get("alg_params", {})
            resolved = resolve_preset(path, selector, checkpoint_context=checkpoint_context)
            run = resolved["algorithm_config"]
            for key, value in params.items():
                if not key.startswith(("inner_", "wandb_")):
                    assert run["alg_params"][key] == value
            cfg = _build_cfg(run)
            assert cfg.train_unroll_horizon == horizon
            assert cfg.outer_planning_horizon == cfg.inner_rollout_horizon == 3
            assert cfg.inner_horizon_ratio == 3 / horizon
            assert cfg.temporal_loss_normalization == "reference_weighted_mean"
            assert cfg.temporal_loss_reference_horizon == 3
            updates = 3 * int(variant[1:]) if kind == "inner" else 0
            assert cfg.inner_critic_updates_per_action == cfg.inner_actor_updates_per_action == cfg.inner_temperature_updates_per_action == updates
            if kind == "inner":
                assert cfg.inner_rollouts_per_round == cfg.inner_steps_per_update == cfg.inner_batch_size == 512
                assert cfg.inner_update_timing == "step"
                assert cfg.inner_finite_horizon is False
                assert cfg.inner_bootstrap_source == "inner_target"
                assert cfg.inner_replay_capacity == 9216
    assert checkpoint_context == before


@pytest.mark.parametrize("horizon", [5, 10])
def test_verified_backbone_labels_distinguish_training_and_inner_horizon(horizon):
    identity = {"backbone": SOURCES[horizon], "planner": {"type": "sac", "settings": {
        "inner_rounds": 3, "inner_rollouts_per_round": 512, "inner_rollout_horizon": 3,
        "inner_update_timing": "step", "inner_steps_per_update": 512}}}
    assert backbone_display_label(identity) == f"AMBI prior-only backbone (train H{horizon})"
    registry = {"identity": identity, "run_id": "abcd1234", "attempt_label": "one-update"}
    label = concise_curve_label(registry)
    assert f"AMBI train H{horizon}" in label
    assert "H3" in evaluation_run_name(registry)
    unknown = {"backbone": "someone/else/" + SOURCES[horizon].split("/")[-1]}
    assert backbone_display_label(unknown) == "Backbone " + unknown["backbone"]


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def campaign_env(launch_env, tmp_path):
    env = dict(launch_env)
    env.pop("EVAL_RUN_MAP")
    python = Path(env["AMBI_DMC_PYTHON"])
    python.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
from pathlib import Path
if sys.argv[1] == '-':
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
args = sys.argv[1:]
with open(os.environ['TEST_CALLS'], 'a') as stream:
    stream.write(json.dumps(args) + '\\n')
if '--bundle-dir' in args:
    bundle = Path(args[args.index('--bundle-dir') + 1])
    bundle.mkdir(parents=True)
    (bundle / 'manifest.json').write_text('{}')
if '--eval-series-spec-dir' in args:
    directory = Path(args[args.index('--eval-series-spec-dir') + 1])
    directory.mkdir(parents=True)
    selector = args[args.index('--preset') + 1]
    (directory / (selector.replace('/', '__') + '.json')).write_text('{}')
''')
    rows = []
    for horizon in (5, 10):
        entries = []
        inventory = tmp_path / f"h{horizon}-inventory.json"
        for step in range(50000, 500001, 50000):
            checkpoint = tmp_path / f"h{horizon}-checkpoint-{step}"
            checkpoint.write_bytes(f"weights-{horizon}-{step}".encode())
            sidecar = Path(str(checkpoint) + ".metadata.json")
            sidecar.write_text(json.dumps({"checkpoint": {"step": step}, "trial_run_params": {
                "alg": "AMBITDMPC2/AMBITDMPC2", "alg_params": {
                    "train_unroll_horizon": horizon, "inner_operator": "none"}}}))
            entries.append({"step": step, "sha256": _digest(checkpoint), "metadata_sha256": _digest(sidecar)})
            for selector in ("named_run/prior", "step_rounds/j1", "step_rounds/j3", "step_rounds/j6"):
                run = tmp_path / f"run-h{horizon}-{selector.replace('/', '-') }"
                run.mkdir(exist_ok=True)
                (run / "run.json").write_text("{}")
                rows.append({"backbone": f"h{horizon}", "stage": "prior" if selector == "named_run/prior" else "inner",
                             "step": step, "checkpoint": str(checkpoint), "checkpoint_inventory": str(inventory),
                             "selector": selector, "eval_run_dir": str(run)})
        inventory.write_text(json.dumps({"source_run": SOURCES[horizon], "checkpoints": entries}))
    manifest = tmp_path / "campaign.json"
    manifest.write_text(json.dumps({"schema_version": 1, "rows": rows}))
    env["AMBI_EVAL_MANIFEST"] = str(manifest)
    return env


def _row(env, index):
    return json.loads(Path(env["AMBI_EVAL_MANIFEST"]).read_text())["rows"][index]


def _calls(env):
    return [json.loads(line) for line in Path(env["TEST_CALLS"]).read_text().splitlines()]


def _launch(env, index, *args):
    return subprocess.run(["bash", str(LAUNCHER), *args], env={**env, "SLURM_ARRAY_TASK_ID": str(index)},
                          capture_output=True, text=True)


@pytest.mark.parametrize("index", [0, 36, 40, 76])
def test_prior_stage_publishes_only_source_matched_prior(campaign_env, index):
    env = campaign_env
    row = _row(env, index)
    result = _launch(env, index)
    assert result.returncode == 0, result.stderr
    spec, append, evaluate, report = _calls(env)
    assert "--eval-series-spec-dir" in spec and append[0:2] == ["eval_series.py", "append"]
    assert evaluate[evaluate.index("--matrix") + 1] == f"configs/research/ambi_humanoid_{row['backbone']}_prior_benchmark.json"
    assert evaluate[evaluate.index("--preset") + 1] == "named_run/prior"
    assert "--reference-bundle" not in evaluate and "--eval-run-map" not in evaluate
    assert evaluate[evaluate.index("--eval-run-dir") + 1] == row["eval_run_dir"]
    assert evaluate[evaluate.index("--max-steps") + 1] == "500"
    assert "--wandb" not in evaluate and report[0] == "report_ambi_benchmark.py"
    assert _launch(env, index).returncode != 0
    assert len(_calls(env)) == 4


@pytest.mark.parametrize("index", [1, 22, 39, 41, 62, 79])
def test_inner_stage_requires_and_reuses_same_backbone_prior(campaign_env, index):
    env = campaign_env
    row = _row(env, index)
    missing = _launch(env, index)
    assert missing.returncode != 0 and "completed paired prior missing" in missing.stderr
    assert not Path(env["TEST_CALLS"]).exists()
    prior = Path(env["AMBI_BENCHMARK_OUTPUT_ROOT"]) / row["backbone"] / f"step_{row['step']}" / "prior"
    prior.mkdir(parents=True)
    (prior / "manifest.json").write_text("{}")
    result = _launch(env, index)
    assert result.returncode == 0, result.stderr
    spec, append, evaluate, report = _calls(env)
    assert evaluate[evaluate.index("--preset") + 1] == row["selector"]
    assert evaluate[evaluate.index("--matrix") + 1] == f"configs/research/ambi_humanoid_{row['backbone']}_inner_step_rounds.json"
    assert evaluate[evaluate.index("--reference-bundle") + 1] == str(prior)
    assert evaluate[evaluate.index("--eval-run-dir") + 1] == row["eval_run_dir"]
    assert evaluate[evaluate.index("--seeds") + 1:evaluate.index("--max-steps")] == ["101", "102", "103", "104", "105"]
    assert not any("named_run/prior" in call for call in _calls(env))
    assert not any("--eval-run-map" in call or "--wandb" in call for call in _calls(env))


@pytest.mark.parametrize("index", [21, 22, 23, 61, 62, 63])
def test_smoke_has_private_short_prior_and_no_publication(campaign_env, index):
    env = campaign_env
    row = _row(env, index)
    manifest = json.loads(Path(env["AMBI_EVAL_MANIFEST"]).read_text())
    manifest["rows"][index].pop("eval_run_dir")
    Path(env["AMBI_EVAL_MANIFEST"]).write_text(json.dumps(manifest))
    result = _launch(env, index, "--smoke")
    assert result.returncode == 0, result.stderr
    spec, prior, evaluate, report = _calls(env)
    assert prior[prior.index("--preset") + 1] == "named_run/prior"
    reference = Path(evaluate[evaluate.index("--reference-bundle") + 1])
    assert reference.parent.name == row["selector"].split("/")[1]
    for call in (spec, prior, evaluate):
        assert call[call.index("--max-steps") + 1] == "3"
        assert call[call.index("--seeds") + 1:call.index("--max-steps")] == ["101", "102"]
        assert not {"--eval-run-dir", "--eval-run-map", "--wandb"}.intersection(call)
    assert not any(call[0] == "eval_series.py" for call in _calls(env))


@pytest.mark.parametrize("issue", ["source", "weight", "sidecar", "horizon", "step", "duplicate", "selector", "assignment"])
def test_invalid_campaign_fails_before_evaluation(campaign_env, issue):
    env = campaign_env
    path = Path(env["AMBI_EVAL_MANIFEST"])
    manifest = json.loads(path.read_text())
    row = manifest["rows"][0]
    inventory_path = Path(row["checkpoint_inventory"])
    inventory = json.loads(inventory_path.read_text())
    if issue == "source":
        inventory["source_run"] = SOURCES[10]
    elif issue == "weight":
        Path(row["checkpoint"]).write_bytes(b"wrong")
    elif issue in {"sidecar", "horizon"}:
        sidecar = Path(row["checkpoint"] + ".metadata.json")
        metadata = json.loads(sidecar.read_text())
        metadata["trial_run_params"]["alg_params"]["train_unroll_horizon"] = 10
        sidecar.write_text(json.dumps(metadata))
        if issue == "horizon":
            inventory["checkpoints"][0]["metadata_sha256"] = _digest(sidecar)
    elif issue == "step":
        row["step"] = 525000
    elif issue == "duplicate":
        manifest["rows"].append(deepcopy(row))
    elif issue == "selector":
        row["selector"] = "step_rounds/j6"
    else:
        row.pop("eval_run_dir")
    path.write_text(json.dumps(manifest))
    inventory_path.write_text(json.dumps(inventory))
    result = _launch(env, 0)
    assert result.returncode != 0
    assert not Path(env["TEST_CALLS"]).exists()


def test_launcher_syntax_and_l40s_resources_have_no_arbitrary_array_throttle():
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True, capture_output=True)
    directives = [line for line in LAUNCHER.read_text().splitlines() if line.startswith("#SBATCH ")]
    assert "#SBATCH --gres=gpu:l40s:1" in directives
    assert "#SBATCH --cpus-per-task=6" in directives
    assert "#SBATCH --mem=32G" in directives
    assert not any("--array" in line for line in directives)
