import copy
import json
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
ALGORITHM_ROOT = ROOT / "configs/dmcontrol/algs"
EXPERIMENT_ROOT = ROOT / "configs/dmcontrol/experiments"
SLURM_ROOT = ROOT / "slurm"

AMBI_BASE = "ambi_humanoid_walk_outer_prior_no_inner_checkpoint_bank_1p5m"
AMBI_BANKS = {
    5: (
        "ambi_humanoid_walk_outer_prior_no_inner_checkpoint_bank_1p5m_train_h5"
    ),
    10: (
        "ambi_humanoid_walk_outer_prior_no_inner_checkpoint_bank_1p5m_train_h10"
    ),
}
TDMPC2_TANH_H3_BANK = (
    "ambi_humanoid_walk_outer_prior_no_inner_checkpoint_bank_1p5m_"
    "train_h3_tdmpc2_tanh"
)
LAUNCHERS = {
    5: SLURM_ROOT
    / "run_ambi_humanoid_walk_outer_prior_no_inner_checkpoint_bank_1p5m_train_h5_hydra.sbatch",
    10: SLURM_ROOT
    / "run_ambi_humanoid_walk_outer_prior_no_inner_checkpoint_bank_1p5m_train_h10_hydra.sbatch",
    3: SLURM_ROOT
    / "run_ambi_humanoid_walk_outer_prior_no_inner_checkpoint_bank_1p5m_train_h3_tdmpc2_tanh_hydra.sbatch",
}
COMMON_RUNNER = (
    SLURM_ROOT / "run_prior_horizon_checkpoint_bank_1p5m_hydra_common.sh"
)
STORAGE_PREFLIGHT = (
    SLURM_ROOT
    / "preflight_ambi_prior_horizon_checkpoint_banks_storage_hydra.sh"
)
SMOKE_LAUNCHER = (
    SLURM_ROOT / "run_ambi_prior_horizon_checkpoint_banks_smoke_hydra.sbatch"
)
SHA_PREFLIGHT = SLURM_ROOT / "require_expected_action_modes_sha.sh"
SUBMISSION_GATE = (
    SLURM_ROOT / "submit_ambi_prior_horizon_checkpoint_banks_hydra.sh"
)


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(root, stem):
    return json.loads(
        (root / f"{stem}.json").read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )


def _leaf_differences(left, right, prefix=()):
    if isinstance(left, dict) and isinstance(right, dict):
        paths = set()
        for key in set(left) | set(right):
            if key not in left or key not in right:
                paths.add(prefix + (key,))
            else:
                paths.update(
                    _leaf_differences(left[key], right[key], prefix + (key,))
                )
        return paths
    return set() if left == right else {prefix}


@pytest.mark.parametrize("train_horizon", [5, 10])
def test_ambi_banks_change_only_training_unroll_and_wandb_labels(
    train_horizon,
):
    baseline = _load_json(ALGORITHM_ROOT, AMBI_BASE)
    actual = _load_json(ALGORITHM_ROOT, AMBI_BANKS[train_horizon])

    assert _leaf_differences(baseline, actual) == {
        ("alg_params", "train_unroll_horizon"),
        ("alg_params", "wandb_run_name"),
        ("alg_params", "wandb_tags"),
    }
    params = actual["alg_params"]
    assert params["train_unroll_horizon"] == train_horizon
    assert params["outer_planning_horizon"] == 3
    assert params["inner_rollout_horizon"] == 3
    assert params["temporal_loss_normalization"] == "reference_weighted_mean"
    assert params["temporal_loss_reference_horizon"] == 3
    assert params["wandb_entity"] == "rwgao_b-brown-university"
    assert params["wandb_project"] == "ambi"
    assert params["wandb_mode"] == "online"
    assert params["wandb_group"] == (
        "ambi-humanoid-walk-no-inner-checkpoint-bank-1p5m"
    )
    assert params["wandb_run_name"] == (
        "AMBITDMPC2-humanoid-walk-outer-prior-no-inner-checkpoint-"
        f"bank-1p5m-train-h{train_horizon}-seed55"
    )
    tags = params["wandb_tags"]
    assert len(tags) == len(set(tags))
    assert f"train-unroll-horizon-{train_horizon}" in tags
    assert "outer-planning-horizon-3" in tags
    assert "temporal-loss-reference-horizon-3" in tags
    assert not ({"train-unroll-horizon-3", "train-unroll-horizon-5", "train-unroll-horizon-10"} - {f"train-unroll-horizon-{train_horizon}"}) & set(tags)


@pytest.mark.parametrize("train_horizon", [5, 10])
def test_ambi_horizon_manifests_keep_the_checkpoint_bank_protocol(
    train_horizon,
):
    manifest = _load_json(EXPERIMENT_ROOT, AMBI_BANKS[train_horizon])
    note = manifest["study_note"].lower()

    assert manifest["study_type"] == (
        f"single_seed_exploratory_no_inner_checkpoint_bank_train_h{train_horizon}"
    )
    assert manifest["configs"] == [AMBI_BANKS[train_horizon]]
    assert manifest["overrides_alg"] == {
        "seed": 55,
        "device": "cuda",
        "env": "DMControl-v0",
        "total_steps": 1_500_000,
        "episodes": None,
    }
    assert manifest["env_params"] == {
        "task": "humanoid-walk",
        "obs": "state",
        "render_mode": None,
    }
    assert manifest["trials"] == 1
    assert manifest["logs"] == "none"
    assert manifest["save_trials"] == "none"
    assert manifest["checkpoint_every"] == 25_000
    assert manifest["save_strat"] == ["all"]
    assert 1_500_000 // manifest["checkpoint_every"] == 60
    assert f"train_unroll_horizon={train_horizon}" in note
    assert "outer_planning_horizon=3" in note
    assert "inner_rollout_horizon=3" in note
    assert "temporal_loss_reference_horizon=3" in note
    assert "zero inner model steps and zero inner optimizer updates" in note
    assert "separate evaluation episodes are disabled" in note
    assert "no step-zero checkpoint" in note


def test_h3_tdmpc2_tanh_prior_changes_only_mapping_minimum_and_labels():
    baseline = _load_json(ALGORITHM_ROOT, AMBI_BASE)
    actual = _load_json(ALGORITHM_ROOT, TDMPC2_TANH_H3_BANK)
    expected = copy.deepcopy(baseline)
    params = expected["alg_params"]
    actual_params = actual["alg_params"]
    params["log_std_mapping"] = "tdmpc2_tanh"
    params["log_std_min"] = -10
    params["wandb_run_name"] = (
        "AMBITDMPC2-humanoid-walk-outer-prior-no-inner-checkpoint-bank-"
        "1p5m-train-h3-actor-tdmpc2-tanh-upstream-bounds-seed55"
    )
    params["wandb_tags"] = actual_params["wandb_tags"]

    assert actual == expected
    assert _leaf_differences(baseline, actual) == {
        ("alg_params", "log_std_mapping"),
        ("alg_params", "log_std_min"),
        ("alg_params", "wandb_run_name"),
        ("alg_params", "wandb_tags"),
    }
    assert actual["alg"] == "AMBITDMPC2/AMBITDMPC2"
    assert actual_params["train_unroll_horizon"] == 3
    assert actual_params["outer_planning_horizon"] == 3
    assert actual_params["inner_rollout_horizon"] == 3
    assert actual_params["temporal_loss_reference_horizon"] == 3
    assert actual_params["compile"] is True
    assert actual_params["mpc"] is False
    assert actual_params["eval_freq"] is None
    assert actual_params["inner_operator"] == "none"
    assert actual_params["log_std_mapping"] == "tdmpc2_tanh"
    assert actual_params["log_std_min"] == -10
    assert actual_params["log_std_max"] == 2
    assert actual_params["wandb_entity"] == "rwgao_b-brown-university"
    assert actual_params["wandb_project"] == "ambi"
    assert actual_params["wandb_mode"] == "online"
    assert actual_params["wandb_group"] == (
        "ambi-humanoid-walk-no-inner-checkpoint-bank-1p5m"
    )
    assert actual_params["wandb_tags"] == [
        "ambi",
        "dmcontrol",
        "humanoid-walk",
        "state",
        "base-v2",
        "tdmpc2-aligned-recipe",
        "single-seed-exploratory",
        "1p5m-decisions",
        "checkpoint-bank",
        "checkpoints-every-25k",
        "train-unroll-horizon-3",
        "outer-planning-horizon-3",
        "temporal-loss-reference-horizon-3",
        "actor-log-std-tdmpc2-tanh",
        "actor-log-std-bounds-neg10-2",
        "outer-prior-execution",
        "no-action-local-improvement",
        "inner-operator-none",
        "zero-inner-model-steps",
        "policy-sample-training",
        "outer-critic-lr3e-4",
        "outer-actor-lr3e-4",
        "q-min-pair",
        "q-heads-5",
        "q-pair-size-2",
        "outer-critic-entropy-augmented",
        "inner-critic-entropy-augmented",
        "outer-alpha-auto",
        "outer-alpha-lr3e-4",
        "actor-loss-scale-none",
        "no-online-evaluation",
        "seed55",
    ]

def test_h3_tdmpc2_tanh_prior_manifest_discloses_exact_scope():
    manifest = _load_json(EXPERIMENT_ROOT, TDMPC2_TANH_H3_BANK)
    note = manifest["study_note"].lower()

    assert manifest["study_type"] == (
        "single_seed_exploratory_no_inner_checkpoint_bank_h3_tdmpc2_tanh_prior"
    )
    assert manifest["configs"] == [TDMPC2_TANH_H3_BANK]
    assert manifest["overrides_alg"] == {
        "seed": 55,
        "device": "cuda",
        "env": "DMControl-v0",
        "total_steps": 1_500_000,
        "episodes": None,
    }
    assert manifest["env_params"] == {
        "task": "humanoid-walk",
        "obs": "state",
        "render_mode": None,
    }
    assert manifest["trials"] == 1
    assert manifest["logs"] == "none"
    assert manifest["save_trials"] == "none"
    assert manifest["checkpoint_every"] == 25_000
    assert manifest["save_strat"] == ["all"]
    assert 1_500_000 // manifest["checkpoint_every"] == 60
    assert "it remains the ambitdmpc2 no-inner h=3 bank" in note
    assert "only scientific changes are log_std_mapping=tdmpc2_tanh" in note
    assert "log_std_min=-10" in note
    assert "already-matched log_std_max=2" in note
    assert "does not use the native tdmpc2baseline architecture" in note
    assert "does not execute mppi" in note
    assert "zero-inner-work contract" in note
    assert "no step-zero checkpoint" in note


@pytest.mark.parametrize("train_horizon", [3, 5, 10])
def test_launchers_are_guarded_single_l40s_jobs(train_horizon):
    launcher = LAUNCHERS[train_horizon]
    contents = launcher.read_text(encoding="utf-8")

    assert os.access(launcher, os.X_OK)
    subprocess.run(["bash", "-n", str(launcher)], check=True)
    for contract in (
        "#SBATCH --partition=gpus",
        "#SBATCH --nodelist=gpu2501",
        "#SBATCH --constraint=l40s",
        "#SBATCH --gres=gpu:nvidia_l40s:1",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --mem=32G",
        "#SBATCH --time=5-00:00:00",
        "#SBATCH --no-requeue",
        f"readonly EXPECTED_TRAIN_HORIZON={train_horizon}",
        'readonly EXPECTED_NODE="gpu2501"',
        "run_prior_horizon_checkpoint_bank_1p5m_hydra_common.sh",
    ):
        assert contract in contents
    assert "#SBATCH --array" not in contents
    assert "SLURM_ARRAY_" not in contents


def test_shared_launcher_body_rechecks_science_identity_and_immutable_paths():
    contents = COMMON_RUNNER.read_text(encoding="utf-8")

    subprocess.run(["bash", "-n", str(COMMON_RUNNER)], check=True)
    for contract in (
        "git status --porcelain --untracked-files=normal",
        "AMBI_DMC_PYTHON",
        "require_expected_action_modes_sha.sh",
        'readonly SOURCE_COMMIT="$("$SHA_PREFLIGHT")"',
        "AMBI horizon bank is not an exact no-inner bank derivative",
        "TD-MPC2-tanh policy prior is not an exact no-inner bank derivative",
        '"inner_operator": "none"',
        '"mpc": False',
        '"outer_planning_horizon": 3',
        '"inner_rollout_horizon": 3',
        '"temporal_loss_reference_horizon": 3',
        '"wandb_entity": "rwgao_b-brown-university"',
        "checkpoint bank must contain exactly 60 snapshots",
        "preflight_ambi_prior_horizon_checkpoint_banks_storage_hydra.sh",
        '"$STORAGE_PREFLIGHT" --single-bank',
        "Refusing to reuse immutable artifact root",
        "Agent decisions: 1500000",
        "Raw simulator control steps: 3000000",
        "Checkpoint cadence: 25000",
        "Expected numbered checkpoints: 60",
        "export WANDB_MODE=online",
        "export WANDB_DISABLE_CODE=true",
        "--alg-index 0",
        "--trial-index 0",
        "--num-runs 1",
    ):
        assert contract in contents


def test_production_sha_preflight_rejects_missing_and_mismatched_sha():
    assert os.access(SHA_PREFLIGHT, os.X_OK)
    subprocess.run(["bash", "-n", str(SHA_PREFLIGHT)], check=True)
    current_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()

    missing_env = os.environ.copy()
    missing_env.pop("EXPECTED_ACTION_MODES_SHA", None)
    missing = subprocess.run(
        ["bash", str(SHA_PREFLIGHT)],
        cwd=ROOT,
        env=missing_env,
        text=True,
        capture_output=True,
    )
    assert missing.returncode != 0
    assert "EXPECTED_ACTION_MODES_SHA is required" in missing.stderr

    abbreviated_env = os.environ.copy()
    abbreviated_env["EXPECTED_ACTION_MODES_SHA"] = current_sha[:12]
    abbreviated = subprocess.run(
        ["bash", str(SHA_PREFLIGHT)],
        cwd=ROOT,
        env=abbreviated_env,
        text=True,
        capture_output=True,
    )
    assert abbreviated.returncode != 0
    assert "full lowercase 40-character Git SHA" in abbreviated.stderr

    mismatch_env = os.environ.copy()
    mismatch_env["EXPECTED_ACTION_MODES_SHA"] = "0" * 40
    mismatched = subprocess.run(
        ["bash", str(SHA_PREFLIGHT)],
        cwd=ROOT,
        env=mismatch_env,
        text=True,
        capture_output=True,
    )
    assert mismatched.returncode != 0
    assert "wrong action-modes commit" in mismatched.stderr
    assert f"Actual:   {current_sha}" in mismatched.stderr

    matching_env = os.environ.copy()
    matching_env["EXPECTED_ACTION_MODES_SHA"] = current_sha
    matched = subprocess.run(
        ["bash", str(SHA_PREFLIGHT)],
        cwd=ROOT,
        env=matching_env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert matched.stdout.strip() == current_sha
    assert matched.stderr == ""


def test_smoke_launcher_derives_all_cells_only_in_job_local_storage():
    contents = SMOKE_LAUNCHER.read_text(encoding="utf-8")

    assert os.access(SMOKE_LAUNCHER, os.X_OK)
    subprocess.run(["bash", "-n", str(SMOKE_LAUNCHER)], check=True)
    for contract in (
        "#SBATCH --nodelist=gpu2501",
        "#SBATCH --constraint=l40s",
        "#SBATCH --gres=gpu:nvidia_l40s:1",
        "#SBATCH --array",
        "AMBI_DMC_PYTHON",
        "require_expected_action_modes_sha.sh",
        'readonly SOURCE_COMMIT="$("$SHA_PREFLIGHT")"',
        "SLURM_TMPDIR",
        'readonly NODE_LOCAL_SCRATCH_BASE="/ltmp"',
        '[[ ! -w "$NODE_LOCAL_SCRATCH_BASE" ]]',
        '${NODE_LOCAL_SCRATCH_BASE}/rgao48-ambi-prior-banks-smoke-',
        "mktemp -d",
        'readonly SMOKE_TOTAL_STEPS=3000',
        'smoke_algorithm["total_steps"] = total_steps',
        'smoke_algorithm["alg_params"]["wandb"] = False',
        'resolved_seed_steps = params.get("seed_steps", 2_500)',
        'total_steps <= resolved_seed_steps',
        'smoke_manifest["checkpoint_every"] = None',
        'smoke_manifest["save_strat"] = "none"',
        'smoke_manifest["save_trials"] = "none"',
        "export WANDB_MODE=disabled",
        'export TMPDIR="$SMOKE_ROOT/tmp"',
        'export XDG_CACHE_HOME="$SMOKE_ROOT/cache"',
        'export TORCHINDUCTOR_CACHE_DIR="$SMOKE_ROOT/torchinductor"',
        'for stem in "${CONFIG_STEMS[@]}"',
        'tee "$cell_stdout"',
        'grep -Fq "Pretraining TD-MPC2 on seed data..."',
        "Smoke cell did not cross into learned updates",
        '--alg-dir "$SMOKE_ALGORITHM_ROOT"',
        '--log-dir "$readonly_cell_result_root"',
        "All three node-local smoke cells completed without checkpoint artifacts",
    ):
        if contract == "#SBATCH --array":
            assert contract not in contents
        else:
            assert contract in contents

    for stem in (*AMBI_BANKS.values(), TDMPC2_TANH_H3_BANK):
        assert f'  "{stem}"' in contents
    assert (
        "/cs/home/rgao48/projects/ambi-runs/"
        "ambi-prior-horizon-checkpoint-banks-1p5m"
    ) not in contents


def test_storage_preflight_budgets_all_three_banks_plus_headroom():
    contents = STORAGE_PREFLIGHT.read_text(encoding="utf-8")

    assert os.access(STORAGE_PREFLIGHT, os.X_OK)
    subprocess.run(["bash", "-n", str(STORAGE_PREFLIGHT)], check=True)
    assert 'readonly EXPECTED_CAMPAIGN_BANKS=3' in contents
    assert 'readonly ESTIMATED_BANK_MIB=4200' in contents
    assert 'readonly CAMPAIGN_HEADROOM_MIB=4096' in contents
    assert '--aggregate)' in contents
    assert '--single-bank)' in contents
    assert "PREFLIGHT_BANKS * ESTIMATED_BANK_MIB" in contents
    assert "available_kib < MIN_CAMPAIGN_FREE_KIB" in contents
    assert 'readonly PREFLIGHT_SCOPE="all three banks"' in contents
    assert 'readonly PREFLIGHT_SCOPE="one remaining bank"' in contents
    assert (
        "/cs/home/rgao48/projects/ambi-runs/"
        "ambi-prior-horizon-checkpoint-banks-1p5m"
    ) in contents
    assert "Refusing symlinked campaign artifact root" in contents


def test_submission_gate_checks_aggregate_once_then_passes_exact_sha():
    contents = SUBMISSION_GATE.read_text(encoding="utf-8")

    assert os.access(SUBMISSION_GATE, os.X_OK)
    subprocess.run(["bash", "-n", str(SUBMISSION_GATE)], check=True)
    assert '"$STORAGE_PREFLIGHT" --aggregate' in contents
    assert contents.count('"$STORAGE_PREFLIGHT" --aggregate') == 1
    assert 'readonly SOURCE_COMMIT="$("$SHA_PREFLIGHT")"' in contents
    assert '--export=ALL,EXPECTED_ACTION_MODES_SHA="$SOURCE_COMMIT"' in contents
    assert 'for launcher in "${LAUNCHERS[@]}"' in contents
    assert 'for stem in "${CONFIG_STEMS[@]}"' in contents
    assert 'configs/dmcontrol/algs/${stem}.json' in contents
    assert 'configs/dmcontrol/experiments/${stem}.json' in contents
    for launcher in LAUNCHERS.values():
        assert f'  "slurm/{launcher.name}"' in contents


def test_artifact_and_wandb_identities_are_unique_across_the_campaign():
    artifact_slugs = []
    config_stems = []
    for launcher in LAUNCHERS.values():
        contents = launcher.read_text(encoding="utf-8")
        for line in contents.splitlines():
            if line.startswith("readonly ARTIFACT_SLUG="):
                artifact_slugs.append(line.partition("=")[2])
            if line.startswith("readonly CONFIG_STEM="):
                config_stems.append(line.partition("=")[2])

    assert len(artifact_slugs) == len(set(artifact_slugs)) == 3
    assert len(config_stems) == len(set(config_stems)) == 3

    identities = []
    for stem in (*AMBI_BANKS.values(), TDMPC2_TANH_H3_BANK):
        params = _load_json(ALGORITHM_ROOT, stem)["alg_params"]
        identities.append(
            (params["wandb_group"], params["wandb_run_name"])
        )
    assert len(identities) == len(set(identities)) == 3
