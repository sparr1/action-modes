import json
from pathlib import Path

import numpy as np
import pytest

from evaluate_ambi_critic_bypass import ARMS, fingerprint, load_config, task_list, timing_summary

CONFIG = Path(__file__).resolve().parents[1] / "configs/research/ambi_800k_critic_bypass.json"


def test_production_grid_complete_and_smoke_keeps_learning_and_selection_work():
    full, smoke = load_config(CONFIG), load_config(CONFIG, smoke=True)
    tasks = task_list(full)
    assert len(tasks) == len({row["task_id"] for row in tasks}) == 100
    for arm in ARMS:
        assert [row["env_seed"] for row in tasks if row["arm"] == arm] == list(range(101, 121))
    assert len(task_list(smoke)) == 5
    assert smoke["max_steps"] == 4 and full["max_steps"] == 500
    for key in ("horizon", "rounds", "mc_rollouts", "validation_rollouts", "max_expanded_batch", "mppi"):
        assert smoke[key] == full[key]
    assert fingerprint(smoke) != fingerprint(full)


@pytest.mark.parametrize("key,value", [("rounds", 4), ("max_steps", 100), ("controller_seed", 56),
                                      ("seeds", [101, 101]), ("mc_rollouts", 8)])
def test_science_contract_cannot_silently_change(tmp_path, key, value):
    config = json.loads(CONFIG.read_text())
    config[key] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="protocol"):
        load_config(path)


def test_duplicate_config_keys_rejected(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"protocol": "a", "protocol": "b"}')
    with pytest.raises(ValueError, match="Duplicate"):
        load_config(path)


def test_timing_percentile_uses_all_decisions_and_rejects_bad_samples():
    values = [0.1, 0.2, 0.3, 2.0]
    mean, p95 = timing_summary(values)
    assert mean == pytest.approx(np.mean(values))
    assert p95 == pytest.approx(np.percentile(values, 95))
    for invalid in ([], [float("nan")], [-1.], [[1.]]):
        with pytest.raises(ValueError):
            timing_summary(invalid)


def test_fingerprint_stable_to_key_order_but_sensitive_to_content():
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})
    assert fingerprint({"a": 1}) != fingerprint({"a": 2})


def test_gpu_launcher_requires_pinned_clean_source_and_full_budget_smoke():
    root = CONFIG.parents[2]
    script = (root / "slurm/run_ambi_critic_bypass.sbatch").read_text()
    assert 'git status --porcelain --untracked-files=normal' in script
    assert 'BYPASS_EXPECTED_COMMIT' in script
    assert '--no-requeue' in script
    assert 'WANDB_MODE=disabled' in script
    assert '--smoke' in script
    assert '--gres=gpu:nvidia_rtx_a5000:1' in script
