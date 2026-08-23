import csv
import json
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

import main as training_main
from RL.XQC import OFFICIAL_XQC_COMMIT, XQC


ROOT = Path(__file__).resolve().parents[1]
FULL_ALGORITHM = ROOT / "configs/dmcontrol/algs/xqc_walker_walk_state.json"
FULL_MANIFEST = (
    ROOT / "configs/dmcontrol/experiments/xqc_walker_walk_state.json"
)
SMOKE_ALGORITHM = (
    ROOT / "configs/dmcontrol/algs/xqc_walker_walk_state_smoke.json"
)
SMOKE_MANIFEST = (
    ROOT / "configs/dmcontrol/experiments/xqc_walker_walk_state_smoke.json"
)
HYDRA_LAUNCHER = ROOT / "slurm/run_xqc_validation_hydra.sbatch"
REFERENCE_DOC = ROOT / "docs/xqc.md"
UPSTREAM_LICENSE = ROOT / "docs/licenses/XQC-LICENSE.txt"


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )


class CountingEnv(gym.Env):
    metadata = {}

    def __init__(self, episode_length=20, reward=3.5):
        self.observation_space = gym.spaces.Box(
            -10.0, 10.0, shape=(2,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            -2.0, 2.0, shape=(1,), dtype=np.float32
        )
        self.episode_length = int(episode_length)
        self.reward = float(reward)
        self.total_steps = 0
        self.actions = []
        self.reset_seeds = []
        self.closed = False
        self._episode_step = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_seeds.append(seed)
        self._episode_step = 0
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        self._episode_step += 1
        self.total_steps += 1
        self.actions.append(float(np.asarray(action).reshape(-1)[0]))
        truncated = self._episode_step >= self.episode_length
        return (
            np.full(2, self.total_steps, dtype=np.float32),
            self.reward,
            False,
            truncated,
            {},
        )

    def close(self):
        self.closed = True


def _small_model(env, **params):
    settings = {
        "device": "cpu",
        "seed": 11,
        "buffer_size": 64,
        "learning_starts": 3,
        "batch_size": 2,
        "train_freq": 1,
        "gradient_steps": 2,
        "updates_per_step": 2,
        "num_interactions": 8,
        "actor_net_arch": [8],
        "critic_net_arch": [8],
        "num_atoms": 5,
        "vmin": -2.0,
        "vmax": 2.0,
        "eval_freq": None,
        "wandb": False,
    }
    settings.update(params)
    return XQC(
        "XQC",
        env,
        settings,
        {"seed": 11, "device": "cpu", "env": "CountingEnv", "total_steps": 8},
        {},
    )


def test_xqc_warmup_update_dose_action_scaling_and_raw_reward_observation():
    env = CountingEnv()
    model = _small_model(env)
    act_calls = []
    observed = []
    update_calls = []

    def fixed_action(_observation, deterministic=False):
        act_calls.append(bool(deterministic))
        return np.array([0.5], dtype=np.float32)

    def record_update(replay, gradient_steps, batch_size):
        update_calls.append(
            (model.num_timesteps, gradient_steps, batch_size, replay.size)
        )
        return {}

    model.agent.act = fixed_action
    model.agent.observe_reward = (
        lambda reward, terminated, truncated: observed.append(
            (float(reward), bool(terminated), bool(truncated))
        )
    )
    model.agent.update = record_update
    model.learn(total_timesteps=5)

    # Official loop: random at i < 3, actor starting at i == 3, learning at i > 3.
    assert act_calls == [False, False, False]
    assert update_calls == [(4, 2, 2, 4), (5, 2, 2, 5)]
    assert observed == [(3.5, False, False)] * 5
    assert model.replay_buffer.size == 5
    np.testing.assert_array_equal(
        model.replay_buffer.rewards[:5],
        np.full((5, 1), 3.5, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        model.replay_buffer.actions[2:5],
        np.full((3, 1), 0.5, dtype=np.float32),
    )
    assert env.actions[2:5] == [1.0, 1.0, 1.0]


def test_xqc_evaluation_uses_seed_plus_42_and_never_steps_training_env(tmp_path):
    eval_csv = tmp_path / "evaluation.csv"
    train_env = CountingEnv(episode_length=3, reward=-1.0)
    eval_env = CountingEnv(episode_length=2, reward=2.0)
    model = _small_model(
        train_env,
        learning_starts=0,
        eval_freq=2,
        eval_episodes=1,
        eval_csv_path=eval_csv,
    )
    model._build_evaluation_env = lambda: eval_env
    deterministic_flags = []

    def action(_observation, deterministic=False):
        deterministic_flags.append(bool(deterministic))
        return np.array([0.25 if deterministic else -0.5], dtype=np.float32)

    model.agent.act = action
    model.agent.update = lambda *_args, **_kwargs: {}
    model.learn(total_timesteps=4)

    assert train_env.total_steps == 4
    assert eval_env.total_steps == 6
    assert deterministic_flags.count(False) == 4
    assert deterministic_flags.count(True) == 6
    assert eval_env.reset_seeds == [53, None, None]
    assert train_env.reset_seeds[0] == 11
    assert eval_env.closed is True
    with eval_csv.open(newline="") as stream:
        assert list(csv.DictReader(stream)) == [
            {"step": "1", "reward": "4.0", "seed": "11"},
            {"step": "2", "reward": "4.0", "seed": "11"},
            {"step": "4", "reward": "4.0", "seed": "11"},
        ]


def test_xqc_checkpoint_round_trip_and_preflight_before_mutation(tmp_path):
    model = _small_model(CountingEnv(), policy_delay=3)
    with torch.no_grad():
        next(model.agent.actor.parameters()).add_(0.125)
    path = model.save(tmp_path, "xqc")

    restored = _small_model(CountingEnv(), policy_delay=3)
    restored.load(path)
    assert restored.agent.update_step == 0
    for expected, actual in zip(
        model.agent.actor.state_dict().values(),
        restored.agent.actor.state_dict().values(),
    ):
        torch.testing.assert_close(actual, expected)

    incompatible = _small_model(CountingEnv(), policy_delay=2)
    before = {
        key: value.detach().clone()
        for key, value in incompatible.agent.actor.state_dict().items()
    }
    with pytest.raises(ValueError, match="configuration mismatch"):
        incompatible.load(path)
    for key, value in incompatible.agent.actor.state_dict().items():
        torch.testing.assert_close(value, before[key])
    with pytest.raises(ValueError, match="cannot safely resume"):
        restored.load(path, resume=True)


def test_main_knows_xqc_owns_its_authoritative_seeded_reset():
    assert training_main._learn_resets_env_with_seed("XQC/XQC") is True


def test_xqc_profiles_freeze_the_official_and_smoke_schedules():
    full = _load_json(FULL_ALGORITHM)
    smoke = _load_json(SMOKE_ALGORITHM)
    full_manifest = _load_json(FULL_MANIFEST)
    smoke_manifest = _load_json(SMOKE_MANIFEST)

    for config in (full, smoke):
        params = config["alg_params"]
        assert config["seed"] == 1
        assert config["env"] == "DMControl-v0"
        assert config["alg"] == "XQC/XQC"
        assert config["device"] == "cuda"
        assert config["episodes"] is None
        assert params["obs"] == "state"
        assert params["learning_starts"] == 5_000
        assert params["batch_size"] == 256
        assert params["train_freq"] == 1
        assert params["gradient_steps"] == params["updates_per_step"] == 2
        assert params["buffer_size"] in {10_000, 1_000_000}
        assert params["actor_net_arch"] == [256] * 4
        assert params["critic_net_arch"] == [512] * 4
        assert params["num_atoms"] == 101
        assert (params["vmin"], params["vmax"]) == (-5.0, 5.0)
        assert params["gamma"] == 0.99
        assert params["tau"] == 0.005
        assert params["target_update_interval"] == 1
        assert params["policy_delay"] == 3
        assert params["learning_rate"] == params["actor_lr"] == 3e-4
        assert params["critic_lr"] == 3e-4
        assert params["lr_end"] == 3e-5
        assert params["init_temperature"] == 0.01
        assert params["target_entropy"] == "auto"
        assert params["adam_eps"] == 1e-8
        assert params["weight_decay"] == 0.0
        assert params["eval_freq"] == 50_000
        assert params["wandb"] is False

    assert full["total_steps"] == full["alg_params"]["num_interactions"] == 500_000
    assert full["alg_params"]["buffer_size"] == 1_000_000
    assert full["alg_params"]["eval_episodes"] == 10
    assert smoke["total_steps"] == smoke["alg_params"]["num_interactions"] == 5_020
    assert smoke["alg_params"]["buffer_size"] == 10_000
    assert smoke["alg_params"]["eval_episodes"] == 1
    assert 2 * (smoke["total_steps"] - 5_000) == 40

    for manifest, config_name in (
        (full_manifest, "xqc_walker_walk_state"),
        (smoke_manifest, "xqc_walker_walk_state_smoke"),
    ):
        assert manifest["env_params"] == {
            "task": "walker-walk",
            "obs": "state",
            "render_mode": None,
        }
        assert manifest["trials"] == 1
        assert manifest["configs"] == [config_name]
        assert manifest["logs"] == "none"
        assert manifest["save_trials"] == "none"
        assert manifest["save_strat"] == ["latest"]


def test_hydra_validation_launcher_is_clean_pinned_and_scratch_only():
    contents = HYDRA_LAUNCHER.read_text(encoding="utf-8")

    assert "#SBATCH --partition=gpus" in contents
    assert "#SBATCH --gres=gpu:1" in contents
    assert "#SBATCH --cpus-per-task=8" in contents
    assert "#SBATCH --mem=32G" in contents
    assert "#SBATCH --time=01:00:00" in contents
    assert "--nodelist" not in contents
    assert "#SBATCH -w" not in contents
    assert OFFICIAL_XQC_COMMIT in contents
    assert "EXPECTED_ACTION_MODES_SHA" in contents
    assert "status --porcelain=v1 --untracked-files=all" in contents
    assert 'SCRATCH_BASE="${SLURM_TMPDIR:-/tmp}"' in contents
    assert "WANDB_MODE=disabled" in contents
    assert "WANDB_DISABLED=true" in contents
    assert "XLA_PYTHON_CLIENT_PREALLOCATE=false" in contents
    assert "MUJOCO_GL=egl" in contents
    assert "torch.cuda.is_available()" in contents
    assert 'jax.default_backend() == "gpu"' in contents
    assert "from importlib.metadata import version" in contents
    assert "sha256sum" in contents
    assert "generate_xqc_official_fixture.py" in contents
    assert "run_official_xqc_smoke.py" in contents
    assert "--expected-updates 40" in contents
    assert "JAX_PLATFORMS=cpu srun" in contents
    assert "cmp \"$JOB_SCRATCH/xqc_official_fixture.json\"" in contents
    assert "nvidia-smi" in contents
    assert "srun --ntasks=1" in contents
    assert "run_xqc_validation_hydra.sbatch" in contents
    assert "xqc_walker_walk_state_smoke.json" in contents
    assert 'agent["update_step"]' in contents
    assert '(40,14,14)' in contents
    assert "torch.isfinite" in contents
    assert "residual<=1e-6" in contents
    assert "WANDB_MODE=online" not in contents
    assert "conda activate" not in contents
    assert "uv sync" not in contents
    assert "--nodelist" not in contents
    assert "/logs/" not in contents
    assert "/results/" not in contents


def test_xqc_provenance_and_license_are_pinned():
    reference = REFERENCE_DOC.read_text(encoding="utf-8")
    license_text = UPSTREAM_LICENSE.read_text(encoding="utf-8")
    assert OFFICIAL_XQC_COMMIT in reference
    assert "https://arxiv.org/abs/2509.25174" in reference
    assert "https://github.com/danielpalenicek/xqc" in reference
    assert "MIT License" in license_text
    assert "Copyright (c) 2026 Daniel Palenicek" in license_text


@pytest.mark.parametrize("path", [HYDRA_LAUNCHER])
def test_shell_launchers_parse(path):
    # Parse without submitting on Linux, including the Hydra validation job.
    if sys.platform == "darwin":
        pytest.skip(
            "forking bash after the macOS Torch/MuJoCo runtime can abort in libomp"
        )
    subprocess.run(["bash", "-n", path], check=True)
