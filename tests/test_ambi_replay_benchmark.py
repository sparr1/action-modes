"""Guard the replay timing comparison's capacity, data and paired protocol."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from slurm import benchmark_ambi_replay as bench


@pytest.mark.parametrize("manifest", list(bench.ENTROPY_MANIFESTS.values()))
def test_recipe_keeps_production_capacity_shapes_and_update_rule(manifest):
    name, original_manifest, original = bench.validate_campaign(manifest)
    selected, manifest, run, overrides = bench.benchmark_recipe(manifest)
    expected = deepcopy(original)
    expected["alg_params"].update(wandb=False, seed_steps=500, pretrain_steps=2)
    expected.update(checkpoint_every=None, save_strat="none")
    assert name == selected and run == expected
    assert run["total_steps"] == run["alg_params"]["buffer_size"] == 1_000_000
    assert (run["alg_params"]["batch_size"], run["alg_params"]["train_unroll_horizon"],
            run["alg_params"]["utd"]) == (256, 3, 1)
    expected_manifest = deepcopy(original_manifest)
    expected_manifest.update(checkpoint_every=None, save_strat="none", save_trials="none", logs="none")
    assert manifest == expected_manifest


def test_benchmark_rejects_other_training_recipes_and_invalid_occupancies():
    from slurm.ambi_reward_training import MANIFESTS
    with pytest.raises(ValueError, match="no-inner"):
        bench.benchmark_recipe(MANIFESTS["sac"])
    assert bench.replay_rows("10000,750000") == (10000, 750000)
    for invalid in ("", "500", "1000000", "10000,10000", "-1", "ten"):
        with pytest.raises(ValueError):
            bench.replay_rows(invalid)


def fixture_episode():
    import torch
    from tensordict import TensorDict
    rows = 5
    td = TensorDict({"obs": torch.arange(rows).float().reshape(rows, 1),
                     "action": torch.arange(rows).float().reshape(rows, 1),
                     "reward": torch.arange(rows).float(),
                     "terminated": torch.zeros(rows)}, batch_size=[rows])
    for key in ("action", "reward", "terminated"):
        td[key][0] = float("nan")
    return td


def make_buffer(capacity=100):
    from RL.tdmpc2_core.common.buffer import Buffer
    return Buffer(SimpleNamespace(device="cpu", buffer_size=capacity, steps=capacity,
                                  batch_size=2, train_unroll_horizon=3, multitask=False, obs="state"))


def test_fixture_preserves_real_rows_distinct_episode_boundaries_and_valid_samples():
    import torch
    buffer, episode = make_buffer(), fixture_episode()
    buffer.add(episode)
    original = episode.clone()
    report = bench.populate_fixture(buffer, episode, 27, chunk_episodes=2)
    assert report["resident_rows"] == 30 and report["resident_transitions"] == 24
    assert report["episode_copies"] == 6 and buffer.num_eps == 6
    stored = buffer._buffer._storage[:]
    assert torch.equal(stored["episode"], torch.arange(6).repeat_interleave(5))
    for key in original.keys():
        torch.testing.assert_close(episode[key], original[key], equal_nan=True)
    for _ in range(8):
        obs, action, reward, terminated, task = buffer.sample()
        assert torch.isfinite(obs).all() and torch.isfinite(action).all()
        assert torch.isfinite(reward).all() and torch.isfinite(terminated).all()
        torch.testing.assert_close(obs[1:] - obs[:-1], torch.ones_like(obs[1:]))
        torch.testing.assert_close(action, obs[1:])
    with pytest.raises(ValueError, match="first complete"):
        bench.populate_fixture(buffer, episode, 40)


def test_fixture_rejects_capacity_overflow_and_invalid_native_values_before_duplication():
    buffer, episode = make_buffer(10), fixture_episode()
    buffer.add(episode)
    with pytest.raises(ValueError, match="fit in replay"):
        bench.populate_fixture(buffer, episode, 11)
    assert buffer.num_eps == 1
    episode["reward"][2] = float("nan")
    with pytest.raises(ValueError, match="Invalid real-episode fixture"):
        bench.populate_fixture(buffer, episode, 10)
    assert buffer.num_eps == 1


def case(index, placement, train_seconds, sample_seconds, fingerprint="real-episode"):
    return {"case_index": index, "placement": placement,
            "fixture": {"requested_rows": 10000, "resident_rows": 10020,
                        "source_episode_sha256": fingerprint},
            "measured": {"seconds_per_decision": train_seconds},
            "replay_sample": {"seconds_per_sample": sample_seconds}}


def test_summary_pairs_reverse_order_without_counting_micro_windows_as_replicates():
    cases = [case(0, "cpu", 4., 10.), case(1, "cuda", 2., 5.),
             case(2, "cuda", 3., 2.), case(3, "cpu", 3., 6.)]
    result = bench.compare_cases(list(reversed(cases)))[0]
    assert result["gpu_training_speedups"] == [2., 1.]
    assert result["gpu_training_speedup_geometric_mean"] == pytest.approx(2 ** .5)
    assert result["gpu_sample_speedups"] == [2., 3.]
    cases[1]["fixture"]["source_episode_sha256"] = "different-data"
    with pytest.raises(AssertionError):
        bench.compare_cases(cases)
