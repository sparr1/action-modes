from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from evaluate_ambi_action_audit import (
    MEANS, PUBLICATION_CANDIDATES, load_config, replay_actions, select_candidates,
    state_hash, task_pairs,
)


CONFIG = Path(__file__).resolve().parents[1] / "configs/research/ambi_800k_action_audit.json"


def test_fixed_protocol_and_separate_smoke():
    production, smoke = load_config(CONFIG), load_config(CONFIG, smoke=True)
    assert len(task_pairs(production)) == 15
    assert len(production["decisions"]) * len(task_pairs(production)) == 60
    assert production["J"] == smoke["J"] == [4, 6, 8]
    assert production["real_tail_steps"] == 500
    assert smoke["real_tail_steps"] == 4
    assert smoke["smoke"] and not production["smoke"]
    assert production["decisions"] == [25, 100, 250, 400]


def test_select_once_keeps_provenance_and_horizon_specific_winners():
    rows = []
    for i, (label, h1, h3) in enumerate([
        *((name, 0, 0) for name in MEANS),
        ("broad/local/0", 3, 1), ("broad/uniform/0", 1, 4),
        ("replay/j4/0", 5, 2), ("replay/j8/0", 2, 6),
        ("policy/sac_j8/0", 99, 99),
    ]):
        rows.append(dict(label=label, action=[i], model={"h1": {"mean": h1}, "h3": {"mean": h3}}))
    before = deepcopy(rows)
    selected = select_candidates({"actions": rows})
    assert selected["labels"] == list(PUBLICATION_CANDIDATES)
    assert selected["provenance"]["broad_best_h1"] == "broad/local/0"
    assert selected["provenance"]["broad_best_h3"] == "broad/uniform/0"
    assert selected["provenance"]["replay_best_h1"] == "replay/j4/0"
    assert selected["provenance"]["replay_best_h3"] == "replay/j8/0"
    assert rows == before


def reference_with_replay():
    replay = SimpleNamespace(size=3, next_sample_id=3, horizon_end=torch.ones(3, 1),
                             z=torch.ones(3, 2), action=torch.arange(6).reshape(3, 2).float())
    return SimpleNamespace(engine=SimpleNamespace(state=SimpleNamespace(replay=None),
                                                  _action_pool=SimpleNamespace(replay=replay))), replay


def test_replay_is_exact_full_root_data_and_copied():
    ref, replay = reference_with_replay()
    actions = replay_actions(ref, 3)
    actions.zero_()
    assert replay.action[1, 0] == 2
    replay.z[1, 0] = 2
    with pytest.raises(RuntimeError, match="other than"):
        replay_actions(ref, 3)


@pytest.mark.parametrize("field,value,message", [
    ("next_sample_id", 4, "overflowed"), ("size", 2, "incomplete"),
    ("horizon_end", torch.zeros(3, 1), "non-boundary"),
])
def test_replay_rejects_invalid_bank(field, value, message):
    ref, replay = reference_with_replay()
    setattr(replay, field, value)
    with pytest.raises(RuntimeError, match=message):
        replay_actions(ref, 3)


def test_rng_hash_tracks_values_independent_of_tensor_storage():
    rng = {"a": torch.arange(10, dtype=torch.uint8), "b": {"counter": 1}}
    copied = deepcopy(rng)
    assert state_hash(rng) == state_hash(copied)
    copied["a"][2] = 9
    assert state_hash(rng) != state_hash(copied)
