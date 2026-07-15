import torch

from RL.tdmpc2_core.common import buffer as buffer_module


def test_pinned_staging_is_only_used_for_cpu_to_cuda_samples():
    uses_pinned_staging = buffer_module.Buffer._uses_pinned_staging

    assert uses_pinned_staging(torch.device("cpu"), torch.device("cuda"))
    assert not uses_pinned_staging(torch.device("cpu"), torch.device("cpu"))
    assert not uses_pinned_staging(torch.device("cuda"), torch.device("cuda"))


def test_replay_reservation_passes_pinned_staging_choice(monkeypatch):
    captured = {}

    def fake_replay_buffer(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(buffer_module, "ReplayBuffer", fake_replay_buffer)
    replay = object.__new__(buffer_module.Buffer)
    replay._sampler = object()
    replay._batch_size = 12
    replay._pin_memory = True
    storage = object()

    replay._reserve_buffer(storage)

    assert captured["storage"] is storage
    assert captured["pin_memory"] is True
    assert captured["prefetch"] == 0
    assert captured["batch_size"] == 12
