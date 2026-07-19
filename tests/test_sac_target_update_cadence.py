import torch

import RL.sac_core as sac_core
from RL.sac_core import SACAgent, SACConfig


class StaticBatch:
    def __init__(self):
        self.batch = {
            "obs": torch.zeros(4, 2),
            "actions": torch.zeros(4, 1),
            "rewards": torch.zeros(4, 1),
            "next_obs": torch.ones(4, 2),
            "dones": torch.zeros(4, 1),
        }

    def sample(self, *_args):
        return self.batch


def _agent() -> SACAgent:
    return SACAgent(
        obs_dim=2,
        action_dim=1,
        config=SACConfig(
            device="cpu",
            seed=7,
            net_arch=(4,),
            ent_coef=0.2,
            target_update_interval=3,
        ),
    )


def test_target_update_interval_keeps_global_phase_across_update_calls(monkeypatch):
    agent = _agent()
    target_update_steps = []

    def record_target_update(*_args):
        target_update_steps.append(agent.num_updates)

    monkeypatch.setattr(sac_core, "polyak_update", record_target_update)

    agent.update(StaticBatch(), gradient_steps=2, batch_size=4)
    agent.update(StaticBatch(), gradient_steps=2, batch_size=4)

    assert agent.num_updates == 4
    assert target_update_steps == [0, 3]


def test_target_update_phase_survives_checkpoint_restore(monkeypatch):
    source = _agent()
    source.update(StaticBatch(), gradient_steps=2, batch_size=4)

    restored = _agent()
    restored.load_state_dict(source.state_dict())
    target_update_steps = []

    def record_target_update(*_args):
        target_update_steps.append(restored.num_updates)

    monkeypatch.setattr(sac_core, "polyak_update", record_target_update)

    restored.update(StaticBatch(), gradient_steps=2, batch_size=4)

    assert restored.num_updates == 4
    assert target_update_steps == [3]
