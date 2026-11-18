"""Check physical-action pairing and additive model/critic error separation."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from diagnose_dog_checkpoints import same_actions, load_bank
from tests.test_ambi_real_calibration import AnalyticEnv
from utils.ambi_benchmark import atomic_json, canonical_hash
from utils.ambi_real_calibration import capture_simulator_snapshot


class LinearObjective:
    discount = .5

    def __init__(self, model_step=1., reward_offset=0., critic_offset=0.):
        self.model = SimpleNamespace(
            encode=lambda obs: obs,
            next=lambda z, a: z + model_step,
            reward=lambda z, a: a + reward_offset,
            decode_reward=lambda reward: reward)
        self.critic_offset = critic_offset

    @staticmethod
    def tensor(value):
        return torch.as_tensor(value, dtype=torch.float64)

    @staticmethod
    def actor(obs, noise):
        # Distinct observable endpoint action: it must be used by both critics.
        return obs + noise

    def q(self, z, action):
        return z + 10*action + self.critic_offset

    def real_q(self, obs, action):
        return (obs + 10*action + self.critic_offset).reshape(-1)


@pytest.mark.parametrize('model_step,reward_offset,critic_offset', [(1.,0.,0.), (2.,3.,7.)])
def test_open_loop_replays_identical_actions_and_separates_terminal_bias(model_step, reward_offset, critic_offset):
    envs = [AnalyticEnv(), AnalyticEnv()]
    envs[0].reset()
    snapshot = capture_simulator_snapshot(envs[0])
    actions = np.array([[[1.], [2.]], [[3.], [4.]], [[5.], [6.]]])
    noise = np.arange(12, dtype=float).reshape(6, 2, 1)/10
    result = same_actions(LinearObjective(model_step, reward_offset, critic_offset), envs,
                         snapshot, [0.], actions, noise, 5)
    for index, row in enumerate(result['rows']):
        np.testing.assert_array_equal(np.stack(envs[index].actions)[:3], actions[:,index])
        assert row['reward_error'] == pytest.approx(1.75*reward_offset)
        # gamma^3 * (model z_H - encoded real z_H), with exactly the same a_H.
        assert row['model_endpoint_value_error'] == pytest.approx(.125*3*(model_step-1))
        actual_tail = sum(.5**t * envs[index].actions[3+t][0] for t in range(3))
        assert row['real_tail_return'] == pytest.approx(actual_tail)
        actual_q = 3 + 10*envs[index].actions[3][0] + critic_offset
        assert row['terminal_critic_error'] == pytest.approx(.125*(actual_q-actual_tail))
        assert row['total_prediction_error'] == pytest.approx(
            row['reward_error']+row['model_endpoint_value_error']+row['terminal_critic_error'])


def test_shared_bank_refuses_corruption_and_environment_mismatch(tmp_path):
    env = AnalyticEnv()
    env.reset()
    environment = {'id': 'DMControl-v0', 'params': {'task': 'dog-stand'}}
    bank = dict(schema=1, kind='dog_checkpoint_common_roots_v1', environment=environment,
                roots=[dict(root_id='root', snapshot=capture_simulator_snapshot(env).to_dict(),
                            observation=[0.], anchor_pretanh_action=[0.])])
    bank['id'] = canonical_hash(bank)
    path = tmp_path/'bank.json'
    atomic_json(path, bank)
    assert load_bank(path, environment)['id'] == bank['id']
    with pytest.raises(ValueError, match='environment'):
        load_bank(path, {'id': 'different'})
    bank['roots'][0]['observation'][0] = 2.
    atomic_json(path, bank, overwrite=True)
    with pytest.raises(ValueError, match='identity'):
        load_bank(path, environment)
