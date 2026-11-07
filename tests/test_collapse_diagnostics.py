import json

import numpy as np
import pytest
import torch

from utils.collapse_diagnostics import gradient_geometry, gradient_probe, FrozenPriorController
from tests.test_ambi_root_local_sac import _tiny_model
from evaluate_ambi_checkpoint import _outer_state_digest
from diagnose_policy_collapse import load_bank
from utils.ambi_benchmark import canonical_hash


def test_gradient_decomposition_translation_scale_and_opposition():
    x = torch.nn.Parameter(torch.tensor([1.,2.]))
    q = (x*torch.tensor([2.,-1.])).sum()
    entropy = -q*.25
    base = gradient_geometry(q/4,entropy,[x])
    shifted = gradient_geometry((q+10000)/4,entropy,[x])
    assert base == shifted
    assert base['cosine'] == pytest.approx(-1)
    assert base['entropy_over_q'] == pytest.approx(1)
    assert base['total_gradient_norm'] == pytest.approx(0)
    changed = gradient_geometry(q/8,entropy,[x])
    assert changed['entropy_over_q'] == pytest.approx(2)
    assert x.grad is None


@pytest.mark.parametrize('distribution',['gaussian','gsde'])
def test_probe_preserves_weights_rng_optimizer_and_existing_grad(distribution):
    model = _tiny_model(actor_distribution=distribution, inner_temperature_mode='fixed',
        inner_temperature_initialization='fixed',inner_temperature=.0021,ent_coef=.0021,
        inner_actor_entropy_mode='squashed',outer_actor_entropy_mode='squashed',
        sac_actor_loss_scale_mode='tdmpc2_percentile_range')
    try:
        # Fresh AMBI critics have zero output layers. Give the probe a real
        # action-dependent Q before testing that it preserves every weight.
        with torch.no_grad():
            for head in model.agent.model._Qs:
                head[-1].weight.fill_(.1)
        before = _outer_state_digest(model)
        rng = torch.random.get_rng_state().clone()
        params = tuple(model.agent.model._pi.parameters())
        for p in params:p.grad = torch.ones_like(p)
        first = gradient_probe(model,np.zeros(3,np.float32),samples=8)
        second = gradient_probe(model,np.zeros(3,np.float32),samples=8)
        assert first == second
        assert first['parameter_count'] == sum(p.numel() for p in params)
        assert first['q_gradient_norm'] > 0
        assert first['entropy_gradient_norm'] > 0
        assert _outer_state_digest(model) == before
        assert torch.equal(rng,torch.random.get_rng_state())
        assert all(torch.equal(p.grad,torch.ones_like(p)) for p in params)
    finally:model.env.close()


@pytest.mark.parametrize('mode,draws',[('mean',0),('sample',2),('iid',9)])
def test_frozen_gsde_noise_lifetime_and_episode_replay(mode,draws):
    model = _tiny_model(actor_distribution='gsde',gsde_execution_resample_steps=8)
    try:
        controller = FrozenPriorController(model,mode)
        controller.reset(123)
        obs = np.zeros(3,np.float32)
        first = [controller.action(obs) for _ in range(9)]
        assert controller.draws == draws
        if mode=='sample':
            np.testing.assert_array_equal(first[0],first[7])
            assert not np.array_equal(first[0],first[8])
        controller.reset(123)
        second = [controller.action(obs) for _ in range(9)]
        np.testing.assert_array_equal(first,second)
    finally:model.env.close()


def test_shared_bank_rejects_tampering_and_non_upright_states(tmp_path):
    bank = dict(kind='collapse_upright_bank_v1',requested_roots=1,
        roots=[dict(head_height=1.4,torso_upright=.95,observation=[1.,2.])])
    bank['id']=canonical_hash(bank)
    path=tmp_path/'bank.json';path.write_text(json.dumps(bank))
    assert load_bank(path)['id']==bank['id']
    bank['roots'][0]['observation'][0]=3
    path.write_text(json.dumps(bank))
    with pytest.raises(ValueError,match='changed'):load_bank(path)
    bank.pop('id');bank['roots'][0]['head_height']=.2;bank['id']=canonical_hash(bank)
    path.write_text(json.dumps(bank))
    with pytest.raises(ValueError,match='upright'):load_bank(path)
