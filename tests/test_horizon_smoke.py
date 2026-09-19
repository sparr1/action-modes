"""Prepared matrix and actual evaluator/publisher smoke wiring, using tiny weights locally."""

from argparse import Namespace
import hashlib
import json
from pathlib import Path

import pytest

from slurm import ambi_horizon_diagnostics_smoke as smoke
from slurm import ambi_aux_hj_sweep as sweep
from tests.test_ambi_root_local_sac import _tiny_model, _tiny_params
from tests.test_ambi_config_decoupling import _build_cfg


def test_eight_settings_and_only_conditioning_differs_within_pairs():
    matrix=json.loads(smoke.MATRIX.read_text())
    cells=sweep.cells(smoke.MATRIX)
    assert len(cells)==8
    assert {(c['H'],c['J']) for c in cells}=={(3,2),(3,4),(3,8),(1,4)}
    for off,on in zip(cells[::2],cells[1::2]):
        a=dict(off['params']);b=dict(on['params'])
        assert a.pop('inner_horizon_conditioning')=='none'
        assert b.pop('inner_horizon_conditioning')=='one_hot'
        assert a==b and a['inner_horizon_diagnostics']
    for cell in cells:
        params={**matrix['shared_alg_params'], **cell['params']}
        cfg=_build_cfg(**{**{k:v for k,v in params.items() if v is not None},'aux_return_mode':'sac'})
        assert cfg.inner_critic_updates_per_round==16 and cfg.inner_actor_updates_per_round==4
        assert cfg.inner_actor_lr==cfg.inner_critic_lr==3e-4 and cfg.inner_critic_target_tau==.01
        assert cfg.inner_model_step_budget<=cfg.inner_replay_capacity==3072
        assert cfg.inner_sac_critic_target=='entropy_augmented' and cfg.inner_terminal_entropy=='outer'


@pytest.mark.parametrize('index',[0,1])
def test_short_smoke_evaluator_all_conditioning_and_logging_arms(tmp_path,monkeypatch,index):
    options=dict(aux_return_mode='sac',log_std_mapping='direct_clamp',sac_actor_loss_scale_mode='none',
                 inner_operator='none',inner_rounds=0,inner_rollouts_per_round=0,inner_updates_per_round=0,
                 ent_coef='auto_0.004345251712948084')
    model=_tiny_model(**options)
    checkpoint=tmp_path/'model_625000'
    model.agent.save(checkpoint);model.env.close()
    metadata=dict(schema_version=1,checkpoint=dict(kind='periodic',step=625000,episode=50,best_score=None,best_window=100),
                  trial_run_params=dict(alg='AMBITDMPC2/AMBITDMPC2',env='Pendulum-v1',seed=55,device='cpu',
                                        total_steps=2000000,alg_params=_tiny_params(**options)),
                  experiment_params=dict(env_params=dict(max_episode_steps=3)))
    Path(str(checkpoint)+'.metadata.json').write_text(json.dumps(metadata))
    matrix=json.loads(smoke.MATRIX.read_text());matrix['shared_alg_params']['compile']=False
    path=tmp_path/'template.json';path.write_text(json.dumps(matrix))
    monkeypatch.setattr(smoke,'MATRIX',path)
    monkeypatch.setattr(sweep,'CHECKPOINT_SHA',hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    smoke.run(Namespace(checkpoint=checkpoint,inventory=None,output=tmp_path/'out',index=index,device='cpu'))
    receipt=json.loads((tmp_path/'out/validation.json').read_text())
    assert receipt['status']=='complete' and len(receipt['cases'])==4
    assert all(c['result']['outer_state_unchanged'] for c in receipt['cases'])
