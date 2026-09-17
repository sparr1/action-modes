"""Real H3/J4 solves, exact replay retention and complete diagnostic coverage."""
import copy
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from evaluate_ambi_checkpoint import evaluate_matrix
from slurm.ambi_aux_hj_sweep import MATRIX, cells, training_summary, validate
from tests.test_ambi_root_local_sac import _tiny_model, _tiny_params


def test_complete_grid_and_objectives():
    panel=cells()
    assert len(panel)==36 and len({c['name'] for c in panel})==36
    assert {(c['H'],c['J']) for c in panel}=={(h,j) for h in (1,2,3) for j in (1,2,4)}
    assert max(128*c['H']*c['J'] for c in panel)==1536
    assert panel[0]['J']==4 and panel[-1]['J']==1
    for c in panel:
        p=c['params']
        assert (p['inner_terminal_entropy']=='outer') == c['name'].startswith('soft_soft_')
        assert (p['inner_sac_critic_target']=='entropy_augmented') == c['name'].startswith('soft_')
        assert p['inner_entropy_enabled'] == ('return_return_zero' not in c['name'])


@pytest.fixture(scope='module',params=['soft_soft_h3_j4','soft_return_h3_j4',
                                     'return_return_alpha_h3_j4','return_return_zero_h3_j4',
                                     'soft_soft_h2_j2','soft_return_h1_j1'])
def panel(request,tmp_path_factory):
    root=tmp_path_factory.mktemp('hj')
    cell=next(c for c in cells() if c['name']==request.param)
    options=dict(aux_return_mode='sac',log_std_mapping='direct_clamp',sac_actor_loss_scale_mode='none',
                 inner_operator='none',inner_rounds=0,inner_rollouts_per_round=0,inner_updates_per_round=0,
                 ent_coef='auto_0.2')
    model=_tiny_model(**options)
    checkpoint=root/'model_625000'
    model.agent.save(checkpoint);model.env.close()
    metadata=dict(schema_version=1,checkpoint=dict(kind='periodic',step=625000,episode=50,best_score=None,best_window=100),
                  trial_run_params=dict(alg='AMBITDMPC2/AMBITDMPC2',env='Pendulum-v1',seed=55,device='cpu',
                                        total_steps=2000000,alg_params=_tiny_params(**options)),
                  experiment_params=dict(env_params=dict(max_episode_steps=3)))
    Path(str(checkpoint)+'.metadata.json').write_text(json.dumps(metadata))
    matrix=json.loads(MATRIX.read_text());matrix['shared_alg_params']['compile']=False
    path=root/'matrix.json';path.write_text(json.dumps(matrix))
    evaluate_matrix(path,checkpoint,selectors=['sweep/prior'],seeds=[101,102],max_steps=3,
                    device='cpu',bundle_dir=root/'prior')
    evaluate_matrix(path,checkpoint,selectors=[cell['selector']],seeds=[101,102],max_steps=3,
                    device='cpu',bundle_dir=root/'bundle',reference_bundle=root/'prior')
    return root,cell,hashlib.sha256(checkpoint.read_bytes()).hexdigest()


def test_real_replay_and_complete_update_metrics(panel):
    root,cell,sha=panel
    manifest=validate(root/'bundle',cell,seeds=[101,102],steps=3,checkpoint_sha=sha)
    summary=training_summary(root/'bundle',cell,expected_steps=3)
    assert len(summary['per_seed_decisions'])==6
    assert len(summary['decision_curves'])==3
    critic=[r for r in summary['update_curves'] if r['axis']=='critic_update']
    actor=[r for r in summary['update_curves'] if r['axis']=='actor_update']
    assert [r['index'] for r in critic]==list(range(1,32*cell['J']+1))
    assert [r['index'] for r in actor]==list(range(1,4*cell['J']+1))
    assert all(r['metrics']['critic_loss']['count']==6 for r in critic)
    assert all(r['metrics']['actor_loss']['count']==6 for r in actor)
    assert all(r['metrics']['decision/reward']['count']==2 for r in summary['decision_curves'])
    assert manifest['runs'][0]['result']['model_metrics']['inner_buffer_size']['mean']==128*cell['H']*cell['J']


def test_missing_update_or_wrong_replay_is_rejected(panel,tmp_path):
    root,cell,_=panel
    manifest=json.loads((root/'bundle/manifest.json').read_text())
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    for name in manifest['runs'][0]['trace_files']:
        target=tmp_path/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((root/'bundle'/name).read_bytes())
    name=manifest['runs'][0]['trace_files'][0]
    rows=[json.loads(line) for line in gzip.open(tmp_path/name,'rt')]
    original=copy.deepcopy(rows)
    index=next(i for i,r in enumerate(rows) if r.get('updated_critic'))
    del rows[index]
    with gzip.open(tmp_path/name,'wt') as f:
        f.write('\n'.join(json.dumps(r) for r in rows)+'\n')
    with pytest.raises(AssertionError): training_summary(tmp_path,cell,expected_steps=3)
    rows=original
    next(r for r in rows if r['phase']=='collection')['replay_size']=1
    with gzip.open(tmp_path/name,'wt') as f:
        f.write('\n'.join(json.dumps(r) for r in rows)+'\n')
    with pytest.raises(AssertionError): training_summary(tmp_path,cell,expected_steps=3)
