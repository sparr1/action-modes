"""J10 scope and real serial/seed-sharded evaluation equivalence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from slurm.ambi_aux_hj_sweep import MATRIX, cells, read, validate, training_summary
from slurm.ambi_aux_j_extension import match_j_identity
from slurm.ambi_aux_seed_panels import seed_groups, shard_location
from tests.test_aux_j_extension import baseline
from tests.test_aux_round_budget import identity

PATH = MATRIX.with_name('ambi_aux_soft_j10_625k.json')


def test_exact_three_cell_grid_and_seed_partition():
    from utils.ambi_research import load_preset_matrix
    matrix = load_preset_matrix(PATH)
    panel = cells(PATH)
    assert {(c['H'],c['J'],c['params']['inner_critic_updates_per_round']) for c in panel} == {(h,10,16) for h in (1,2,3)}
    assert len(panel) == 3 and matrix['comparisons']['sweep']['reference'] == 'prior'
    assert seed_groups(matrix) == [[101,102],[103,104],[105]]
    assert not matrix.get('execution')
    assert shard_location(matrix, {'directory':'/tmp/cell'}, 1) == (Path('/tmp/cell/shards/1'),[103,104])


@pytest.mark.parametrize('cell',cells(PATH),ids=lambda c:c['name'])
def test_j10_changes_only_rounds_and_nonevicting_capacity(cell):
    old,path = baseline(cell)
    match_j_identity(identity(cell,PATH),identity(old,path),10)
    prev_path = MATRIX.with_name('ambi_aux_soft_j1216_625k.json')
    previous = next(x for x in cells(prev_path) if x['H']==cell['H'] and x['J']==12 and x['params']['inner_critic_updates_per_round']==16)
    assert read(PATH)['shared_alg_params']==read(prev_path)['shared_alg_params']
    assert cell['params']=={**previous['params'],'inner_rounds':10}


@pytest.mark.parametrize('groups', [[[101,102],[102,103,104,105]], [[101,102],[103,104]],
                                  [[101,102],[103,104,106]], [[],[101,102,103,104,105]]])
def test_rejects_duplicate_missing_or_unexpected_seeds(groups):
    with pytest.raises(AssertionError): seed_groups({'seed_shards':groups})


def test_real_j10_serial_and_merged_panels_match(tmp_path):
    from evaluate_ambi_checkpoint import evaluate_matrix
    from tests.test_ambi_root_local_sac import _tiny_model, _tiny_params
    from utils.ambi_seed_shards import seal_episode_bundle, merge_episode_bundles
    from utils.eval_series_data import load_records
    cell = cells(PATH)[0]
    options=dict(aux_return_mode='sac',log_std_mapping='direct_clamp',sac_actor_loss_scale_mode='none',
                 inner_operator='none',inner_rounds=0,inner_rollouts_per_round=0,inner_updates_per_round=0,ent_coef='auto_0.2')
    model=_tiny_model(**options); checkpoint=tmp_path/'model_625000'
    model.agent.save(checkpoint);model.env.close()
    metadata=dict(schema_version=1,checkpoint=dict(kind='periodic',step=625000,episode=50,best_score=None,best_window=100),
        trial_run_params=dict(alg='AMBITDMPC2/AMBITDMPC2',env='Pendulum-v1',seed=55,device='cpu',
            total_steps=2000000,alg_params=_tiny_params(**options)),experiment_params=dict(env_params=dict(max_episode_steps=3)))
    Path(str(checkpoint)+'.metadata.json').write_text(json.dumps(metadata))
    matrix=read(PATH);matrix['shared_alg_params']['compile']=False
    matrix.pop('source_run')  # This synthetic checkpoint is not the production W&B backbone.
    matrix_path=tmp_path/'matrix.json';matrix_path.write_text(json.dumps(matrix))
    evaluate_matrix(matrix_path,checkpoint,selectors=['sweep/prior'],seeds=[101,102],max_steps=3,
                    device='cpu',bundle_dir=tmp_path/'prior')
    def evaluate(name,seeds):
        p=tmp_path/name
        evaluate_matrix(matrix_path,checkpoint,selectors=[cell['selector']],seeds=seeds,max_steps=3,
                        device='cpu',bundle_dir=p,reference_bundle=tmp_path/'prior')
        seal_episode_bundle(p)
        return p
    serial=evaluate('serial',[101,102])
    merged=merge_episode_bundles([evaluate('a',[101]),evaluate('b',[102])],tmp_path/'merged',expected_seeds=[101,102])
    sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    original=validate(serial,cell,seeds=[101,102],steps=3,checkpoint_sha=sha)
    actual=validate(merged,cell,seeds=[101,102],steps=3,checkpoint_sha=sha)
    for a,b in zip(original['runs'][0]['episodes'],actual['runs'][0]['episodes']):
        for key in ('seed','solver_seed','return','length','paired_return_delta'):
            assert a[key]==b[key]
        # Wall time changes between executions; all scientific probe values
        # and per-round work counters must still match exactly.
        summaries = [deepcopy(x['togo_round_summaries']) for x in (a,b)]
        for summary in summaries:
            for row in summary:
                row['metrics'].pop('probe_seconds',None)
        assert summaries[0]==summaries[1]
    inventory=tmp_path/'test-checkpoint-manifest.json'
    inventory.write_text(json.dumps({'source_run':'tests/local/j10',
                                    'checkpoints':[{'step':625000,'sha256':sha}]}))
    serial_record,=load_records(serial,inventory_path=inventory)
    merged_record,=load_records(merged,inventory_path=inventory)
    assert serial_record['identity']==merged_record['identity']
    s=training_summary(merged,cell,expected_steps=3)
    assert len(s['per_seed_decisions'])==6
    assert [x['index'] for x in s['update_curves'] if x['axis']=='critic_update']==list(range(1,161))
    assert [x['index'] for x in s['update_curves'] if x['axis']=='actor_update']==list(range(1,41))
