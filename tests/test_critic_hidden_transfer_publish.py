"""Historical comparison is explicit and read-only; current identities stay separate."""
from copy import deepcopy
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from slurm import ambi_transfer_sweep_publish as publication
from tests.test_transfer_sweep_publish import campaign, completed, FakeWandb
from tests.test_wandb_results_layout import Service, sample_spec
from utils import wandb_transfer_sweep_layout as layout


def hidden_campaign():
    value = campaign()
    value['cells'] = [dict(cell, transfer_mode='critic_hidden',
                          name=cell['name'].replace('critic_only', 'critic_hidden'))
                      for cell in value['cells'] if cell['transfer_mode'] == 'critic_only']
    for index, cell in enumerate(value['cells']):
        cell['index'] = index
    value['campaign_mode'] = 'critic-hidden'
    return value


def references(tmp_path):
    value = campaign()
    value['source_commit'] = publication.HISTORICAL_COMMIT
    data = {cell['name']: completed(index) for index, cell in enumerate(value['cells'])}
    return dict(campaign=value, completed=data,
                state={'cells': {name: {'status': 'published', 'run_id': name} for name in data}},
                binding={'campaign_root': str(tmp_path), 'evaluation_commit': value['source_commit']})


def test_initial_hidden_comparison_shows_eight_pending_plus25_historical_results(tmp_path):
    snapshot = publication.comparison_snapshot(tmp_path, hidden_campaign(), {'cells': {}}, {}, {},
                                               references=references(tmp_path))
    assert snapshot['total'] == 33 and snapshot['completed'] == snapshot['published'] == 25
    assert snapshot['new_completed'] == 0 and snapshot['new_total'] == 8
    assert len(snapshot['episodes']) == 125
    assert all(row['state'] == 'pending' and row['origin'] == 'new_evaluation'
               and row['return_mean'] is None for row in snapshot['results'][:8])
    assert all(row['state'] == 'published' and row['origin'] == 'historical_reference'
               and row['evaluation_commit'] == publication.HISTORICAL_COMMIT for row in snapshot['results'][8:])
    payload = publication.overview_payload(FakeWandb(), snapshot, hidden_comparison=True)
    assert len(payload['critic_hidden_sweep/settings']['data']) == 33
    assert not any(key.startswith('transfer_sweep/') for key in payload)
    charts = [value for key, value in payload.items() if '_return_vs_' in key]
    assert len(charts) == 8
    assert all(len(chart['keys']) == 4 and not chart['xs'][3] for chart in charts)
    assert 'evaluation_commit' in payload['critic_hidden_sweep/results']['columns']


def test_hidden_pairs_full_critic_fresh_actor_prior_by_both_seeds(tmp_path):
    value = hidden_campaign(); name = value['cells'][0]['name']
    ready = {name: completed(100)}
    ready[name]['episodes'].reverse()
    snapshot = publication.comparison_snapshot(tmp_path, value, {'cells': {}}, ready, {},
                                               references=references(tmp_path))
    pairs = [pair for pair in snapshot['paired_effects'] if pair['setting'] == name]
    assert {pair['comparison'] for pair in pairs} == {'vs_fresh', 'vs_prior', 'vs_full_critic', 'vs_actor'}
    assert all(pair['paired_episodes'] == 5 for pair in pairs)
    assert snapshot['new_completed'] == 1 and snapshot['completed'] == 26
    ready[name]['episodes'][0]['solver_seed'] += 1
    with pytest.raises(ValueError, match='five-seed'):
        publication.comparison_snapshot(tmp_path, value, {'cells': {}}, ready, {}, references=references(tmp_path))


def test_hidden_layout_preserves_old_sections_and_installs_idempotently(tmp_path):
    original = layout.patch_transfer_sweep_spec(sample_spec())
    hidden = layout.patch_transfer_sweep_spec(original, hidden_comparison=True)
    assert layout._without_owned(hidden, hidden_comparison=True) == original
    assert layout._installed(hidden) and layout._installed(hidden, hidden_comparison=True)
    assert layout.patch_transfer_sweep_spec(hidden, hidden_comparison=True) == hidden
    panels = [p for s in layout.transfer_sweep_sections(hidden_comparison=True) for p in s['panels']]
    assert len(panels) == 13
    intro = panels[0]['config']['value']
    for text in ('33 progress rows', 'eight new', '25 completed historical', 'including the first',
                 'cross-revision', 'Pending values are null', 'immutable run identities'):
        assert text in intro
    service = Service(); service.views[0]['spec'] = json.dumps(original)
    api = SimpleNamespace(_service_api=service)
    for attempt in range(2):
        receipt = layout.ensure_transfer_sweep_results_layout(api, entity='entity', project='project',
            receipt_dir=tmp_path, run_id='new-hidden-overview', hidden_comparison=True)
        assert receipt['changed'] == (attempt == 0)
        assert receipt['layout_version'] == layout.HIDDEN_LAYOUT_VERSION
        assert all(key.startswith('critic_hidden_sweep/') for key in receipt['expected_table_keys'])
    assert service.writes == 1
    assert layout._without_owned(json.loads(service.views[0]['spec']), hidden_comparison=True) == original


def reference_fixture(tmp_path, monkeypatch):
    from utils import eval_series
    root = tmp_path/'old-campaign'; root.mkdir()
    pub = tmp_path/'old-publication'; pub.mkdir()
    reference = campaign()
    reference.update(source_commit=publication.HISTORICAL_COMMIT, metadata_sha256='b'*64,
                     seeds=publication.SEEDS, controller_seed=55, max_steps=500, science={'old': True})
    (root/'campaign.json').write_text(json.dumps(reference))
    audit = dict(schema_version=1, scope='comparison_only_no_identity_reuse',
                 new_source_commit='e'*40,
                 cpu_default_path_proof={'status': 'passed'},
                 candidate_source_sha256={'RL/tdmpc2_core/inner_improvement.py': hashlib.sha256(b'fixture').hexdigest()},
                 reference_commit=publication.HISTORICAL_COMMIT,
                 reference_campaign_sha256=publication.digest(root/'campaign.json'))
    audit_path=tmp_path/'audit.json'; audit_path.write_text(json.dumps(audit))
    state=dict(campaign_root=str(root), campaign_sha256=publication.digest(root/'campaign.json'),
               evaluation_commit=publication.HISTORICAL_COMMIT, overview_run_id='old-overview', cells={})
    registries = {}
    for index, cell in enumerate(reference['cells']):
        name = cell['name']; directory = pub/'registry'/name; directory.mkdir(parents=True)
        record = completed(index)['record']
        registries[str(directory)] = dict(identity=record['identity'], run_id=name)
        (directory/'publication.json').write_text(json.dumps({'records': {record['record_id']: {'status': 'published'}}}))
        state['cells'][name] = dict(status='published', record_id=record['record_id'], run_id=name, run_dir=str(directory))
    (pub/'publication.json').write_text(json.dumps(state))
    calls=[]
    def load(root_arg, original, cell):
        assert original == reference and root_arg == root
        calls.append(cell['name'])
        return completed(cell['index'])
    monkeypatch.setattr(publication.subprocess, 'check_output', lambda *a, **k: b'fixture')
    monkeypatch.setattr(publication, 'load_completed', load)
    monkeypatch.setattr(eval_series, 'load_run', lambda path: registries[str(path)])
    args=SimpleNamespace(reference_root=root, reference_publication_root=pub, reference_audit=audit_path)
    new={**reference, **hidden_campaign()}
    return args, new, calls


def test_reference_ingestion_validates_original_identities_and_never_changes_old_files(tmp_path, monkeypatch):
    args, new, calls=reference_fixture(tmp_path, monkeypatch)
    files = {str(path): path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    result=publication.load_references(args,new)
    assert len(result['completed']) == len(calls) == 25
    assert result['binding']['evaluation_commit'] == publication.HISTORICAL_COMMIT
    assert result['binding']['policy'] == 'comparison_only_no_identity_reuse'
    assert {str(path): path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()} == files


@pytest.mark.parametrize('corruption', ['audit_scope','audit_sha','source_commit','publication_id','registry_ack','checkpoint','proof','candidate_sha','candidate_commit'])
def test_reference_ingestion_rejects_mismatches_without_summary_fallback(tmp_path, monkeypatch, corruption):
    args,new,_=reference_fixture(tmp_path,monkeypatch)
    if corruption in ('proof', 'candidate_sha', 'candidate_commit'):
        data=publication.read(args.reference_audit)
        if corruption == 'proof': data['cpu_default_path_proof']['status']='pending'
        elif corruption == 'candidate_commit': data['new_source_commit']='changed'
        else: data['candidate_source_sha256']['RL/tdmpc2_core/inner_improvement.py']='changed'
        args.reference_audit.write_text(json.dumps(data))
    elif corruption.startswith('audit'):
        data=publication.read(args.reference_audit)
        data['scope' if corruption=='audit_scope' else 'reference_campaign_sha256']='changed'
        args.reference_audit.write_text(json.dumps(data))
    elif corruption=='source_commit':
        path=args.reference_root/'campaign.json';data=publication.read(path);data['source_commit']='c'*40
        path.write_text(json.dumps(data))
    elif corruption=='publication_id':
        path=args.reference_publication_root/'publication.json';data=publication.read(path)
        next(iter(data['cells'].values()))['record_id']='changed';path.write_text(json.dumps(data))
    elif corruption=='registry_ack':
        path=next((args.reference_publication_root/'registry').glob('*/publication.json'))
        path.write_text(json.dumps({'records':{}}))
    else:
        new['checkpoint_sha256']='changed'
    with pytest.raises(ValueError):
        publication.load_references(args,new)


def test_hidden_watch_publishes_only_new_conditions(tmp_path, monkeypatch):
    import sys
    root=tmp_path/'campaign';root.mkdir();output=tmp_path/'publication'
    value=hidden_campaign();(root/'campaign.json').write_text(json.dumps(value))
    reference=references(tmp_path/'old'); fake=FakeWandb()
    monkeypatch.setitem(sys.modules,'wandb',fake)
    monkeypatch.setattr(publication,'load_references',lambda *a:reference)
    monkeypatch.setattr(publication,'publisher_commit',lambda:'p'*40)
    monkeypatch.setattr(publication,'gpu_jobs_active',lambda ids:False)
    monkeypatch.setattr(publication,'load_completed',lambda *a:completed(100))
    for cell in value['cells']:
        directory=root/'settings'/cell['name'];directory.mkdir(parents=True)
        (directory/'worker-completion.json').write_text('{}')
    published=[]
    def publish(output,state,cell,data):
        published.append(cell['name']);state['cells'][cell['name']]={'status':'published','run_id':'new'}
    monkeypatch.setattr(publication,'publish_cell',publish)
    def install(wandb,run,directory,**kwargs):
        assert kwargs=={'hidden_comparison':True}
        assert len(fake.logs[0]['critic_hidden_sweep/settings']['data'])==33
        return {'url':'https://wandb.ai/verified','status':'verified'}
    monkeypatch.setattr(publication,'install_layout',install)
    args=SimpleNamespace(root=root,publication_root=output,gpu_job_id=['1'],once=True,poll_seconds=.1,terminal_grace=0,
        reference_root=tmp_path/'old',reference_publication_root=tmp_path/'oldpub',reference_audit=tmp_path/'audit')
    publication.watch(args)
    assert len(published)==8 and all('critic_hidden' in name for name in published)
    assert fake.summary['status']=='complete' and fake.summary['completed_settings']==33
    assert fake.summary['new_completed_settings']==8
    assert 'comparison_reference' in fake.init_args['config']
