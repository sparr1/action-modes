"""A dedicated campaign view must not inherit unrelated project filters."""
from copy import deepcopy
import errno
import hashlib
import json
import re
from types import SimpleNamespace

import pytest

from utils import wandb_transfer_discovery_layout as layout
from tests.test_wandb_results_layout import view


class Service:
    def __init__(self, *, timeout=False, apply=True, concurrent=False):
        self.views = [view(), view(name='nw-other-v', id='other')]
        self.timeout = timeout; self.apply = apply; self.concurrent = concurrent
        self.reads = 0; self.writes = 0

    def execute_graphql(self, query, variables):
        if 'mutation ' in query:
            self.writes += 1
            assert variables['entityName'] == 'entity'
            assert variables['projectName'] == 'project'
            assert 'id' not in variables
            if self.apply:
                self.views.append(dict(variables, id='saved'))
            if self.timeout:
                raise TimeoutError('Response lost')
            return {'upsertView':{'view':{'id':'saved','name':variables['name']},'inserted':True}}
        self.reads += 1
        if self.concurrent and self.reads == 2:
            self.views[0]['displayName'] = 'Concurrent user edit'
        return {'project':{'allViews':{'edges':[{'node':deepcopy(v)} for v in self.views]}}}


def install(tmp_path, service, run_id='publication123'):
    return layout.ensure_discovery_saved_view(SimpleNamespace(_service_api=service),
        entity='entity', project='project', receipt_dir=tmp_path, run_id=run_id)


def test_saved_view_selects_exact_publication_and_preserves_all_existing_views(tmp_path):
    service = Service(); before = deepcopy(service.views)
    receipt = install(tmp_path, service)
    assert receipt['changed'] and receipt['preserved_existing_views'] == 2
    assert service.views[:2] == before
    spec = json.loads(service.views[-1]['spec'])
    assert layout._saved_installed(spec, 'publication123')
    runset, = spec['section']['runSets']
    assert runset['filters']['filters'] == [{'key':{'section':'config','name':'publication_id'},
        'op':'=', 'value':'publication123', 'disabled':False}]
    assert runset['search'] == {'query':''} and runset['enabled']
    assert runset['selections'] == {'root':1, 'bounds':[], 'tree':[]}
    assert spec['section']['panelBankConfig']['panelPlacementOverrides'] == {}
    panels = [p for s in layout._bank(spec)['sections'] for p in s['panels']]
    assert sum(p['viewType'] == 'Vega2' for p in panels) == 6
    assert sum(p['viewType'] == 'Media Browser' for p in panels) == 3
    assert receipt['url'] == 'https://wandb.ai/entity/project?nw=transfer575publication123'
    assert receipt['view_name'] == 'nw-transfer575publication123-v'
    assert '/runs/publication123?' in receipt['run_url']
    assert not install(tmp_path, service)['changed'] and service.writes == 1


@pytest.mark.parametrize('run_id', ['publication123', 'trial-1', 'trial_1'])
def test_saved_view_slug_is_alphanumeric_and_keeps_exact_publication_filter(tmp_path, run_id):
    service = Service()
    receipt = install(tmp_path, service, run_id)
    slug = receipt['view_name'][3:-2]
    assert re.fullmatch(r'[A-Za-z0-9]+', slug)
    suffix = (run_id if run_id.isalnum()
              else 'h' + hashlib.sha256(run_id.encode('utf-8')).hexdigest())
    assert slug == 'transfer575' + suffix
    assert receipt['url'] == f'https://wandb.ai/entity/project?nw={slug}'
    assert receipt['run_url'] == f'https://wandb.ai/entity/project/runs/{run_id}?nw={slug}'
    assert layout._saved_installed(json.loads(service.views[-1]['spec']), run_id)
    assert not install(tmp_path, service, run_id)['changed'] and service.writes == 1


def test_legacy_hyphenated_view_is_preserved_when_creating_native_view(tmp_path):
    service = Service()
    legacy = view(name='nw-transfer575-publication123-v', id='legacy-discovery')
    service.views.append(legacy)
    before = deepcopy(service.views)
    receipt = install(tmp_path, service)
    assert service.views[:3] == before
    assert receipt['changed'] and receipt['preserved_existing_views'] == 3
    assert service.views[-1]['name'] == 'nw-transfer575publication123-v'
    assert not install(tmp_path, service)['changed'] and service.writes == 1


@pytest.mark.parametrize('change', ['filter', 'search', 'selection', 'panel', 'type', 'duplicate'])
def test_existing_modified_view_is_never_overwritten(tmp_path, change):
    service = Service(); install(tmp_path, service)
    node = service.views[-1]; spec = json.loads(node['spec'])
    if change == 'filter': spec['section']['runSets'][0]['filters']['filters'][0]['value'] = 'other'
    elif change == 'search': spec['section']['runSets'][0]['search']['query'] = 'exclude'
    elif change == 'selection': spec['section']['runSets'][0]['selections']['root'] = 0
    elif change == 'panel': layout._bank(spec)['sections'][0]['panels'].pop()
    elif change == 'type': node['type'] = 'run-view'
    elif change == 'duplicate': service.views.append(deepcopy(node))
    node['spec'] = json.dumps(spec)
    with pytest.raises(layout.ResultsLayoutError, match='differs'):
        install(tmp_path, service)
    assert service.writes == 1


def test_uncertain_applied_create_is_reconciled_and_retry_is_idempotent(tmp_path):
    service = Service(timeout=True)
    assert install(tmp_path, service)['uncertain_response_reconciled']
    assert not install(tmp_path, service)['changed'] and service.writes == 1


def test_unapplied_create_records_uncertainty_and_can_retry(tmp_path):
    service = Service(timeout=True, apply=False)
    with pytest.raises(layout.ResultsLayoutError, match='could not be verified'):
        install(tmp_path, service)
    assert json.loads((tmp_path/'results-layout-receipt.json').read_text())['status'] == 'uncertain'
    service.apply = True; service.timeout = False
    assert install(tmp_path, service)['status'] == 'verified'
    assert len(service.views) == 3


def test_concurrent_edit_aborts_before_create(tmp_path):
    service = Service(concurrent=True)
    with pytest.raises(layout.ResultsLayoutError, match='changed during preparation'):
        install(tmp_path, service)
    assert service.writes == 0


def test_invalid_run_id_does_not_access_service(tmp_path):
    service = Service()
    with pytest.raises(layout.ResultsLayoutError, match='valid explicit'):
        install(tmp_path, service, '../other')
    assert service.reads == service.writes == 0


@pytest.mark.parametrize('error_number', [errno.ESTALE, errno.EACCES])
def test_receipt_write_retries_only_estale_with_a_bounded_budget(tmp_path, monkeypatch, error_number):
    calls = []; delays = []
    def fail(*args, **kwargs):
        calls.append(1)
        raise OSError(error_number, 'test I/O failure')
    monkeypatch.setattr(layout, 'atomic_json', fail)
    monkeypatch.setattr(layout.time, 'sleep', delays.append)
    with pytest.raises(OSError):
        layout._saved_receipt_write(tmp_path/'receipt.json', {})
    assert len(calls) == (5 if error_number == errno.ESTALE else 1)
    assert delays == ([1, 2, 4, 8] if error_number == errno.ESTALE else [])
