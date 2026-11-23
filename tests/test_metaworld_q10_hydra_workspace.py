"""Verify the Hydra Q10 workspace without logging in or changing remote data."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from slurm import metaworld_q10_hydra_workspace as layout


def public_api(by_group):
    def runs(project, *, filters, per_page):
        assert project == f'{layout.ENTITY}/{layout.PROJECT}' and per_page == 2
        assert filters['group'] in ('metaworld-push-mrsq-q10-s55', 'metaworld-stick-pull-mrsq-q10-s55')
        return iter(by_group.get(filters['group'], []))

    return SimpleNamespace(runs=runs)


def test_exact_groups_reject_duplicate_registration():
    pending = layout.discover(public_api({}), [])
    assert [row['task'] for row in pending] == ['push', 'stick-pull']
    assert all(row['run_id'] is None for row in pending)
    group = pending[0]['group']
    run = SimpleNamespace(name=pending[0]['name'], group=group)
    with pytest.raises(layout.WorkspaceError, match='duplicate W&B run'):
        layout.discover(public_api({group: [run, run]}), pending)


def test_pending_layout_roundtrips_with_locked_sdk(tmp_path, monkeypatch):
    ws = pytest.importorskip('wandb_workspaces.workspaces')
    pytest.importorskip('wandb_workspaces.reports.v2')
    store = {}
    url = f'https://wandb.ai/{layout.ENTITY}/{layout.PROJECT}?nw=mockview'
    monkeypatch.setattr(ws.Workspace, 'url', property(lambda self: url))
    monkeypatch.setattr(ws.Workspace, 'save', lambda self: store.update(model=self._to_model()))
    monkeypatch.setattr(ws.Workspace, 'from_url', classmethod(lambda cls, _: cls._from_model(deepcopy(store['model']))))

    receipt = tmp_path / 'workspace.json'
    pending = layout.install(receipt, api=public_api({}))
    assert pending['api_verified'] and pending['panel_count'] == 4
    push = pending['rows'][0]
    run = SimpleNamespace(group=push['group'], name=push['name'], state='running', id='a' * 32,
                          summary={'env_step': 25_000})
    refreshed = layout.install(receipt, api=public_api({run.group: [run]}))
    assert refreshed['api_verified'] and refreshed['url'] == pending['url']
    assert refreshed['rows'][0]['run_id'] == run.id and refreshed['rows'][1]['run_id'] is None
    wire = store['model'].model_dump_json()
    assert all(row['group'] in wire for row in pending['rows'])
    assert all(metric in wire for _, metric, _ in layout.CURVES)
