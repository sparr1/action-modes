"""Check duplicate prevention and the pinned workspace SDK round trip offline."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from slurm import metaworld_assembly_q10_workspace as layout


def public_api(rows):
    def runs(project, *, filters, per_page):
        assert project == f'{layout.ENTITY}/{layout.PROJECT}'
        assert filters == {'group': layout.GROUP} and per_page == 2
        return iter(rows)

    return SimpleNamespace(runs=runs)


def native_run(tmp_path):
    run_id = 'a' * 32
    lineage = tmp_path / 'LINEAGE.json'
    lineage.write_text(json.dumps({'metadata': {'total_steps': 1_000_000, 'initial_wandb_run_id': run_id}}))
    config = {
        'alg_params': dict(**layout.REDUCTIONS, eval_freq=25_000, eval_episodes=10,
                           compile=True, compile_strict=True, inner_cuda_graph=True,
                           critic_geometry_diagnostics=True),
        'run_params': dict(seed=55, total_steps=1_000_000, env='MetaWorld-v0', resolved_runtime={
            'observation': dict(task='assembly', episode_length=100, action_repeat=2)}),
    }
    run = SimpleNamespace(group=layout.GROUP, name=layout.RUN_NAME, id=run_id, config=config,
                          state='running', summary={'env_step': 31_000})
    return run, lineage


def test_discovery_rejects_duplicate_and_wrong_lineage(tmp_path):
    run, lineage = native_run(tmp_path)
    assert layout.discover(public_api([run]), None, lineage)['run_id'] == run.id
    with pytest.raises(layout.WorkspaceError, match='Duplicate runs'):
        layout.discover(public_api([run, run]), None, lineage)
    with pytest.raises(layout.WorkspaceError, match='unexpected run identity'):
        layout.discover(public_api([run]), 'b' * 32, lineage)
    run.id = 'c' * 32
    with pytest.raises(layout.WorkspaceError, match='differs from the native'):
        layout.discover(public_api([run]), None, lineage)


def test_sdk_reuses_pending_view_and_preserves_user_sections(tmp_path, monkeypatch):
    ws = pytest.importorskip('wandb_workspaces.workspaces')
    pytest.importorskip('wandb_workspaces.reports.v2')
    run, lineage = native_run(tmp_path)
    store = {}
    url = f'https://wandb.ai/{layout.ENTITY}/{layout.PROJECT}?nw=mockview'
    monkeypatch.setattr(ws.Workspace, 'url', property(lambda self: url))
    monkeypatch.setattr(ws.Workspace, 'save', lambda self: store.update(model=self._to_model()))
    monkeypatch.setattr(ws.Workspace, 'from_url', classmethod(lambda cls, _: cls._from_model(deepcopy(store['model']))))

    receipt = tmp_path / 'workspace.json'
    pending = layout.install(receipt, api=public_api([]), lineage=lineage)
    assert pending['api_verified'] and pending['panel_count'] == 4 and pending['run']['run_id'] is None
    assert layout.GROUP in store['model'].model_dump_json()
    workspace = ws.Workspace.from_url(url)
    workspace.sections.append(ws.Section(name='User notes', is_open=True))
    workspace.save()
    bound = layout.install(receipt, api=public_api([run]), lineage=lineage)
    assert bound['api_verified'] and bound['url'] == pending['url'] and bound['run']['run_id'] == run.id
    assert 'User notes' in store['model'].model_dump_json()
    assert run.id in store['model'].model_dump_json()
    assert all(metric in store['model'].model_dump_json() for _, metric, _ in layout.CURVES)
