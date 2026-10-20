"""Install and verify connected native checkpoint curves from a stable catalog.

Run only in the isolated wandb==0.28.1 / wandb-workspaces==0.4.5 runtime:
  python slurm/ambi_entropy_kl_curves_layout.py --catalog curves.json --receipt layout.json

Catalog fields: entity, project, workspace_url, curves[{run_id, name, color,
mode, run_dir}], and optional other_workspace_urls. This changes workspace
presentation only; it never uploads, merges, or rewrites scientific results.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import re
import sys
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import eval_series as series
from slurm.ambi_entropy_kl_mppi500k_publish import METRICS

PREFIX = 'Entropy and KL checkpoint curves / '
REPLACED_PREFIX = 'MPPI 500k objectives / '
MODES = {'prior', 'entropy', 'kl', 'both'}
PANEL_COUNT = len(METRICS) + 3
WORKSPACE_NAME = 'MPPI-trained backbone: entropy and KL checkpoint curves'


class LayoutError(ValueError):
    """Catalog or persisted workspace does not satisfy the native curve contract."""


def _workspace_id(url, entity, project):
    parsed = urlparse(url)
    path = parsed.path.rstrip('/')
    if parsed.hostname == 'forge.coreweave.com' and path.startswith('/wandb/'):
        path = path[len('/wandb'):]
    view = parse_qs(parsed.query).get('nw', [])
    if (parsed.scheme != 'https' or parsed.hostname not in {'wandb.ai', 'forge.coreweave.com'}
            or path != f'/{entity}/{project}' or len(view) != 1
            or not re.fullmatch(r'[A-Za-z0-9_-]+', view[0])):
        raise LayoutError('Workspace URL must identify an existing named view in the catalog project')
    return view[0]


def validate_catalog(catalog):
    if not isinstance(catalog, dict):
        raise LayoutError('Catalog must be an object')
    for key in ('entity', 'project'):
        if not isinstance(catalog.get(key), str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', catalog[key]):
            raise LayoutError('Invalid catalog ' + key)
    rows = catalog.get('curves')
    if not isinstance(rows, list) or len(rows) != 4 or {r.get('mode') for r in rows} != MODES:
        raise LayoutError('Catalog must contain the four independently identified methods')
    for row in rows:
        if (not isinstance(row.get('run_id'), str) or not re.fullmatch(r'[A-Za-z0-9_-]+', row['run_id'])
                or not isinstance(row.get('name'), str) or not row['name'].strip()
                or not isinstance(row.get('run_dir'), str) or not Path(row['run_dir']).is_absolute()
                or not isinstance(row.get('color'), str) or not re.fullmatch(r'#[0-9A-Fa-f]{6}', row['color'])):
            raise LayoutError('Each method needs a run ID, readable name, absolute registry directory and hex color')
        if re.search(r'\b\d+(?:\.\d+)?[kKmM]\b|\b\d{5,}\b', row['name']):
            raise LayoutError('Stable curve names must not contain a checkpoint number')
    for key in ('run_id', 'name', 'color', 'run_dir'):
        values = [r[key].lower() if key == 'color' else r[key] for r in rows]
        if len(set(values)) != len(rows):
            raise LayoutError('Methods must have distinct ' + key)
    urls = [catalog.get('workspace_url'), *catalog.get('other_workspace_urls', [])]
    if any(not isinstance(url, str) for url in urls):
        raise LayoutError('Workspace URLs must be strings')
    views = [_workspace_id(url, catalog['entity'], catalog['project']) for url in urls]
    if len(set(views)) != len(views):
        raise LayoutError('Duplicate workspace view')
    return urls


def _color(value):
    # Server read-back expands line colors into {color, transparentColor}.
    return value.get('color') if isinstance(value, dict) else value


def _filter_model(expression, expr=None):
    if isinstance(expression, str):
        model = expr.expr_to_filters(expression)
    elif isinstance(expression, list):
        if len(expression) != 1:
            return {'invalid_filter_count': len(expression)}
        model = expression[0].to_model()
    else:
        model = expression.to_model()
    wire = model.model_dump(exclude_none=True) if hasattr(model, 'model_dump') else model.dict(exclude_none=True)
    def normalize(node):
        if isinstance(node, dict):
            node = {k: normalize(v) for k, v in node.items() if not (k == 'disabled' and v is False)}
            if node.get('op') in ('AND', 'OR') and len(node.get('filters', [])) == 1 and 'key' not in node:
                return node['filters'][0]
        elif isinstance(node, list):
            return [normalize(v) for v in node]
        return node
    return normalize(wire)


def _sections(catalog, ws, wr):
    rows = catalog['curves']
    markdown = '''**Connected checkpoint curves for the MPPI-trained backbone.**

Each color is one independently identified controller, with its actual measured
checkpoint results connected on the source-training-decision axis. Native
hover and legend labels identify prior, entropy-only, KL-only and joint
refinement. No averaging across controllers, smoothing, or synthetic results
are applied. Missing checkpoints remain missing; a single measurement is one
point until another compatible checkpoint result is available.

Adapted controllers use H3/J8/C16/A4/N128/B256, sampled execution, five matched
environment seeds and frozen outer state. Where KL is enabled, beta equals the
saved outer alpha for that checkpoint; this is the same coefficient rule across
checkpoints, not a claim that its numeric value is constant across training.
Automatic entropy uses alpha learning rate0.0003. KL-only has no temperature
updates. Paired gains use the sampled prior from that same checkpoint and seeds.

Return SD describes variability across environment seeds, not training seeds.
Entropy comes from actor-update batches; the maximum is averaged per solve.
Final KL is measured against the frozen prior. Prior policy diagnostics and
prior timing comparisons are omitted: historical prior evaluations used
different compilation and probe settings. Adapted-controller timings use
measured episode controller time. Full traces remain on Oscar.
'''
    def plot(title, metric):
        return wr.LinePlot(title=title, x=series.X_AXIS, y=[metric], aggregate=False,
            smoothing_type='none', legend_template='${run:name}', max_runs_to_show=len(rows),
            title_x='Source training decisions', layout=wr.Layout(w=12, h=8),
            line_colors={r['run_id'] + ':' + metric: r['color'] for r in rows},
            line_marks={r['run_id'] + ':' + metric: 'solid' for r in rows},
            line_titles={r['run_id'] + ':' + metric: r['name'] for r in rows})
    return [ws.Section(name=PREFIX + 'Protocol and progress', is_open=True, panels=[
        wr.MarkdownPanel(markdown=markdown, layout=wr.Layout(w=24, h=10)),
        wr.ScalarChart(title='Published checkpoints per method', metric='progress/checkpoints_complete',
                       layout=wr.Layout(w=12, h=6)),
        wr.ScalarChart(title='Failed checkpoint cells', metric='progress/failed_cells',
                       layout=wr.Layout(w=12, h=6))]),
        ws.Section(name=PREFIX + 'Return and improvement', is_open=True,
                   panels=[plot(*item) for item in METRICS[:5]]),
        ws.Section(name=PREFIX + 'Runtime and policy diagnostics', is_open=True,
                   panels=[plot(*item) for item in METRICS[5:]])]


def _verify(work, catalog, url, expected_filter, wr, expr):
    rows = catalog['curves']
    if (work.name != WORKSPACE_NAME or work.entity != catalog['entity'] or work.project != catalog['project']
            or _workspace_id(work.url, work.entity, work.project) != _workspace_id(url, work.entity, work.project)):
        raise LayoutError('Saved workspace target changed')
    owned = [section for section in work.sections if section.name.startswith(PREFIX)]
    if (len(owned) != 3 or any(section.name.startswith(REPLACED_PREFIX) for section in work.sections)
            or sum(len(section.panels) for section in owned) != PANEL_COUNT):
        raise LayoutError('Connected curve sections are missing, duplicated or stale')
    settings = work.runset_settings
    if (settings.groupby or settings.query
            or _filter_model(settings.filters, expr) != expected_filter):
        raise LayoutError('Saved run filter or independent-run grouping changed')
    for row in rows:
        run = settings.run_settings.get(row['run_id'])
        if run is None or run.disabled or _color(run.color) != row['color']:
            raise LayoutError('Saved run colors or visibility changed')
    if (work.settings.x_axis != series.X_AXIS or work.settings.remove_legends_from_panels
            or not work.settings.tooltip_color_run_names):
        raise LayoutError('Saved checkpoint axis or native legend changed')
    plots = [p for section in owned for p in section.panels if isinstance(p, wr.LinePlot)]
    expected_metrics = {metric for _, metric in METRICS}
    actual_metrics = set()
    for panel in plots:
        if (panel.aggregate or getattr(panel.x, 'name', panel.x) != series.X_AXIS
                or panel.smoothing_type != 'none' or panel.legend_template != '${run:name}'
                or panel.max_runs_to_show < len(rows) or len(panel.y) != 1):
            raise LayoutError('Saved panel lost unsmoothed independent native checkpoint curves')
        metric = getattr(panel.y[0], 'name', panel.y[0])
        if metric not in expected_metrics or metric in actual_metrics:
            raise LayoutError('Saved panel metrics changed or duplicated')
        actual_metrics.add(metric)
        for row in rows:
            key = row['run_id'] + ':' + metric
            if (panel.line_titles.get(key) != row['name'] or panel.line_marks.get(key) != 'solid'
                    or _color(panel.line_colors.get(key)) != row['color']):
                raise LayoutError('Saved panel lost readable names, solid lines or exact colors')
    if actual_metrics != expected_metrics:
        raise LayoutError('Saved layout lacks required metrics')
    return {'url': work.url, 'view_id': _workspace_id(url, work.entity, work.project),
            'panel_count': PANEL_COUNT, 'api_verified': True}


def install_catalog(catalog, receipt_path, *, backend=None):
    """Update existing views only; fail clearly if the server drops any styling."""
    urls = validate_catalog(catalog)
    if backend is None:
        for package, version in (('wandb', '0.28.1'), ('wandb-workspaces', '0.4.5')):
            if importlib.metadata.version(package) != version:
                raise LayoutError(f'Layout requires isolated {package}=={version}')
        import wandb_workspaces.workspaces as ws
        import wandb_workspaces.reports.v2 as wr
        from wandb_workspaces import expr
    else:
        ws, wr, expr = backend
    rows = catalog['curves']
    wanted_filter = expr.Metric('ID').isin([row['run_id'] for row in rows])
    expected_filter = _filter_model(wanted_filter)
    receipt = {'catalog_sha256': series._digest(catalog), 'url': urls[0], 'views': [],
               'updated_at': datetime.now(timezone.utc).isoformat(), 'api_verified': False,
               'curves': deepcopy(rows)}
    series._atomic_json(receipt_path, receipt)
    try:
        for url in urls:
            work = ws.Workspace.from_url(url)
            if work.entity != catalog['entity'] or work.project != catalog['project']:
                raise LayoutError('Workspace target differs from catalog')
            work.name = WORKSPACE_NAME
            sections = _sections(catalog, ws, wr)
            # Replace our old and new sections in place, retaining every unrelated section.
            replacement, inserted = [], False
            for section in work.sections:
                if section.name.startswith((PREFIX, REPLACED_PREFIX)):
                    if not inserted:
                        replacement.extend(sections)
                        inserted = True
                else:
                    replacement.append(section)
            work.sections = replacement if inserted else replacement + sections
            work.runset_settings = ws.RunsetSettings(filters=[wanted_filter], groupby=[],
                run_settings={r['run_id']: ws.RunSettings(color=r['color'], disabled=False) for r in rows})
            work.settings.x_axis = series.X_AXIS
            work.settings.max_runs = len(rows)
            work.settings.remove_legends_from_panels = False
            work.settings.tooltip_number_of_runs = 'all_runs'
            work.settings.tooltip_color_run_names = True
            work._auto_generate_panels = False  # SDK0.4.5 exposes a read-only alias.
            work.save()
            verified = _verify(ws.Workspace.from_url(url), catalog, url, expected_filter, wr, expr)
            receipt['views'].append(verified)
            series._atomic_json(receipt_path, receipt)
    except Exception as exc:
        receipt.update(error_type=type(exc).__name__, error=str(exc))
        series._atomic_json(receipt_path, receipt)
        raise
    receipt.update(api_verified=True, panel_count=PANEL_COUNT)
    series._atomic_json(receipt_path, receipt)
    return receipt


def _verify_view_spec(spec, catalog):
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except json.JSONDecodeError as exc:
            raise LayoutError('Saved native view has invalid JSON') from exc
    wire = json.dumps(spec, ensure_ascii=False)
    tokens = [PREFIX, series.X_AXIS, 'progress/checkpoints_complete', 'progress/failed_cells',
              *[metric for _, metric in METRICS],
              *[row[key] for row in catalog['curves'] for key in ('run_id', 'name', 'color')]]
    if any(token not in wire for token in tokens):
        raise LayoutError('Native workspace lost required panels, runs, names or colors')
    def dictionaries(value):
        if isinstance(value, dict):
            yield value
            for item in value.values():
                yield from dictionaries(item)
        elif isinstance(value, list):
            for item in value:
                yield from dictionaries(item)
    configs = [node['config'] for node in dictionaries(spec)
               if node.get('viewType') == 'Run History Line Plot' and isinstance(node.get('config'), dict)]
    for _, metric in METRICS:
        matches = [config for config in configs if config.get('metrics') == [metric]
                   and config.get('xAxis') == series.X_AXIS and config.get('aggregate') is False
                   and all(config.get('overrideMarks', {}).get(row['run_id'] + ':' + metric) == 'solid'
                           and config.get('overrideSeriesTitles', {}).get(row['run_id'] + ':' + metric) == row['name']
                           and _color(config.get('overrideColors', {}).get(row['run_id'] + ':' + metric)) == row['color']
                           for row in catalog['curves'])]
        if not matches:
            raise LayoutError('Native workspace lost connected independent curves for ' + metric)


def verify_installed_catalog(catalog, receipt_path, wandb_module=None):
    """Read-only CPU publication gate; no wandb_workspaces dependency is imported.

    The installer performs complete SDK validation. This check verifies its
    exact catalog receipt and checks the live native panels through the Public
    API, supporting both the older client and newer service API transports.
    """
    urls = validate_catalog(catalog)
    if not Path(receipt_path).exists():
        raise LayoutError('Install and verify the native curve layout before publication')
    saved = series._read(receipt_path)
    if (saved.get('api_verified') is not True or saved.get('catalog_sha256') != series._digest(catalog)
            or saved.get('curves') != catalog['curves'] or saved.get('panel_count') != PANEL_COUNT):
        raise LayoutError('Native curve receipt does not match the verified catalog')
    expected_views = {_workspace_id(url, catalog['entity'], catalog['project']) for url in urls}
    views = saved.get('views', [])
    if (len(views) != len(expected_views)
            or {view.get('view_id') for view in views} != expected_views
            or any(view.get('api_verified') is not True or view.get('panel_count') != PANEL_COUNT
                   or _workspace_id(view.get('url', ''), catalog['entity'], catalog['project']) != view['view_id']
                   for view in views)):
        raise LayoutError('Native curve receipt does not cover every requested workspace')
    if wandb_module is None:
        import wandb as wandb_module
    api = wandb_module.Api(timeout=30)
    query = ('query View($entityName:String,$name:String,$viewName:String){'
             'project(name:$name,entityName:$entityName){allViews(viewType:"project-view",'
             'viewName:$viewName){edges{node{id spec}}}}}')
    for view_id in sorted(expected_views):
        variables = {'entityName': catalog['entity'], 'name': catalog['project'],
                     'viewName': 'nw-' + view_id + '-v'}
        if hasattr(api, '_service_api'):
            response = api._service_api.execute_graphql(query, variables=variables)
        else:
            try:
                gql = importlib.import_module('wandb_gql').gql
            except ImportError:
                gql = importlib.import_module('wandb.vendor.gql').gql
            response = api.client.execute(gql(query), variable_values=variables)
        edges = (response.get('project') or {}).get('allViews', {}).get('edges', [])
        if len(edges) != 1:
            raise LayoutError('Verified native curve workspace is missing remotely')
        _verify_view_spec(edges[0]['node']['spec'], catalog)
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    result = install_catalog(series._read(args.catalog), args.receipt)
    print(series._json(result))


if __name__ == '__main__':
    main()
