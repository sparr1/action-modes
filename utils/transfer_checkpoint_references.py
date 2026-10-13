"""Read-only, content-pinned references for the selected checkpoint follow-ups."""
from copy import deepcopy
import hashlib
import math
from pathlib import Path

from utils.eval_series import load_run, validate_record, _record_fingerprint
from utils.transfer_checkpoint_publication import (
    digest, identity, require, settings, table_rows,
)
from utils.transfer_mppi_comparison import STYLES

SELECTION = 'h2-j8-post500k-bernoulli-v1'
KIND = 'h2-j8-after500k-selected-references'
SETTING = 'h2_j8_bernoulli_a0_c05'
REFERENCE = 'h1_j8_bernoulli_a0_c05'
HORIZON_SELECTION = 'horizon-j8-post500k-followup-v1'
HORIZON_KIND = 'horizon-j8-after500k-selected-references'
HORIZON_TRANSFER = 'h3_j8_bernoulli_a0_c05'
HORIZON_FRESH = 'h2_j8_fresh'
SELECTED_KINDS = (KIND, HORIZON_KIND)


def selected_campaign(campaign):
    return campaign.get('selection_id') in (SELECTION, HORIZON_SELECTION)


def _verify_protocol(campaign, previous):
    require(all(campaign.get(k) == previous.get(k) for k in
        ('protocol', 'source_run', 'seeds', 'controller_seed', 'max_steps', 'scientific_source', 'diagnostics', 'gpu_hardware')),
        'Selected reference science, episode seeds, diagnostics or timing hardware differ.')
    previous_cps = {cp['step']: cp for cp in previous['checkpoints']}
    require(len(previous_cps) == len(previous['checkpoints']), 'Repeated reference checkpoint.')
    for cp in campaign['checkpoints']:
        require(cp['step'] in previous_cps and all(cp.get(k) == previous_cps[cp['step']].get(k)
            for k in ('checkpoint_sha256', 'metadata_sha256', 'prior_manifest_code', 'prior_pin')),
            'Selected reference checkpoint or prior pin differs.')


def selected_reference_receipt(campaign, campaign_path, host_root, reference_root, mppi_data, entity, project):
    from slurm.ambi_transfer_checkpoint_publish import read, fingerprint
    from slurm.publish_ambi_transfer_j10 import _validate_episodes
    require(campaign.get('selection_id') == SELECTION and len(campaign['candidates']) == 1 and
        campaign['candidates'][0]['setting_id'] == SETTING and
        (campaign['candidates'][0]['H'], campaign['candidates'][0]['J'], campaign['candidates'][0]['arm']) == (2, 8, 'bernoulli_a0_c05'),
        'Selected references require the exact H2 J8 Bernoulli critic follow-up.')
    require([cp['step'] for cp in campaign['checkpoints']] == list(range(525000, 2000001, 25000)),
        'Selected H2 references require the 60 checkpoints after 500k.')
    require(reference_root and mppi_data, 'H2 requires the completed H1 J8 publication and pinned MPPI overlay data.')
    files = {}; steps = {cp['step'] for cp in campaign['checkpoints']}
    def pinned(path):
        path = Path(path).resolve(); value = read(path); files[str(path)] = digest(path); return value
    host_root, reference_root = Path(host_root).resolve(), Path(reference_root).resolve()
    host, reference = pinned(host_root/'publication.json'), pinned(reference_root/'publication.json')
    original = pinned(Path(host['campaign_root'])/'campaign.json')
    previous = pinned(Path(reference['campaign_root'])/'campaign.json')
    for root, state, old, expected_count in ((host_root, host, original, 400), (reference_root, reference, previous, 180)):
        require(state.get('prepared') is True and state.get('entity') == entity and state.get('project') == project
            and state['campaign_sha256'] == digest(Path(state['campaign_root'])/'campaign.json'),
            'Selected reference publication identity changed.')
        completion = pinned(root/'publisher-complete.json')
        require(completion.get('status') == 'complete' and completion.get('completed') == completion.get('total') == expected_count,
            'Selected reference publication is incomplete.')
        _verify_protocol(campaign, old)
    ref_setting = [s for s in settings(previous) if s['setting_id'] == REFERENCE]
    require(len(ref_setting) == 1 and ref_setting[0]['arm_definition'] == campaign['candidates'][0]['arm_definition'],
        'H1/H2 Bernoulli definitions differ.')
    reference_curves = {}; prior_pins = {}; prior_run = None
    for state, old, setting in ((host, original, settings(original)[0]), (reference, previous, ref_setting[0])):
        key = setting['setting_id']; entry = state['runs'][key]; directory = Path(entry['run_dir'])
        registry = load_run(directory); pinned(directory/'run.json'); journal = pinned(directory/'publication.json')
        require(registry['run_id'] == entry['run_id'] and registry['identity'] == identity(old, setting)
            and registry['entity'] == entity and registry['project'] == project, 'Selected reference registry differs.')
        records = {}
        for rid, accepted in journal['records'].items():
            step = accepted['checkpoint_step']
            if step not in steps: continue
            require(step not in records and accepted.get('status') == 'published', 'Duplicate or unpublished selected checkpoint.')
            path = directory/'records'/(rid+'.json'); record = validate_record(pinned(path), registry['identity'])
            cp = next(c for c in campaign['checkpoints'] if c['step'] == step)
            require(record['record_id'] == rid and record['checkpoint'] == dict(step=step, sha256=cp['checkpoint_sha256'])
                and _record_fingerprint(record, accepted['artifact_sha256']) == accepted['record_sha256'],
                'Selected reference fingerprint/checkpoint differs.')
            _validate_episodes(record, campaign, prior=key == 'prior')
            records[step] = record
            if key == 'prior':
                alias = 'record-' + hashlib.sha256(rid.encode()).hexdigest()[:20]
                prior_pins[str(step)] = dict(record_id=rid, record_sha256=accepted['record_sha256'],
                    record_file_sha256=digest(path), checkpoint=deepcopy(record['checkpoint']),
                    artifact=f"{entity}/{project}/eval-{registry['run_id']}-{step}:{alias}")
        require(set(records) == steps, 'Selected reference does not cover all requested checkpoints.')
        # The H1 fresh-control comparison is outside this five-curve view.
        point_campaign = dict(old, checkpoints=campaign['checkpoints'], publication_without_fresh_control=True)
        points, progress = table_rows(point_campaign, setting, records, {})
        if key == 'prior':
            # Historical prior timing is outside the compiled L40S comparison.
            for point in points:
                for field in point:
                    if field.startswith(('control_', 'late_')):
                        point[field] = None
        reference_curves[key] = dict(points=points, progress=progress, completed=len(steps), total=len(steps))
        if key == 'prior':
            prior_run = dict(run_dir=str(directory), run_id=registry['run_id'], identity_sha256=registry['identity_sha256'])
    overlay = pinned(host_root/'mppi-overlay.json'); data = pinned(mppi_data)
    require(overlay.get('status') == 'published' and overlay.get('publication_id') == host['publication_id']
        and overlay.get('campaign_sha256') == host['campaign_sha256'] and fingerprint(data) == overlay.get('content_sha256')
        and data.get('curves') == overlay.get('curves') == STYLES, 'MPPI compact data differs from its published receipt.')
    cps = {cp['step']: cp['checkpoint_sha256'] for cp in campaign['checkpoints']}
    for style, source_key in zip(STYLES, ('return_q', 'soft_q')):
        source = data['sources'][source_key]
        require(source['run_id'] == overlay['source_runs'][source_key], 'MPPI source run differs.')
        require(all(source['records'][str(s)]['checkpoint_sha256'] == sha and
            data['prior']['records'][str(s)]['checkpoint_sha256'] == sha for s, sha in cps.items()),
            'MPPI/prior checkpoint hashes differ.')
        points = [p for p in data['points'] if p['setting'] == style['setting_id'] and p['step'] in steps]
        progress = [p for p in data['progress'] if p['setting'] == style['setting_id'] and p['step'] in steps]
        require(len(points) == len(progress) == len(steps) and {p['step'] for p in points} == {p['step'] for p in progress} == steps,
            'MPPI selected curve coverage differs.')
        require(all(p['state'] == 'complete' and p['episodes'] == 5 and p['control_seconds'] is None and p['late_seconds'] is None
            and all(type(p[k]) in (int, float) and math.isfinite(p[k]) for k in
                ('return_mean', 'return_std', 'gain_mean', 'gain_std')) for p in points), 'MPPI statistics or timing omission invalid.')
        reference_curves[style['setting_id']] = dict(points=points, progress=progress, completed=len(steps), total=len(steps))
    return dict(format_version=1, kind=KIND, campaign_sha256=digest(campaign_path),
        host_publication_root=str(host_root), host_publication_id=host['publication_id'], prior_run=prior_run,
        prior_records=prior_pins, reference_files=files, reference_curves=reference_curves,
        selected_reference_styles=[{k:ref_setting[0][k] for k in ('setting_id', 'label', 'color', 'role')}, *deepcopy(STYLES)])


def validate_selected_reference_receipt(host, campaign, state):
    require(host.get('format_version') == 1 and host.get('kind') == (HORIZON_KIND if campaign.get('selection_id') == HORIZON_SELECTION else KIND)
        and selected_campaign(campaign)
        and host['campaign_sha256'] == state['campaign_sha256'] and
        state.get('comparison_publication_id') == state['publication_id'], 'Selected reference binding differs.')
    require(set(host['prior_records']) == {str(cp['step']) for cp in campaign['checkpoints']}, 'Selected prior coverage changed.')
    require(all(digest(path) == sha for path, sha in host['reference_files'].items()), 'Pinned selected reference changed.')
    prior = state['runs']['prior']
    require(prior.get('read_only') is True and all(prior[k] == host['prior_run'][k] for k in ('run_id', 'run_dir')),
        'Selected prior must remain read-only.')
    return host


def horizon_reference_receipt(campaign, campaign_path, host_root, entity, project):
    """Reuse the final H2 publication and its pinned references without new artifacts."""
    from slurm.ambi_transfer_checkpoint_publish import read, fingerprint, validate_comparison_host
    from slurm.publish_ambi_transfer_j10 import _validate_episodes
    require(campaign.get('selection_id') == HORIZON_SELECTION and
        {(s['setting_id'], s['H'], s['J'], s['arm'], s['role']) for s in campaign['candidates']} ==
        {(HORIZON_TRANSFER, 3, 8, 'bernoulli_a0_c05', 'transfer'), (HORIZON_FRESH, 2, 8, 'fresh', 'fresh')}
        and len(campaign['candidates']) == 2, 'Horizon references require exactly H3 transfer and H2 fresh at J8.')
    require([cp['step'] for cp in campaign['checkpoints']] == list(range(525000, 2000001, 25000)),
        'Horizon references require the 60 checkpoints after 500k.')
    require(len(campaign['cells']) == 120 and
        {(c['setting_id'], c['step']) for c in campaign['cells']} ==
        {(s['setting_id'], cp['step']) for s in campaign['candidates'] for cp in campaign['checkpoints']},
        'Horizon references require exactly the 120 distinct setting/checkpoint cells.')
    root = Path(host_root).resolve(); files = {}
    def pinned(path):
        path = Path(path).resolve(); value = read(path); files[str(path)] = digest(path); return value
    state = pinned(root/'publication.json')
    oldpath = Path(state['campaign_root'])/'campaign.json'; previous = pinned(oldpath)
    require(previous.get('selection_id') == SELECTION and len(previous['candidates']) == 1
        and tuple(previous['candidates'][0].get(k) for k in ('setting_id', 'H', 'J', 'arm', 'role')) ==
        (SETTING, 2, 8, 'bernoulli_a0_c05', 'transfer'),
        'Horizon comparison host must be the completed selected H2 J8 campaign.')
    require(state.get('prepared') is True and state.get('entity') == entity and state.get('project') == project
        and state['campaign_sha256'] == digest(oldpath), 'H2 reference publication identity changed.')
    _verify_protocol(campaign, previous)
    transfer = next(s for s in campaign['candidates'] if s['setting_id'] == HORIZON_TRANSFER)
    require(transfer['arm_definition'] == previous['candidates'][0]['arm_definition'],
        'H2/H3 Bernoulli definitions differ.')
    inherited = validate_comparison_host(previous, state, root); pinned(root/'comparison-host.json')
    require(inherited and inherited['kind'] == KIND, 'H2 inherited references are missing.')
    files.update(inherited['reference_files'])
    completion = pinned(root/'publisher-complete.json'); snapshot = pinned(root/'snapshot.json')
    require(completion.get('status') == 'complete' and completion.get('completed') == completion.get('total') == 60
        and snapshot.get('completed') == snapshot.get('total') == 60 and not snapshot.get('failures'),
        'H2 reference publication must be complete without failures.')
    require(state.get('layout', {}).get('status') == 'verified' and
        state['layout'].get('layout_version') == 'transfer-checkpoint-selected-references-v1'
        and state.get('overview', {}).get('snapshot_sha256') == fingerprint(dict(snapshot=snapshot, layout=state['layout'])),
        'H2 overview must contain its verified final snapshot.')
    setting = next(s for s in settings(previous) if s['setting_id'] == SETTING)
    entry = state['runs'][SETTING]; directory = Path(entry['run_dir'])
    registry = load_run(directory); pinned(directory/'run.json'); journal = pinned(directory/'publication.json')
    require(registry['run_id'] == entry['run_id'] and registry['identity'] == identity(previous, setting)
        and registry['entity'] == entity and registry['project'] == project, 'H2 reference registry differs.')
    steps = {cp['step']: cp['checkpoint_sha256'] for cp in campaign['checkpoints']}
    records = {}; record_paths = {}
    for rid, accepted in journal['records'].items():
        step = accepted['checkpoint_step']
        require(step in steps and step not in records and accepted.get('status') == 'published',
            'Unexpected, duplicate or unpublished H2 reference checkpoint.')
        path = directory/'records'/(rid+'.json'); record = validate_record(pinned(path), registry['identity'])
        require(record['record_id'] == rid and record['checkpoint'] == dict(step=step, sha256=steps[step])
            and _record_fingerprint(record, accepted['artifact_sha256']) == accepted['record_sha256'],
            'H2 reference fingerprint/checkpoint differs.')
        _validate_episodes(record, campaign, horizon=2)
        records[step] = record; record_paths[str(step)] = str(path)
    require(set(records) == set(steps), 'H2 reference does not cover all requested checkpoints.')
    points, progress = table_rows(previous, setting, records, {})
    curve = dict(points=points, progress=progress, completed=60, total=60)
    expected = dict(inherited['reference_curves'], **{SETTING: curve})
    require(snapshot['curves'] == expected, 'H2 final snapshot differs from its accepted references/results.')
    return dict(format_version=1, kind=HORIZON_KIND, campaign_sha256=digest(campaign_path),
        host_publication_root=str(root), host_publication_id=state['publication_id'],
        prior_run=deepcopy(inherited['prior_run']), prior_records=deepcopy(inherited['prior_records']),
        reference_files=files, reference_curves=expected, paired_reference_records=record_paths,
        paired_reference_setting=deepcopy(setting),
        selected_reference_styles=[{k:setting[k] for k in ('setting_id', 'label', 'color', 'role')},
                                   *deepcopy(inherited['selected_reference_styles'])])


def reference_curves_with_fresh(campaign, host, fresh_records):
    """Derived paired display only; reused scientific histories remain immutable."""
    from slurm.ambi_transfer_checkpoint_publish import read
    curves = deepcopy(host['reference_curves'])
    if host['kind'] == HORIZON_KIND:
        setting = host['paired_reference_setting']
        records = {int(step): read(path) for step, path in host['paired_reference_records'].items()}
        matched = dict(campaign, candidates=[*campaign['candidates'], setting])
        points, progress = table_rows(matched, setting, records, {}, fresh_records=fresh_records)
        curves[SETTING] = dict(points=points, progress=progress, completed=len(records), total=len(campaign['checkpoints']))
    return curves


def compact_scientific_record(record):
    """Keep complete raw traces on Oscar, retaining their verified receipt hashes."""
    value = deepcopy(record)
    traces = {name: path for name, path in value['artifact_files'].items() if name.startswith('decisions-seed-')}
    require(len(traces) == 5, 'Compact publication requires all five validated traces.')
    receipt = value['provenance']['worker_receipt']; hashes = receipt['file_sha256']
    require(all(name in hashes and digest(path) == hashes[name] for name, path in traces.items()), 'Compact trace hash mismatch.')
    value['provenance']['raw_trace_references'] = {name: dict(path=path, sha256=hashes[name]) for name, path in traces.items()}
    value['provenance']['artifact_policy'] = 'compact results, manifest and receipt; full decision traces retained on Oscar'
    value['artifact_files'] = {name: path for name, path in value['artifact_files'].items()
        if name in ('manifest.json', 'results.json', 'worker-receipt.json')}
    require(len(value['artifact_files']) == 3, 'Compact scientific metadata files missing.')
    return validate_record(value)
