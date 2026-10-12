"""Read-only, content-pinned references for the selected H2 checkpoint follow-up."""
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


def selected_campaign(campaign):
    return campaign.get('selection_id') == SELECTION


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
    require(selected_campaign(campaign) and len(campaign['candidates']) == 1 and
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
    require(host.get('format_version') == 1 and host.get('kind') == KIND and selected_campaign(campaign)
        and host['campaign_sha256'] == state['campaign_sha256'] and
        state.get('comparison_publication_id') == state['publication_id'], 'Selected reference binding differs.')
    require(set(host['prior_records']) == {str(cp['step']) for cp in campaign['checkpoints']}, 'Selected prior coverage changed.')
    require(all(digest(path) == sha for path, sha in host['reference_files'].items()), 'Pinned selected reference changed.')
    prior = state['runs']['prior']
    require(prior.get('read_only') is True and all(prior[k] == host['prior_run'][k] for k in ('run_id', 'run_dir')),
        'Selected prior must remain read-only.')
    return host


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
