"""Assemble independently seeded H/J workers before the existing publication path."""
from pathlib import Path


def seed_groups(campaign):
    from slurm.ambi_aux_hj_sweep import SEEDS
    groups = campaign['seed_shards']
    assert isinstance(groups,list) and groups and all(isinstance(g,list) and g for g in groups)
    flat = [s for group in groups for s in group]
    assert all(type(s) is int for s in flat) and sorted(flat) == SEEDS
    return groups


def shard_location(campaign, cell, index):
    groups = seed_groups(campaign)
    assert type(index) is int and 0 <= index < len(groups)
    return Path(cell['directory'])/'shards'/str(index), groups[index]


def merge(args):
    from slurm.ambi_aux_hj_sweep import SEEDS, read, write, digest, validate
    from utils.ambi_seed_shards import merge_episode_bundles
    from utils.eval_series import load_run
    from utils.eval_series_data import load_records
    campaign = read(args.root/'campaign.json')
    assert not campaign.get('execution')
    cell = campaign['cells'][args.index]
    assert not cell['reused']
    sources, receipts = [], []
    for index, seeds in enumerate(seed_groups(campaign)):
        directory, _ = shard_location(campaign, cell, index)
        source = directory/'bundle'
        receipt = read(directory/'worker-completion.json')
        assert receipt['status'] == 'complete' and receipt['cell'] == cell['name']
        assert receipt['seeds'] == seeds and receipt['seed_shard_index'] == index
        assert receipt['selector'] == cell['selector']
        assert digest(source/'manifest.json') == receipt['manifest_sha256']
        assert all(digest(source/n) == sha for n,sha in receipt['trace_sha256'].items())
        sources.append(source); receipts.append(receipt)
    bundle = merge_episode_bundles(sources, cell['bundle'], expected_seeds=SEEDS)
    manifest = validate(bundle, cell)
    record, = load_records(bundle, inventory_path=campaign['inventory'])
    assert record['identity'] == load_run(cell['run_dir'])['identity']
    assert record['metrics']['eval/paired_episodes'] == 5
    write(Path(cell['directory'])/'worker-completion.json',dict(
        status='complete',cell=cell['name'],manifest_sha256=digest(bundle/'manifest.json'),
        trace_sha256={n:digest(bundle/n) for n in manifest['runs'][0]['trace_files']},
        selector=cell['selector'],bundle=str(bundle),reused=False,execution=None,
        seeds=SEEDS,gpu='seed-sharded',shard_gpus=[r['gpu'] for r in receipts],
        seed_shards=campaign['seed_shards']))
    print('MERGED '+cell['name'],flush=True)
