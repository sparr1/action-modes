"""Five independent episode workers followed by one complete-panel publisher.

Workers never publish. All seeds, paired prior gains and both probe rounds must
be present before the merger can stage a result or publish diagnostics.
"""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MATRIX = ROOT / 'configs/research/ambi_aux_closed_loop_625k.json'
SELECTOR = 'critic/soft_q'
CRITIC_SOURCES = {'soft_q': 'sac', 'return_q': 'aux_return'}
SEEDS = [101, 102, 103, 104, 105]
ATTEMPT = 'aux625k-soft-init-soft-tail-j1-alpha0-20260916'
MATRICES = {'zero': MATRIX, 'inherit_outer': ROOT / 'configs/research/ambi_aux_closed_loop_625k_inherited_alpha.json'}
ATTEMPTS = {'zero': ATTEMPT, 'inherit_outer': 'aux625k-soft-init-soft-tail-j1-inherited-alpha-20260916'}
MATRICES['auto'] = ROOT / 'configs/research/ambi_aux_closed_loop_625k_auto_alpha.json'
ATTEMPTS['auto'] = 'aux625k-soft-init-soft-tail-j1-auto-alpha-20260916'


def checkpoint_alpha(path):
    """Read the saved outer temperature without constructing a learner."""
    import torch
    state = torch.load(path, map_location='cpu', weights_only=True)
    if 'log_ent_coef' in state:
        assert 'ent_coef_optim' in state, 'Incomplete saved automatic temperature'
        alpha = state['log_ent_coef'].exp().clamp_min(1e-8).item()
    else:
        alpha = state['fixed_ent_coef'].item()
    assert math.isfinite(alpha) and alpha >= 0
    return alpha


def validate_bundle(path, seeds, max_steps, *, paired=True, expected_alpha=0., alpha_mode=None,
                    critic_mode='soft_q'):
    from utils.ambi_diagnostic_series import record_from_model_bundle
    if alpha_mode is None:
        alpha_mode = 'inherit_outer' if expected_alpha > 0 else 'zero'
    assert alpha_mode in MATRICES
    selector = f'critic/{critic_mode}'
    critic_source = CRITIC_SOURCES[critic_mode]
    manifest = json.loads((Path(path) / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    assert len(manifest['runs']) == 1
    run, = manifest['runs']
    result, cfg = run['result'], run['resolved_config']
    assert run['selector'] == selector and run['status'] == 'complete'
    assert result['outer_state_unchanged']
    assert result['outer_updates_before'] == result['outer_updates_after']
    assert not result['nonfinite_model_metrics'] and not result['nonfinite_trace_metrics']
    assert result['environment_seeds'] == seeds and result['controller_seed'] == 55
    assert cfg['inner_critic_source'] == cfg['inner_horizon_critic_source'] == critic_source
    assert cfg['inner_actor_source'] == cfg['inner_horizon_actor_source'] == 'sac'
    routing = result['value_routing']
    assert routing['inner_critic_source'] == routing['inner_horizon_critic_source'] == critic_source
    assert routing['inner_actor_source'] == routing['inner_horizon_actor_source'] == 'sac'
    assert cfg['inner_actor_initialization'] == cfg['inner_critic_initialization'] == 'prior'
    assert cfg['inner_critic_target_initialization'] == 'online'
    assert cfg['inner_finite_horizon'] and cfg['inner_sac_critic_target'] == 'reward_only'
    assert cfg['inner_entropy_enabled'] == (expected_alpha > 0)
    assert cfg['inner_temperature_mode'] == ('auto' if alpha_mode == 'auto' else 'inherit_outer')
    assert cfg['inner_temperature_initialization'] == cfg['inner_target_entropy'] == 'inherit_outer'
    assert cfg['inner_temperature_scope'] == cfg['inner_temperature_optimizer_scope'] == 'action'
    if alpha_mode == 'auto':
        assert expected_alpha > 0 and cfg['inner_temperature_lr'] == 3e-4
    assert cfg['inner_execution_action'] == 'mean'
    assert cfg['inner_log_std_mapping'] == 'direct_clamp'
    assert cfg['inner_batch_size'] == 256 and cfg['inner_rollout_horizon'] == 1
    assert cfg['inner_rounds'] == 1 and cfg['inner_rollouts_per_round'] == 128
    for key, value in [('inner_model_steps', 128), ('inner_actor_optimizer_steps', 4),
                       ('inner_critic_optimizer_steps', 32),
                       ('inner_temperature_optimizer_steps', 4 if alpha_mode == 'auto' else 0),
                       ('inner_compile_fallback', 0)]:
        for statistic in ('mean', 'min', 'max'):
            assert result['model_metrics'][key][statistic] == value, (key, statistic)
    fixed_keys = ('inner_alpha_initial',) if alpha_mode == 'auto' else (
        'inner_alpha', 'inner_alpha_initial', 'inner_alpha_final')
    for key in fixed_keys:
        for statistic in ('mean', 'min', 'max'):
            assert math.isclose(result['model_metrics'][key][statistic], expected_alpha,
                                rel_tol=1e-6, abs_tol=1e-10), (key, statistic)
    if alpha_mode == 'auto':
        final = result['model_metrics']['inner_alpha_final']
        delta = result['model_metrics']['inner_alpha_delta']
        assert final['min'] > 0
        assert max(abs(delta['min']), abs(delta['max'])) > 1e-10, 'Alpha never changed'
        for statistic in ('mean', 'min', 'max'):
            assert result['model_metrics']['inner_alpha'][statistic] == final[statistic]
    probe = run['togo_return_probe']
    assert probe['rollouts'] == 32 and probe['horizon'] == 1
    assert probe['cadence'] == 'initial_and_after_each_round' and not probe['entropy_bonus']
    assert probe['tail_q_reduction'] == 'mean_pair'
    assert [e['seed'] for e in run['episodes']] == seeds
    for episode in run['episodes']:
        assert episode['length'] == max_steps
        if paired:
            assert not episode['truncated_by_evaluator']
            assert 'paired_return_delta' in episode
        points = episode['togo_round_summaries']
        assert [(r['round_index'], r['actor_updates'], r['critic_updates']) for r in points] == [(0, 0, 0), (1, 4, 32)]
        assert all(s['count'] == max_steps for r in points for s in r['metrics'].values())
    attempt = ATTEMPTS[alpha_mode]
    if critic_mode == 'return_q':
        attempt = attempt.replace('soft-init-soft-tail', 'return-init-return-tail')
    diagnostics = record_from_model_bundle(path, selector, attempt, bootstrap_resamples=2000,
                                          bootstrap_seed=20260912)
    assert diagnostics['status'] == 'complete'
    assert len(diagnostics['rows']) == len(seeds) * max_steps * 2
    return manifest, diagnostics


def worker(args):
    import torch
    from evaluate_ambi_checkpoint import evaluate_matrix
    from utils.ambi_seed_shards import seal_episode_bundle
    assert torch.cuda.is_available()
    assert args.seed in SEEDS
    selector = f'critic/{args.critic_mode}'
    saved_alpha = checkpoint_alpha(args.checkpoint)
    expected_alpha = saved_alpha if args.alpha_mode != 'zero' else 0.
    if args.alpha_mode != 'zero':
        assert expected_alpha > 0, 'Inherited-alpha condition requires a saved positive coefficient'
    directory = args.root / ('smoke' if args.smoke else f'shards/seed-{args.seed}')
    directory.mkdir(parents=True, exist_ok=False)
    result = evaluate_matrix(MATRICES[args.alpha_mode], args.checkpoint, selectors=[selector], seeds=[args.seed],
                             controller_seed=55, max_steps=20 if args.smoke else 500, device='cuda',
                             bundle_dir=directory / 'bundle', checkpoint_inventory=args.inventory,
                             reference_bundle=None if args.smoke else args.reference)
    (directory / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    manifest, _ = validate_bundle(directory / 'bundle', [args.seed], 20 if args.smoke else 500,
                                  paired=not args.smoke, expected_alpha=expected_alpha,
                                  alpha_mode=args.alpha_mode, critic_mode=args.critic_mode)
    cfg = manifest['runs'][0]['resolved_config']
    assert cfg['compile'] and cfg['compile_strict']
    assert result['results'][0]['resolved_device'].startswith('cuda')
    assert manifest['checkpoint']['metadata']['checkpoint']['step'] == 625000
    seal_episode_bundle(directory / 'bundle')
    receipt = dict(status='complete', seed=args.seed, smoke=args.smoke,
                   alpha_mode=args.alpha_mode, critic_mode=args.critic_mode, selector=selector,
                   saved_outer_alpha=saved_alpha,
                   inner_alpha_initial=expected_alpha,
                   inner_alpha_final_mean=manifest['runs'][0]['result']['model_metrics']['inner_alpha_final']['mean'],
                   checkpoint_sha256=manifest['checkpoint']['sha256'], gpu=torch.cuda.get_device_name(0),
                   episodes=manifest['runs'][0]['episodes'])
    (directory / 'worker-completion.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'episodes'}), flush=True)


def merge(args):
    from utils.ambi_seed_shards import merge_episode_bundles
    from utils.ambi_diagnostic_series import write_diagnostic_bundle
    from utils.eval_series import load_run
    from utils.eval_series_data import load_records
    from report_ambi_benchmark import load_bundles, render_html
    sources = [args.root / f'shards/seed-{seed}/bundle' for seed in SEEDS]
    alphas = []
    for seed, source in zip(SEEDS, sources):
        receipt = json.loads((source.parent / 'worker-completion.json').read_text())
        assert receipt['status'] == 'complete' and receipt['seed'] == seed and not receipt['smoke']
        assert receipt.get('alpha_mode', 'zero') == args.alpha_mode
        assert receipt.get('critic_mode', 'soft_q') == args.critic_mode
        alphas.append(receipt.get('inner_alpha_initial', receipt.get('inner_alpha', 0.)))
    assert len(set(alphas)) == 1
    bundle = merge_episode_bundles(sources, args.root / 'production/bundle', expected_seeds=SEEDS)
    manifest, diagnostic = validate_bundle(bundle, SEEDS, 500, expected_alpha=alphas[0],
                                           alpha_mode=args.alpha_mode, critic_mode=args.critic_mode)
    record, = load_records(bundle, inventory_path=args.inventory)
    assert record['checkpoint']['step'] == 625000
    assert record['identity'] == load_run(args.run_dir)['identity']
    assert record['metrics']['eval/paired_episodes'] == 5
    output = bundle.parent
    (output / 'comparison.html').write_text(render_html(load_bundles([bundle])))
    (output / 'results.json').write_text(json.dumps({'results': [r['result'] for r in manifest['runs']]}, indent=2) + '\n')
    write_diagnostic_bundle(output / 'model-series', diagnostic)
    receipt = dict(status='complete', checkpoint_step=625000, seeds=SEEDS,
                   alpha_mode=args.alpha_mode, critic_mode=args.critic_mode,
                   selector=f'critic/{args.critic_mode}', inner_alpha_initial=alphas[0],
                   inner_alpha_final_mean=manifest['runs'][0]['result']['model_metrics']['inner_alpha_final']['mean'],
                   metrics=record['metrics'], diagnostic_series_id=diagnostic['series_id'])
    (output / 'merge-completion.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if args.publish:
        publish(args)
    else:
        print(json.dumps(receipt, indent=2), flush=True)


def publish(args):
    """Publish an already complete panel; safe after a pre-publication failure."""
    from utils.ambi_benchmark import stage_completed_bundle
    from utils.ambi_diagnostic_series import read_diagnostic_bundle, publish_diagnostic_bundle
    from utils.eval_series import publish_run
    output = args.root / 'production'
    receipt = json.loads((output / 'merge-completion.json').read_text())
    assert receipt['status'] == 'complete' and receipt['checkpoint_step'] == 625000
    assert receipt['seeds'] == SEEDS
    critic_mode = getattr(args, 'critic_mode', 'soft_q')
    selector = f'critic/{critic_mode}'
    assert receipt.get('critic_mode', 'soft_q') == critic_mode
    assert receipt.get('selector', SELECTOR) == selector
    diagnostic_path = output / 'model-series'
    diagnostic = read_diagnostic_bundle(diagnostic_path)
    assert diagnostic['series_id'] == receipt['diagnostic_series_id']
    assert diagnostic['status'] == 'complete' and len(diagnostic['rows']) == 5000
    # The staging receipt is JSON; the existing helper expects string paths.
    staged = stage_completed_bundle(output / 'bundle', {selector: str(args.run_dir)},
                                    inventory_path=args.inventory)
    assert staged[selector]['status'] == 'queued', staged
    receipt['episode_publication'] = publish_run(args.run_dir, owner='oscar-rgao48')
    receipt['diagnostic_publication'] = publish_diagnostic_bundle(
        diagnostic_path, entity='rwgao_b-brown-university', mode='online')
    (output / 'publication-completion.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['worker', 'merge', 'publish'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--alpha-mode', choices=list(MATRICES), default='zero')
    parser.add_argument('--critic-mode', choices=list(CRITIC_SOURCES), default='soft_q')
    args = parser.parse_args()
    {'worker': worker, 'merge': merge, 'publish': publish}[args.mode](args)


if __name__ == '__main__':
    main()
