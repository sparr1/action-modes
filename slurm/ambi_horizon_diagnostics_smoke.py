"""Three-decision actual-checkpoint smoke; never launches a full evaluation."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MATRIX = ROOT / 'configs/research/ambi_aux_horizon_conditioning_625k.json'


def tensor_digest(value):
    import torch
    digest = hashlib.sha256()
    def visit(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((tensor.dtype, tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item):
                digest.update(str(key).encode()); visit(item[key])
        elif isinstance(item, (tuple, list)):
            for child in item: visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def smoke_matrix(horizon, rounds):
    matrix = json.loads(MATRIX.read_text())
    variants = matrix['comparisons']['sweep']['variants']
    selected = {}
    for mode in ('none', 'one_hot'):
        original = f'soft_soft_h{horizon}_j{rounds}_c16_{mode}'
        for enabled in (False, True):
            name = original + ('_diag_on' if enabled else '_diag_off')
            selected[name] = deepcopy(variants[original])
            selected[name]['alg_params']['inner_horizon_diagnostics'] = enabled
    matrix['comparisons']['sweep']['variants'] = selected
    matrix['comparisons']['sweep']['reference'] = next(iter(selected))
    matrix['evaluation'].update(default_presets=['sweep/'+name for name in selected],
                                seeds=[101], max_steps=3)
    return matrix


def run(args):
    import numpy as np
    import torch
    import evaluate_ambi_checkpoint as evaluator
    from slurm.ambi_aux_hj_sweep import cells, training_summary, validate, CHECKPOINT_SHA, digest
    from utils.ambi_benchmark import atomic_json

    assert digest(args.checkpoint) == CHECKPOINT_SHA, 'Wrong 625k checkpoint'
    args.output.mkdir(parents=True, exist_ok=False)
    horizon, rounds = ((1, 4), (3, 8))[args.index]
    matrix = smoke_matrix(horizon, rounds)
    matrix_path = args.output/'matrix.json'
    atomic_json(matrix_path, matrix)
    original_initialize = evaluator._initialize_frozen_model
    cases = []
    for cell in cells(matrix_path):
        torch._dynamo.reset()
        decisions = []
        def initialize(*positional, **keywords):
            model, config = original_initialize(*positional, **keywords)
            original_predict = model.predict
            def predict(*a, **kw):
                result = original_predict(*a, **kw)
                if kw.get('trace') is None:  # Existing unscored compiler warmup.
                    return result
                pool = model.agent.inner_engine._action_pool
                replay = pool.replay
                labelled = (cell['params']['inner_horizon_diagnostics']
                            or cell['params']['inner_horizon_conditioning'] != 'none')
                assert (replay.remaining_horizon is not None) == labelled
                labels = None
                if labelled:
                    labels = [int((replay.remaining_horizon[:replay.size] == h).sum())
                              for h in range(1, horizon+1)]
                    assert labels == [128*rounds]*horizon
                    assert torch.equal(replay.horizon_end[:replay.size].bool(),
                                       replay.remaining_horizon[:replay.size] == 1)
                decisions.append(dict(
                    action=np.asarray(result[0]).tolist(), replay_size=replay.size,
                    horizon_counts=labels,
                    learner_rng=tensor_digest(model.agent.inner_engine.rng.training_state_dict()),
                    graphs=int(torch._dynamo.utils.counters['stats']['unique_graphs']),
                ))
                return result
            model.predict = predict
            return model, config
        evaluator._initialize_frozen_model = initialize
        try:
            directory = args.output/cell['name']
            evaluator.evaluate_matrix(matrix_path, args.checkpoint, selectors=[cell['selector']],
                seeds=[101], controller_seed=55, max_steps=3, device=args.device,
                bundle_dir=directory/'bundle', checkpoint_inventory=args.inventory,
                stage_results=False)
        finally:
            evaluator._initialize_frozen_model = original_initialize
        manifest = validate(directory/'bundle', cell, seeds=[101], steps=3, paired=False,
                            checkpoint_sha=CHECKPOINT_SHA)
        summary = training_summary(directory/'bundle', cell, expected_steps=3)
        atomic_json(directory/'training-summary.json', summary)
        assert len(decisions) == 3
        assert decisions[0]['graphs'] == decisions[1]['graphs'] == decisions[2]['graphs']
        expected_diagnostics = cell['params']['inner_horizon_diagnostics']
        for row in summary['update_curves']:
            component = row['axis'].removesuffix('_update')
            counts = [row['metrics'].get(f'{component}_horizon_{h}_sample_count') for h in range(1,horizon+1)]
            assert all(count is not None for count in counts) == expected_diagnostics
            if expected_diagnostics:
                assert sum(count['sample_count'] for count in counts) == 3*256
        case = dict(name=cell['name'], conditioning=cell['params']['inner_horizon_conditioning'],
                    diagnostics=expected_diagnostics, decisions=decisions,
                    result=manifest['runs'][0]['result'],
                    manifest_sha256=digest(directory/'bundle/manifest.json'))
        cases.append(case)
        atomic_json(directory/'receipt.json', case)
        print('COMPLETE '+cell['name'], flush=True)
    pairs=[]
    for mode in ('none','one_hot'):
        off,on=[case for case in cases if case['conditioning']==mode]
        for a,b in zip(off['decisions'],on['decisions']):
            np.testing.assert_allclose(a['action'],b['action'],rtol=2e-5,atol=2e-6)
            assert a['learner_rng']==b['learner_rng']
        pairs.append(dict(conditioning=mode, actions_match=True, rng_match=True))
    atomic_json(args.output/'validation.json',dict(status='complete', H=horizon,J=rounds,
        checkpoint_sha256=CHECKPOINT_SHA,cases=cases,paired_diagnostic_checks=pairs,
        gpu=torch.cuda.get_device_name(0) if args.device=='cuda' else None,
        scope='Three decisions per case; not a performance evaluation.'))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--inventory',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--index',type=int,choices=(0,1),required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    run(parser.parse_args())
