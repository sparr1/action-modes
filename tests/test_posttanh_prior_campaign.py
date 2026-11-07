"""Preserve the approved backbones and require real GPU validation receipts."""
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest
import torch
from tests.test_ambi_root_local_sac import _build_cfg
from RL.tdmpc2_core.whole_inner_graph import validate_training_graph
from slurm.posttanh_prior_fixes_oscar import MANIFEST, GPU_PARITY_TESTS, parity_cases

ROOT = Path(__file__).resolve().parents[1]
METHOD_KEYS = {'outer_actor_objective', 'outer_behavior_policy_objective',
               'outer_behavior_policy_weighting', 'outer_behavior_policy_kl_schedule',
               'outer_behavior_policy_kl_coef', 'outer_behavior_policy_compile_mode',
               'outer_boom_coef', 'outer_boom_min_std', 'outer_boom_compile_mode'}

def read(name):
    return json.loads((ROOT / 'configs/dmcontrol/algs' / (name+'.json')).read_text())

@pytest.mark.parametrize('backbone', ['outer_adapt', 'outer_frozen'])
@pytest.mark.parametrize('method', ['baseline', 'tdmpc_square', 'boom'])
def test_only_approved_prior_objective_changes(backbone, method):
    source=read('ambi_inner_entropy_posttanh_'+backbone)
    run=read(f'ambi_posttanh_prior_{backbone}_{method}')
    assert {k:v for k,v in run.items() if k!='alg_params'} == {k:v for k,v in source.items() if k!='alg_params'}
    def stable(p):
        return {k:v for k,v in p.items() if not k.startswith('wandb') and k not in METHOD_KEYS}
    p=run['alg_params']
    assert stable(p)==stable(source['alg_params'])
    cfg=_build_cfg(**p)
    assert cfg.inner_cuda_graph and cfg.compile and cfg.compile_strict
    assert cfg.critic_ensemble_backend=='vmap' and cfg.critic_geometry_diagnostics
    assert cfg.ent_coef==.0021 and cfg.outer_actor_entropy_mode==cfg.inner_actor_entropy_mode=='squashed'
    assert cfg.inner_actor_loss_scale_scope=='action'
    assert cfg.inner_actor_loss_scale_update==('per_update' if backbone=='outer_adapt' else 'per_action')
    assert (cfg.inner_critic_updates_per_action,cfg.inner_actor_updates_per_action)==(128,32)
    validate_training_graph(SimpleNamespace(cfg=cfg,device=torch.device('cuda'),_active_trace=None,_retrace_enabled=False))
    assert cfg.store_behavior_policy == (method!='baseline')
    if method=='boom':
        assert cfg.outer_actor_objective=='boom' and cfg.outer_boom_coef==.42
        assert cfg.outer_boom_compile_mode=='reduce-overhead'
    elif method=='tdmpc_square':
        assert cfg.outer_actor_objective=='q_maximization'
        assert cfg.outer_behavior_policy_weighting=='tdmpc_square'
        assert cfg.outer_behavior_policy_kl_schedule=='quantile_gate'
        assert cfg.outer_behavior_policy_kl_coef==1
        assert cfg.outer_behavior_policy_compile_mode=='reduce-overhead'
    else:
        assert cfg.outer_actor_objective=='q_maximization' and cfg.outer_behavior_policy_kl_schedule=='none'

def test_exact_six_cells():
    m=json.loads(MANIFEST.read_text())
    assert m['configs']==[f'ambi_posttanh_prior_{b}_{m}' for b in ('outer_adapt','outer_frozen') for m in ('baseline','tdmpc_square','boom')]
    assert m['trials']==1 and m['checkpoint_every']==25000 and m['overrides_alg']['total_steps']==1000000

@pytest.mark.parametrize('bad', [None, 'skipped', 'failure', 'error', 'missing'])
def test_gpu_receipt_rejects_missing_or_skipped_coverage(tmp_path,bad):
    root=ET.Element('testsuite')
    for node,count in zip(GPU_PARITY_TESTS,[4,6,4,1]):
        for i in range(count):
            c=ET.SubElement(root,'testcase',name=node.split('::')[-1]+(f'[{i}]' if count>1 else ''))
    if bad=='missing': root.remove(list(root)[0])
    elif bad: ET.SubElement(list(root)[-1],bad)
    path=tmp_path/'parity.xml';ET.ElementTree(root).write(path)
    if bad:
        with pytest.raises(AssertionError): parity_cases(path)
    else: assert len(parity_cases(path))==15

@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires scheduled CUDA GPU')
@pytest.mark.parametrize('update', ['per_action', 'per_update'])
@pytest.mark.parametrize('method', ['tdmpc_square', 'boom'])
def test_cuda_graph_captures_correct_behavior_labels_without_changing_solve(update,method):
    from copy import deepcopy
    from tests.test_inner_entropy_graph import make_agent, assert_tree_close, inner_snapshot
    from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
    from RL.tdmpc2_core.whole_inner_graph import WholeInnerGraph
    cfg=deepcopy(make_agent(update=update).cfg)
    cfg.device='cuda';cfg.compile=cfg.compile_strict=True;cfg.store_behavior_policy=True
    cfg.outer_actor_objective='boom' if method=='boom' else 'q_maximization'
    cfg.outer_boom_coef=.42
    if method=='tdmpc_square':
        cfg.outer_behavior_policy_objective='action_space_cross_entropy'
        cfg.outer_behavior_policy_weighting='tdmpc_square'
        cfg.outer_behavior_policy_kl_schedule='quantile_gate'
    native,captured=(AMBITDMPC2Agent(deepcopy(cfg)) for _ in range(2))
    captured.load(deepcopy(native.checkpoint_state()))
    native.model.eval();captured.model.eval()
    root=torch.zeros(1,cfg.latent_dim,device='cuda')
    control=WholeInnerGraph(native.inner_engine,root,capture=False,return_behavior_policy=True)
    graph=WholeInnerGraph(captured.inner_engine,root,return_behavior_policy=True)
    for index in range(3):
        initial=torch.full_like(native.actor_loss_scale,4.+index)
        expected=control(root+.1*index,initial_actor_loss_scale=initial)
        actual=graph(root+.1*index,initial_actor_loss_scale=initial)
        assert_tree_close(actual,expected,rtol=5e-5,atol=5e-7)
        assert_tree_close(inner_snapshot(captured.inner_engine),inner_snapshot(native.inner_engine),rtol=5e-5,atol=5e-7)
        with torch.no_grad():
            _,info=captured.model.pi(root+.1*index,policy=captured.inner_engine._action_pool.actor,
                deterministic=True,**captured.inner_engine._inner_policy_kwargs())
        torch.testing.assert_close(actual[3]['pre_tanh_mean'],info['pre_tanh_mean'][0],rtol=5e-5,atol=5e-7)
        torch.testing.assert_close(actual[3]['log_std'],info['log_std'][0],rtol=5e-5,atol=5e-7)
        torch.testing.assert_close(graph.rng_state,control.rng_state,rtol=0,atol=0)
