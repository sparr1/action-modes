"""Scientific contracts for the six fixed-entropy normalization runs."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from tests.test_ambi_inner_decoupling import _fixed_reward_model
from tests.test_ambi_root_local_sac import _tiny_model, _build_cfg

ROOT = Path(__file__).resolve().parents[1]

def model(**kwargs):
    return _fixed_reward_model(
        sac_actor_loss_scale_mode="tdmpc2_percentile_range",
        inner_actor_loss_scale_update="per_update",
        inner_actor_loss_scale_scope="run", sac_actor_loss_scale_tau=0.5,
        **kwargs,
    )

def test_carry_is_separate_from_outer_and_survives_episodes_and_resume():
    first, restored = model(), model()
    try:
        engine = first.agent.inner_engine
        first.agent.actor_loss_scale.fill_(3.75)
        first.agent.act(torch.zeros(3), t0=True, collect_diagnostics=False)
        metrics = first.agent.last_inner_metrics
        assert metrics['inner_actor_loss_scale_initial'] == pytest.approx(3.75)
        previous = metrics['inner_actor_loss_scale_final']
        assert previous != pytest.approx(3.75)
        first.agent.actor_loss_scale.fill_(100)
        first.agent.act(torch.zeros(3), t0=True, collect_diagnostics=False)
        metrics = first.agent.last_inner_metrics
        assert metrics['inner_actor_loss_scale_initial'] == pytest.approx(previous)
        assert metrics['inner_actor_loss_scale_outer'] == pytest.approx(100)
        torch.testing.assert_close(first.agent.actor_loss_scale, torch.tensor([100.]))
        engine.prepare_training_resume_boundary()
        state = deepcopy(engine.training_state_dict())
        restored.agent.inner_engine.load_training_state_dict(state)
        torch.testing.assert_close(restored.agent.inner_engine._carried_actor_loss_scale,
                                   engine._carried_actor_loss_scale, rtol=0, atol=0)
        expected = state['actor_loss_scale_carry'].item()
        restored.agent.act(torch.zeros(3), t0=True, collect_diagnostics=False)
        assert restored.agent.last_inner_metrics['inner_actor_loss_scale_initial'] == pytest.approx(expected)
        restored.agent.inner_engine.reset_for_evaluation(123)
        assert restored.agent.inner_engine._carried_actor_loss_scale is None
    finally:
        for m in (first, restored):
            m.close(); m.env.close()

def test_held_actions_do_not_adapt_carried_scale():
    m = _tiny_model(inner_solve_interval=3, inner_rounds=1,
                   inner_updates_per_round=2, ent_coef=0.5,
                   outer_critic_target='reward_only', inner_sac_critic_target='reward_only',
                   inner_temperature_mode='inherit_outer',
                   sac_actor_loss_scale_mode='tdmpc2_percentile_range',
                   inner_actor_loss_scale_update='per_update',
                   inner_actor_loss_scale_scope='run')
    try:
        m.agent.actor_loss_scale.fill_(3.75)
        m.agent.act(torch.zeros(3), t0=True, collect_diagnostics=False)
        engine = m.agent.inner_engine
        value = engine._carried_actor_loss_scale.clone()
        for _ in range(2):
            m.agent.act(torch.zeros(3), collect_diagnostics=False)
            torch.testing.assert_close(engine._carried_actor_loss_scale, value, rtol=0, atol=0)
        m.agent.act(torch.zeros(3), collect_diagnostics=False)
        assert m.agent.last_inner_metrics['inner_actor_loss_scale_initial'] == pytest.approx(value.item())
    finally:
        m.close(); m.env.close()

def test_corrupt_carry_resume_rejected_without_mutation():
    m = model()
    try:
        engine = m.agent.inner_engine
        m.agent.act(torch.zeros(3), t0=True, collect_diagnostics=False)
        engine.prepare_training_resume_boundary()
        state = deepcopy(engine.training_state_dict())
        before = engine._carried_actor_loss_scale.clone()
        for bad in (float('nan'), 0.0):
            state['actor_loss_scale_carry'].fill_(bad)
            with pytest.raises(ValueError, match='scale'):
                engine.load_training_state_dict(state)
            torch.testing.assert_close(engine._carried_actor_loss_scale, before, rtol=0, atol=0)
    finally:
        m.close(); m.env.close()

def test_six_configs_preserve_controls_and_change_only_entropy_and_scale():
    manifest = json.loads((ROOT/'configs/dmcontrol/experiments/ambi_inner_entropy.json').read_text())
    assert len(manifest['configs']) == 6
    seen = set()
    common = None
    for name in manifest['configs']:
        cfg = json.loads((ROOT/f'configs/dmcontrol/algs/{name}.json').read_text())
        p = cfg['alg_params']
        resolved = _build_cfg(**p)
        assert (resolved.inner_critic_updates_per_action, resolved.inner_actor_updates_per_action) == (128,32)
        assert (cfg['seed'], cfg['total_steps'], cfg['checkpoint_every'], cfg['save_strat']) == (55,1000000,25000,'all')
        assert (p['inner_rollout_horizon'],p['inner_rounds'],p['inner_solve_interval']) == (3,8,1)
        assert (p['inner_rollouts_per_round'],p['inner_critic_updates_per_round'],p['inner_actor_updates_per_round'],p['inner_batch_size']) == (128,16,4,256)
        assert p['inner_replay_capacity']==3072 and p['inner_finite_horizon']
        assert p['outer_critic_target']==p['inner_sac_critic_target']=='reward_only'
        assert p['aux_return_mode']=='off' and p['inner_temperature_mode']=='inherit_outer'
        entropy=p['outer_actor_entropy_mode']
        assert p['inner_actor_entropy_mode']==entropy
        assert p['ent_coef']==(0.0001 if entropy=='tdmpc2_scaled' else 0.0021)
        assert p['inner_temperature']==p['ent_coef']
        seen.add((entropy,p['inner_actor_loss_scale_scope'],p['inner_actor_loss_scale_update']))
        comparable={k:v for k,v in p.items() if k not in {
            'outer_actor_entropy_mode','inner_actor_entropy_mode','ent_coef','inner_temperature',
            'inner_actor_loss_scale_scope','inner_actor_loss_scale_update','wandb_run_name','wandb_tags'}}
        if common is None: common=comparable
        assert comparable==common
    assert len(seen)==6

def test_resource_plan_uses_measurements_and_prefers_one_cpu():
    from slurm.inner_entropy_oscar import select_plan
    rows=[{'index':0,'physical_cpus':n,'checkpoint_bytes':72_500_000,
           'measured':{'seconds_per_decision':s}} for n,s in ((1,.4),(2,.39))]
    plan=select_plan(rows)
    assert plan['cpus']==1 and plan['wall_hours']==140
    rows[1]['measured']['seconds_per_decision']=.3
    assert select_plan(rows)['cpus']==2

def test_checkpoint_quota_uses_user_limit_not_shared_free_space(monkeypatch,tmp_path):
    from slurm import inner_entropy_oscar as launch
    monkeypatch.setattr(launch.subprocess,'check_output',lambda *a,**k:
                        'rgao48 /oscar/scratch 500.00 97.65 512 10240 1 1 1 OK None\n')
    with pytest.raises(AssertionError):
        launch.checkpoint_space(tmp_path,20*2**30)

def test_pending_resource_changes_are_held_and_all_fields_verified(monkeypatch):
    from slurm import inner_entropy_oscar as launch
    records=iter([
        {'JobState':'PENDING','UserId':'rgao48(1)','WorkDir':str(launch.ROOT)},
        {'JobState':'PENDING','NumCPUs':'1','CPUs/Task':'1','MinCPUsNode':'1',
         'TresPerTask':'cpu=1','TimeLimit':'5-20:00:00'},
    ])
    calls=[]
    monkeypatch.setattr(launch,'slurm_fields',lambda job:next(records))
    monkeypatch.setattr(launch.subprocess,'run',lambda args,**kw:calls.append(args))
    launch.resize_pending(123,{'cpus':1,'wall_hours':140})
    assert calls[0]==['scontrol','hold','123']
    assert {'NumCPUs=1','CPUsPerTask=1','TresPerTask=cpu:1','MinCPUsNode=1'} <= set(calls[1])
    assert calls[2]==['scontrol','release','123']
