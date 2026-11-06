"""Single-axis interval contracts and real controller refresh boundaries."""
from copy import deepcopy
from types import SimpleNamespace
import gymnasium as gym
import numpy as np
import pytest
import torch
from RL.AMBITDMPC2 import AMBITDMPC2
from slurm.gsde_resample_intervals_oscar import INTERVALS, MANIFEST, validate_recipe
from slurm.ambi_transfer_training import recipe
from tests.test_ambi_gsde_lifecycle import _model


@pytest.mark.parametrize('index', range(4))
def test_interval_recipe_resolves_the_exact_reference_with_one_changed_interval(index):
    _, manifest, run = validate_recipe(index)
    params = deepcopy(run['alg_params'])
    params.update(device='cpu', wandb=False)
    algorithm = object.__new__(AMBITDMPC2)
    algorithm.env = SimpleNamespace(observation_type='state',
        observation_space=gym.spaces.Box(-np.inf,np.inf,shape=(67,),dtype=np.float32),
        action_space=gym.spaces.Box(-1.,1.,shape=(21,),dtype=np.float32),
        spec=SimpleNamespace(max_episode_steps=500))
    algorithm.run_params = {**run,'device':'cpu'}
    algorithm.custom_params = params
    cfg = algorithm._build_cfg(params)
    assert (cfg.inner_gsde_resample_steps,cfg.gsde_execution_resample_steps)==INTERVALS[index]
    assert cfg.inner_gsde_noise_scope=='real_steps' and cfg.actor_distribution=='gsde'
    assert cfg.outer_actor_entropy_mode==cfg.inner_actor_entropy_mode=='squashed'
    assert cfg.ent_coef==cfg.inner_temperature==.0021
    assert cfg.outer_actor_objective=='q_maximization'
    assert cfg.inner_actor_updates_per_action==32 and cfg.inner_critic_updates_per_action==128
    assert cfg.inner_rollout_horizon==3 and cfg.inner_rounds==8 and cfg.inner_rollouts_per_round==128
    assert cfg.compile and cfg.compile_strict and cfg.inner_cuda_graph
    assert cfg.critic_ensemble_backend=='vmap' and cfg.critic_geometry_diagnostics
    assert cfg.seed==55 and cfg.steps==1000000
    assert manifest['checkpoint_every']==run['checkpoint_every']==25000


def test_four_arms_are_unique_single_axis_changes_and_reuse_existing_baseline():
    assert len(set(INTERVALS))==4 and (8,8) not in INTERVALS
    names=[]
    for index,(imagined,execution) in enumerate(INTERVALS):
        assert (imagined==8) != (execution==8)
        name,manifest,_=recipe(index,MANIFEST)
        names.append(name)
        assert manifest['reference_runs']['gsde_real_steps_8_8']['id']=='590be7c27a9b40eaa9a68218cdb74d0a'
    assert len(set(names))==4


@pytest.mark.parametrize('imagined,execution',INTERVALS)
def test_actual_controller_keeps_independent_intervals_through_parameter_updates(imagined,execution):
    model=_model(inner_gsde_noise_scope='real_steps',inner_gsde_resample_steps=imagined,
                 gsde_execution_resample_steps=execution,inner_rounds=1)
    engine=model.agent.inner_engine
    previous=None
    try:
        for step in range(2*max(imagined,execution)+1):
            model.agent.act(torch.zeros(3),t0=step==0,collect_diagnostics=False)
            current=(engine._gsde_collection_noise.clone(),engine._gsde_execution_noise.clone())
            if previous is not None:
                assert torch.equal(current[0],previous[0]) == (step%imagined!=0)
                assert torch.equal(current[1],previous[1]) == (step%execution!=0)
            # Updating the current prior must not restart either base-noise clock.
            with torch.no_grad(): model.agent.model._pi.log_scale.add_(1e-4)
            previous=current
        model.agent.act(torch.zeros(3),t0=True,collect_diagnostics=False)
        assert engine._gsde_decision_index==engine._gsde_execution_uses==1
        assert not torch.equal(engine._gsde_collection_noise,previous[0])
        assert not torch.equal(engine._gsde_execution_noise,previous[1])
    finally:
        model.env.close()
