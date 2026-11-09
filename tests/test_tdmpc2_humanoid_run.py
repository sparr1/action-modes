"""The new native baseline preserves the H3 learner and planner settings."""
from copy import deepcopy
from types import SimpleNamespace
import gymnasium as gym
import numpy as np
from RL.TDMPC2 import TDMPC2Baseline
from slurm.tdmpc2_humanoid_run_oscar import recipe,verify_recipe,select_plan


def test_native_baseline_configuration_and_resolved_planner():
    verify_recipe()
    _,manifest,run=recipe();params=deepcopy(run['alg_params']);params.update(wandb=False,device='cpu')
    algorithm=object.__new__(TDMPC2Baseline)
    algorithm.env=SimpleNamespace(observation_type='state',
        observation_space=gym.spaces.Box(-np.inf,np.inf,shape=(67,),dtype=np.float32),
        action_space=gym.spaces.Box(-1.,1.,shape=(21,),dtype=np.float32),
        spec=SimpleNamespace(max_episode_steps=500))
    algorithm.run_params={**run,'device':'cpu'};algorithm.custom_params=params
    cfg=algorithm._build_cfg(params)
    assert cfg.seed==55 and cfg.steps==1000000 and cfg.mpc and not cfg.compile
    assert (cfg.train_unroll_horizon,cfg.outer_planning_horizon,cfg.iterations)==(3,3,6)
    # The agent constructor adds two iterations for action_dim >= 20; the GPU gate checks eight.
    assert (cfg.num_samples,cfg.num_elites,cfg.num_pi_trajs)==(512,64,24)
    assert cfg.entropy_coef==.0001 and cfg.num_q==5 and cfg.eval_freq is None


def test_sizing_uses_smallest_core_count_without_losing_throughput():
    rows=[dict(cpus=n,seconds_per_decision=s,peak_rss_kib=2*2**20) for n,s in [(1,.08),(2,.079)]]
    assert select_plan(rows)==dict(cpus=1,seconds_per_decision=.08,wall_hours=27,memory_gib=8)
    rows[1]['seconds_per_decision']=.06
    assert select_plan(rows)['cpus']==2
