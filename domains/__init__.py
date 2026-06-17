from gymnasium.envs.registration import register
from .ant_variable_legs import AntVariableLegsEnv
from .ant_leg_adaptation import AntLegAdaptationEnv
from .ant_3leg_deadstump_env import Ant3LegDeadStumpEnv

register(id="VarLegsAnt-v0",
    entry_point="domains:AntVariableLegsEnv",
    max_episode_steps=1000,
    order_enforce=True,
    kwargs={"exclude_current_positions_from_observation":False,"num_legs":8, "contact_cost_weight":.0, "render_mode":"human"}
    )

# Adaptation env: 4 legs -> 3 legs at switch_fraction of total_timesteps.
# total_timesteps should match the value passed to model.learn().
register(id="LegAdaptAnt-v0",
    entry_point="domains:AntLegAdaptationEnv",
    max_episode_steps=1000,
    order_enforce=True,
    kwargs={"exclude_current_positions_from_observation":False, "total_timesteps":1000000, "start_legs":4, "end_legs":3, "switch_fraction":0.5, "contact_cost_weight":0.0}
    )

# Dead-stump ant: canonical 4-leg body geometry with right_back_leg rigidly
# welded (hip_4/ankle_4 joints removed). nu=6, nq=13, nv=12. Mass identical
# to the healthy ant. Designed for transfer/fine-tuning from a 4-leg policy.
register(id="Ant3LegDeadStump-v0",
    entry_point="domains:Ant3LegDeadStumpEnv",
    max_episode_steps=1000,
    order_enforce=True,
    kwargs={"exclude_current_positions_from_observation":False, "contact_cost_weight":0.0}
    )