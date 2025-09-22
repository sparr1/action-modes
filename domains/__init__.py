from gymnasium.envs.registration import register
from .ant_variable_legs import AntVariableLegsEnv

register(id="VarLegsAnt-v0",
    entry_point="domains:AntVariableLegsEnv",          # "<module>:<class or callable>"
    max_episode_steps=1000,              # optional
    order_enforce=True,                  # default; keeps Gymnasium API strict
    kwargs={"exclude_current_positions_from_observation":False,"num_legs":8, "contact_cost_weight":.0, "render_mode":"human"}
    )