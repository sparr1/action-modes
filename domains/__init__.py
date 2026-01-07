from gymnasium.envs.registration import register
from .ant_variable_legs import AntVariableLegsEnv
from .VAntMaze import VAntMazeEnv
from .AntMaze import MyAntMazeEnv

register(id="VarLegsAnt-v0",
    entry_point="domains:AntVariableLegsEnv",          # "<module>:<class or callable>"
    max_episode_steps=1000,              # optional
    order_enforce=True,                  # default; keeps Gymnasium API strict
    kwargs={"exclude_current_positions_from_observation":False,"num_legs":8, "contact_cost_weight":.0, "render_mode":"human"}
    )
register(id="2AntMaze-v0",
    entry_point="domains:VAntMazeEnv",          # "<module>:<class or callable>"
    max_episode_steps=1000,              # optional
    order_enforce=True,                  # default; keeps Gymnasium API strict
    kwargs={"num_legs":2, "reward_type": "sparse", "render_mode":"human"}
    )
register(id="2AntMazeDense-v0",
    entry_point="domains:VAntMazeEnv",          # "<module>:<class or callable>"
    max_episode_steps=1000,              # optional
    order_enforce=True,                  # default; keeps Gymnasium API strict
    kwargs={"num_legs":2, "reward_type": "dense", "render_mode":"human"}
)
register(id="4AntMaze-v0",
    entry_point="domains:VAntMazeEnv",          # "<module>:<class or callable>"
    max_episode_steps=1000,              # optional
    order_enforce=True,                  # default; keeps Gymnasium API strict
    kwargs={"num_legs":4, "render_mode":"human"}
    )
register(id="MyAntMaze-v0",
    entry_point="domains:MyAntMazeEnv",          # "<module>:<class or callable>"
    max_episode_steps=1000,              # optional
    order_enforce=True,                  # default; keeps Gymnasium API strict
    kwargs={"render_mode":"human"}
    )
register(id="4AntMazeDense-v0",
    entry_point="domains:VAntMazeEnv",          # "<module>:<class or callable>"
    max_episode_steps=1000,              # optional
    order_enforce=True,                  # default; keeps Gymnasium API strict
    kwargs={"num_legs":4, "reward_type": "dense", "render_mode":"human"}
)
register(id="6AntMaze-v0",
    entry_point="domains:VAntMazeEnv",          # "<module>:<class or callable>"
    max_episode_steps=1000,              # optional
    order_enforce=True,                  # default; keeps Gymnasium API strict
    kwargs={"num_legs":6, "render_mode":"human"}
    )
register(id="6AntMazeDense-v0",
    entry_point="domains:VAntMazeEnv",          # "<module>:<class or callable>"
    max_episode_steps=1000,              # optional
    order_enforce=True,                  # default; keeps Gymnasium API strict
    kwargs={"num_legs":6, "reward_type": "dense", "render_mode":"human"}
    )
register(id="8AntMaze-v0",
    entry_point="domains:VAntMazeEnv",          # "<module>:<class or callable>"
    max_episode_steps=1000,              # optional
    order_enforce=True,                  # default; keeps Gymnasium API strict
    kwargs={"num_legs":8, "render_mode":"human"}
    )
register(id="8AntMazeDense-v0",
    entry_point="domains:VAntMazeEnv",          # "<module>:<class or callable>"
    max_episode_steps=1000,              # optional
    order_enforce=True,                  # default; keeps Gymnasium API strict
    kwargs={"num_legs":8, "reward_type": "dense", "render_mode":"human"}
    )