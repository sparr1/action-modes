from importlib import import_module
from importlib.util import find_spec

from gymnasium.envs.registration import register

_ANT_MODULES = {
    "AntVariableLegsEnv": ".ant_variable_legs",
    "Ant3LegDeadStumpEnv": ".ant_3leg_deadstump_env",
    "AntLegAdaptationEnv": ".ant_leg_adaptation",
}
__all__ = ["register", *_ANT_MODULES]
_has_adaptation = find_spec(f"{__name__}.ant_leg_adaptation") is not None
if not _has_adaptation:
    AntLegAdaptationEnv = None


def __getattr__(name):
    # Keep the public Ant imports and Gym entry points available without
    # loading their native MuJoCo bindings in the legacy Meta-World process.
    if name not in _ANT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(_ANT_MODULES[name], __name__), name)


register(id="VarLegsAnt-v0",
    entry_point="domains:AntVariableLegsEnv",
    max_episode_steps=1000,
    order_enforce=True,
    kwargs={"exclude_current_positions_from_observation":False,"num_legs":8, "contact_cost_weight":.0, "render_mode":None}
    )

if _has_adaptation:
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

# Keep this entry point as a string: importing the existing ``domains``
# package must not require DMControl, MuJoCo 3, or a rendering backend.
register(
    id="DMControl-v0",
    entry_point="domains.dmcontrol:DMControlEnv",
    max_episode_steps=500,
    order_enforce=True,
)

# Meta-World has its own pinned MuJoCo runtime; keep the optional import lazy.
register(
    id="MetaWorld-v0",
    entry_point="domains.metaworld:MetaWorldEnv",
    max_episode_steps=100,
    order_enforce=True,
)
