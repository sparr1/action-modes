import copy
import importlib
import json
from collections.abc import Mapping, Sequence

import gymnasium as gym

SUPPORTED_WRAPPERS = ("Subtask", "AntPlane", "ScaledStateWrapper", "PlatformFlattenedActionWrapper", "ScaledParameterisedActionWrapper")
SUPPORTED_LOG_SETTINGS = ("none", "overwrite", "warn", "timestamp", "overwrite-safe")
SUPPORTED_LOG_TYPES = ("detailed", "summary")
_RENDER_MODE_UNSET = object()


def _validate_wrapper_config(wrapper, location):
    if not isinstance(wrapper, Mapping):
        raise ValueError(f"{location} must be a mapping.")

    wrapper_name = wrapper.get("name")
    if not isinstance(wrapper_name, str) or not wrapper_name:
        raise ValueError(f"{location} must define a non-empty string 'name'.")
    short_name = wrapper_name.split(":")[-1]
    if short_name not in SUPPORTED_WRAPPERS:
        raise ValueError(
            f"Unsupported wrapper {wrapper_name!r} in {location}; "
            f"expected one of {SUPPORTED_WRAPPERS}."
        )
    if short_name == "Subtask" and wrapper_name != "Subtask":
        raise ValueError(
            f"{location} must name the task-aware wrapper exactly 'Subtask'."
        )
    if wrapper_name not in {"Subtask", "AntPlane"}:
        parts = wrapper_name.split(":")
        if len(parts) != 2 or not all(parts):
            raise ValueError(
                f"{location} wrapper {wrapper_name!r} must use the fully "
                "qualified 'module:ClassName' form."
            )

    if "wrapper_params" not in wrapper:
        raise ValueError(f"{location} must define 'wrapper_params'.")
    wrapper_params = wrapper["wrapper_params"]
    if not isinstance(wrapper_params, Mapping):
        raise ValueError(f"{location} 'wrapper_params' must be a mapping.")

    return wrapper_name, wrapper_params


def build_env(run_params, experiment_params, *, render_mode=_RENDER_MODE_UNSET):
    """Build an environment with the same wrapper order used for training.

    When supplied, ``render_mode`` is authoritative, including when it is
    explicitly ``None``. Omitting it preserves the experiment configuration so
    existing training behavior remains unchanged.
    """
    env_params = copy.deepcopy(experiment_params.get("env_params", {}))
    if not isinstance(env_params, Mapping):
        raise ValueError("experiment 'env_params' must be a mapping.")
    env_params = dict(env_params)
    if render_mode is not _RENDER_MODE_UNSET:
        env_params["render_mode"] = render_mode

    domain = gym.make(run_params["env"], **env_params)
    try:
        wrappers = run_params.get("env_wrappers", ())
        if not isinstance(wrappers, Sequence) or isinstance(wrappers, (str, bytes)):
            raise ValueError("'env_wrappers' must be a sequence of wrapper mappings.")

        for index, wrapper in enumerate(wrappers):
            wrapper_name, wrapper_params = _validate_wrapper_config(
                wrapper, f"env_wrappers[{index}]"
            )
            domain = setup_wrapper(domain, wrapper_name, wrapper_params)

        if "env_wrapper" in run_params:
            wrapper_name, wrapper_params = _validate_wrapper_config(
                run_params["env_wrapper"], "env_wrapper"
            )
            domain = setup_wrapper(domain, wrapper_name, wrapper_params)
    except BaseException as exc:
        try:
            domain.close()
        except BaseException as cleanup_error:
            # Preserve the wrapper construction/validation failure as the
            # actionable error even if cleanup itself also fails.
            note = f"Additional environment cleanup failure: {cleanup_error}"
            add_note = getattr(exc, "add_note", None)
            if callable(add_note):
                add_note(note)
            else:
                notes = list(getattr(exc, "__notes__", ()))
                notes.append(note)
                exc.__notes__ = notes
        raise

    return domain


def initialize_alg(alg_string, alg_params, domain, custom_action_space = None, full_run_params=None, experiment_params=None):
    baseline = False
    if '/' in alg_string:
        parts = alg_string.split('/')
        file_name, alg_name = "/".join(parts[:-1]), parts[-1]
        # print(file_name)
        # print(alg_name)
        if "baselines" in file_name:
            baseline = True
            try:
                # Keep environment reconstruction and non-SB3 renderers usable
                # without importing the optional SB3 stack eagerly.
                from RL.baselines import Baseline

                model = Baseline(
                    alg_name,
                    domain,
                    alg_params,
                    full_run_params,
                    experiment_params,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize baseline algorithm '{alg_string}'.") from e
        elif file_name == "PAMDP":
            try:
                module = importlib.import_module("RL.PAMDP")
                alg_class = getattr(module, alg_name)
                model = alg_class(alg_name, domain, alg_params, custom_action_space = custom_action_space)
            except Exception as e:
                raise RuntimeError(f"Failed to initialize PAMDP algorithm '{alg_string}'.") from e
        elif file_name == "modes":
            try:
                module = importlib.import_module("RL.modes")
                alg_class = getattr(module, alg_name)
                model = alg_class(alg_name, domain, **alg_params)
            except Exception as e:
                raise RuntimeError(f"Failed to initialize modes algorithm '{alg_string}'.") from e
        else:
            module = importlib.import_module("RL."+file_name.replace('/','.')) #last ditch, just try to load it!
            alg_class = getattr(module, alg_name)
            model = alg_class(alg_name, domain, alg_params, full_run_params, experiment_params)
    else:
        try:
            module = importlib.import_module("RL.alg")
            alg_class = getattr(module, alg_string)
            model = alg_class(alg_string, domain, alg_params)
            alg_name = alg_string
        except Exception as e:
            raise RuntimeError(f"Failed to initialize algorithm '{alg_string}'.") from e
    return model, baseline, alg_name

#TODO: currently does nothing. either add functionality or delete
def handle_settings(path): #processes the settings.json file
    with open(path) as f:
        contents = json.load(f)
    print(contents) 

def setup_wrapper(domain, wrapper_name, wrapper_params):
    if wrapper_name == 'Subtask':      
        try:
            from modes.tasks import Subtask

            print(wrapper_params["task"])
            module_name,task_name = wrapper_params["task"].split(':') 
            # print(module)
            module = importlib.import_module(module_name)
            task_class = getattr(module,task_name) #grab the specific task
            p = wrapper_params["task_params"]
            task = task_class(**p)
            domain = Subtask(
                domain,
                task,
                task_info=wrapper_params.get("task_info"),
            ) #replace the reward function and termination conditions based on task, then return the new wrapped domain.
        except (ModuleNotFoundError, AttributeError) as e:
            raise ValueError(f"Could not find model class '{task_name}' in module '{module_name}': {e}")
    elif wrapper_name == 'AntPlane':
        from domains.AntPlane import AntPlane

        domain = AntPlane(domain, **wrapper_params)
    else:
        print("setting up default wrapper ", wrapper_name, "with params", wrapper_params)
        module_name,raw_wrapper_name = wrapper_name.split(':') #this is likely to error out
        module = importlib.import_module(module_name)
        wrapper_class = getattr(module, raw_wrapper_name)
        domain = wrapper_class(domain, **wrapper_params)
        print("wrapping appears to have been successful.")

    return domain
