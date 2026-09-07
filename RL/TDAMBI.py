"""Evaluation-only TD-MPC2 actor--critic adaptation on imagined transitions.

The native checkpoint supplies the model and both Q ensembles. Only the copied
actor and critics learn inside the shared inner engine; this wrapper has no
training or writeback entry point.
"""

import copy
import math
from collections.abc import Mapping

import torch

from RL.AMBITDMPC2 import AMBITDMPC2
from RL.TDMPC2 import _DEFAULTS as _NATIVE_DEFAULTS
from RL.tdmpc2_core import MODEL_SIZE
from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
from RL.tdmpc2_core.common.layers import api_model_conversion


# These are fixed native semantics, not additional SAC ablations. Only the
# canonical collection/update budget and explicitly mapped learning rates vary.
_NATIVE_CONTRACT = {
    "mpc": False,
    "q_representation": "distributional",
    "log_std_mapping": "tdmpc2_tanh",
    "inner_log_std_mapping": "tdmpc2_tanh",
    "inner_operator": "tdambi",
    "inner_actor_adaptation": "clone",
    "inner_critic_adaptation": "clone",
    "inner_critic_dropout_enabled": True,
    "inner_bootstrap_source": "inner_target",
    "inner_q_target_reduction": "min_pair",
    "inner_q_actor_reduction": "mean_pair",
    "inner_sac_critic_target": "reward_only",
    "outer_critic_target": "reward_only",
    "inner_finite_horizon": False,
    "inner_outer_replay_fraction": 0.0,
    "inner_explorer_mode": "none",
    "inner_behavior_action": "policy_sample",
    "inner_behavior_std_scale": 1.0,
    "inner_behavior_noise_std": 0.0,
    "inner_execution_action": "mean",
    "inner_execution_std_scale": 1.0,
    "inner_execution_noise_std": 0.0,
    "inner_execution_policy_source": "primary",
    "inner_temperature_mode": "fixed",
    "inner_temperature": 1.0,
    "inner_actor_writeback_coef": 0.0,
    "inner_critic_writeback_coef": 0.0,
    "inner_outer_policy_kl_coef": 0.0,
    "inner_outer_action_l2_coef": 0.0,
    "inner_critic_target_update_interval": 1,
    "inner_adam_eps": 1e-8,
    "inner_actor_adam_eps": 1e-5,
    "inner_diagnostic_rollouts": 0,
    "compile": False,
    "compile_strict": False,
    "episodic": False,
    "outer_behavior_policy_kl_schedule": "none",
    "sac_actor_loss_scale_mode": "none",
    "value_equivalence_diagnostics": False,
    "value_equivalence_loss_coef": 0.0,
    "eval_inner_comparison": False,
}
for _component in ("actor", "critic", "temperature", "replay", "actor_optimizer",
                   "critic_optimizer", "temperature_optimizer"):
    _NATIVE_CONTRACT[f"inner_{_component}_scope"] = "action"


def native_evaluation_params(params):
    """Translate saved native settings without inventing a learned SAC state."""
    result = copy.deepcopy(params)
    # A saved mpc flag describes training collection, not this controller.
    result["mpc"] = False
    native = {**_NATIVE_DEFAULTS, **params}
    if native["model_size"] is not None:
        # Unlike AMBI's explicit-head override, native TD-MPC2's model-size
        # preset determines the actual ensemble even if num_q was also saved.
        size = int(native["model_size"])
        if size not in MODEL_SIZE:
            raise ValueError(f"Unsupported native TD-MPC2 model_size={size}.")
        result["num_q"] = MODEL_SIZE[size]["num_q"]
    for key in ("log_std_min", "log_std_max", "tau", "lr", "grad_clip_norm",
                "entropy_coef", "value_coef"):
        result.setdefault(key, native[key])
    result.update({key: value for key, value in _NATIVE_CONTRACT.items()
                   if key not in result})
    result.setdefault("inner_actor_lr", float(native["lr"]))
    result.setdefault("inner_critic_lr", float(native["lr"]))
    result.setdefault("inner_critic_target_tau", float(native["tau"]))
    result.setdefault("inner_actor_grad_clip_norm", float(native["grad_clip_norm"]))
    result.setdefault("inner_critic_grad_clip_norm", float(native["grad_clip_norm"]))
    result.setdefault("inner_log_std_min", float(native["log_std_min"]))
    result.setdefault("inner_log_std_max", float(native["log_std_max"]))
    result.setdefault("tdambi_entropy_coef", float(native["entropy_coef"]))
    result.setdefault("tdambi_value_coef", float(native["value_coef"]))
    result.setdefault("tdambi_scale_tau", float(native["tau"]))
    result.setdefault("inner_rounds", 6)
    result.setdefault("inner_rollouts_per_round", 512)
    result.setdefault("inner_rollout_horizon", 3)
    result.setdefault("inner_update_timing", "round")
    result.setdefault("inner_updates_per_round", None if result.get("inner_steps_per_update") is not None else 3)
    result.setdefault("inner_batch_size", 512)
    result.setdefault("inner_replay_capacity", 12_288)
    return result


class TDAMBIAgent(AMBITDMPC2Agent):
    """The shared action engine with a strict native weight-only loader."""

    def update(self, *args, **kwargs):
        raise RuntimeError("TDAMBI is evaluation-only; outer learning is disabled.")

    def load(self, fp):
        state = fp if isinstance(fp, Mapping) else torch.load(
            fp, map_location=self.device, weights_only=False
        )
        if not isinstance(state, Mapping) or "model" not in state:
            raise ValueError("TDAMBI requires a native TD-MPC2 structured model checkpoint.")
        if any(key in state for key in ("policy_spec", "entropy_spec", "log_alpha", "fixed_alpha")):
            raise ValueError("TDAMBI requires a native TD-MPC2 checkpoint, not an AMBI SAC checkpoint.")
        saved_observation = state.get("observation_spec")
        if saved_observation is not None and saved_observation != self.observation_signature():
            raise ValueError("TDAMBI checkpoint observation specification is incompatible before load.")
        expected = self.model.state_dict()
        source_model = state["model"]
        if not isinstance(source_model, Mapping) or not any(
            key.startswith(("_target_Qs.", "_target_Qs_params.")) for key in source_model
        ):
            raise ValueError("TDAMBI requires the saved target critic; online-Q fallback is not permitted.")
        incoming = dict(api_model_conversion(expected, source_model))
        # Native integer-valued bounds may be stored as int64 buffers. The
        # shared policy implementation keeps equivalent float32 scalar buffers.
        for key in ("log_std_min", "log_std_dif"):
            value = incoming.get(key)
            if torch.is_tensor(value) and value.shape == expected[key].shape:
                incoming[key] = value.to(dtype=expected[key].dtype)
        missing = sorted(set(expected) - set(incoming))
        unexpected = sorted(set(incoming) - set(expected))
        incompatible = sorted(key for key in set(expected) & set(incoming)
                              if not torch.is_tensor(incoming[key])
                              or incoming[key].shape != expected[key].shape
                              or incoming[key].dtype != expected[key].dtype)
        if missing or unexpected or incompatible:
            raise ValueError("TDAMBI checkpoint architecture is incompatible before load: "
                             f"missing={missing[:5]}, unexpected={unexpected[:5]}, "
                             f"shape_or_dtype={incompatible[:5]}.")
        for key in ("log_std_min", "log_std_dif"):
            if not torch.equal(incoming[key].to(expected[key].device), expected[key]):
                raise ValueError(f"TDAMBI checkpoint policy bounds {key} are incompatible before load.")
        if not all(bool(torch.isfinite(value).all()) for value in incoming.values()
                   if torch.is_floating_point(value)):
            raise ValueError("TDAMBI checkpoint contains non-finite model weights.")
        updates = state.get("num_updates", 0)
        if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
            raise ValueError("TDAMBI checkpoint num_updates must be a non-negative integer.")
        # The saved target ensemble must not be reset from the online ensemble.
        self.model.load_state_dict(incoming, strict=True)
        self.model.requires_grad_(False)
        self.model.eval()
        self.num_updates = updates
        self.outer_version = updates
        self.inner_engine.clear_all()
        self.last_inner_metrics = {}
        self.last_inner_rollout_lengths = []
        self._resume_boundary_prepared = False


class TDAMBI(AMBITDMPC2):
    """Frozen native TD-MPC2 backbone, freshly copied actor--critic per action."""

    def __init__(self, name, env, custom_params=None, run_params=None, experiment_params=None):
        if not (experiment_params or {}).get("frozen_checkpoint_evaluation", False):
            raise ValueError("TDAMBI is evaluation-only; use evaluate_ambi_checkpoint.py.")
        super().__init__(name, env, custom_params, run_params, experiment_params)

    def _build_cfg(self, params):
        params = native_evaluation_params(params)
        for key, required in _NATIVE_CONTRACT.items():
            if params.get(key) != required:
                raise ValueError(f"TDAMBI requires {key}={required!r}.")
        if params.get("obs", "state") != "state" or params.get("multitask", False):
            raise ValueError("TDAMBI currently supports single-task state observations only.")
        if params.get("inner_steps_per_update") is not None and params["inner_update_timing"] != "step":
            raise ValueError("TDAMBI inner_steps_per_update requires inner_update_timing='step'.")
        unsupported = [key for key in params if (
            key in {"inner_critic_updates_per_round",
                    "inner_actor_updates_per_round", "inner_model_step_budget",
                    "inner_critic_updates_per_action", "inner_actor_updates_per_action",
                    "inner_temperature_updates_per_action", "inner_iterations",
                    "inner_rollouts", "inner_horizon", "inner_updates_per_iteration"}
            and params[key] is not None)]
        if unsupported:
            raise ValueError(f"TDAMBI requires the canonical paired-update schedule; remove {unsupported}.")
        for key in ("inner_log_std_min", "inner_log_std_max"):
            if params[key] != params[key.removeprefix("inner_")]:
                raise ValueError("TDAMBI must preserve the checkpoint policy distribution.")
        for key in ("q_num_bins", "q_vmin", "q_vmax"):
            native_key = key.removeprefix("q_")
            if params.get(key) is not None and params[key] != params.get(native_key, _NATIVE_DEFAULTS[native_key]):
                raise ValueError("TDAMBI must preserve the native distributional Q support.")
        if params.get("q_pair_size", 2) != 2:
            raise ValueError("TDAMBI uses native random two-head critic reductions.")
        for key in ("tdambi_entropy_coef", "tdambi_value_coef", "tdambi_scale_tau"):
            value = float(params[key])
            if not math.isfinite(value) or value < 0 or (key == "tdambi_scale_tau" and value > 1):
                raise ValueError(f"{key} must be finite and nonnegative (scale tau at most one).")
        for key, native_key in (("tdambi_entropy_coef", "entropy_coef"),
                                ("tdambi_value_coef", "value_coef"), ("tdambi_scale_tau", "tau")):
            if params[key] != params[native_key]:
                raise ValueError(f"TDAMBI {key} must inherit native {native_key}.")
        # Reuse the existing canonical budget/lifecycle validation. The resolved
        # operator changes before constructing any agent or optimization kernel.
        params["inner_operator"] = "sac"
        params["ent_coef"] = 1.0  # inert compatibility buffer; never optimized
        params["actor_lr"] = float(params["lr"])
        params["critic_lr"] = float(params["lr"])
        cfg = super()._build_cfg(params)
        if cfg.obs != "state":
            raise ValueError("TDAMBI currently supports single-task state observations only.")
        cfg.inner_operator = "tdambi"
        cfg.inner_temperature_updates_per_action = 0
        cfg.inner_primary_temperature_updates_per_round = 0
        cfg.inner_primary_optimizer_steps_per_action = (
            cfg.inner_actor_updates_per_action + cfg.inner_critic_updates_per_action
        )
        cfg.inner_total_optimizer_steps_per_action = cfg.inner_primary_optimizer_steps_per_action
        return cfg

    def _make_agent(self, cfg):
        return TDAMBIAgent(cfg)

    def learn(self, *args, **kwargs):
        raise RuntimeError("TDAMBI is evaluation-only; outer learning is disabled.")

    def predict(self, observation, deterministic=True, episode_start=None, *,
                collect_diagnostics=True, trace=None):
        if not deterministic:
            raise ValueError("TDAMBI executes the adapted actor's central action.")
        if trace is not None and trace.probes:
            raise ValueError("TDAMBI does not support SAC shared-observation quality probes.")
        return super().predict(observation, deterministic=True, episode_start=episode_start,
                               collect_diagnostics=collect_diagnostics, trace=trace)
