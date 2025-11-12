import torch, copy, time
import torch.nn as nn
import numpy as np
from RL.alg import Algorithm
from utils import core as utils_core
from utils.utils import setup_logs


def _snapshot_env_state(env):
    try:
        sim = getattr(env, "sim", None)
        if sim is not None:
            return sim.get_state()
    except Exception:
        pass
    for fn in ("get_state", "_get_state", "state_vector"):
        if hasattr(env, fn):
            try:
                return getattr(env, fn)()
            except Exception:
                pass
    return None


def _restore_env_state(env, state):
    if state is None:
        return False
    try:
        sim = getattr(env, "sim", None)
        if sim is not None:
            sim.set_state(state)
            if hasattr(sim, "forward"):
                try:
                    sim.forward()
                except Exception:
                    pass
            return True
    except Exception:
        pass
    for fn in ("set_state", "_set_state", "set_state_vector"):
        if hasattr(env, fn):
            try:
                getattr(env, fn)(state)
                return True
            except Exception:
                pass
    return False


class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank=4, alpha=1.0):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        for param in self.original_layer.parameters():
            param.requires_grad = False
        if isinstance(original_layer, nn.Linear):
            self.lora_A = nn.Parameter(
                torch.randn(original_layer.in_features, rank) * 0.01
            )
            self.lora_B = nn.Parameter(torch.zeros(rank, original_layer.out_features))
        else:
            self.lora_A = None
            self.lora_B = None

    def forward(self, x):
        original_output = self.original_layer(x)
        if self.lora_A is not None and self.lora_B is not None:
            lora_output = (x @ self.lora_A) @ self.lora_B * (self.alpha / self.rank)
            return original_output + lora_output
        return original_output


def apply_lora_to_model(model, rank=4, alpha=1.0, target_modules=None):
    if target_modules is None:
        target_modules = [""]

    def should_apply_lora(name):
        if not target_modules:
            return True
        return any(pattern in name for pattern in target_modules)

    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and should_apply_lora(name):
            *parent_path, attr_name = name.split(".")
            parent = model
            for part in parent_path:
                parent = getattr(parent, part)
            lora_layer = LoRALayer(module, rank=rank, alpha=alpha)
            setattr(parent, attr_name, lora_layer)
    return model


def get_lora_parameters(model):
    lora_params = []
    for module in model.modules():
        if isinstance(module, LoRALayer):
            if module.lora_A is not None:
                lora_params.append(module.lora_A)
            if module.lora_B is not None:
                lora_params.append(module.lora_B)
    return lora_params


def get_all_optimizers(agent):
    optimizers = []
    optimizers_attrs = [
        "optimizer",
        "optimizers",
        "actor_optimizer",
        "critic_optimizer",
        "policy_optimizer",
        "q_optimizer",
        "value_optimizer",
    ]
    for attr in optimizers_attrs:
        if hasattr(agent, attr):
            opt = getattr(agent, attr)
            if opt is not None:
                if isinstance(opt, (list, tuple)):
                    optimizers.extend(opt)
                else:
                    optimizers.append(opt)
    if hasattr(agent, "policy"):
        for attr in optimizers_attrs:
            if hasattr(agent.policy, attr):
                opt = getattr(agent.policy, attr)
                if opt is not None:
                    if isinstance(opt, (list, tuple)):
                        optimizers.extend(opt)
                    else:
                        optimizers.append(opt)
    if hasattr(agent, "model"):
        for attr in optimizers_attrs:
            if hasattr(agent.model, attr):
                opt = getattr(agent.model, attr)
                if opt is not None:
                    if isinstance(opt, (list, tuple)):
                        optimizers.extend(opt)
                    else:
                        optimizers.append(opt)
    return list(set(optimizers))


def replace_optimizer_params_with_lora(agent, lora_params, base_lr=3e-4):
    if not lora_params:
        return False
    optimizers = get_all_optimizers(agent)
    replaced_any = False
    for opt in optimizers:
        try:
            lr = opt.defaults.get("lr", base_lr)
            new_optimizer = torch.optim.Adam(lora_params, lr=lr)
            for attr in dir(agent):
                if getattr(agent, attr, None) is opt:
                    setattr(agent, attr, new_optimizer)
                    replaced_any = True
                    break
            if hasattr(agent, "policy"):
                for attr in dir(agent.policy):
                    if getattr(agent.policy, attr, None) is opt:
                        setattr(agent.policy, attr, new_optimizer)
                        replaced_any = True
                        break
            if hasattr(agent, "model"):
                for attr in dir(agent.model):
                    if getattr(agent.model, attr, None) is opt:
                        setattr(agent.model, attr, new_optimizer)
                        replaced_any = True
                        break
        except Exception:
            continue
    return replaced_any


class AMBI(Algorithm):
    def __init__(self, name, env, custom_params=None):
        super().__init__(name, env, custom_params=custom_params)
        cp = custom_params or {}

        self.outer_alg_str = cp.get("outer_alg", cp.get("alg", None))
        self.outer_alg_params = cp.get("outer_alg_params", {})
        self.inner_alg_str = cp.get("inner_alg", self.outer_alg_str)
        self.inner_alg_params = cp.get("inner_alg_params", {})

        self.use_lora = bool(cp.get("use_lora", True))
        self.lora_rank = int(cp.get("lora_rank", 4))
        self.lora_alpha = float(cp.get("lora_alpha", 1.0))
        self.lora_target_modules = cp.get("lora_target_modules", None)

        self.inner_rollouts = int(cp.get("inner_rollouts", 6))
        self.inner_horizon = int(cp.get("inner_horizon", 32))
        self.inner_reinit_every_step = bool(cp.get("inner_reinit_every_step", True))
        self.inner_updates_per_rollout = int(cp.get("inner_updates_per_rollout", 1))

        self.max_episode_steps = int(cp.get("max_episode_steps", 250))
        self.outer_update_frequency = int(cp.get("outer_update_frequency", 1000))
        self.render = bool(cp.get("render", False))

        if self.outer_alg_str is None:
            raise ValueError("AMBI requires 'outer_alg' string in custom_params")

        _outer = utils_core.initialize_alg(
            self.outer_alg_str, self.outer_alg_params, env
        )

        if isinstance(_outer, tuple) or isinstance(_outer, list):
            try:
                self.outer_agent = _outer[0]
            except Exception:
                self.outer_agent = _outer
        else:
            self.outer_agent = _outer
        self.outer_buffer = []
        self.inner_buffer = []
        self.alg_logger = None
        self._persistent_inner = None

    def set_logger(self, logger):
        self.alg_logger = logger
        try:
            self.outer_agent.set_logger(logger)
        except Exception:
            pass

    def predict(self, obs):
        return self.outer_agent.predict(obs)

    def save(self, save_path, name):
        try:
            self.outer_agent.save(save_path, name)
        except Exception:
            pass

    def load(self, load_path):
        try:
            self.outer_agent.load(load_path)
        except Exception:
            pass

    def _run_inner_rollout_once(self, env_copy, inner_agent, init_obs):
        obs = init_obs
        cum_reward = 0.0
        rollout_data = []
        for step in range(self.inner_horizon):
            action, _ = inner_agent.predict(obs)
            ret = env_copy.step(action)
            if len(ret) == 5:
                obs_next, reward, terminated, truncated, info = ret
                done = bool(terminated or truncated)
            else:
                obs_next, reward, done, info = ret
            self.inner_buffer.append((obs, action, reward, obs_next, done))
            obs = obs_next
            cum_reward += float(reward)
            if done:
                ret_reset = env_copy.reset()
                if isinstance(ret_reset, tuple):
                    obs = ret_reset[0]
                else:
                    obs = ret_reset
        return obs, cum_reward

    def _get_model_from_agent(self, agent):
        if hasattr(agent, "model"):
            return agent.model
        if hasattr(agent, "policy"):
            return agent.policy
        if hasattr(agent, "actor"):
            return agent.actor
        if hasattr(agent, "q_network") or hasattr(agent, "qf"):
            return getattr(agent, "q_network", None) or agent.qf
        if hasattr(agent, "network"):
            return agent.network
        if hasattr(agent, "get_model"):
            try:
                return agent.get_model()
            except Exception:
                pass
        return agent

    def _copy_model_weights(self, source_agent, target_agent):
        source_model = self._get_model_from_agent(source_agent)
        target_model = self._get_model_from_agent(target_agent)
        if source_model is None or target_model is None:
            return False
        try:
            if hasattr(source_model, "state_dict") and hasattr(
                target_model, "load_state_dict"
            ):
                target_model.load_state_dict(source_model.state_dict())
                return True
        except Exception:
            pass
        try:
            source_params = dict(source_model.named_parameters())
            target_params = dict(target_model.named_parameters())
            for name, param in target_params.items():
                if name in source_params:
                    param.data.copy_(source_params[name].data)
            return True
        except Exception:
            pass
        return False

    def _build_inner_agent(self, agent_env):
        # Create inner agent bound to the provided agent_env (should be a model/copy env)
        _inner = utils_core.initialize_alg(
            self.inner_alg_str, self.inner_alg_params, agent_env
        )
        if isinstance(_inner, tuple) or isinstance(_inner, list):
            try:
                inner_agent = _inner[0]
            except Exception:
                inner_agent = _inner
        else:
            inner_agent = _inner

        weights_copied = self._copy_model_weights(self.outer_agent, inner_agent)

        if self.use_lora:
            inner_model = self._get_model_from_agent(inner_agent)

            if inner_model is not None and isinstance(inner_model, nn.Module):
                # Apply LoRA to the model
                inner_model = apply_lora_to_model(
                    inner_model,
                    rank=self.lora_rank,
                    alpha=self.lora_alpha,
                    target_modules=self.lora_target_modules,
                )

                lora_params = get_lora_parameters(inner_model)

                if lora_params:
                    # Replace optimizers to only train LoRA params
                    replaced = replace_optimizer_params_with_lora(
                        inner_agent,
                        lora_params,
                        base_lr=self.inner_alg_params.get("learning_rate", 3e-4),
                    )
        return inner_agent

    def _update_inner_agent(self, inner_agent):
        """
        Update inner agent using imagined experience.

        This method tries multiple update strategies to work with any algorithm:
        1. Direct update() method with buffer
        2. Learn() method with small timesteps
        3. Step-by-step updates
        4. Replay buffer population + training
        """
        if not self.inner_buffer:
            return False

        updated = False

        for _ in range(self.inner_updates_per_rollout):
            # Strategy 1: Direct update with buffer (custom implementations)
            try:
                if hasattr(inner_agent, "update") and callable(inner_agent.update):
                    inner_agent.update(self.inner_buffer)
                    updated = True
                    break
            except Exception:
                pass

            # Strategy 2: Use learn() method (stable-baselines3 style)
            try:
                if hasattr(inner_agent, "learn") and callable(inner_agent.learn):
                    # Small number of gradient steps
                    inner_agent.learn(total_timesteps=len(self.inner_buffer))
                    updated = True
                    break
            except Exception:
                pass

            # Strategy 3: Populate replay buffer and train (stable-baselines3)
            try:
                if hasattr(inner_agent, "replay_buffer") and hasattr(
                    inner_agent, "train"
                ):
                    # Add transitions to replay buffer
                    for obs, action, reward, next_obs, done in self.inner_buffer:
                        try:
                            # Try stable-baselines3 style add
                            inner_agent.replay_buffer.add(
                                obs, next_obs, action, reward, done, [{}]
                            )
                        except Exception:
                            try:
                                # Try simpler add signature
                                inner_agent.replay_buffer.add(
                                    obs, action, reward, next_obs, done
                                )
                            except Exception:
                                pass

                    # Train on the buffer
                    if len(inner_agent.replay_buffer) > 0:
                        inner_agent.train(gradient_steps=self.inner_updates_per_rollout)
                        updated = True
                        break
            except Exception:
                pass

            # Strategy 4: Step-by-step updates (custom implementations)
            try:
                if hasattr(inner_agent, "step") and callable(inner_agent.step):
                    for obs, action, reward, next_obs, done in self.inner_buffer:
                        try:
                            inner_agent.step(obs, action, reward, next_obs, None, done)
                        except Exception:
                            try:
                                inner_agent.step(obs, action, reward, next_obs, done)
                            except Exception:
                                pass
                    updated = True
                    break
            except Exception:
                pass

        if not updated:
            print(
                "  Warning: Could not update inner agent - no compatible update method found"
            )

        return updated

    def _make_model_env_copy(self):
        try:
            env_copy = copy.deepcopy(self.env)
            return env_copy, False, None
        except Exception:
            snapshot = _snapshot_env_state(self.env)
            return self.env, True, snapshot

    def learn(self, total_timesteps=10000):
        t = 0
        episodes = 0
        outer_obs = None
        reset_return = self.env.reset()
        if isinstance(reset_return, tuple):
            outer_obs, _info = reset_return
        else:
            outer_obs = reset_return
        start_time = time.time()
        while t < total_timesteps:
            env_model_copy, uses_snapshot, snapshot = self._make_model_env_copy()
            if self.inner_reinit_every_step:
                inner_agent = self._build_inner_agent(env_model_copy)
            else:
                if self._persistent_inner is None:
                    self._persistent_inner = self._build_inner_agent(env_model_copy)
                inner_agent = self._persistent_inner
            self.inner_buffer = []
            for b in range(self.inner_rollouts):
                if uses_snapshot:
                    _restore_env_state(env_model_copy, snapshot)
                    reset_return = env_model_copy.reset()
                    if isinstance(reset_return, tuple):
                        obs_model = reset_return[0]
                    else:
                        obs_model = reset_return
                    obs_model = outer_obs
                else:
                    try:
                        outer_snapshot = _snapshot_env_state(self.env)
                        if outer_snapshot is not None:
                            _restore_env_state(env_model_copy, outer_snapshot)
                            reset_return = env_model_copy.reset()
                            if isinstance(reset_return, tuple):
                                obs_model = reset_return[0]
                            else:
                                obs_model = reset_return
                            obs_model = outer_obs
                        else:
                            obs_model = outer_obs
                    except Exception:
                        obs_model = outer_obs
                final_obs, cum = self._run_inner_rollout_once(
                    env_model_copy, inner_agent, obs_model
                )
                self._update_inner_agent(inner_agent)

            try:
                real_action, _ = inner_agent.predict(outer_obs)
            except Exception:
                real_action, _ = self.outer_agent.predict(outer_obs)
            ret = self.env.step(real_action)
            if len(ret) == 5:
                outer_obs_next, reward, terminated, truncated, info = ret
                done = bool(terminated or truncated)
            else:
                outer_obs_next, reward, done, info = ret
            if self.alg_logger:
                data = setup_logs(
                    reward,
                    outer_obs,
                    real_action,
                    [terminated if "terminated" in locals() else False, done],
                    [info],
                )
                self.alg_logger.on_step(data)
            self.outer_buffer.append(
                (outer_obs, real_action, reward, outer_obs_next, done)
            )
            t += 1
            outer_obs = outer_obs_next
            if done:
                episodes += 1
                reset_return = self.env.reset()
                if isinstance(reset_return, tuple):
                    outer_obs, _info = reset_return
                else:
                    outer_obs = reset_return
            if t % self.outer_update_frequency == 0 or t >= total_timesteps:
                try:
                    self.outer_agent.update(self.outer_buffer)
                except Exception:
                    try:
                        self.outer_agent.learn(total_timesteps=0)
                    except Exception:
                        pass
            if self.render:
                try:
                    self.env.render()
                except Exception:
                    pass
        total_time = time.time() - start_time
        print(
            f"AMBI Training completed in {total_time:.2f} seconds over {episodes} episodes and {t} timesteps."
        )
        return self.outer_agent
