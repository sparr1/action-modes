"""Observational policy-collapse diagnostics; never update the outer learner."""
from contextlib import contextmanager
import time

import numpy as np
import torch



@contextmanager
def evaluation_modes(*modules):
    modes = [(m, m.training) for module in modules for m in module.modules()]
    try:
        for module in modules:
            module.eval()
        yield
    finally:
        for module, mode in modes:
            module.training = mode


def gradient_geometry(q_objective, entropy_objective, parameters):
    """Separate ascent gradients, without touching .grad or optimizer state."""
    parameters = tuple(p for p in parameters if p.requires_grad)
    if not parameters:
        raise ValueError('No trainable actor parameters for gradient measurement.')

    def derivative(objective):
        values = torch.autograd.grad(objective, parameters, retain_graph=True, allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1)
                          for p, g in zip(parameters, values)])

    q, h = derivative(q_objective), derivative(entropy_objective)
    if not torch.isfinite(q).all() or not torch.isfinite(h).all():
        raise ValueError('Nonfinite actor gradient contribution.')
    qn, hn = q.norm(), h.norm()
    dot = torch.dot(q, h)
    return dict(q_gradient_norm=float(qn), entropy_gradient_norm=float(hn),
                total_gradient_norm=float((q+h).norm()), dot=float(dot),
                cosine=float(dot/(qn*hn)) if qn > 1e-12 and hn > 1e-12 else None,
                entropy_over_q=float(hn/qn) if qn > 1e-12 else None,
                q_zero=bool(qn <= 1e-12), entropy_zero=bool(hn <= 1e-12),
                parameter_count=int(q.numel()))


def gradient_probe(wrapper, observations, *, policy=None, critic=None, scale=None,
                   alpha=None, samples=32, seed=1729, pair=(0, 1)):
    """Fixed-input marginal samples; critic and encoder receive no updates.

    R here is explicitly conditional action spread when observations has one
    row. It is not mislabeled as the training minibatch's cross-state R.
    """
    model, cfg = wrapper.agent.model, wrapper.cfg
    policy = model._pi if policy is None else policy
    critic = model._Qs if critic is None else critic
    scale = wrapper.agent.actor_loss_scale if scale is None else scale
    alpha = wrapper.agent.alpha if alpha is None else alpha
    scale = torch.as_tensor(scale, device=wrapper.agent.device).detach().reshape(())
    alpha = torch.as_tensor(alpha, device=wrapper.agent.device).detach().reshape(())
    if not bool(torch.isfinite(scale) & (scale > 0)):
        raise ValueError('Scale must be finite and positive.')
    obs = torch.as_tensor(observations, device=wrapper.agent.device, dtype=torch.float32)
    if obs.ndim == 1:
        obs = obs[None]
    with evaluation_modes(model, policy, critic), torch.enable_grad():
        with torch.no_grad():
            z = model.encode(obs).detach().repeat_interleave(samples, dim=0)
        generator = torch.Generator(device=z.device).manual_seed(seed)
        noise = torch.randn((len(z), cfg.action_dim), generator=generator, device=z.device)
        actions, info = model.pi(z, policy=policy, noise=noise,
            log_std_mapping=cfg.inner_log_std_mapping, log_std_min=cfg.inner_log_std_min,
            log_std_max=cfg.inner_log_std_max)
        indices = torch.tensor(pair, device=z.device, dtype=torch.long)
        q = model.Q(z, actions, qs=critic, detach=True, reduction=cfg.inner_q_actor_reduction,
                    pair_indices=indices, trusted_pair_indices=True)
        # This campaign explicitly selects the true squashed-entropy recipes.
        if cfg.inner_actor_entropy_mode != 'squashed':
            raise ValueError('This probe requires squashed-action entropy.')
        h = -info['log_prob']
        action_derivative = torch.autograd.grad(q.sum(), actions, retain_graph=True)[0]
        result = gradient_geometry(q.mean()/scale, alpha*h.mean(), policy.parameters())
        spread = torch.quantile(q.detach().flatten(), q.new_tensor([.05, .95]))
        spread = spread[1]-spread[0]
        result.update(q_mean=float(q.detach().mean()), entropy=float(h.detach().mean()),
            scale=float(scale), alpha=float(alpha), raw_entropy_coefficient=float(alpha*scale),
            q_action_spread=float(spread), q_action_spread_floored=float(spread.clamp_min(1)),
            action_spread_over_scale=float(spread/scale),
            action_gradient_rms=float(action_derivative.detach().square().mean().sqrt()),
            action_mean_rms=float(info['mean'].detach().square().mean().sqrt()),
            action_std_mean=float(info['log_std'].detach().exp().mean()),
            noise_seed=seed, critic_pair=list(pair), samples_per_state=samples,
            state_count=len(obs), critic_dropout='disabled_for_fixed_state_measurement')
        return result


def adapted_workspace(engine):
    """Read the finished action-scoped solve, retained in the allocation pool.

    act() expires public workspace fields before returning. Their modules and
    final weights live in the action pool until the next solve initializes it.
    Never silently substitute the outer networks when measuring adaptation.
    """
    state = engine._action_pool
    if state.actor is None or state.critic is None:
        raise RuntimeError('No retained adapted actor/critic after the fresh inner solve.')
    return state


class FrozenPriorController:
    """Fixed network parameters; gSDE matrices are held or independently drawn."""
    def __init__(self, wrapper, mode, interval=None):
        if mode not in {'mean', 'sample', 'iid'}:
            raise ValueError(mode)
        self.wrapper, self.mode = wrapper, mode
        self.gsde = wrapper.cfg.actor_distribution == 'gsde'
        if mode == 'iid' and not self.gsde:
            raise ValueError('iid is a separate arm only for gSDE.')
        self.interval = int(interval or wrapper.cfg.gsde_execution_resample_steps) if self.gsde else 1
        if mode == 'iid':
            self.interval = 1
        if self.interval < 1:
            raise ValueError('Noise interval must be positive.')
        self.reset(55)

    def reset(self, seed):
        self.generator = torch.Generator(device=self.wrapper.agent.device).manual_seed(seed)
        self.matrix, self.steps, self.draws = None, 0, 0

    @torch.no_grad()
    def action(self, observation, *, t0=False):
        del t0
        w = self.wrapper
        z = w.agent.model.encode(w._obs_to_tensor(observation).to(w.agent.device)[None])
        options = dict(deterministic=self.mode == 'mean')
        if self.mode != 'mean':
            if self.gsde:
                if self.matrix is None or self.steps % self.interval == 0:
                    self.matrix = torch.randn(w.agent.model.gsde_noise_shape(batch_shape=(1,)),
                        device=z.device, generator=self.generator)
                    self.draws += 1
                options['sde_noise'] = self.matrix
            else:
                options['noise'] = torch.randn((1, w.cfg.action_dim), device=z.device,
                                               generator=self.generator)
                self.draws += 1
            self.steps += 1
        action = w.agent.model.pi_action(z, **options)[0].cpu().numpy()
        return w._unscale_action(action)


class PairedInnerController:
    """One captured learner shared by mean/sample arms, with paired solve RNG.

    The graph always produces both its sampled action and final actor mean.
    Mean evaluation discards the sampled action; that unused execution draw
    keeps the inner learner RNG protocol identical between paired arms. gSDE
    collection and execution matrices retain their native, separate lifetimes.
    """
    def __init__(self, wrapper, observation, *, seed=55, capture=True):
        from RL.tdmpc2_core.inner_improvement import InnerImprovementEngine
        from RL.tdmpc2_core.whole_inner_graph import WholeInnerGraph
        self.wrapper = wrapper
        # Each controller owns its live gSDE lifetime/RNG state, including when
        # graph and uncaptured controllers are compared in the same process.
        self.engine = InnerImprovementEngine(wrapper.agent)
        with torch.no_grad():
            root = wrapper.agent.model.encode(wrapper._obs_to_tensor(observation).to(wrapper.agent.device)[None])
        worker = InnerImprovementEngine(wrapper.agent)
        worker._whole_graph_worker = True
        self.runner = WholeInnerGraph(worker, root.detach(), capture=capture, seed=seed,
                                     capture_teacher_policy=True)
        self.setup_seconds = self.runner.warmup_seconds + self.runner.capture_seconds
        self.mode = 'sample'

    def reset(self, seed, mode='sample'):
        if mode not in {'mean', 'sample'}:
            raise ValueError(mode)
        self.mode = mode
        self.engine.reset_for_evaluation(seed, reuse_action_pool=True)
        self.runner.seed = int(seed)
        self.runner.reset_rng()
        self.runner.calls = 0

    def action(self, observation, *, t0=False):
        from RL.tdmpc2_core.whole_inner_graph import _gsde_inputs_for_action
        w, engine = self.wrapper, self.engine
        with torch.no_grad():
            root = w.agent.model.encode(w._obs_to_tensor(observation).to(w.agent.device)[None]).detach()
        if engine._gsde_active:
            engine._begin_gsde_action(t0=t0, eval_mode=False)
            collection, execution = _gsde_inputs_for_action(engine, root)
            result = self.runner(root, gsde_collection=collection, gsde_execution=execution)
            if execution is not None:
                engine._gsde_execution_uses += 1
            engine._finish_gsde_action()
        else:
            result = self.runner(root)
        engine.action_index += 1
        selected = result[-1]['pre_tanh_mean'].tanh() if self.mode == 'mean' else result[0]
        return w._unscale_action(selected.detach().cpu().numpy().reshape(-1))


def episode(env, controller, *, env_seed, noise_seed, max_steps=500):
    observation, _ = env.reset(seed=int(env_seed))
    total, actions, heights, uprights = 0., [], [], []
    started, cpu_started = time.perf_counter(), time.process_time()
    ended = False
    for step in range(max_steps):
        physics = env.unwrapped._env.physics
        heights.append(float(physics.head_height()))
        uprights.append(float(physics.torso_upright()))
        action = controller.action(observation, t0=step == 0)
        actions.append(np.asarray(action, dtype=np.float32))
        observation, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        if terminated or truncated:
            ended = True
            break
    array = np.asarray(actions)
    result = dict(seed=int(env_seed), noise_seed=int(noise_seed), return_value=total, steps=step+1,
        seconds=time.perf_counter()-started, cpu_seconds=time.process_time()-cpu_started,
        environment_ended=ended, head_height_mean=float(np.mean(heights)),
        upright_fraction=float(np.mean((np.asarray(heights)>=1.2) & (np.asarray(uprights)>=.8))),
        action_saturation_fraction=float(np.mean(np.abs(array)>=.99)))
    if len(array)>1:
        result.update(action_delta_rms=float(np.sqrt(np.mean(np.diff(array,axis=0)**2))),
                      action_sign_flip_fraction=float(np.mean(array[1:]*array[:-1]<0)))
    return result
