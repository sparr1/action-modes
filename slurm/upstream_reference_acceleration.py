"""CUDA execution regions around the released single-task reference models."""
from __future__ import annotations

from copy import deepcopy
import importlib
import json
import random
import time

import numpy as np
import torch
import torch.nn.functional as F


def rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all()


def restore_rng(state):
    random.setstate(state[0]); np.random.set_state(state[1])
    torch.set_rng_state(state[2]); torch.cuda.set_rng_state_all(state[3])


class PlannerGraph:
    """Capture MPPI's iterations while retaining native final action sampling.

    The native NumPy head pairs are drawn in their original order before the
    captured region. The final NumPy/Torch categorical draw remains native.
    All policy/noise draws inside the graph use Torch's graph-aware generator.
    """
    def __init__(self, agent, method, compile_region=False, capture=True):
        self.agent = agent
        self.cfg = agent.cfg
        self.model = agent.model
        self.method = method
        self.math = importlib.import_module(("boom" if method == "boom" else "tdmpc_square") + ".common.math")
        self.native_pi = type(self.model).pi
        self.z = torch.zeros(1, self.cfg.latent_dim, device=agent.device)
        self.previous = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=agent.device)
        self.reset = torch.ones((), device=agent.device, dtype=torch.bool)
        self.heads = torch.zeros(self.cfg.iterations, 2, device=agent.device, dtype=torch.long)
        self.compile_region = compile_region
        self.capture = capture
        self.graph = None
        self.capture_seconds = None

    def pi(self, z):
        return self.native_pi(self.model, z, None)

    def q(self, z, action, pair):
        out = self.model._Qs(torch.cat((z, action), dim=-1))
        pair_q = out.index_select(0, pair)
        return (self.math.two_hot_inv(pair_q[0], self.cfg) +
                self.math.two_hot_inv(pair_q[1], self.cfg)) / 2

    @torch.no_grad()
    def kernel(self, z, previous, reset, heads):
        cfg = self.cfg
        pi_actions = torch.empty(cfg.horizon, cfg.num_pi_trajs, cfg.action_dim, device=z.device)
        latent = z.repeat(cfg.num_pi_trajs, 1)
        for t in range(cfg.horizon - 1):
            pi_actions[t] = self.pi(latent)[1]
            latent = self.model.next(latent, pi_actions[t], None)
        pi_actions[-1] = self.pi(latent)[1]
        z = z.repeat(cfg.num_samples, 1)
        zeros = torch.zeros(cfg.horizon, cfg.action_dim, device=z.device)
        shifted = torch.cat((previous[1:], zeros[:1]), dim=0)
        mean = torch.where(reset, zeros, shifted)
        std = cfg.max_std * torch.ones_like(mean)
        actions = torch.empty(cfg.horizon, cfg.num_samples, cfg.action_dim, device=z.device)
        actions[:, :cfg.num_pi_trajs] = pi_actions
        for iteration in range(cfg.iterations):
            actions[:, cfg.num_pi_trajs:] = (mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
                cfg.horizon, cfg.num_samples - cfg.num_pi_trajs, cfg.action_dim, device=z.device)).clamp(-1, 1)
            latent = z
            value, discount = 0, 1
            for t in range(cfg.horizon):
                reward = self.math.two_hot_inv(self.model.reward(latent, actions[t], None), cfg)
                latent = self.model.next(latent, actions[t], None)
                value = value + discount * reward
                discount *= self.agent.discount
            value = (value + discount * self.q(latent, self.pi(latent)[1], heads[iteration])).nan_to_num(0)
            indices = torch.topk(value.squeeze(1), cfg.num_elites, dim=0).indices
            elite_value, elite_actions = value[indices], actions[:, indices]
            score = torch.exp(cfg.temperature * (elite_value - elite_value.max(0)[0]))
            score = score / score.sum(0)
            if self.method == "boom":
                score = score.squeeze()
                score_sum = score.sum() + 1e-9
                mean = torch.einsum("e,hea->ha", score, elite_actions) / score_sum
                variance = torch.einsum("e,hea->ha", score, (elite_actions - mean.unsqueeze(1)) ** 2) / score_sum
                std = torch.sqrt(variance).clamp(cfg.min_std, cfg.max_std)
            else:
                mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
                std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2,
                                           dim=1) / (score.sum(0) + 1e-9)).clamp(cfg.min_std, cfg.max_std)
        previous.copy_(mean)
        return elite_actions, score.reshape(-1), std[0]

    @torch.no_grad()
    def build(self):
        state = rng_state()
        previous = self.previous.clone()
        start = time.perf_counter()
        fn = self.kernel
        if self.compile_region:
            # Native random operators retain the released sampling semantics.
            torch._inductor.config.fallback_random = True
            fn = torch.compile(fn, fullgraph=True, dynamic=False)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3): self.outputs = fn(self.z, self.previous, self.reset, self.heads)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.outputs = fn(self.z, self.previous, self.reset, self.heads)
        torch.cuda.synchronize()
        self.previous.copy_(previous)
        restore_rng(state)
        self.capture_seconds = time.perf_counter() - start

    @torch.no_grad()
    def __call__(self, z, t0=False, eval_mode=False, task=None):
        assert task is None and not self.cfg.multitask
        if self.capture and self.graph is None: self.build()
        self.z.copy_(z)
        self.reset.fill_(t0)
        if not t0 and hasattr(self.agent, "_prev_mean") and self.agent._prev_mean.data_ptr() != self.previous.data_ptr():
            self.previous.copy_(self.agent._prev_mean)
        pairs = np.stack([np.random.choice(self.cfg.num_q, 2, replace=False) for _ in range(self.cfg.iterations)])
        self.heads.copy_(torch.as_tensor(pairs, device=self.agent.device))
        if self.capture:
            self.graph.replay()
            elites, score, std = self.outputs
        else:
            elites, score, std = self.kernel(self.z, self.previous, self.reset, self.heads)
        if self.method == "boom":
            index = torch.multinomial(score, 1).item()
        else:
            probabilities = score.cpu().numpy()
            index = np.random.choice(np.arange(probabilities.shape[0]), p=probabilities)
        mu = elites[0, index]
        action = mu if eval_mode else mu + std * torch.randn(self.cfg.action_dim, device=std.device)
        self.agent._prev_mean = self.previous
        # Copies preserve outputs across a later replay, including evaluation.
        return action.clamp(-1, 1).clone(), mu.clone(), std.clone()


def install_planner(agent, method, acceleration):
    if acceleration == "native": return
    agent.plan = PlannerGraph(agent, method, compile_region=acceleration == "compiled-planner")
    if acceleration == "model-planner-graph":
        agent.update = ModelUpdateGraph(agent, method)


@torch.no_grad()
def validate_planner(agent, method, output, acceleration):
    """Compare translated and captured execution against the released planner."""
    native = agent.plan
    z = agent.model.encode(torch.randn(1, 67, device=agent.device), None)
    native(z, t0=True)  # Initialize native compilation and cached math constants.
    translated = PlannerGraph(agent, method, capture=False)
    captured = PlannerGraph(agent, method, compile_region=acceleration == "compiled-planner")
    captured.build()
    rows = []
    for t0, eval_mode in ((True, False), (False, False), (False, True)):
        previous = torch.randn_like(translated.previous)
        state = rng_state()
        outputs, states = [], []
        for fn in (native, translated, captured):
            agent._prev_mean = previous.clone()
            restore_rng(state)
            result = fn(z, t0=t0, eval_mode=eval_mode)
            outputs.append(tuple(x.clone() for x in result) + (agent._prev_mean.clone(),))
            states.append(rng_state())
        deltas = []
        for i in (1, 2):
            for a, b in zip(outputs[0], outputs[i]):
                torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
            assert states[0][0] == states[i][0]
            assert states[0][1][0] == states[i][1][0]
            assert np.array_equal(states[0][1][1], states[i][1][1])
            assert states[0][1][2:] == states[i][1][2:]
            assert torch.equal(states[0][2], states[i][2])
            assert all(torch.equal(a, b) for a, b in zip(states[0][3], states[i][3]))
            deltas.append(max(float((a-b).abs().max()) for a, b in zip(outputs[0], outputs[i])))
        rows.append({"t0": t0, "eval_mode": eval_mode, "rng_exact": True,
                     "eager_max_absolute_difference": deltas[0],
                     "captured_max_absolute_difference": deltas[1]})
    receipt = {"checks": rows, "acceleration": acceleration,
               "capture_seconds": captured.capture_seconds}
    (output / "planner_parity.json").write_text(json.dumps(receipt, indent=2))
    print("PLANNER_PARITY_PASSED", json.dumps(receipt), flush=True)


class ModelUpdateGraph:
    """Capture native world-model forward/backward; keep both optimizers native.

    The actor update (including its original threshold branch), model-wide
    clipping with retained actor gradients, and target update remain upstream.
    """
    def __init__(self, agent, method):
        self.agent, self.model, self.cfg = agent, agent.model, agent.cfg
        self.math = importlib.import_module(("boom" if method == "boom" else "tdmpc_square") + ".common.math")
        self.method = method
        self.heads = torch.zeros(2, device=agent.device, dtype=torch.long)
        self.parameters = [p for group in agent.optim.param_groups for p in group["params"]]
        self.graph = None

    def kernel(self):
        obs, action, _, _, reward, _ = self.batch
        cfg, model = self.cfg, self.model
        with torch.no_grad():
            next_z = model.encode(obs[1:], None)
            pi = model.pi(next_z, None)[1]
            out = model._target_Qs(torch.cat((next_z, pi), dim=-1)).index_select(0, self.heads)
            q1, q2 = (self.math.two_hot_inv(out[i], cfg) for i in (0, 1))
            targets = reward + self.agent.discount * torch.min(q1, q2)
        self.agent.optim.zero_grad(set_to_none=False)
        model.train()
        zs = torch.empty(cfg.horizon + 1, cfg.batch_size, cfg.latent_dim, device=self.agent.device)
        z = model.encode(obs[0], None)
        zs[0] = z
        consistency = 0
        for t in range(cfg.horizon):
            z = model.next(z, action[t], None)
            consistency += F.mse_loss(z, next_z[t]) * cfg.rho**t
            zs[t + 1] = z
        latent = zs[:-1]
        qs = model.Q(latent, action, None, return_type="all")
        rewards = model.reward(latent, action, None)
        reward_loss, value_loss = 0, 0
        for t in range(cfg.horizon):
            reward_loss += self.math.soft_ce(rewards[t], reward[t], cfg).mean() * cfg.rho**t
            for q in range(cfg.num_q):
                value_loss += self.math.soft_ce(qs[q][t], targets[t], cfg).mean() * cfg.rho**t
        consistency *= 1 / cfg.horizon
        reward_loss *= 1 / cfg.horizon
        value_loss *= 1 / (cfg.horizon * cfg.num_q)
        loss = cfg.consistency_coef * consistency + cfg.reward_coef * reward_loss + cfg.value_coef * value_loss
        loss.backward()
        return latent.detach(), torch.stack((consistency.detach(), reward_loss.detach(), value_loss.detach(), loss.detach()))

    def build(self, batch):
        self.batch = tuple(x.detach().clone() if x is not None else None for x in batch)
        assert self.batch[-1] is None and not self.cfg.multitask
        state = rng_state()
        grads = [None if p.grad is None else p.grad.clone() for p in self.parameters]
        for p in self.parameters:
            if p.grad is None: p.grad = torch.zeros_like(p)
        # Stable buffers are also used for restoration in the parity check.
        self.grad_buffers = [p.grad for p in self.parameters]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        start = time.perf_counter()
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.model.eval()
                self.outputs = self.kernel()
        torch.cuda.current_stream().wait_stream(stream)
        self.model.eval()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.outputs = self.kernel()
        torch.cuda.synchronize()
        for p, original in zip(self.parameters, grads):
            if original is None: p.grad.zero_()
            else: p.grad.copy_(original)
        self.model.eval()
        restore_rng(state)
        print("MODEL_GRAPH_CAPTURE_SECONDS", time.perf_counter() - start, flush=True)

    def __call__(self, batch, step):
        start = time.perf_counter()
        agent = self.agent
        if not isinstance(batch, tuple): batch = batch.sample()
        if self.graph is None: self.build(batch)
        for dst, src in zip(self.batch, batch):
            if dst is not None: dst.copy_(src)
        # Native target-Q selection is the first NumPy draw in an update.
        self.heads.copy_(torch.as_tensor(np.random.choice(self.cfg.num_q, 2, replace=False), device=agent.device))
        for p, grad in zip(self.parameters, self.grad_buffers): p.grad = grad
        self.graph.replay()
        latent, losses = self.outputs
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
        agent.optim.step()
        self.model.train()
        pi_loss, q_loss, prior_loss = agent.update_pi(latent, batch[1].detach(), batch[2].detach(),
                                                     batch[3].detach(), None, step)
        self.model.soft_update_target_Q()
        self.model.eval()
        consistency, reward, value, loss = losses.tolist()
        metrics = dict(consistency_loss=consistency, reward_loss=reward, value_loss=value,
                       pi_loss=pi_loss, pi_q_loss=q_loss, pi_fkl_loss=prior_loss,
                       total_loss=loss, grad_norm=float(norm), pi_scale=agent.scale.value)
        if self.method == "tdmpc_square":
            metrics.update(pi_loss_q=q_loss, pi_loss_prior=prior_loss)
        if agent.observe and agent.recorder is not None:
            agent.recorder.update(step, {**metrics, **agent.actor_metrics,
                "gate_active": float(agent.scale.value > 2),
                "weighted_alignment_loss": self.cfg.action_dim * agent.lamda * prior_loss,
                "update_seconds": time.perf_counter() - start})
        return metrics


def validate_model_update(agent, method, output):
    batch = (torch.randn(4, 256, 67, device=agent.device),
             torch.rand(3, 256, 21, device=agent.device) * 2 - 1,
             torch.rand(3, 256, 21, device=agent.device) * 2 - 1,
             torch.ones(3, 256, 21, device=agent.device),
             torch.rand(3, 256, 1, device=agent.device), None)
    native = agent.update
    native(batch, 0)
    captured = ModelUpdateGraph(agent, method)
    captured.build(batch)
    rows = []
    for scale in (1., 3.):
        model = deepcopy(agent.model.state_dict())
        grads = {n: None if p.grad is None else p.grad.clone() for n, p in agent.model.named_parameters()}
        optim, pi_optim = deepcopy(agent.optim.state_dict()), deepcopy(agent.pi_optim.state_dict())
        state = rng_state()
        outputs = []
        for fn in (native, native, captured):
            agent.model.load_state_dict(model)
            for name, p in agent.model.named_parameters():
                p.grad = None if grads[name] is None else grads[name].clone()
            agent.optim.load_state_dict(deepcopy(optim)); agent.pi_optim.load_state_dict(deepcopy(pi_optim))
            agent.scale._value.fill_(scale)
            restore_rng(state)
            metrics = fn(batch, 0)
            outputs.append((deepcopy(agent.model.state_dict()), deepcopy(agent.optim.state_dict()),
                            deepcopy(agent.pi_optim.state_dict()), metrics, rng_state(),
                            {n: None if p.grad is None else p.grad.clone() for n, p in agent.model.named_parameters()}))
        differences = []
        def equal(a, b):
            if isinstance(a, torch.Tensor):
                torch.testing.assert_close(a, b, rtol=2e-5, atol=1e-7)
                differences.append(float((a.double() - b.double()).abs().max()))
            elif isinstance(a, np.ndarray): assert np.array_equal(a, b)
            elif isinstance(a, dict):
                assert a.keys() == b.keys()
                for k in a: equal(a[k], b[k])
            elif isinstance(a, (tuple, list)):
                assert len(a) == len(b)
                for x, y in zip(a, b): equal(x, y)
            elif isinstance(a, float):
                assert abs(a-b) <= 1e-7 + 2e-5 * abs(a), (a, b)
                differences.append(abs(a-b))
            else: assert a == b, (a, b)
        equal(outputs[0], outputs[1])
        repeat = max(differences)
        differences.clear()
        equal(outputs[1], outputs[2])
        rows.append(dict(initial_scale=scale, numerical_parity=True, rng_exact=True,
                         native_repeat_max_absolute_difference=repeat,
                         captured_max_absolute_difference=max(differences)))
    (output / "model_parity.json").write_text(json.dumps({"checks": rows}, indent=2))
    print("MODEL_UPDATE_PARITY_PASSED", json.dumps(rows), flush=True)
