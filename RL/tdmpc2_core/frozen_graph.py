"""Explicit mean-action CUDA graph execution for frozen checkpoint evaluation."""

import torch

from .inner_improvement import InnerImprovementEngine
from .whole_inner_graph import WholeInnerGraph, validate_training_graph, _gsde_inputs_for_action


PROTOCOL = "frozen_mean_cuda_graph_v1"


class MeanActionGraph(WholeInnerGraph):
    """Keep stochastic imagined learning and execute the final actor mean."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, eval_mode=True)


class FrozenGraphController:
    """Use a private captured workspace through the ordinary prediction API.

    Episode resets affect the native engine, never the captured allocations.
    Independent-seed evaluation resets the graph-private stream per episode;
    consecutive evaluation resets it once and advances it across all episodes.
    Native phase RNG and this graph protocol are distinct publication identities.
    """

    def __init__(self, model, observation, seed):
        self.model = model
        self.engine = model.agent.inner_engine
        validate_training_graph(self.engine, frozen_mean_evaluation=True)
        if model.cfg.inner_actor_loss_scale_scope != "action":
            raise ValueError("Frozen graph evaluation requires action-scoped Q normalization.")
        with torch.no_grad():
            obs = model._obs_to_tensor(observation).to(model.agent.device).unsqueeze(0)
            root = model.agent.model.encode(obs).detach()
        worker = InnerImprovementEngine(model.agent)
        worker._whole_graph_worker = True
        self.runner = MeanActionGraph(worker, root, seed=seed, warmup=4)
        self.original_act = self.engine.act
        self.engine.act = self.act

    def reset_episode(self, seed):
        self.runner.seed = int(seed)
        self.runner.reset_rng()

    def act(self, root, *, t0=False, eval_mode=False, collect_diagnostics=True,
            trace=None, apply_inner_writeback=False, return_behavior_policy=False,
            capture_teacher_policy=False, **kwargs):
        if (not eval_mode or collect_diagnostics or trace is not None
                or apply_inner_writeback or return_behavior_policy or capture_teacher_policy):
            raise ValueError("Frozen graph requires mean execution without traces, probes or writeback.")
        if self.engine._gsde_active:
            self.engine._begin_gsde_action(t0=t0, eval_mode=True)
            collection, execution = _gsde_inputs_for_action(self.engine, root, eval_mode=True)
            action, metrics, lengths = self.runner(
                root, gsde_collection=collection, gsde_execution=execution)
            self.engine._finish_gsde_action()
        else:
            action, metrics, lengths = self.runner(root)
        self.engine.action_index += 1
        return action, dict(metrics, inner_cuda_graph_used=1.0), lengths

    def close(self):
        self.engine.act = self.original_act
