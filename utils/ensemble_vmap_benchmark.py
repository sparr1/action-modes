"""Temporary head-vectorization backend for controlled ensemble benchmarks.

This does not change production defaults or checkpoint/optimizer ownership.
Unlike upstream's persistent stacked parameters, each call differentiably
stacks the existing Parameters. Its allocation and stacking cost is included in
timings; results are not a bound on a future persistent-stack implementation.
Create/compile/use/discard models within the context: compiled callables made
there must not escape it. The process-wide patch is not thread-safe.
"""

from contextlib import contextmanager

import torch
import torch._dynamo
from torch import nn
from torch.nn import functional as F

from RL.tdmpc2_core import pave
from RL.tdmpc2_core.common.layers import Ensemble, NormedLinear


def _layer_spec(layer):
    if type(layer) not in (nn.Linear, NormedLinear):
        raise TypeError("vmap benchmark supports only Linear/NormedLinear critic layers")
    base = (layer.in_features, layer.out_features, layer.bias is not None)
    if type(layer) is nn.Linear:
        return (*base, None)
    if type(layer.act) not in (nn.Mish, nn.Identity):
        raise TypeError("vmap benchmark supports only Mish/Identity critic activations")
    if type(layer.ln) is not nn.LayerNorm:
        raise TypeError("vmap benchmark requires standard LayerNorm")
    if layer.dropout is not None and type(layer.dropout) is not nn.Dropout:
        raise TypeError("vmap benchmark requires standard Dropout")
    dropout = layer.dropout
    norm = (tuple(layer.ln.normalized_shape), layer.ln.eps,
            layer.ln.weight is not None, layer.ln.bias is not None,
            type(layer.act) is nn.Mish,
            0.0 if dropout is None else dropout.p,
            False if dropout is None else dropout.training)
    return (*base, norm)


def _specifications(ensemble):
    if not isinstance(ensemble, Ensemble) or len(ensemble.modules_list) == 0:
        raise TypeError("vmap benchmark requires a nonempty compatibility Ensemble")
    if any(type(head) is not nn.Sequential for head in ensemble.modules_list):
        raise TypeError("vmap benchmark requires Sequential critic heads")
    specs = tuple(_layer_spec(layer) for layer in ensemble.modules_list[0])
    for head in ensemble.modules_list:
        if tuple(_layer_spec(layer) for layer in head) != specs:
            raise ValueError("vmap benchmark requires homogeneous heads and dropout modes")
        if any(True for _ in head.buffers()):
            raise TypeError("vmap benchmark does not support stateful critic buffers")
    return specs


def vectorized_forward(ensemble, inputs, *, head_inputs=False, detached=False):
    """Evaluate homogeneous heads with shared or independent head inputs.

    ``head_inputs=True`` takes [heads, ..., features], preserving PAVE's
    independent action coordinates. Otherwise all leading input axes are shared
    by the heads (including time and batch). Dropout masks are independent over
    heads via ``randomness='different'``; their RNG order differs from the loop.
    """
    specs = _specifications(ensemble)
    if head_inputs and inputs.shape[0] != len(ensemble.modules_list):
        raise ValueError("Independent inputs require one leading row per critic head")
    parameter_maps = tuple(dict(head.named_parameters()) for head in ensemble.modules_list)
    # Never use stack_module_state: it creates unrelated leaf Parameters.
    parameters = {
        name: torch.stack([
            mapping[name].detach() if detached else mapping[name]
            for mapping in parameter_maps
        ], dim=0)
        for name in parameter_maps[0]
    }

    return _tensor_mlp_vmap(parameters, inputs, specs, head_inputs)


@torch._dynamo.allow_in_graph
def _tensor_mlp_vmap(parameters, inputs, specs, head_inputs):
    # TensorDict 0.3 monkeypatches vmap's input helper with a context manager
    # that Dynamo 2.3 cannot inline. This pure-tensor boundary skips that Python
    # plumbing in Dynamo; AOTAutograd still traces the actual torch operations.
    # Every tensor dependency is an explicit argument, never a closure/module.
    def head_forward(params, x):
        for index, (_, _, bias, norm) in enumerate(specs):
            prefix = str(index) + "."
            x = F.linear(x, params[prefix + "weight"],
                         params[prefix + "bias"] if bias else None)
            if norm is not None:
                shape, eps, ln_weight, ln_bias, mish, dropout, training = norm
                if dropout:
                    x = F.dropout(x, p=dropout, training=training, inplace=False)
                x = F.layer_norm(
                    x, shape, params[prefix + "ln.weight"] if ln_weight else None,
                    params[prefix + "ln.bias"] if ln_bias else None, eps,
                )
                if mish:
                    x = F.mish(x, inplace=False)
        return x

    return torch.vmap(
        head_forward, in_dims=(0, 0 if head_inputs else None), randomness="different",
    )(parameters, inputs)


def _forward_eager(self, inputs):
    return vectorized_forward(self, inputs)


def _forward_detached_eager(self, inputs):
    return vectorized_forward(self, inputs, detached=True)


def _head_predictions(critic, latents, actions):
    expanded = latents.unsqueeze(0).expand(actions.shape[0], *latents.shape)
    return vectorized_forward(critic, torch.cat((expanded, actions), dim=-1), head_inputs=True)


@contextmanager
def vectorized_ensemble_benchmark():
    """Temporarily replace loop forwards; production implementation stays default.

    Enter before compiling any relevant models, and discard those models before
    leaving. All class methods and the eager PAVE seam are restored on failure.
    Existing parameter objects, state-dict keys and optimizer slots stay intact.
    """
    original_forward = Ensemble._forward_eager
    original_detached = Ensemble._forward_detached_eager
    original_geometry = pave._head_predictions
    Ensemble._forward_eager = _forward_eager
    Ensemble._forward_detached_eager = _forward_detached_eager
    pave._head_predictions = _head_predictions
    try:
        yield {
            "implementation": "torch_vmap_differentiable_parameter_stack_v1",
            "parameter_ownership": "original_modulelist_parameters",
            "parameter_stacking": "each_forward_included_in_timing",
            "ordinary_forward": "vmap_shared_inputs",
            "detached_actor_forward": "vmap_shared_inputs_detached_parameters",
            "pave_forward": "eager_vmap_independent_head_actions_separate_views",
            "dropout_randomness": "different_per_head_distribution_equivalent_rng_order_differs",
            "persistent_stacked_parameters": False,
            "view_fusion": False,
            "compile_boundary": "pure_tensor_vmap_allow_in_graph_aot_traced",
            "context_seams_installed": (
                Ensemble._forward_eager is _forward_eager
                and Ensemble._forward_detached_eager is _forward_detached_eager
                and pave._head_predictions is _head_predictions
            ),
        }
    finally:
        Ensemble._forward_eager = original_forward
        Ensemble._forward_detached_eager = original_detached
        pave._head_predictions = original_geometry
