"""Vectorized dense critic heads with ordinary ModuleList parameter ownership.

Parameters are stacked differentiably on each call. They remain owned by the
original modules, so optimizers, Polyak updates and state-dict keys are unchanged.
The supported path is deliberately explicit: custom/LoRA layers, buffers,
hooks, overridden instance forwards, and heterogeneous dropout modes use the
caller's ordinary module loop instead of silently changing their behavior.
"""

import torch
import torch._dynamo
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import module as module_hooks


def _layer_spec(layer, normed_linear_type):
    if type(layer) not in (nn.Linear, normed_linear_type):
        return None
    base = (layer.in_features, layer.out_features, layer.bias is not None)
    if type(layer) is nn.Linear:
        return (*base, None)
    if type(layer.act) not in (nn.Mish, nn.Identity) or type(layer.ln) is not nn.LayerNorm:
        return None
    if layer.dropout is not None and type(layer.dropout) is not nn.Dropout:
        return None
    dropout = layer.dropout
    norm = (tuple(layer.ln.normalized_shape), layer.ln.eps,
            layer.ln.weight is not None, layer.ln.bias is not None,
            type(layer.act) is nn.Mish,
            0.0 if dropout is None else dropout.p,
            False if dropout is None else dropout.training)
    return (*base, norm)


def ensemble_vmap_spec(modules, normed_linear_type):
    """Return functional specifications, or None for the explicit loop path.

    This check runs before any tensor work or random draws. Exceptions from a
    supported vectorized forward propagate normally; there is no runtime retry
    that could consume extra randomness or hide numerical/backend failures.
    """
    if (len(modules) == 0 or module_hooks._global_forward_hooks
            or module_hooks._global_forward_pre_hooks or module_hooks._global_backward_hooks
            or module_hooks._global_backward_pre_hooks):
        return None
    specifications = None
    for head in modules:
        if type(head) is not nn.Sequential or len(head) == 0:
            return None
        if tuple(head._modules) != tuple(str(index) for index in range(len(head))):
            return None
        specs = tuple(_layer_spec(layer, normed_linear_type) for layer in head)
        if any(spec is None for spec in specs):
            return None
        if specifications is not None and specifications != specs:
            return None
        specifications = specs
        if any(True for _ in head.buffers()):
            return None
        for module in head.modules():
            if (module._forward_hooks or module._forward_pre_hooks or module._backward_hooks
                    or module._backward_pre_hooks
                    or getattr(module.forward, "__func__", None) is not type(module).forward):
                return None
    return specifications


def vectorized_forward(modules, inputs, specifications, *, head_inputs=False, detached=False):
    """Execute supported heads with shared inputs or independent leading heads.

    Dropout uses independent masks across heads. It has the same distribution
    as the module loop, but a different RNG order and floating-point reductions.
    """
    parameter_maps = tuple(dict(head.named_parameters(remove_duplicate=False)) for head in modules)
    # stack_module_state would create unrelated leaf Parameters. Stack these
    # existing tensors instead so gradients reach every original optimizer slot.
    parameters = {
        name: torch.stack([
            mapping[name].detach() if detached else mapping[name]
            for mapping in parameter_maps
        ], dim=0)
        for name in parameter_maps[0]
    }
    return _tensor_mlp_vmap(parameters, inputs, specifications, head_inputs)


@torch._dynamo.allow_in_graph
def _tensor_mlp_vmap(parameters, inputs, specifications, head_inputs):
    # TensorDict 0.3 monkeypatches vmap input processing with a Python context
    # manager Dynamo 2.3 cannot inline. AOTAutograd traces the tensor operations
    # behind this boundary. All tensor dependencies are explicit arguments;
    # there are no hidden module parameters, buffers, or tensor closures.
    def head_forward(params, x):
        for index, (_, _, bias, norm) in enumerate(specifications):
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
