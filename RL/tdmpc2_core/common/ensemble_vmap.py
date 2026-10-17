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


def grouped_vmap_spec(ensembles, normed_linear_type):
    """Check whether independent ensembles can share one dense-head vmap.

    The grouping bypasses the individual ensemble wrappers, so their hooks and
    overridden forwards must be checked as well as the constituent heads.
    Selection is entirely structural and consumes no tensor RNG. Run this
    preflight eagerly and pass its result into compiled tensor regions: Dynamo
    does not preserve wrapper bound-method identity during inspection.
    """
    # Imported lazily because layers imports the ordinary vmap helpers above.
    from .layers import Ensemble

    if not ensembles:
        return None
    for ensemble in ensembles:
        if (type(ensemble) is not Ensemble or ensemble.backend != "vmap"
                or len(ensemble) == 0 or ensemble._forward_hooks
                or ensemble._forward_pre_hooks or ensemble._backward_hooks
                or ensemble._backward_pre_hooks
                or getattr(ensemble.forward, "__func__", None) is not Ensemble.forward
                or getattr(ensemble._forward_eager, "__func__", None) is not Ensemble._forward_eager
                or any(True for _ in ensemble.buffers())):
            return None
    modules = tuple(head for ensemble in ensembles for head in ensemble.modules_list)
    return ensemble_vmap_spec(modules, normed_linear_type)


def grouped_vectorized_forward(ensembles, inputs, specifications, *, dropout_masks=None):
    """Evaluate checked ensembles together while retaining their ownership.

    ``inputs`` supplies one tensor per ensemble. Identical tensor objects use a
    shared-input vmap; distinct inputs (including detached representations or
    separately sampled actions) are expanded over their own ensemble heads.
    Outputs retain one leading head axis per original ensemble.
    """
    if not ensembles or len(ensembles) != len(inputs):
        raise ValueError("Grouped critics require one input tensor per ensemble.")
    counts = tuple(len(ensemble.modules_list) for ensemble in ensembles)
    if any(count == 0 for count in counts):
        raise ValueError("Grouped critic ensembles must not be empty.")
    if any(value.shape != inputs[0].shape for value in inputs):
        raise ValueError("Grouped critic inputs must have the same shape.")
    modules = tuple(head for ensemble in ensembles for head in ensemble.modules_list)
    shared = all(value is inputs[0] for value in inputs)
    values = inputs[0] if shared else torch.cat([
        value.unsqueeze(0).expand(count, *value.shape)
        for value, count in zip(inputs, counts)
    ], dim=0)
    predictions = vectorized_forward(
        modules, values, specifications, head_inputs=not shared,
        dropout_masks=dropout_masks,
    )
    return predictions.split(counts, dim=0)


def vectorized_forward(
    modules, inputs, specifications, *, head_inputs=False, detached=False,
    dropout_masks=None,
):
    """Execute supported heads with shared inputs or independent leading heads.

    Dropout uses independent masks across heads. It has the same distribution
    as the module loop, but a different RNG order and floating-point reductions.
    Optional deterministic masks map active dropout layer indices to already
    scaled multiplicative tensors shaped [heads, ..., layer_output_features].
    They support arithmetic parity checks without consuming dropout randomness.
    """
    if dropout_masks is not None:
        active = {
            index for index, (_, _, _, norm) in enumerate(specifications)
            if norm is not None and norm[-2] and norm[-1]
        }
        if set(dropout_masks) != active:
            raise ValueError("Dropout masks must cover exactly the active dropout layers.")
        batch_shape = inputs.shape[1:-1] if head_inputs else inputs.shape[:-1]
        for index, mask in dropout_masks.items():
            expected = (len(modules), *batch_shape, specifications[index][1])
            if tuple(mask.shape) != expected:
                raise ValueError("Dropout masks require one full-shaped mask per critic head.")
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
    return _tensor_mlp_vmap(parameters, inputs, specifications, head_inputs, dropout_masks)


@torch._dynamo.allow_in_graph
def _tensor_mlp_vmap(parameters, inputs, specifications, head_inputs, dropout_masks=None):
    # TensorDict 0.3 monkeypatches vmap input processing with a Python context
    # manager Dynamo 2.3 cannot inline. AOTAutograd traces the tensor operations
    # behind this boundary. All tensor dependencies are explicit arguments;
    # there are no hidden module parameters, buffers, or tensor closures.
    def head_forward(params, x, masks):
        for index, (_, _, bias, norm) in enumerate(specifications):
            prefix = str(index) + "."
            x = F.linear(x, params[prefix + "weight"],
                         params[prefix + "bias"] if bias else None)
            if norm is not None:
                shape, eps, ln_weight, ln_bias, mish, dropout, training = norm
                if dropout and training:
                    x = (
                        F.dropout(x, p=dropout, training=True, inplace=False)
                        if masks is None else x * masks[index]
                    )
                x = F.layer_norm(
                    x, shape, params[prefix + "ln.weight"] if ln_weight else None,
                    params[prefix + "ln.bias"] if ln_bias else None, eps,
                )
                if mish:
                    x = F.mish(x, inplace=False)
        return x

    return torch.vmap(
        head_forward,
        in_dims=(0, 0 if head_inputs else None, 0 if dropout_masks is not None else None),
        randomness="different",
    )(parameters, inputs, dropout_masks)
