"""Opt-in CUDA capture of MPR/action-smoothness losses and critic gradients.

Sampling and encoding stay eager, using the existing stateless private RNG.
Capture evaluates the existing geometry and its parameter derivatives; it never
steps an optimizer or writes Parameter.grad. A first-order autograd bridge adds
those derivatives to the ordinary loss before the existing joint clipping and
Adam step. No approximation, subsampling change, or stale parameter cache is
introduced. Graph buffers are disposable and are rebuilt after checkpoint load.
"""

from copy import deepcopy

import torch
from torch.autograd.function import once_differentiable

from utils.pave_identity import pave_training_signature, resolve_pave_controls
from .pave import (
    _geometry_mode, _view_batch_spec, selected_pave_critics,
    prepare_outer_pave_inputs, prepared_pave_loss,
)


class _CapturedGradientLoss(torch.autograd.Function):
    """A scalar with snapshotted first derivatives, including arbitrary VJPs.

    Save parameters as well as gradients so PyTorch detects an optimizer step
    between this forward and backward. Independent snapshots allow multiple
    outstanding forwards, repeated backward, and retained metric dictionaries.
    Higher-order differentiation through this bridge is intentionally excluded;
    all derivatives needed to train the regularizer are computed inside capture.
    """

    @staticmethod
    def forward(ctx, value, packed, layout, *parameters):
        ctx.layout = layout
        ctx.save_for_backward(packed, *parameters)
        ctx.set_materialize_grads(False)
        return value

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        packed, *parameters = ctx.saved_tensors  # Also validates tensor versions.
        if grad_output is None:
            return (None,) * (3 + len(parameters))
        scaled = packed * grad_output
        offset, gradients = 0, []
        for parameter, used in zip(parameters, ctx.layout):
            if used:
                size = parameter.numel()
                gradients.append(scaled.narrow(0, offset, size).view_as(parameter))
                offset += size
            else:
                gradients.append(None)
        return (None, None, None, *gradients)


def captured_gradient_loss(value, gradients, parameters):
    """Snapshot a precomputed derivative without touching live .grad buffers."""
    layout = tuple(gradient is not None for gradient in gradients)
    pieces = [gradient.detach().reshape(-1) for gradient in gradients if gradient is not None]
    packed = torch.cat(pieces) if pieces else value.new_empty(0)
    return _CapturedGradientLoss.apply(value.detach().clone(), packed, layout, *parameters)


def _tensor_spec(value):
    return None if value is None else (value.shape, value.dtype, value.device)


def _parameter_spec(parameter):
    return (id(parameter), parameter.data_ptr(), parameter.shape, parameter.stride(),
            parameter.dtype, parameter.device, parameter.requires_grad)


class PaveGraph:
    """One fixed-shape, deterministic loss-and-gradient graph per outer learner.

    Only standard dense vmap critics and the standard scalar/distributional
    decoder are accepted. Unsupported configurations fail explicitly; capture
    errors never silently fall back to a different implementation.
    """

    def __init__(self, model, cfg):
        self.model, self.cfg = model, cfg
        controls = resolve_pave_controls(cfg)
        if not controls["outer_pave_cuda_graph"]:
            raise ValueError("PaveGraph requires outer_pave_cuda_graph=True.")
        self.signature = deepcopy(pave_training_signature(cfg))
        self.selected = selected_pave_critics(model, cfg)
        self.parameters = tuple(dict.fromkeys(
            parameter for _, critic, _ in self.selected for parameter in critic.parameters()
        ))
        if not self.parameters or any(not p.requires_grad for p in self.parameters):
            raise ValueError("PAVE graph requires trainable selected critics.")
        self.device, self.dtype = self.parameters[0].device, self.parameters[0].dtype
        if self.device.type != "cuda":
            raise ValueError("PAVE CUDA graph requires CUDA.")
        if self.dtype not in (torch.float32, torch.float64) or any(
            p.device != self.device or p.dtype != self.dtype for p in self.parameters
        ):
            raise ValueError("PAVE graph requires one CUDA device and a uniform float32/float64 dtype.")
        self.parameter_specs = tuple(_parameter_spec(p) for p in self.parameters)
        self.critic_ids = tuple(id(critic) for _, critic, _ in self.selected)
        self.decoder = model.q_backend
        self.decoder_signature = deepcopy((self.decoder.signature, self.decoder.codec))
        self.structures = self._structures()
        self.graph = None
        self.capture_parameters = None
        self.capture_selected = None
        self.static_inputs = None
        self.outputs = None
        self.calls = 0

    def _structures(self):
        structures = []
        for _, critic, _ in self.selected:
            with _geometry_mode(critic):
                spec = _view_batch_spec(critic, self.model.q_backend.decode)
            if spec is None:
                raise ValueError("PAVE graph requires standard dense vmap critics and decoder, without hooks.")
            structures.append(spec)
        return tuple(structures)

    def validate(self):
        """Reject changed static assumptions before sampling or live updates."""
        if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
            raise ValueError("PAVE graph requires ordinary gradient-enabled training.")
        if torch.is_autocast_enabled():
            raise ValueError("PAVE graph does not support autocast.")
        if pave_training_signature(self.cfg) != self.signature:
            raise ValueError("PAVE graph configuration changed; create a fresh graph/lineage.")
        selected = selected_pave_critics(self.model, self.cfg)
        parameters = tuple(dict.fromkeys(
            p for _, critic, _ in selected for p in critic.parameters()
        ))
        if (tuple(id(critic) for _, critic, _ in selected) != self.critic_ids
                or tuple(_parameter_spec(p) for p in parameters) != self.parameter_specs):
            raise ValueError("PAVE graph critic parameters or storage changed; rebuild the graph.")
        if any(p._backward_hooks or getattr(p, "_post_accumulate_grad_hooks", None)
               for p in parameters):
            raise ValueError("PAVE graph does not support parameter gradient hooks.")
        if (self.model.q_backend is not self.decoder
                or (self.decoder.signature, self.decoder.codec) != self.decoder_signature
                or self._structures() != self.structures):
            raise ValueError("PAVE graph critic structure or decoder changed.")

    def _loss_and_gradients(self):
        loss, metrics = prepared_pave_loss(self.model, self.static_inputs, self.cfg, self.capture_selected)
        gradients = torch.autograd.grad(loss, self.capture_parameters, allow_unused=True)
        layout = tuple(gradient is not None for gradient in gradients)
        pieces = [gradient.reshape(-1) for gradient in gradients if gradient is not None]
        packed = torch.cat(pieces) if pieces else loss.new_empty(0)
        return loss, metrics, packed, layout

    def _capture(self, inputs):
        self.static_inputs = {
            key: value.detach().clone(memory_format=torch.contiguous_format) if value is not None else None
            for key, value in inputs.items()
        }
        current = torch.cuda.current_stream(self.device)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(current)
        # No optimizer mutation, .grad accumulation, or random sampling occurs
        # in warmup/capture. Existing ordinary-loss autograd graphs remain valid.
        with torch.cuda.stream(stream):
            # Live leaves may already have AccumulateGrad nodes associated with
            # the default stream (including the pending ordinary TD backward).
            # Give capture independent autograd leaves, sharing the SAME weight
            # storage. Initialize their gradient tape on this capture stream.
            # deepcopy's memo substitutes aliases without copying any weights;
            # no live modules, optimizer slots, or .grad buffers are replaced.
            memo = {id(p): torch.nn.Parameter(p.detach(), requires_grad=True)
                    for p in self.parameters}
            self.capture_parameters = tuple(memo[id(p)] for p in self.parameters)
            self.capture_selected = [
                (prefix, deepcopy(critic, memo), coefficients)
                for prefix, critic, coefficients in self.selected
            ]
            for _ in range(3):
                self._loss_and_gradients()
        current.wait_stream(stream)
        torch.cuda.synchronize(self.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs = self._loss_and_gradients()
        current.wait_stream(stream)
        self.outputs, self.graph = outputs, graph

    def __call__(self, obs, action, update_number):
        self.validate()
        inputs = prepare_outer_pave_inputs(
            self.model, obs, action, self.cfg, update_number, self.selected,
        )
        if any(value is not None and (value.device != self.device or value.dtype != self.dtype)
               for value in inputs.values()):
            raise ValueError("PAVE graph inputs must match the critic device and dtype.")
        if self.graph is None:
            self._capture(inputs)
        else:
            if {key: _tensor_spec(value) for key, value in inputs.items()} != {
                key: _tensor_spec(value) for key, value in self.static_inputs.items()
            }:
                raise ValueError("PAVE graph input shapes/dtypes/devices changed.")
            for key, value in inputs.items():
                if value is not None:
                    self.static_inputs[key].copy_(value)
        self.graph.replay()
        self.calls += 1
        loss, metrics, packed, layout = self.outputs
        # A single packed clone owns all gradients for this forward. Replaying
        # another batch cannot overwrite a pending or retained autograd result.
        result = _CapturedGradientLoss.apply(
            loss.detach().clone(), packed.detach().clone(), layout, *self.parameters,
        )
        info = {key: value.detach().clone() for key, value in metrics.items()}
        info["outer_pave_cuda_graph_used"] = loss.new_ones(())
        return result, info
