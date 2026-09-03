from collections import OrderedDict
import warnings
import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from .compile_regions import _capture_rng_state, _restore_rng_state


_DETACHED_PARAMETER_VIEWS = weakref.WeakKeyDictionary()


def detached_module_forward(module, *args, **kwargs):
	"""Evaluate ``module`` with frozen parameters and live input gradients.

	The stateless parameter mapping prevents gradients from accumulating on the
	module while leaving buffers, training mode, and differentiation with respect
	to tensor inputs unchanged. Unlike the ensemble-specific fast path below,
	this general helper intentionally creates fresh detached views so it remains
	correct across arbitrary device and dtype moves.
	"""
	detached_parameters = {
		name: parameter.detach()
		for name, parameter in module.named_parameters()
	}
	return functional_call(module, detached_parameters, args, kwargs)


class Ensemble(nn.Module):
	"""
	Q-function ensemble.

	The official TD-MPC2 code vectorizes this with newer tensordict helpers.
	AMBI keeps the same ensemble semantics with a plain ModuleList so the
	vendored core works on older torch/tensordict stacks.
	"""

	def __init__(self, modules, **kwargs):
		super().__init__()
		self.modules_list = nn.ModuleList(modules)
		self._repr = str(modules[0])
		self._compile_enabled = False
		self._compile_strict = False
		self._compile_failed = False
		self._detached_compile_failed = False
		# Compiled wrappers close over bound methods and therefore over ``self``.
		# Keeping them in a global weak-key mapping would still root the key through
		# the mapping's strong value. Runtime-only instance attributes form an
		# ordinary collectable cycle and are reset across deepcopy/pickle.
		self._compiled_forward = None
		self._compiled_detached_forward = None

	def __getstate__(self):
		state = super().__getstate__()
		state["_compiled_forward"] = None
		state["_compiled_detached_forward"] = None
		return state

	def __setstate__(self, state):
		super().__setstate__(state)
		# Compiled callables are process- and object-identity-specific. Older
		# checkpoints also do not contain these non-persistent runtime fields.
		object.__setattr__(self, "_compiled_forward", None)
		object.__setattr__(self, "_compiled_detached_forward", None)
		for name, default in (
			("_compile_enabled", False),
			("_compile_strict", False),
			("_compile_failed", False),
			("_detached_compile_failed", False),
		):
			if not hasattr(self, name):
				object.__setattr__(self, name, default)

	def __len__(self):
		return len(self.modules_list)

	def __iter__(self):
		return iter(self.modules_list)

	def __getitem__(self, idx):
		return self.modules_list[idx]

	def _apply(self, fn, recurse=True):
		result = super()._apply(fn, recurse=recurse)
		# Module._apply may replace Parameter objects or their backing storage.
		# Recreate cached detached aliases before the next actor-gradient pass.
		for module in self.modules_list:
			_DETACHED_PARAMETER_VIEWS.pop(module, None)
		return result

	def _forward_eager(self, *args, **kwargs):
		return torch.stack([m(*args, **kwargs) for m in self.modules_list], dim=0)

	def enable_compile(self, *, strict=False):
		"""Compile the fixed ensemble forward while retaining state-dict keys."""
		strict = bool(strict)
		if strict != self._compile_strict:
			# Compiled wrappers encode the fullgraph policy used to create them.
			# A deliberate mode change starts a fresh compile attempt, including
			# after a sticky non-strict fallback.
			object.__setattr__(self, "_compiled_forward", None)
			object.__setattr__(self, "_compiled_detached_forward", None)
			self._compile_failed = False
			self._detached_compile_failed = False
		self._compile_strict = strict
		# A backend failure is sticky for this module. Re-enabling on every action
		# would retry the same unsupported graph, repeat warnings, and recompile.
		if self._compile_failed:
			self._compile_enabled = False
			return self
		self._compile_enabled = hasattr(torch, "compile")
		if not self._compile_enabled and self._compile_strict:
			raise RuntimeError("torch.compile is unavailable in this PyTorch build.")
		return self

	def disable_compile(self, *, reset_failure=False):
		"""Disable and release runtime compilation without changing parameters."""
		self._compile_enabled = False
		object.__setattr__(self, "_compiled_forward", None)
		object.__setattr__(self, "_compiled_detached_forward", None)
		if reset_failure:
			self._compile_failed = False
			self._detached_compile_failed = False
		return self

	@property
	def compile_failed(self):
		return self._compile_failed

	def _disable_compilation(self, exc, *, detached):
		# Compilation is sticky-disabled for this ensemble, so release both
		# process-local wrappers immediately.
		object.__setattr__(self, "_compiled_forward", None)
		object.__setattr__(self, "_compiled_detached_forward", None)
		self._compile_failed = True
		if detached:
			self._detached_compile_failed = True
		self._compile_enabled = False
		label = "detached critic ensemble" if detached else "critic ensemble"
		warnings.warn(
			f"Falling back to eager {label} after compile failure: {exc}",
			RuntimeWarning,
			stacklevel=3,
		)

	def forward(self, *args, **kwargs):
		# A larger compiled region owns this Python ModuleList loop when Dynamo is
		# already tracing. Starting a nested torch.compile here would graph-break
		# (or fail in strict mode) and create a second compilation cache.
		if not self._compile_enabled or (
			hasattr(torch, "compiler")
			and hasattr(torch.compiler, "is_compiling")
			and torch.compiler.is_compiling()
		):
			return self._forward_eager(*args, **kwargs)
		compiled = self._compiled_forward
		rng_snapshot = (
			_capture_rng_state(
				args,
				kwargs,
				include_host_globals=compiled is None,
			)
			if not self._compile_strict
			else None
		)
		if compiled is None:
			construction_rng_snapshot = (
				rng_snapshot
				if rng_snapshot is not None
				else _capture_rng_state(
					args, kwargs, include_host_globals=True
				)
			)
			try:
				compiled = torch.compile(
					self._forward_eager,
					fullgraph=self._compile_strict,
					dynamic=False,
				)
			except Exception as exc:
				_restore_rng_state(construction_rng_snapshot)
				if self._compile_strict:
					raise
				self._disable_compilation(exc, detached=False)
				return self._forward_eager(*args, **kwargs)
			# Lazy wrapper construction is process-local control work and may be
			# repeated after resume. Only the compiled call may consume RNG.
			_restore_rng_state(construction_rng_snapshot)
			object.__setattr__(self, "_compiled_forward", compiled)
		try:
			result = compiled(*args, **kwargs)
		except Exception as exc:
			if self._compile_strict:
				raise
			_restore_rng_state(rng_snapshot)
			self._disable_compilation(exc, detached=False)
			return self._forward_eager(*args, **kwargs)
		return result

	def _forward_detached_eager(self, *args, **kwargs):
		"""Evaluate Qs with frozen Q parameters while preserving input gradients."""
		outputs = []
		for module in self.modules_list:
			detached_parameters = _DETACHED_PARAMETER_VIEWS.get(module)
			if detached_parameters is None:
				detached_parameters = {
					name: parameter.detach()
					for name, parameter in module.named_parameters()
				}
				_DETACHED_PARAMETER_VIEWS[module] = detached_parameters
			outputs.append(
				functional_call(module, detached_parameters, args, kwargs)
			)
		return torch.stack(outputs, dim=0)

	def forward_detached(self, *args, **kwargs):
		if (
			not self._compile_enabled
			or self._detached_compile_failed
			or (
				hasattr(torch, "compiler")
				and hasattr(torch.compiler, "is_compiling")
				and torch.compiler.is_compiling()
			)
		):
			return self._forward_detached_eager(*args, **kwargs)
		compiled = self._compiled_detached_forward
		rng_snapshot = (
			_capture_rng_state(
				args,
				kwargs,
				include_host_globals=compiled is None,
			)
			if not self._compile_strict
			else None
		)
		if compiled is None:
			construction_rng_snapshot = (
				rng_snapshot
				if rng_snapshot is not None
				else _capture_rng_state(
					args, kwargs, include_host_globals=True
				)
			)
			try:
				compiled = torch.compile(
					self._forward_detached_eager,
					fullgraph=self._compile_strict,
					dynamic=False,
				)
			except Exception as exc:
				_restore_rng_state(construction_rng_snapshot)
				if self._compile_strict:
					raise
				self._disable_compilation(exc, detached=True)
				return self._forward_detached_eager(*args, **kwargs)
			_restore_rng_state(construction_rng_snapshot)
			object.__setattr__(self, "_compiled_detached_forward", compiled)
		try:
			result = compiled(*args, **kwargs)
		except Exception as exc:
			if self._compile_strict:
				raise
			_restore_rng_state(rng_snapshot)
			self._disable_compilation(exc, detached=True)
			return self._forward_detached_eager(*args, **kwargs)
		return result

	def __repr__(self):
		return f'{len(self)}x ' + self._repr


class ShiftAug(nn.Module):
	"""
	Random shift image augmentation.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	def __init__(self, pad=3):
		super().__init__()
		self.pad = pad
		self.padding = tuple([self.pad] * 4)

	def forward(self, x):
		x = x.float()
		n, _, h, w = x.size()
		assert h == w
		x = F.pad(x, self.padding, 'replicate')
		eps = 1.0 / (h + 2 * self.pad)
		arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
		arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
		base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
		base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
		shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
		shift *= 2.0 / (h + 2 * self.pad)
		grid = base_grid + shift
		return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)


class PixelPreprocess(nn.Module):
	"""
	Normalizes pixel observations to [-0.5, 0.5].
	"""

	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x.div(255.).sub(0.5)


class SimNorm(nn.Module):
	"""
	Simplicial normalization.
	Adapted from https://arxiv.org/abs/2204.00616.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.dim = cfg.simnorm_dim

	def forward(self, x):
		shp = x.shape
		x = x.view(*shp[:-1], -1, self.dim)
		x = F.softmax(x, dim=-1)
		return x.view(*shp)

	def __repr__(self):
		return f"SimNorm(dim={self.dim})"


class NormedLinear(nn.Linear):
	"""
	Linear layer with LayerNorm, activation, and optionally dropout.
	"""

	def __init__(self, *args, dropout=0., act=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.ln = nn.LayerNorm(self.out_features)
		if act is None:
			act = nn.Mish(inplace=False)
		self.act = act
		self.dropout = nn.Dropout(dropout, inplace=False) if dropout else None

	def forward(self, x):
		x = super().forward(x)
		if self.dropout:
			x = self.dropout(x)
		return self.act(self.ln(x))

	def __repr__(self):
		repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
		return f"NormedLinear(in_features={self.in_features}, "\
			f"out_features={self.out_features}, "\
			f"bias={self.bias is not None}{repr_dropout}, "\
			f"act={self.act.__class__.__name__})"


def mlp(in_dim, mlp_dims, out_dim, act=None, dropout=0.):
	"""
	Basic building block of TD-MPC2.
	MLP with LayerNorm, Mish activations, and optionally dropout.
	"""
	if isinstance(mlp_dims, int):
		mlp_dims = [mlp_dims]
	dims = [in_dim] + mlp_dims + [out_dim]
	mlp = nn.ModuleList()
	for i in range(len(dims) - 2):
		mlp.append(NormedLinear(dims[i], dims[i+1], dropout=dropout*(i==0)))
	mlp.append(NormedLinear(dims[-2], dims[-1], act=act) if act else nn.Linear(dims[-2], dims[-1]))
	return nn.Sequential(*mlp)


def conv(in_shape, num_channels, act=None):
	"""
	Basic convolutional encoder for TD-MPC2 with raw image observations.
	4 layers of convolution with ReLU activations, followed by a linear layer.
	"""
	assert in_shape[-1] == 64 # assumes rgb observations to be 64x64
	layers = [
		ShiftAug(), PixelPreprocess(),
		nn.Conv2d(in_shape[0], num_channels, 7, stride=2), nn.ReLU(inplace=False),
		nn.Conv2d(num_channels, num_channels, 5, stride=2), nn.ReLU(inplace=False),
		nn.Conv2d(num_channels, num_channels, 3, stride=2), nn.ReLU(inplace=False),
		nn.Conv2d(num_channels, num_channels, 3, stride=1), nn.Flatten()]
	if act:
		layers.append(act)
	return nn.Sequential(*layers)


def enc(cfg, out=None):
	"""
	Returns a dictionary of encoders for each observation in the dict.
	"""
	if out is None:
		out = {}
	for k in cfg.obs_shape.keys():
		if k == 'state':
			out[k] = mlp(cfg.obs_shape[k][0] + cfg.task_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim, act=SimNorm(cfg))
		elif k == 'rgb':
			out[k] = conv(cfg.obs_shape[k], cfg.num_channels, act=SimNorm(cfg))
		else:
			raise NotImplementedError(f"Encoder for observation type {k} not implemented.")
	return nn.ModuleDict(out)


def api_model_conversion(target_state_dict, source_state_dict):
    """Convert official vectorized TD-MPC2 critic keys to this port's modules.

    Current port checkpoints already use ``modules_list`` and pass through.
    Official checkpoints stack each critic parameter along dimension zero under
    ``_Qs.params`` / ``_target_Qs_params``; the compatibility ensemble stores
    one ordinary module per critic instead.
    """
    if any(key.startswith("_Qs.modules_list.") for key in source_state_dict):
        return source_state_dict
    if not any(key.startswith("_Qs.params.") for key in source_state_dict):
        return source_state_dict

    converted = OrderedDict()
    official_prefixes = ("_Qs.params.", "_detach_Qs_params.", "_target_Qs_params.")
    for key, value in source_state_dict.items():
        if not key.startswith(official_prefixes):
            converted[key] = value

    def unpack(source_prefix, target_prefix):
        found = False
        for key, value in source_state_dict.items():
            if not key.startswith(source_prefix):
                continue
            remainder = key[len(source_prefix):]
            if remainder.startswith("__"):
                continue
            if not torch.is_tensor(value) or "." not in remainder:
                continue
            layer, field = remainder.split(".", 1)
            for critic_index in range(value.shape[0]):
                target_key = f"{target_prefix}.modules_list.{critic_index}.{layer}.{field}"
                if target_key not in target_state_dict:
                    raise ValueError(
                        f"Official TD-MPC2 checkpoint critic key {key!r} cannot map to {target_key!r}."
                    )
                if target_state_dict[target_key].shape != value[critic_index].shape:
                    raise ValueError(
                        f"Checkpoint shape mismatch for {target_key}: "
                        f"expected {tuple(target_state_dict[target_key].shape)}, "
                        f"got {tuple(value[critic_index].shape)}."
                    )
                converted[target_key] = value[critic_index]
            found = True
        return found

    unpack("_Qs.params.", "_Qs")
    has_target = unpack("_target_Qs_params.", "_target_Qs")
    if not has_target:
        for key, value in list(converted.items()):
            if key.startswith("_Qs.modules_list."):
                converted[key.replace("_Qs.modules_list.", "_target_Qs.modules_list.", 1)] = value

    missing = [key for key in target_state_dict if key not in converted]
    unexpected = [key for key in converted if key not in target_state_dict]
    if missing or unexpected:
        raise ValueError(
            "Converted TD-MPC2 checkpoint does not match this model. "
            f"Missing keys: {missing[:8]}; unexpected keys: {unexpected[:8]}."
        )
    return converted
