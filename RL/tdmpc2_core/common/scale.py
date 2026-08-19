import torch
from .device import resolve_device


def linear_percentiles(x, percentiles):
	"""Return linearly interpolated percentiles along the leading dimension.

	This is the tensor-only portion of TD-MPC2's running scale estimator.  It is
	kept separate from :class:`RunningScale` so algorithms that need the same
	statistics can own and checkpoint their buffers without inheriting the
	upstream class's custom ``state_dict`` contract.
	"""
	if not torch.is_tensor(x) or x.ndim == 0:
		raise ValueError("x must be a tensor with a non-empty leading dimension.")
	if x.shape[0] == 0:
		raise ValueError("x must have a non-empty leading dimension.")
	if not torch.is_tensor(percentiles) or percentiles.ndim != 1:
		raise ValueError("percentiles must be a one-dimensional tensor.")

	x_dtype, x_shape = x.dtype, x.shape
	flat = x.reshape(x.shape[0], -1)
	sorted_values = torch.sort(flat, dim=0).values
	positions = percentiles.to(device=x.device) * (x.shape[0] - 1) / 100
	floored = torch.floor(positions)
	ceiled = torch.clamp(floored + 1, max=x.shape[0] - 1)
	weight_ceiled = positions - floored
	weight_floored = 1.0 - weight_ceiled
	d0 = sorted_values[floored.long()] * weight_floored.unsqueeze(1)
	d1 = sorted_values[ceiled.long()] * weight_ceiled.unsqueeze(1)
	return (d0 + d1).reshape(-1, *x_shape[1:]).to(x_dtype)


def percentile_range(x, percentiles, *, minimum=1.0):
	"""Return a detached-friendly robust range using two percentiles."""
	if not torch.is_tensor(percentiles) or percentiles.numel() != 2:
		raise ValueError("percentile_range requires exactly two percentiles.")
	values = linear_percentiles(x, percentiles)
	return torch.clamp(values[1] - values[0], min=minimum)


class RunningScale(torch.nn.Module):
	"""Running trimmed scale estimator."""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		device = resolve_device(getattr(cfg, 'device', None), warn=False)
		self.register_buffer('value', torch.ones(1, dtype=torch.float32, device=device))
		self.register_buffer('_percentiles', torch.tensor([5, 95], dtype=torch.float32, device=device))

	def state_dict(self):
		return dict(value=self.value, percentiles=self._percentiles)

	def load_state_dict(self, state_dict):
		self.value.copy_(state_dict['value'])
		self._percentiles.copy_(state_dict['percentiles'])

	def _positions(self, x_shape):
		positions = self._percentiles * (x_shape-1) / 100
		floored = torch.floor(positions)
		ceiled = floored + 1
		ceiled = torch.where(ceiled > x_shape - 1, x_shape - 1, ceiled)
		weight_ceiled = positions-floored
		weight_floored = 1.0 - weight_ceiled
		return floored.long(), ceiled.long(), weight_floored.unsqueeze(1), weight_ceiled.unsqueeze(1)

	def _percentile(self, x):
		return linear_percentiles(x, self._percentiles)

	def update(self, x):
		value = percentile_range(x.detach(), self._percentiles, minimum=1.)
		self.value.data.lerp_(value, self.cfg.tau)

	def forward(self, x, update=False):
		if update:
			self.update(x)
		return x / self.value

	def __repr__(self):
		return f'RunningScale(S: {self.value})'
