import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


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

	def __len__(self):
		return len(self.modules_list)

	def __iter__(self):
		return iter(self.modules_list)

	def __getitem__(self, idx):
		return self.modules_list[idx]

	def forward(self, *args, **kwargs):
		return torch.stack([m(*args, **kwargs) for m in self.modules_list], dim=0)

	def forward_detached(self, *args, **kwargs):
		"""Evaluate Qs with frozen Q parameters while preserving input gradients."""
		requires_grad = [p.requires_grad for p in self.parameters()]
		for p in self.parameters():
			p.requires_grad_(False)
		try:
			return self.forward(*args, **kwargs)
		finally:
			for p, req_grad in zip(self.parameters(), requires_grad):
				p.requires_grad_(req_grad)

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
