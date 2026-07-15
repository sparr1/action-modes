from functools import lru_cache

import torch
import torch.nn.functional as F
try:
	from tensordict import TensorDict
except ImportError:  # tensordict<newer API compatibility
	from tensordict.tensordict import TensorDict


def soft_ce(pred, target, cfg):
	"""Computes the cross entropy loss between predictions and soft targets."""
	log_probabilities = F.log_softmax(pred, dim=-1)
	if cfg.num_bins <= 1:
		target = two_hot(target, cfg)
		return -(target * log_probabilities).sum(-1, keepdim=True)

	lower, upper, lower_weight, upper_weight = _two_hot_bin_weights(target, cfg)
	return -(
		lower_weight * log_probabilities.gather(-1, lower)
		+ upper_weight * log_probabilities.gather(-1, upper)
	)


def log_std(x, low, dif):
	return low + 0.5 * dif * (torch.tanh(x) + 1)


def gaussian_logprob(eps, log_std):
	"""Compute Gaussian log probability."""
	residual = -0.5 * eps.pow(2) - log_std
	log_prob = residual - 0.9189385175704956
	return log_prob.sum(-1, keepdim=True)


def squash(mu, pi, log_pi):
	"""Apply squashing function."""
	mu = torch.tanh(mu)
	pi = torch.tanh(pi)
	squashed_pi = torch.log(F.relu(1 - pi.pow(2)) + 1e-6)
	log_pi = log_pi - squashed_pi.sum(-1, keepdim=True)
	return mu, pi, log_pi


def int_to_one_hot(x, num_classes):
	"""
	Converts an integer tensor to a one-hot tensor.
	Supports batched inputs.
	"""
	one_hot = torch.zeros(*x.shape, num_classes, device=x.device)
	one_hot.scatter_(-1, x.unsqueeze(-1), 1)
	return one_hot


def symlog(x):
	"""
	Symmetric logarithmic function.
	Adapted from https://github.com/danijar/dreamerv3.
	"""
	return torch.sign(x) * torch.log(1 + torch.abs(x))


def symexp(x):
	"""
	Symmetric exponential function.
	Adapted from https://github.com/danijar/dreamerv3.
	"""
	return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def _two_hot_bin_weights(x, cfg):
	"""Return the two occupied discrete-regression bins and their weights."""
	x = torch.clamp(symlog(x), cfg.vmin, cfg.vmax)
	position = (x - cfg.vmin) / cfg.bin_size
	lower = torch.floor(position)
	upper_weight = position - lower
	lower = lower.long()
	upper = (lower + 1) % cfg.num_bins
	return lower, upper, 1 - upper_weight, upper_weight


@lru_cache(maxsize=64)
def _cached_categorical_support(vmin, vmax, num_bins, device, dtype):
	"""Build one immutable categorical support per value/device/dtype tuple."""
	return torch.linspace(vmin, vmax, num_bins, device=device, dtype=dtype)


def categorical_support(reference, cfg):
	"""Return the cached reward support outside compiled tensor regions."""
	if cfg.num_bins <= 1:
		return reference.new_empty(0)
	return _cached_categorical_support(
		float(cfg.vmin),
		float(cfg.vmax),
		int(cfg.num_bins),
		reference.device,
		reference.dtype,
	)


def two_hot(x, cfg):
	"""Converts a batch of scalars to soft two-hot encoded targets for discrete regression."""
	if cfg.num_bins == 0:
		return x
	elif cfg.num_bins == 1:
		return symlog(x)
	lower, upper, lower_weight, upper_weight = _two_hot_bin_weights(x, cfg)
	soft_two_hot = torch.zeros(
		*x.shape[:-1], cfg.num_bins, device=x.device, dtype=x.dtype
	)
	soft_two_hot.scatter_add_(-1, lower, lower_weight)
	soft_two_hot.scatter_add_(-1, upper, upper_weight)
	return soft_two_hot


def two_hot_inv(x, cfg, support=None):
	"""Converts a batch of soft two-hot encoded vectors to scalars."""
	if cfg.num_bins == 0:
		return x
	elif cfg.num_bins == 1:
		return symexp(x)
	dreg_bins = categorical_support(x, cfg) if support is None else support
	x = F.softmax(x, dim=-1)
	x = torch.sum(x * dreg_bins, dim=-1, keepdim=True)
	return symexp(x)


def gumbel_softmax_sample(p, temperature=1.0, dim=0):
	"""Sample from the Gumbel-Softmax distribution."""
	logits = p.log()
	gumbels = (
		-torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
	)  # ~Gumbel(0,1)
	gumbels = (logits + gumbels) / temperature  # ~Gumbel(logits,tau)
	y_soft = gumbels.softmax(dim)
	return y_soft.argmax(-1)


def termination_statistics(pred, target, eps=1e-9):
	"""Compute episode termination statistics."""
	pred = pred.squeeze(-1)
	target = target.squeeze(-1)
	rate = target.sum() / len(target)
	tp = ((pred > 0.5) & (target == 1)).sum()
	fn = ((pred <= 0.5) & (target == 1)).sum()
	fp = ((pred > 0.5) & (target == 0)).sum()
	recall = tp / (tp + fn + eps)
	precision = tp / (tp + fp + eps)
	f1 = 2 * (precision * recall) / (precision + recall + eps)
	return TensorDict({'termination_rate': rate,
			'termination_f1': f1})
