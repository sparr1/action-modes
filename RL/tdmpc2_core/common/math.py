from functools import lru_cache
from math import isfinite

import torch
import torch.nn.functional as F
try:
	from tensordict import TensorDict
except ImportError:  # tensordict<newer API compatibility
	from tensordict.tensordict import TensorDict


TEMPORAL_LOSS_NORMALIZATIONS = {
	"divide_horizon",
	"reference_weighted_mean",
}


def temporal_loss_uses_legacy_order(
	normalization,
	horizon,
	reference_horizon,
):
	"""Whether a reducer must execute the historical arithmetic order.

	The compatibility mode always does so. The new normalization also takes this
	path at its reference horizon, which preserves the old value *and* floating-
	point/gradient operation order exactly at the anchor.
	"""
	normalization = str(normalization).lower()
	if normalization not in TEMPORAL_LOSS_NORMALIZATIONS:
		raise ValueError(
			"temporal loss normalization must be one of "
			f"{sorted(TEMPORAL_LOSS_NORMALIZATIONS)}, got {normalization!r}."
		)
	return normalization == "divide_horizon" or int(horizon) == int(
		reference_horizon
	)


def temporal_loss_weights(
	horizon,
	rho,
	*,
	normalization="reference_weighted_mean",
	reference_horizon=3,
	include_terminal=False,
	device=None,
	dtype=torch.float32,
):
	"""Return normalized geometric weights for one temporal objective.

	Transition objectives contain ``horizon`` terms. Actor objectives contain the
	additional terminal latent and therefore set ``include_terminal=True``. The
	reference-weighted mode preserves the aggregate weight of the configured
	reference horizon while redistributing it across the requested depth.

	The implementation sums the finite geometric sequence directly instead of
	using ``(1-rho**n)/(1-rho)``, so ``rho=1`` is well-defined.
	"""
	horizon = int(horizon)
	reference_horizon = int(reference_horizon)
	if horizon <= 0:
		raise ValueError("horizon must be positive.")
	if reference_horizon <= 0:
		raise ValueError("reference_horizon must be positive.")
	rho = float(rho)
	if not isfinite(rho):
		raise ValueError(f"rho must be finite, got {rho}.")
	normalization = str(normalization).lower()
	if normalization not in TEMPORAL_LOSS_NORMALIZATIONS:
		raise ValueError(
			"temporal loss normalization must be one of "
			f"{sorted(TEMPORAL_LOSS_NORMALIZATIONS)}, got {normalization!r}."
		)

	term_offset = int(bool(include_terminal))
	num_terms = horizon + term_offset
	reference_terms = reference_horizon + term_offset
	base = torch.as_tensor(rho, device=device, dtype=dtype)
	weights = torch.pow(
		base,
		torch.arange(num_terms, device=device, dtype=dtype),
	)
	if normalization == "divide_horizon" or horizon == reference_horizon:
		# The reference branch is deliberately identical to the legacy coefficient
		# construction at H=H0.
		return weights / num_terms

	reference_weights = torch.pow(
		base,
		torch.arange(reference_terms, device=device, dtype=dtype),
	)
	requested_sum = weights.sum()
	if not bool(torch.isfinite(requested_sum)) or float(requested_sum) == 0.0:
		raise ValueError(
			"Temporal weights must have a finite, non-zero sum; "
			f"got rho={rho}, horizon={horizon}."
		)
	reference_mean_weight = reference_weights.sum() / reference_terms
	return weights * (reference_mean_weight / requested_sum)


def reduce_temporal_loss(
	per_time_losses,
	rho,
	*,
	normalization="reference_weighted_mean",
	reference_horizon=3,
	include_terminal=False,
	legacy_order="sequential",
	weights=None,
):
	"""Reduce scalar losses over time with exact anchor compatibility.

	``legacy_order`` records the historical expression used by the caller:

	* ``sequential``: accumulate ``loss[t] * rho**t`` and divide afterwards.
	* ``vector_mean``: ``(losses * rho_weights).mean()`` (outer actors).
	* ``vector_sum_divide``: vectorized weighted sum followed by division.

	At the reference horizon, and for ``divide_horizon`` at every horizon, these
	paths intentionally preserve the former floating-point operation order. At
	other horizons the supplied normalized weights implement the fixed aggregate
	temporal weight.
	"""
	if legacy_order not in {"sequential", "vector_mean", "vector_sum_divide"}:
		raise ValueError(f"Unsupported legacy temporal reduction {legacy_order!r}.")
	values = per_time_losses if torch.is_tensor(per_time_losses) else None
	terms = (
		tuple(per_time_losses.unbind(0))
		if values is not None
		else tuple(per_time_losses)
	)
	if not terms:
		raise ValueError("per_time_losses must contain at least one temporal term.")
	if any(term.ndim != 0 for term in terms):
		raise ValueError("Each temporal loss must already be reduced to a scalar.")
	num_terms = len(terms)
	horizon = num_terms - int(bool(include_terminal))
	if horizon <= 0:
		raise ValueError("Resolved temporal horizon must be positive.")

	if temporal_loss_uses_legacy_order(
		normalization,
		horizon,
		reference_horizon,
	):
		if legacy_order == "sequential":
			total = 0
			for index, loss in enumerate(terms):
				total = total + loss * float(rho) ** index
			return total / num_terms
		if values is None:
			values = torch.stack(terms, dim=0)
		raw_weights = torch.pow(
			float(rho),
			torch.arange(num_terms, device=values.device),
		).to(dtype=values.dtype)
		weighted = values * raw_weights
		if legacy_order == "vector_mean":
			return weighted.mean()
		return weighted.sum() / num_terms

	if values is None:
		values = torch.stack(terms, dim=0)
	if weights is None:
		weights = temporal_loss_weights(
			horizon,
			rho,
			normalization=normalization,
			reference_horizon=reference_horizon,
			include_terminal=include_terminal,
			device=values.device,
			dtype=values.dtype,
		)
	weights = weights[:num_terms].to(device=values.device, dtype=values.dtype)
	return (values * weights).sum()


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


def diagonal_gaussian_reverse_kl(
	current_mean,
	current_log_std,
	behavior_mean,
	behavior_log_std,
	*,
	sum_action_dim=True,
):
	"""Return ``KL(current || behavior)`` for diagonal Gaussians.

	The behavior distribution is a fixed target: its arguments are detached so
	gradients flow only through the current distribution. Inputs may broadcast,
	which permits evaluating one current policy against multiple behavior-policy
	components. By default the per-dimension KL is summed over the final action
	dimension while retaining that dimension as a singleton.

	The standard-deviation term is evaluated from log-standard-deviation
	differences. This avoids forming tiny variances or dividing by a clamped
	behavior variance when log standard deviations span the policy bounds.
	"""
	if not isinstance(sum_action_dim, bool):
		raise TypeError("sum_action_dim must be bool.")
	behavior_mean = behavior_mean.detach()
	behavior_log_std = behavior_log_std.detach()
	log_std_ratio = current_log_std - behavior_log_std
	standardized_mean_delta = (
		current_mean - behavior_mean
	) * torch.exp(-behavior_log_std)
	elementwise_kl = (
		0.5 * torch.expm1(2.0 * log_std_ratio)
		- log_std_ratio
		+ 0.5 * standardized_mean_delta.square()
	)
	if sum_action_dim:
		return elementwise_kl.sum(dim=-1, keepdim=True)
	return elementwise_kl


def diagonal_gaussian_cross_entropy(
	current_mean,
	current_log_std,
	behavior_mean,
	behavior_log_std,
	*,
	sum_action_dim=True,
):
	"""Return Gaussian ``-E_current[log behavior]`` in pre-tanh space.

	The behavior distribution is a fixed target, so its arguments are detached.
	Inputs may broadcast, and the final action dimension is summed with a retained
	singleton by default. This is the analytic Gaussian component of squashed-
	action cross entropy; add :func:`tanh_log_abs_det_jacobian` evaluated on a
	current-policy sample to obtain an unbiased action-space estimator.
	"""
	if not isinstance(sum_action_dim, bool):
		raise TypeError("sum_action_dim must be bool.")
	behavior_mean = behavior_mean.detach()
	behavior_log_std = behavior_log_std.detach()
	log_std_ratio = current_log_std - behavior_log_std
	standardized_mean_delta = (
		current_mean - behavior_mean
	) * torch.exp(-behavior_log_std)
	elementwise_cross_entropy = 0.5 * (
		1.8378770664093453
		+ 2.0 * behavior_log_std
		+ torch.exp(2.0 * log_std_ratio)
		+ standardized_mean_delta.square()
	)
	if sum_action_dim:
		return elementwise_cross_entropy.sum(dim=-1, keepdim=True)
	return elementwise_cross_entropy


def tanh_log_abs_det_jacobian(pre_tanh_action, *, sum_action_dim=True):
	"""Return the exact, stable ``log|det J_tanh|``.

	The softplus identity remains finite when ``tanh(pre_tanh_action)`` has
	already rounded to ``-1`` or ``1``. By default independent action-coordinate
	terms are summed while retaining a singleton final dimension.
	"""
	if not isinstance(sum_action_dim, bool):
		raise TypeError("sum_action_dim must be bool.")
	elementwise_log_abs_det = 2.0 * (
		0.6931471805599453
		- pre_tanh_action
		- F.softplus(-2.0 * pre_tanh_action)
	)
	if sum_action_dim:
		return elementwise_log_abs_det.sum(dim=-1, keepdim=True)
	return elementwise_log_abs_det


def tanh_saturation_statistics(pre_tanh_action, action):
	"""Return detached scalar diagnostics for tanh action saturation.

	The third statistic marks pre-tanh coordinates for which the historical
	``+1e-6`` Jacobian floor dominated the true derivative. The fourth records
	coordinates whose already-squashed action rounded exactly to either bound.
	"""
	pre_tanh_abs = pre_tanh_action.detach().abs()
	action_abs = action.detach().abs()
	return (
		pre_tanh_abs.mean(),
		pre_tanh_abs.amax(),
		(pre_tanh_abs >= 7.600902).to(dtype=pre_tanh_abs.dtype).mean(),
		(action_abs == 1.0).to(dtype=action_abs.dtype).mean(),
	)


def squash(mu, pi, log_pi):
	"""Tanh-squash a mean/sample and apply the exact density correction.

	``pi`` is the pre-tanh sample, so its stable Jacobian must be evaluated
	before the returned action is transformed.
	"""
	log_pi = log_pi - tanh_log_abs_det_jacobian(pi)
	return torch.tanh(mu), torch.tanh(pi), log_pi


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
