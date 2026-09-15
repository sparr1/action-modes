"""Named critic components and their meaning, independent of numeric encoding.

Networks emit a packed component axis; codecs only regress numbers. This module
defines what those numbers mean and how to compose them for policy evaluation.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ValueSpecification:
    mode: str
    components: tuple[str, ...]

    @classmethod
    def from_config(cls, cfg):
        mode = str(getattr(cfg, "critic_value_mode", "single")).lower()
        if mode == "single":
            return cls(mode, ("value",))
        if mode == "return_entropy":
            return cls(mode, ("return", "entropy"))
        raise ValueError("critic_value_mode must be 'single' or 'return_entropy'.")

    @property
    def is_split(self):
        return self.mode != "single"

    @property
    def num_components(self):
        return len(self.components)

    def signature(self):
        """Ordered semantic roles for resolved configurations and checkpoints."""
        return {
            "critic_value_mode": self.mode,
            "value_components": list(self.components),
        }

    def component(self, values, name):
        """Extract one decoded component without removing its scalar axis."""
        if not self.is_split:
            if name != "value":
                raise ValueError("A single-value critic has no named return component.")
            return values
        return values[..., self.components.index(name), :]

    def project(self, values, *, projection=None, beta=None, weights=None):
        """Compose decoded components; explicit weights also support inner roles."""
        if not self.is_split:
            if weights is not None or projection not in {None, "value"}:
                raise ValueError("Named value projections require a split critic.")
            return values
        if values.shape[-2:] != (self.num_components, 1):
            raise ValueError(
                "Decoded component values must end in "
                f"({self.num_components}, 1), got {tuple(values.shape)}."
            )
        if weights is not None:
            if projection is not None or beta is not None:
                raise ValueError("Specify component weights or a named projection, not both.")
            if torch.is_tensor(weights):
                weights = weights.to(device=values.device, dtype=values.dtype)
            else:
                weights = torch.stack([
                    torch.as_tensor(weight, device=values.device, dtype=values.dtype).reshape(())
                    for weight in weights
                ])
            if weights.shape != (self.num_components,):
                raise ValueError(f"Expected {self.num_components} composition weights.")
            return (values * weights[:, None]).sum(dim=-2)
        if projection == "return":
            if beta is not None:
                raise ValueError("The return projection does not use beta.")
            return self.component(values, "return")
        if projection == "policy":
            if beta is None:
                raise ValueError("The policy projection requires an explicit beta.")
            return self.component(values, "return") + beta * self.component(values, "entropy")
        raise ValueError("Split critics require projection='return'/'policy' or explicit weights.")

    @torch.no_grad()
    def outer_targets(self, reward, discount, terminated, next_components, next_entropy):
        """Return unweighted reward and future-entropy Bellman targets."""
        if self.mode != "return_entropy":
            raise ValueError("Component Bellman targets require return_entropy semantics.")
        continuation = discount * (1 - terminated)
        # A terminal transition has no continuation, even when evaluating its
        # unused successor produces NaN/Inf. Multiplication by zero alone does
        # not suppress those values.
        terminal = terminated == 1
        next_return = torch.where(
            terminal, 0.0, self.component(next_components, "return")
        )
        next_entropy_value = torch.where(
            terminal, 0.0,
            next_entropy + self.component(next_components, "entropy"),
        )
        return_target = reward + continuation * next_return
        entropy_target = continuation * next_entropy_value
        return torch.stack((return_target, entropy_target), dim=-2)


def reduce_components(
    values, backend, specification, reduction, *, projection=None, beta=None,
    weights=None, pair_indices=None, generator=None, trusted_pair_indices=False,
):
    """Reduce complete members using a scalar projection to select the minimum.

    An independent minimum for each component would fabricate a member that the
    ensemble never predicted. Min reductions gather every component at one index.
    """
    if not specification.is_split:
        raise ValueError("Component reduction requires a split critic.")
    if values.ndim < 3 or values.shape[0] != backend.num_q:
        raise ValueError(f"Expected leading ensemble dimension {backend.num_q}.")
    reduction = {"min": "min_pair", "avg": "mean_pair"}.get(reduction, reduction)
    if reduction not in {"all", "min_pair", "mean_pair", "min_all", "mean_all"}:
        raise ValueError(f"Unknown Q reduction {reduction!r}.")
    if reduction == "all" or reduction.endswith("_all"):
        if pair_indices is not None:
            raise ValueError(f"pair_indices cannot be supplied with reduction={reduction!r}.")
        selected = values
    elif pair_indices is None and backend.pair_size == backend.num_q:
        selected = values
    else:
        selected = values.index_select(0, backend._pair_indices(
            values.device, pair_indices=pair_indices, generator=generator,
            trusted=trusted_pair_indices,
        ))
    if reduction == "all":
        return selected
    if reduction.startswith("mean_"):
        return selected.mean(dim=0)
    score = specification.project(selected, projection=projection, beta=beta, weights=weights)
    index = score.argmin(dim=0, keepdim=True).unsqueeze(-2)
    index = index.expand(1, *selected.shape[1:])
    return selected.gather(0, index).squeeze(0)
