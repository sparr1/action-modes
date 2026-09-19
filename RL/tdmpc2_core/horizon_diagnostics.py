"""Detached, device-resident sufficient statistics for training minibatches."""

import re

import torch


HORIZON_METRIC = re.compile(r"^(critic|actor)_horizon_([1-9][0-9]*)_(sample_count|[a-z_]+_(?:sum|mean))$")


@torch.no_grad()
def horizon_sums(component, remaining_horizon, horizon, **row_values):
    """Count rows, not critic heads; callers reduce heads before entering here.

    No host synchronization, sampling, model evaluation, or autograd is involved.
    Empty groups retain zero counts/sums; only the publisher derives means.
    """
    labels = remaining_horizon.detach().reshape(-1)
    result = {}
    for h in range(1, int(horizon) + 1):
        mask = labels == h
        prefix = f"{component}_horizon_{h}_"
        result[prefix + "sample_count"] = mask.sum()
        for name, values in row_values.items():
            values = values.detach().reshape(-1)
            if values.shape != labels.shape:
                raise ValueError("Horizon diagnostics require one value per sampled row.")
            result[prefix + name + "_sum"] = torch.where(mask, values, 0).sum()
    return result


def horizon_metric_description(name):
    match = HORIZON_METRIC.fullmatch(name)
    if match is None:
        return None
    component, horizon, statistic = match.groups()
    derived = statistic.endswith("_mean")
    if derived:
        statistic = statistic[:-5] + "_sum"
    descriptions = {
        "sample_count": "Number of sampled replay rows (including repeated samples)",
        "td_error_abs_sum": "Sum of absolute TD errors, averaged over critic heads per row",
        "predicted_q_sum": "Sum of decoded predicted Q, averaged over critic heads per row",
        "target_q_sum": "Sum of Bellman target Q",
        "entropy_sum": "Sum of the actor objective's entropy statistic (nats or configured TD-MPC2 scaled statistic)",
        "objective_q_sum": "Sum of the actor objective's Q after its configured reduction and Q scaling",
    }
    description = descriptions.get(statistic, statistic)
    if derived:
        description = description.replace("Sum of", "Sample-weighted mean of", 1)
    return (f"{description} at remaining horizon h={horizon}; "
            f"pre-update {component} training minibatch, not a held-out or real-return probe.")
