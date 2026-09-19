"""Count-weighted training statistics within seeds, equal-weighted across seeds."""

from collections import defaultdict
import math
import re


_KEY = re.compile(r"^((critic|actor)_horizon_[1-9][0-9]*_)(sample_count|[a-z_]+_sum)$")
_FIELDS = {"critic": {"td_error_abs_sum", "predicted_q_sum", "target_q_sum"},
           "actor": {"entropy_sum", "objective_q_sum"}}


def groups(metrics):
    result = defaultdict(dict)
    for name, value in metrics.items():
        match = _KEY.fullmatch(name)
        if match:
            prefix, component, field = match.groups()
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Nonfinite horizon statistic: {name}")
            result[(prefix, component)][field] = value
    for (prefix, component), values in result.items():
        if set(values) != _FIELDS[component] | {"sample_count"}:
            raise ValueError(f"Incomplete horizon statistics: {prefix}")
        count = values["sample_count"]
        if count < 0 or int(count) != count:
            raise ValueError(f"Invalid horizon sample count: {prefix}")
        if count == 0 and any(values[key] != 0 for key in _FIELDS[component]):
            raise ValueError(f"Nonzero sum in an empty horizon group: {prefix}")
    return result


def accumulate(target, grouped):
    for (prefix, _), values in grouped.items():
        for key, value in values.items():
            target[prefix][key] += value


def seed_means(totals):
    result = {}
    for prefix, values in totals.items():
        count = values["sample_count"]
        result[prefix + "sample_count"] = count
        if count:
            for key, value in values.items():
                if key.endswith("_sum"):
                    result[prefix + key[:-4] + "_mean"] = value / count
    return result


def equal_seed_summary(seeds):
    """Missing groups contribute coverage=0 and no mean observation."""
    samples = defaultdict(list)
    coverage = defaultdict(float)
    for totals in seeds:
        for name, value in seed_means(totals).items():
            samples[name].append(value)
        for prefix, values in totals.items():
            coverage[prefix] += values["sample_count"]
    result = {}
    for name, values in samples.items():
        prefix = re.match(r"(?:actor|critic)_horizon_[1-9][0-9]*_", name)[0]
        result[name] = dict(mean=sum(values)/len(values), min=min(values), max=max(values),
                            count=len(values), sample_count=coverage[prefix])
    return result


class HorizonTrainingSummary:
    def __init__(self):
        factory = lambda: defaultdict(lambda: defaultdict(float))
        self.updates = defaultdict(lambda: defaultdict(factory))
        self.decisions = defaultdict(factory)

    def add(self, event):
        grouped = groups(event["metrics"])
        ep, decision = event["episode_id"], event["decision_index"]
        for (prefix, component), values in grouped.items():
            if not event.get("updated_" + component):
                raise ValueError("Horizon statistics must belong to their optimizer update.")
            axis = component + "_update"
            index = event[component + "_updates"]
            group = {(prefix, component): values}
            accumulate(self.updates[(axis, index)][ep], group)
            accumulate(self.decisions[(ep, decision)], group)

    def update_curves(self):
        return {key: equal_seed_summary(seeds.values()) for key, seeds in self.updates.items()}

    def decision_curves(self):
        decisions = defaultdict(list)
        for (_, decision), values in self.decisions.items():
            decisions[decision].append(values)
        return {key: equal_seed_summary(seeds) for key, seeds in decisions.items()}
