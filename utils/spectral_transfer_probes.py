"""Independent, fixed-reference probes for spectral weight transfer.

The actor objective is ``-mean Q_prior(s, mean_actor(s))``: a deterministic
policy surrogate, without entropy. It deliberately does not score log-standard
deviation rows and is not the changing SAC training loss. The critic objective
is decoded-Q squared error, averaged over every head and a fixed action bank,
against frozen-prior model-rollout labels. These labels are model references,
not environment-return ground truth. No candidate or donor selects the bank.

Scoring and heldout contexts use different named random streams. A heldout
context can evaluate initialization and the eventual post-J learner with the
same bank, labels and frozen reference critic. Layer input probes always use
the prior, so their output errors describe local preactivation geometry rather
than exact full-network output preservation.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import math

import numpy as np
import torch

from utils.ambi_benchmark import solver_seed
from utils.transfer_diagnostics import Reference, evaluating, json_value


DEFAULTS = dict(state_count=32, action_count=8, mc_rollouts=8)
ACTOR_OBJECTIVE = "negative frozen-prior Q of candidate deterministic mean; no entropy"
CRITIC_OBJECTIVE = "mean decoded per-head squared error against fixed prior-continuation model returns"


def spectral_probe_settings(settings=None):
    if settings is not None and (not isinstance(settings, dict) or set(settings) - set(DEFAULTS)):
        raise ValueError("Unknown spectral probe settings.")
    values = {**DEFAULTS, **(settings or {})}
    for key, minimum in (("state_count", 2), ("action_count", 2), ("mc_rollouts", 2)):
        if isinstance(values[key], bool) or not isinstance(values[key], int) or values[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}.")
    return values


def _clone(base, state=None, *, gradients=False):
    result = deepcopy(base).eval().requires_grad_(False)
    if state is not None:
        result.load_state_dict(state, strict=True)
    if gradients:
        matrix_names = set(_matrices(result))
        for name, parameter in result.named_parameters():
            parameter.requires_grad_(name in matrix_names)
    return result


def _matrices(module):
    matrices = {f"{name}.weight" if name else "weight": layer.weight
                for name, layer in module.named_modules() if isinstance(layer, torch.nn.Linear)}
    all_matrices = {name for name, value in module.named_parameters() if value.ndim == 2}
    if all_matrices != set(matrices):
        raise ValueError("Spectral probes require every matrix parameter to belong to a Linear layer.")
    return matrices


@contextmanager
def _capture_inputs(module):
    captured, handles = {}, []
    for name, layer in module.named_modules():
        if not isinstance(layer, torch.nn.Linear):
            continue
        key = f"{name}.weight" if name else "weight"
        def capture(_, args, key=key):
            value = args[0].detach().reshape(-1, args[0].shape[-1]).clone()
            if key in captured:
                raise RuntimeError(f"Spectral probe forwarded a matrix more than once: {key}")
            captured[key] = value
        handles.append(layer.register_forward_pre_hook(capture))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def _sha(value):
    value = value.detach().contiguous().cpu()
    header = f"{value.dtype}:{tuple(value.shape)}:".encode()
    return hashlib.sha256(header + value.numpy().tobytes()).hexdigest()


def _objective(reference, component, module, context, reference_critic):
    if component == "actor":
        mean = reference.model.policy_stats(context["states"], policy=module, **reference.bounds)["mean"]
        return -reference.q(reference_critic, context["states"], mean).mean()
    states, actions = context["states"], context["actions"]
    roots = states[:, None].expand(-1, actions.shape[1], -1).flatten(0, 1)
    heads = reference.q_heads(module, roots, actions.flatten(0, 1)).squeeze(-1)
    return (heads - context["labels"].flatten()[None]).square().mean()


def _forward_inputs(reference, component, module, context):
    """Visit all input hooks without constructing an unrequested objective."""
    if component == "actor":
        reference.model.policy_stats(context["states"], policy=module, **reference.bounds)
    else:
        states, actions = context["states"], context["actions"]
        roots = states[:, None].expand(-1, actions.shape[1], -1).flatten(0, 1)
        reference.q_heads(module, roots, actions.flatten(0, 1))


def build_spectral_context(wrapped, observation, *, controller_seed, episode_seed,
                           decision, settings=None, purpose="scoring",
                           components=("actor", "critic"), compute_gradients=True):
    """Make detached input/gradient banks at the frozen prior, independent of donors.

    ``inputs[component][weight_name]`` is ``[samples, input_width]`` and
    ``gradients[component][weight_name]`` matches that matrix. All tensors stay
    on the model device. Neither Python/NumPy/global Torch RNG nor the inner
    learner RNG, parameter gradients or persistent module modes are changed.
    ``purpose='heldout'`` must be used for efficacy diagnostics after selecting
    a transfer from a ``purpose='scoring'`` context. Heldout contexts always
    construct the complete action/return bank. Scoring builds MC returns only
    for requested critic gradients: activation inputs and actor gradients need
    no MC labels. Unrequested ``actions``, ``samples`` and ``labels`` are None,
    with explicit metadata availability flags and None checksums. Unmeasured
    objectives are omitted from ``metadata.reference_losses``, never zeroed.
    """
    settings = spectral_probe_settings(settings)
    if purpose not in {"scoring", "heldout"}:
        raise ValueError("Spectral probe purpose must be scoring or heldout.")
    components = tuple(components)
    if not components or len(set(components)) != len(components) or set(components) - {"actor", "critic"}:
        raise ValueError("Spectral probe components must be unique actor/critic names.")
    if not isinstance(compute_gradients, bool):
        raise ValueError("compute_gradients must be a boolean.")
    if isinstance(decision, bool) or not isinstance(decision, int) or decision < 0:
        raise ValueError("decision must be a nonnegative integer.")
    r = Reference(wrapped, rollouts=settings["mc_rollouts"])
    if (r.cfg.inner_sac_critic_target != "reward_only" or r.cfg.inner_terminal_entropy != "none"
            or getattr(r.cfg, "critic_value_mode", "single") != "single"
            or getattr(r.cfg, "obs", "state") != "state"
            or getattr(r.cfg, "inner_horizon_conditioning", "none") not in {"none", None, False}):
        raise ValueError("Spectral probes require state-observation, unconditioned return/return single critics.")
    seed = solver_seed(controller_seed, "spectral-probe", episode_seed, decision, purpose)
    n, k, horizon = settings["state_count"], settings["action_count"], int(r.cfg.inner_rollout_horizon)
    need_actions = purpose == "heldout" or "critic" in components
    need_returns = purpose == "heldout" or ("critic" in components and compute_gradients)
    compute_objectives = compute_gradients or purpose == "heldout"
    actor = _clone(r.engine._actor_base)
    reference_critic = _clone(r.engine._critic_base) if "actor" in components and compute_objectives else None
    devices = [r.device.index if r.device.index is not None else torch.cuda.current_device()] if r.device.type == "cuda" else []
    context = dict(inputs={}, gradients={}, prior_states={
        component: {name: value.detach().clone() for name, value in getattr(r.engine, f"_{component}_base").state_dict().items()}
        for component in ("actor", "critic")})
    with torch.random.fork_rng(devices=devices), evaluating(r.model, r.engine._horizon_actor, r.engine._horizon_critic):
        with torch.no_grad():
            root = r.encode(np.asarray(observation)[None])
            successor_actions, _ = r.policy(actor)(root.expand(n - 1, -1), r.noise(
                (n - 1, r.cfg.action_dim), solver_seed(seed, "state-bank")))
            _, successors, _ = r.transition(root.expand(n - 1, -1), successor_actions)
            states = torch.cat((root, successors)).detach()
            actions = samples = labels = None
            if need_actions:
                # One prior mean and K-1 independently sampled prior actions.
                # Named streams preserve the exact bank when unused MC work is
                # skipped. Neither donor nor candidate chooses these queries.
                mean = r.model.policy_stats(states, policy=actor, **r.bounds)["mean"]
                action_bank = [mean]
                for index in range(k - 1):
                    action_bank.append(r.policy(actor)(states, r.noise(
                        (n, r.cfg.action_dim), solver_seed(seed, "action-bank", index)))[0])
                actions = torch.stack(action_bank, 1).detach()
            if need_returns:
                roots = states[:, None].expand(-1, k, -1).flatten(0, 1)
                samples = r.returns(roots, actions.flatten(0, 1), actor, (horizon,),
                                    seed=solver_seed(seed, "fixed-targets"))[horizon].reshape(-1, n, k).detach()
                labels = samples.mean(0).detach()
            context.update(states=states, actions=actions, samples=samples, labels=labels)
        losses = {}
        for component in components:
            module = _clone(getattr(r.engine, f"_{component}_base"), gradients=compute_gradients)
            with torch.enable_grad() if compute_gradients else torch.no_grad():
                with _capture_inputs(module) as captured:
                    if compute_objectives:
                        loss = _objective(r, component, module, context, reference_critic)
                    else:
                        _forward_inputs(r, component, module, context)
                matrices = _matrices(module)
                if set(captured) != set(matrices):
                    raise RuntimeError("Spectral input hooks missed one or more matrices.")
                context["inputs"][component] = captured
                if compute_gradients:
                    grads = torch.autograd.grad(loss, tuple(matrices.values()), allow_unused=False)
                    context["gradients"][component] = {name: value.detach().clone() for name, value in zip(matrices, grads)}
            if compute_objectives:
                losses[component] = float(loss.detach())
    for name in ("states", "actions", "samples", "labels"):
        if context[name] is not None and not torch.isfinite(context[name]).all():
            raise ValueError(f"Nonfinite spectral probe {name}.")
    for family in ("inputs", "gradients"):
        if any(not torch.isfinite(value).all() for bank in context[family].values() for value in bank.values()):
            raise ValueError(f"Nonfinite spectral probe {family}.")
    context["metadata"] = dict(protocol="spectral-prior-probes-v1", purpose=purpose,
        seed=int(seed), decision=decision, episode_seed=int(episode_seed), controller_seed=int(controller_seed),
        **settings, horizon=horizon, actor_objective=ACTOR_OBJECTIVE, critic_objective=CRITIC_OBJECTIVE,
        gradient_anchor="frozen_prior", bank="actual root plus independent one-step prior successors",
        continuation="frozen_prior", tail="frozen_horizon_actor_and_critic", target="reward_only",
        input_anchor="frozen_prior", reference_losses=losses, state_sha256=_sha(context["states"]),
        actions_available=need_actions, return_labels_available=need_returns,
        gradients_available=compute_gradients,
        action_sha256=_sha(context["actions"]) if need_actions else None,
        target_sha256=_sha(context["labels"]) if need_returns else None)
    return context


def evaluate_spectral_objectives(wrapped, states, context):
    """Evaluate one actor/critic state on an already fixed context without fitting."""
    if context.get("actions") is None or context.get("labels") is None:
        raise ValueError("Evaluating both spectral objectives requires a complete action/return context.")
    r = Reference(wrapped)
    reference_critic = _clone(r.engine._critic_base, context["prior_states"]["critic"])
    values = {}
    with torch.no_grad(), evaluating(r.model):
        for component in ("actor", "critic"):
            module = _clone(getattr(r.engine, f"_{component}_base"), states[component])
            values[f"{component}_loss"] = float(_objective(r, component, module, context, reference_critic))
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Nonfinite spectral handoff objective.")
    return values


def evaluate_spectral_handoff(wrapped, *, donor, initial_states, final_states=None,
                              context, ranks=(1, 2, 4, 8, 16, 32, 64, 128)):
    """Report spectral geometry and immediate/post-J fixed heldout objectives.

    ``donor`` may be the exported diagnostic state (with ``modules``), a plain
    actor/critic state mapping, or None at the first decision. Energy ratios are
    actual squared parameter norms, not probabilities or rank fractions. Zero
    donor norms produce zero ratios and are explicitly flagged. No nonzero
    donor is assumed at the first decision.
    """
    from utils.spectral_transfer import spectral_energy

    if context["metadata"]["purpose"] != "heldout":
        raise ValueError("Handoff diagnostics require an independent heldout context.")
    if set(context["inputs"]) != {"actor", "critic"} or set(context["gradients"]) != {"actor", "critic"}:
        raise ValueError("Handoff diagnostics require both components and heldout gradients.")
    prior = context["prior_states"]
    donor_states = None if donor is None else donor.get("modules", donor)
    stages = dict(prior=prior, initial=initial_states)
    if donor_states is not None:
        stages["donor"] = donor_states
    if final_states is not None:
        stages["final"] = final_states
    objectives = {stage: evaluate_spectral_objectives(wrapped, states, context) for stage, states in stages.items()}
    summary = {f"{stage}_{name}": value for stage, values in objectives.items() for name, value in values.items()}
    layers = {}
    for component in ("actor", "critic"):
        component_layers = layers[component] = {}
        donor_energy = initial_energy = residual_energy = alignment = 0.
        for name, inputs in context["inputs"][component].items():
            base = prior[component][name]
            delta = torch.zeros_like(base) if donor_states is None else donor_states[component][name].to(base) - base
            carried = initial_states[component][name].to(base) - base
            gradient = context["gradients"][component][name]
            d2, c2 = float(delta.square().sum()), float(carried.square().sum())
            error2 = float((carried - delta).square().sum())
            dot = float((delta * carried).sum())
            output = inputs @ delta.T
            output_error = inputs @ (carried - delta).T
            output2 = float(output.square().mean())
            row = dict(spectrum=spectral_energy(delta, ranks=ranks, gradient=gradient),
                donor_squared_norm=d2, initial_squared_norm=c2, zero_donor=d2 == 0.,
                transferred_energy_ratio=c2 / d2 if d2 else 0.,
                parameter_residual_energy_ratio=error2 / d2 if d2 else 0.,
                donor_initial_cosine=dot / math.sqrt(d2 * c2) if d2 and c2 else 0.,
                donor_projection_coefficient=dot / d2 if d2 else 0.,
                prior_input_output_mse=float(output_error.square().mean()),
                prior_input_output_relative_mse=float(output_error.square().mean()) / output2 if output2 else 0.,
                zero_donor_prior_input_output=output2 == 0.,
                donor_first_order_benefit=-float((gradient * delta).sum()),
                initial_first_order_benefit=-float((gradient * carried).sum()),
                prior_input_samples=int(len(inputs)))
            component_layers[name] = row
            donor_energy += d2
            initial_energy += c2
            residual_energy += error2
            alignment += dot
        summary.update({f"{component}_donor_squared_norm": donor_energy,
            f"{component}_initial_squared_norm": initial_energy,
            f"{component}_transferred_energy_ratio": initial_energy / donor_energy if donor_energy else 0.,
            f"{component}_parameter_residual_energy_ratio": residual_energy / donor_energy if donor_energy else 0.,
            f"{component}_donor_initial_cosine": alignment / math.sqrt(donor_energy * initial_energy) if donor_energy and initial_energy else 0.,
            f"{component}_initial_first_order_benefit": sum(row["initial_first_order_benefit"] for row in component_layers.values()),
            f"{component}_donor_first_order_benefit": sum(row["donor_first_order_benefit"] for row in component_layers.values()),
            f"{component}_prior_input_output_mse": float(np.mean([row["prior_input_output_mse"] for row in component_layers.values()])),
            f"{component}_initial_loss_gain_vs_prior": objectives["prior"][f"{component}_loss"] - objectives["initial"][f"{component}_loss"]})
        for metric in ("stable_rank", "effective_rank", "energy_effective_rank", "rank_50", "rank_90", "rank_95", "rank_99",
                       "positive_benefit_fraction", "positive_benefit_energy_fraction"):
            summary[f"{component}_mean_{metric}"] = float(np.mean([row["spectrum"][metric] for row in component_layers.values()]))
        correlations = [row["spectrum"]["energy_benefit_correlation"] for row in component_layers.values()
                        if row["spectrum"]["energy_benefit_correlation"] is not None]
        # Explicit denominator preserves the distinction between undefined
        # constant-spectrum correlations and a measured zero correlation.
        summary[f"{component}_energy_benefit_correlation_layers"] = len(correlations)
        if correlations:
            summary[f"{component}_mean_energy_benefit_correlation"] = float(np.mean(correlations))
        for rank in ranks:
            summary[f"{component}_mean_energy_at_rank_{rank}"] = float(np.mean([
                row["spectrum"]["energy_at_rank"][str(rank)] for row in component_layers.values()]))
            summary[f"{component}_top_energy_benefit_rank_{rank}"] = sum(
                row["spectrum"]["top_energy_benefit"][str(rank)] for row in component_layers.values())
        if final_states is not None:
            summary[f"{component}_post_j_loss_gain_vs_prior"] = objectives["prior"][f"{component}_loss"] - objectives["final"][f"{component}_loss"]
            summary[f"{component}_within_solve_loss_improvement"] = objectives["initial"][f"{component}_loss"] - objectives["final"][f"{component}_loss"]
    return json_value(dict(reference=context["metadata"], donor_available=donor is not None,
        objectives=objectives, layers=layers, summary=summary,
        semantics="Heldout fixed-reference surrogate objectives; post-J evaluated on the same bank. Spectra and linear preactivation geometry are layerwise, at prior inputs, not full-network behavior or real-return guarantees."))
