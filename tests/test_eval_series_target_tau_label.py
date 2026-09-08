"""SAC target-rate comparisons remain distinguishable in published curves."""

from copy import deepcopy

import pytest

from tests.test_eval_series import label_registry
from utils.eval_series import concise_curve_label, evaluation_run_name
from utils.eval_series_data import planner_identity


@pytest.mark.parametrize("tau", [None, 0.01])
def test_original_default_target_labels_remain_unchanged(tau):
    registry = label_registry(inner_bootstrap_source="inner_target")
    if tau is not None:
        registry["identity"]["planner"]["settings"]["inner_critic_target_tau"] = tau
    assert concise_curve_label(registry) == "AMBI original · SAC C6/A12/T3 inner Q · #abcd"
    assert evaluation_run_name(registry) == (
        "Original AMBI prior-only backbone | SAC C6/A12/T3 inner Q aLR5e-5 J6"
        " | Attempt: actor sweep [abcd]"
    )


@pytest.mark.parametrize("label", [concise_curve_label, evaluation_run_name])
def test_nondefault_inner_target_tau_is_visible_in_both_labels(label):
    registry = label_registry(inner_bootstrap_source="inner_target", inner_critic_target_tau=0.1)
    before = deepcopy(registry)
    assert "inner Q qTau0.1" in label(registry)
    assert registry == before


@pytest.mark.parametrize("kind,bootstrap,terminal", [
    ("sac", "outer_target", None),
    ("sac", "outer_online", None),
    ("xqc", "inner_target", None),
    ("xqc", "inner_target", "outer"),
    ("prior", "inner_target", None),
])
def test_inactive_sac_target_rate_does_not_label_other_controllers(kind, bootstrap, terminal):
    registry = label_registry(inner_bootstrap_source=bootstrap, inner_critic_target_tau=0.1)
    planner = registry["identity"]["planner"]
    planner["type"] = kind
    if terminal:
        planner["settings"]["inner_terminal_bootstrap"] = terminal
    assert "qTau" not in concise_curve_label(registry)
    assert "qTau" not in evaluation_run_name(registry)


def test_label_reads_canonical_target_tau_and_does_not_change_scientific_identity():
    registry = label_registry(inner_bootstrap_source="inner_target", inner_critic_target_tau=0.1)
    config = {"inner_operator": "sac", **registry["identity"]["planner"]["settings"],
              "inner_tau": 0.9, "inner_actor_target_tau": 0.2}
    planner = planner_identity(config, {}, "AMBITDMPC2/AMBITDMPC2", "tanh_mean")
    assert "inner_tau" not in planner["settings"]
    assert "inner_actor_target_tau" not in planner["settings"]
    registry["identity"]["planner"] = planner
    before = deepcopy(planner)
    assert "qTau0.1" in concise_curve_label(registry)
    assert "qTau0.1" in evaluation_run_name(registry)
    assert planner == before
    baseline = planner_identity({**config, "inner_critic_target_tau": 0.01}, {},
                                "AMBITDMPC2/AMBITDMPC2", "tanh_mean")
    expected = deepcopy(planner)
    expected["settings"]["inner_critic_target_tau"] = 0.01
    assert baseline == expected
