"""AMBI's selected policy entropy and Bellman-target provenance."""


def policy_entropy(info, mode):
    """Select an existing statistic without resampling or detaching its graph."""
    if mode == "tdmpc2_scaled":
        return info["scaled_entropy"]
    if mode == "squashed":
        return -info["log_prob"]
    raise ValueError(f"Unknown policy entropy mode: {mode!r}")


def critic_entropy_spec(config, *, historical=False):
    """Describe each Bellman entropy bonus from a configuration mapping.

    Before this contract, entropy-augmented critics always used action entropy,
    even when the corresponding actor selected TD-MPC2 scaled entropy.

    With Q scaling, the raw-return entropy coefficient is alpha times the
    detached current scale S, sampled before the critic update. Outer targets
    use outer S; inner SAC targets use action-local S. Record these units only
    for active Q-scaled entropy bonuses, preserving unscaled, reward-only, and
    historical mappings exactly.
    """
    spec = {}
    coefficient_units = {}
    for learner, target_key in (
        ("outer", "outer_critic_target"), ("inner", "inner_sac_critic_target"),
    ):
        if config.get(target_key, "entropy_augmented") == "reward_only":
            semantics = "none"
        elif (
            learner == "inner"
            and config.get("inner_explorer_mode") == "shared_mixture"
        ):
            semantics = "squashed_mixture_entropy"
        elif (
            not historical
            and config.get(f"{learner}_actor_entropy_mode") == "tdmpc2_scaled"
        ):
            semantics = "tdmpc2_scaled_entropy"
        else:
            semantics = "squashed_action_entropy"
        spec[learner] = semantics
        if (
            not historical
            and config.get("sac_actor_loss_scale_mode") == "tdmpc2_percentile_range"
            and semantics != "none"
            and (learner == "outer" or config.get("inner_operator", "sac") == "sac")
        ):
            coefficient_units[learner] = (
                "alpha_times_outer_q_scale" if learner == "outer"
                else "alpha_times_action_local_q_scale"
            )
    if coefficient_units:
        spec["coefficient_units"] = coefficient_units
    return spec
