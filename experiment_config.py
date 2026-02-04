"""
Experiment configuration for AMBI training.
Modify these parameters to run different experiments.
"""

# ============================================
# EXPERIMENT SETTINGS
# ============================================
EXPERIMENT_NAME = "ambi_ant_change_v1"
TOTAL_TIMESTEPS = int(1e6)  # 1 million steps

# ============================================
# ENVIRONMENT SETTINGS
# ============================================
ENV_CONFIG = {
    "id": "Ant-v4",
    "exclude_current_positions_from_observation": False,
    "max_episode_steps": 400,
    # "render_mode": "human",  # Uncomment for visualization
}

# Task: Train ant to reach high Z-coordinates (jump/rear up)
CHANGE_OBJECTIVE_CONFIG = {
    "target_coords": "Z",
    "desired_coord_minimum": 0.3,
    "desired_coord_maximum": 0.9,
    "survival_bonus": 1.0,
    "margin": 0.025,
    "slope": 3.0,
    "categorical": False,
    "metric": "L2"
}

# ============================================
# AMBI HYPERPARAMETERS
# ============================================
AMBI_CONFIG = {
    # Algorithms
    "outer_alg": "baselines/SAC",
    "inner_alg": "baselines/SAC",

    # Outer agent (real environment) parameters
    "outer_alg_params": {
        "learning_starts": int(1e4),    # Start learning after 10k steps
        "buffer_size": int(1e6),        # 1M experience replay buffer
        "learning_rate": 3e-4,          # SAC learning rate
        "batch_size": 256,              # Batch size for updates
        "tau": 0.005,                   # Soft update coefficient
        "gamma": 0.99,                  # Discount factor
    },

    # Inner agent (imagination) parameters
    "inner_alg_params": {
        "learning_starts": 0,           # Learn immediately from imagined data
        "buffer_size": int(1e5),        # 100k buffer for imagined experience
        "learning_rate": 3e-4,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.99,
    },

    # Inner loop: Imagination settings
    "inner_rollouts": 6,                # B = 6 imagined rollouts per real step
    "inner_reinit_every_step": True,    # Reinitialize inner agent each step
    "inner_updates_per_rollout": 1,     # Update inner agent after each rollout

    # Episode settings
    "max_episode_steps": 400,           # Must match env max_episode_steps
    "seed_episodes": 10,                # Random exploration to fill outer buffer

    # Visualization
    "render": False,                    # Set to True for local testing with GUI
}

# ============================================
# EXPERIMENT VARIANTS
# ============================================
# Uncomment to try different configurations:

# # Variant 1: More imagination rollouts
# AMBI_CONFIG["inner_rollouts"] = 10

# # Variant 2: Longer episodes
# ENV_CONFIG["max_episode_steps"] = 600
# AMBI_CONFIG["max_episode_steps"] = 600

# # Variant 3: No inner agent reinitialization (persistent inner agent)
# AMBI_CONFIG["inner_reinit_every_step"] = False

# # Variant 4: More inner updates per rollout
# AMBI_CONFIG["inner_updates_per_rollout"] = 5

# ============================================
# SAVE SETTINGS
# ============================================
SAVE_PATH = "./"
MODEL_NAME = EXPERIMENT_NAME
