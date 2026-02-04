"""
AMBI Training Script with Configurable Experiments
Uses experiment_config.py for all hyperparameters
"""
import sys
import gymnasium as gym
from datetime import datetime
from RL.AMBI import AMBI
from domains.AntPlane import AntPlane
from modes.tasks import Subtask
from domains.Maze import Change
from experiment_config import (
    EXPERIMENT_NAME,
    TOTAL_TIMESTEPS,
    ENV_CONFIG,
    CHANGE_OBJECTIVE_CONFIG,
    AMBI_CONFIG,
    SAVE_PATH,
    MODEL_NAME,
)

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def print_experiment_info():
    """Print experiment configuration."""
    print("\n" + "="*70)
    print(f"AMBI EXPERIMENT: {EXPERIMENT_NAME}")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"\nEnvironment: {ENV_CONFIG['id']}")
    print(f"Max episode steps: {ENV_CONFIG['max_episode_steps']}")
    print(f"\nTask: Change Z-coordinate")
    print(f"  Target range: {CHANGE_OBJECTIVE_CONFIG['desired_coord_minimum']:.2f} - {CHANGE_OBJECTIVE_CONFIG['desired_coord_maximum']:.2f}")
    print(f"\nAMBI Configuration:")
    print(f"  Outer algorithm: {AMBI_CONFIG['outer_alg']}")
    print(f"  Inner algorithm: {AMBI_CONFIG['inner_alg']}")
    print(f"  Inner rollouts (B): {AMBI_CONFIG['inner_rollouts']}")
    print(f"  Reinit inner every step: {AMBI_CONFIG['inner_reinit_every_step']}")
    print(f"  Inner updates per rollout: {AMBI_CONFIG['inner_updates_per_rollout']}")
    print(f"  Seed episodes: {AMBI_CONFIG['seed_episodes']}")
    print(f"\nOuter Agent (Real Environment):")
    print(f"  Buffer size: {AMBI_CONFIG['outer_alg_params']['buffer_size']:,}")
    print(f"  Learning starts: {AMBI_CONFIG['outer_alg_params']['learning_starts']:,}")
    print(f"  Learning rate: {AMBI_CONFIG['outer_alg_params']['learning_rate']}")
    print(f"\nInner Agent (Imagination):")
    print(f"  Buffer size: {AMBI_CONFIG['inner_alg_params']['buffer_size']:,}")
    print(f"  Learning starts: {AMBI_CONFIG['inner_alg_params']['learning_starts']}")
    print("="*70 + "\n")


def main():
    # Print configuration
    print_experiment_info()

    # Create environment
    print("Creating environment...")
    base_env = gym.make(**ENV_CONFIG)
    base_domain = AntPlane(base_env)
    train_env = Subtask(base_domain, Change(**CHANGE_OBJECTIVE_CONFIG))

    print(f"✓ Environment created")
    print(f"  Observation space: {train_env.observation_space}")
    print(f"  Action space: {train_env.action_space}")

    # Initialize AMBI agent
    print("\nInitializing AMBI agent...")
    model = AMBI("AMBI", train_env, custom_params=AMBI_CONFIG)
    print("✓ AMBI agent initialized")

    # Optional: Load checkpoint if resuming
    # model.load("logs/checkpoint/model")

    # Train
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")

    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        print("\n" + "="*70)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user (Ctrl+C)")

    except Exception as e:
        print(f"\n\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Always save model (even if interrupted)
        try:
            save_name = f"{MODEL_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"\nSaving model as '{save_name}'...")
            model.save(SAVE_PATH, save_name)
            print(f"✓ Model saved to: {SAVE_PATH}{save_name}")
        except Exception as e:
            print(f"✗ Failed to save model: {e}")

        # Close environment
        train_env.close()
        print("\n" + "="*70)
        print(f"Experiment '{EXPERIMENT_NAME}' finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)


if __name__ == "__main__":
    main()
