"""
Plot AMBI training progress from logs.
Creates visualizations for episode rewards, lengths, and learning curves.
"""
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def parse_log_file(log_path):
    """Parse training log and extract metrics."""
    data = {
        'timesteps': [],
        'episodes': [],
        'episode_steps': [],
        'rewards': [],
        'episode_rewards': [],
        'episode_lengths': [],
        'episode_timesteps': [],
    }

    with open(log_path, 'r') as f:
        for line in f:
            # New format: [100/1,000,000] 0.0% | Episodes: 5 | 42.3 steps/s | ETA: 123.4m
            progress_match = re.search(r'\[(\d+(?:,\d+)*)/(\d+(?:,\d+)*)\].+Episodes: (\d+)', line)
            if progress_match:
                timestep = int(progress_match.group(1).replace(',', ''))
                episodes = int(progress_match.group(3))
                data['timesteps'].append(timestep)
                data['episodes'].append(episodes)

            # New format: Episode 5 complete | Steps: 250 | Return: 123.45 | Timestep: 1,250
            episode_match = re.search(r'Episode (\d+) complete \| Steps: (\d+) \| Return: ([-\d.]+) \| Timestep: (\d+(?:,\d+)*)', line)
            if episode_match:
                episode_num = int(episode_match.group(1))
                episode_length = int(episode_match.group(2))
                episode_return = float(episode_match.group(3))
                timestep = int(episode_match.group(4).replace(',', ''))

                data['episode_lengths'].append(episode_length)
                data['episode_rewards'].append(episode_return)
                data['episode_timesteps'].append(timestep)

    return data


def smooth(values, weight=0.9):
    """Exponential moving average smoothing."""
    smoothed = []
    last = values[0] if len(values) > 0 else 0
    for v in values:
        smoothed_val = last * weight + (1 - weight) * v
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_training_curves(data, save_path=None):
    """Create comprehensive training visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('AMBI Training Progress', fontsize=16, fontweight='bold')

    # 1. Episode Returns Over Time
    ax = axes[0, 0]
    if data['episode_rewards'] and data['episode_timesteps']:
        ax.plot(data['episode_timesteps'], data['episode_rewards'],
                alpha=0.3, color='blue', label='Raw')
        if len(data['episode_rewards']) > 10:
            smoothed = smooth(data['episode_rewards'], weight=0.9)
            ax.plot(data['episode_timesteps'], smoothed,
                    color='blue', linewidth=2, label='Smoothed')
        ax.set_xlabel('Timesteps', fontsize=12)
        ax.set_ylabel('Episode Return', fontsize=12)
        ax.set_title('Cumulative Reward per Episode', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No episode data yet',
                ha='center', va='center', transform=ax.transAxes)

    # 2. Episode Lengths
    ax = axes[0, 1]
    if data['episode_lengths'] and data['episode_timesteps']:
        ax.plot(data['episode_timesteps'], data['episode_lengths'],
                color='green', marker='o', markersize=3, alpha=0.6)
        ax.axhline(y=400, color='red', linestyle='--',
                   label='Max Episode Length', alpha=0.5)
        ax.set_xlabel('Timesteps', fontsize=12)
        ax.set_ylabel('Episode Length', fontsize=12)
        ax.set_title('Episode Lengths Over Time', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No episode data yet',
                ha='center', va='center', transform=ax.transAxes)

    # 3. Step-wise Rewards
    ax = axes[1, 0]
    if data['timesteps'] and data['rewards']:
        # Plot every Nth point to avoid overcrowding
        stride = max(1, len(data['timesteps']) // 1000)
        ax.scatter(data['timesteps'][::stride], data['rewards'][::stride],
                   alpha=0.3, s=1, color='orange')

        # Add moving average
        window = min(100, len(data['rewards']) // 10)
        if window > 1:
            rewards_array = np.array(data['rewards'])
            moving_avg = np.convolve(rewards_array,
                                     np.ones(window)/window, mode='valid')
            ax.plot(data['timesteps'][window-1::stride],
                    moving_avg[::stride],
                    color='red', linewidth=2, label=f'MA({window})')
            ax.legend()

        ax.set_xlabel('Timesteps', fontsize=12)
        ax.set_ylabel('Step Reward', fontsize=12)
        ax.set_title('Step-wise Rewards', fontsize=14)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No step data yet',
                ha='center', va='center', transform=ax.transAxes)

    # 4. Recent Performance (Last 50 episodes)
    ax = axes[1, 1]
    if data['episode_rewards'] and len(data['episode_rewards']) > 0:
        recent_returns = data['episode_rewards'][-50:]
        recent_episodes = list(range(len(data['episode_rewards']) - len(recent_returns),
                                     len(data['episode_rewards'])))

        ax.bar(recent_episodes, recent_returns, color='purple', alpha=0.6)
        if len(recent_returns) > 1:
            avg = np.mean(recent_returns)
            ax.axhline(y=avg, color='red', linestyle='--',
                      label=f'Avg: {avg:.2f}', linewidth=2)
            ax.legend()

        ax.set_xlabel('Episode Number', fontsize=12)
        ax.set_ylabel('Episode Return', fontsize=12)
        ax.set_title('Recent Episodes (Last 50)', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No episode data yet',
                ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {save_path}")

    return fig


def print_statistics(data):
    """Print training statistics."""
    print("\n" + "="*60)
    print("TRAINING STATISTICS")
    print("="*60)

    if data['episode_rewards']:
        print(f"\nEpisodes completed: {len(data['episode_rewards'])}")
        print(f"Total timesteps: {max(data['timesteps']) if data['timesteps'] else 0}")

        print(f"\nEpisode Returns:")
        print(f"  Mean: {np.mean(data['episode_rewards']):.2f}")
        print(f"  Std:  {np.std(data['episode_rewards']):.2f}")
        print(f"  Min:  {np.min(data['episode_rewards']):.2f}")
        print(f"  Max:  {np.max(data['episode_rewards']):.2f}")

        if len(data['episode_rewards']) >= 10:
            print(f"\nLast 10 episodes:")
            print(f"  Mean: {np.mean(data['episode_rewards'][-10:]):.2f}")
            print(f"  Max:  {np.max(data['episode_rewards'][-10:]):.2f}")

        if data['episode_lengths']:
            print(f"\nEpisode Lengths:")
            print(f"  Mean: {np.mean(data['episode_lengths']):.1f}")
            print(f"  Max:  {np.max(data['episode_lengths'])}")
    else:
        print("\nNo episodes completed yet.")

    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Plot AMBI training progress')
    parser.add_argument('log_file', type=str,
                       help='Path to training log file')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output path for plot (default: show interactively)')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plot interactively')

    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        return

    print(f"Parsing log file: {log_path}")
    data = parse_log_file(log_path)

    # Print statistics
    print_statistics(data)

    # Create plots
    output_path = args.output
    if output_path is None and not args.no_show:
        # Default: save to same directory as log
        output_path = log_path.parent / f"{log_path.stem}_plot.png"

    fig = plot_training_curves(data, save_path=output_path)

    if not args.no_show:
        print("Displaying plot... (close window to exit)")
        plt.show()


if __name__ == "__main__":
    main()
