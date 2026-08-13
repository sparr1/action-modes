"""
Quick smoke test for LegAdaptAnt-v0 and Ant3LegDeadStump-v0.
"""
import gymnasium as gym
import domains  # registers the envs


def test_adaptation(total_timesteps=300, episode_len=50):
    print("=== LegAdaptAnt-v0 (4 -> 3 legs) ===")
    env = gym.make("LegAdaptAnt-v0", total_timesteps=total_timesteps)
    obs, info = env.reset()
    print(f"  obs shape:    {obs.shape}  (fixed throughout)")
    print(f"  action shape: {env.action_space.shape}  (fixed throughout)")
    print(f"  switch at step: {env.unwrapped._switch_step}")

    step = 0
    switch_logged = False
    while step < total_timesteps + episode_len:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        step += 1

        if info["switched"] and not switch_logged:
            print(f"  [step {step}] switched to {info['num_legs']} legs")
            print(f"  obs shape post-switch:    {obs.shape}")
            switch_logged = True

        if terminated or truncated:
            obs, info = env.reset()
            print(f"  [step {step}] episode reset — num_legs={info['num_legs']}, switched={info['switched']}")

    env.close()
    print("  PASSED\n")


if __name__ == "__main__":
    test_adaptation()
