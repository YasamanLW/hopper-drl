import gymnasium as gym
from stable_baselines3 import PPO

ENV_ID = "Hopper-v5"
MODEL_NAME = "hopper_ppo_final"
SEED = 42


def evaluate():
    env = gym.make(ENV_ID, render_mode="human")
    model = PPO.load(MODEL_NAME)

    obs, info = env.reset(seed=SEED)
    episode_rewards = []
    total_reward = 0

    try:
        for step in range(5000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward

            if terminated or truncated:
                episode_rewards.append(total_reward)
                print(f"Episode {len(episode_rewards)} reward: {total_reward:.2f}")

                obs, info = env.reset()
                total_reward = 0

                if len(episode_rewards) >= 5:
                    break

    finally:
        env.close()

    if episode_rewards:
        average_reward = sum(episode_rewards) / len(episode_rewards)
        print(f"\nAverage evaluation reward: {average_reward:.2f}")


if __name__ == "__main__":
    evaluate()