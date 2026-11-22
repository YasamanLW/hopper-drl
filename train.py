
import gymnasium as gym
from stable_baselines3 import PPO

ENV_ID = "Hopper-v5"
MODEL_NAME = "hopper_ppo_final"
SEED = 42


def train():
    env = gym.make(ENV_ID)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=SEED
    )

    # Training stages
    model.learn(total_timesteps=10_000)
    model.learn(total_timesteps=50_000)
    model.learn(total_timesteps=50_000)

    model.save(MODEL_NAME)
    print(f"Model saved as {MODEL_NAME}.zip")

    env.close()


def evaluate():
    env = gym.make(ENV_ID, render_mode="human")
    model = PPO.load(MODEL_NAME)

    obs, info = env.reset(seed=SEED)

    try:
        for _ in range(5_000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                obs, info = env.reset()
    finally:
        env.close()


if __name__ == "__main__":
    train()
    evaluate()
