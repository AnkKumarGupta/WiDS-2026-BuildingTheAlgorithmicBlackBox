import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from day2 import TradingEnv 
import os

def train_agent():
    print("--- 1. Initialize Environment ---")
    env = TradingEnv()

    print("Checking environment compatibility...")
    check_env(env)
    print("Environment passed checks.")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        batch_size=64,
        ent_coef=0.05  # Encouraging exploration initially
    )

    print("\n--- Start Training (50k Steps) ---")
    model.learn(total_timesteps=50000)

    model_name = "ppo_trading_agent"
    model.save(model_name)
    print(f"\nModel saved to {model_name}.zip")

if __name__ == "__main__":
    train_agent()