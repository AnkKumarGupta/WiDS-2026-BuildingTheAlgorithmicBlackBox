import os
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from day2 import TradingEnv

def run_sanity_check():
    print("--- DAY 5: SANITY CHECK (50k Steps) ---")

    log_dir = "training_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "monitor.csv")

    env = TradingEnv()
    env.transaction_cost = 0.0001
    env.risk_aversion = 0.05      # Penalty for volatility
    env.inventory_penalty = 0.01  # Penalty for hoarding

    env = Monitor(env, filename=os.path.join(log_dir, "monitor"))

    model_path = "ppo_trading_agent.zip"
    if not os.path.exists(model_path):
        print("Error: ppo_trading_agent.zip not found. Run Day 4 first.")
        return

    print("Loading 'Cowboy' Agent...")
    model = PPO.load(model_path)
    model.set_env(env)

    model.learning_rate = 0.0001
    model.ent_coef = 0.01

    print("Starting 50,000 step run...")
    model.learn(total_timesteps=50000)
    print("Training Complete.")

    model.save("ppo_trading_agent_pro")
    print("Saved 'ppo_trading_agent_pro.zip'")

    plot_learning_curve(log_file)

def plot_learning_curve(log_file):
    print("\n--- Generating Learning Curves ---")
    try:
        df = pd.read_csv(log_file + ".csv", skiprows=1)
    except FileNotFoundError:
        print("Log file not found. Check training_logs folder.")
        return

    window = 50
    df['rolling_reward'] = df['r'].rolling(window=window).mean()
    df['rolling_len'] = df['l'].rolling(window=window).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(df['r'], alpha=0.3, color='gray', label='Raw Reward')
    ax1.plot(df['rolling_reward'], color='blue', linewidth=2, label=f'Moving Avg ({window})')
    ax1.set_title("Sanity Check: Mean Episode Reward")
    ax1.set_ylabel("Reward")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(df['l'], alpha=0.3, color='gray')
    ax2.plot(df['rolling_len'], color='green', linewidth=2)
    ax2.set_title("Episode Length (Survival Time)")
    ax2.set_ylabel("Steps")
    ax2.set_xlabel("Episodes")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    print("Check the plots: Does the Blue Line trend UP or Stabilize?")

if __name__ == "__main__":
    run_sanity_check()