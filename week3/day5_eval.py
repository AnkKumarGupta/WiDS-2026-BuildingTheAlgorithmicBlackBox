from stable_baselines3 import PPO
from day2 import TradingEnv
import matplotlib.pyplot as plt
import numpy as np

def evaluate_pro_agent():
    print("--- EVALUATING 'PRO' AGENT ---")

    env = TradingEnv()
    env.transaction_cost = 0.0001
    env.risk_aversion = 0.05
    env.inventory_penalty = 0.01

    model = PPO.load("ppo_trading_agent_pro")

    obs, _ = env.reset(seed=42)
    done = False

    history_inv = []
    history_net_worth = []
    history_price = []

    while not done:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        history_inv.append(info['inventory'])
        history_net_worth.append(info['net_worth'])
        history_price.append(env.last_mid_price)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    ax1.plot(history_net_worth, color='green')
    ax1.set_title("Pro Agent: Net Worth (Risk Adjusted)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(history_inv, color='blue')
    ax2.set_title("Pro Agent: Inventory Control")
    ax2.set_ylabel("Shares Held")
    ax2.grid(True, alpha=0.3)

    ax3.plot(history_price, color='black', alpha=0.5)
    ax3.set_title("Market Price")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"Final Net Worth: ${history_net_worth[-1]:.2f}")

if __name__ == "__main__":
    evaluate_pro_agent()