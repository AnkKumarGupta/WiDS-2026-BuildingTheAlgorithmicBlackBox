from stable_baselines3 import PPO
from day2 import TradingEnv
import numpy as np
import matplotlib.pyplot as plt

def evaluate_agent():
    print("--- Loading Trained Agent ---")
    env = TradingEnv()
    model = PPO.load("ppo_trading_agent")

    obs, _ = env.reset(seed=42)
    done = False

    history_net_worth = []
    history_actions = []
    history_price = []
    history_inventory = []

    print("Running Evaluation Episode...")
    while not done:
        action, _ = model.predict(obs, deterministic=False)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        history_net_worth.append(info['net_worth'])
        history_actions.append(action)
        history_inventory.append(info['inventory'])
        history_price.append(env.last_mid_price)

    # --- VISUALIZATION ---
    print("Generating Analysis Charts...")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Chart 1: Net Worth
    ax1.plot(history_net_worth, label="Net Worth", color='green')
    ax1.set_title("Agent Performance (Net Worth)")
    ax1.grid(True, alpha=0.3)

    # Chart 2: Inventory & Actions
    ax2.plot(history_inventory, label="Inventory", color='blue', alpha=0.6)
    ax2.set_ylabel("Inventory")

    # Overlay Buy/Sell markers
    actions = np.array(history_actions)
    buy_idx = np.where(actions == 1)[0]
    sell_idx = np.where(actions == 2)[0]

    if len(buy_idx) > 0:
        ax2.scatter(buy_idx, [history_inventory[i] for i in buy_idx],
                   marker='^', color='green', label='Buy', zorder=5)
    if len(sell_idx) > 0:
        ax2.scatter(sell_idx, [history_inventory[i] for i in sell_idx],
                   marker='v', color='red', label='Sell', zorder=5)

    ax2.set_title("Inventory Management & Trade Signals")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Chart 3: Price
    ax3.plot(history_price, label="Market Price", color='black', alpha=0.5)
    ax3.set_title("Underlying Market Price")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"Final Net Worth: ${history_net_worth[-1]:.2f}")

    unique, counts = np.unique(actions, return_counts=True)
    print(f"Action Distribution: {dict(zip(unique, counts))}")
    if len(unique) == 1:
        print("WARNING: Agent collapsed to a single action (e.g., Hold).")
    else:
        print("SUCCESS: Agent is taking varied actions.")

if __name__ == "__main__":
    evaluate_agent()