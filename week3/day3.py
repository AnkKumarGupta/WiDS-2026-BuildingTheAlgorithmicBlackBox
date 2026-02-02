from day2 import TradingEnv
import numpy as np
import matplotlib.pyplot as plt

def validate_reward_function():
    print("--- Validating Risk-Aware Reward ---")
    env = TradingEnv()
    env.reset(seed=42)
    
    history_pnl = []
    history_reward = []
    history_penalty = []
    history_inv = []
    
    print("Simulating a 'Reckless Bull' (Always Buy)...")
    
    for _ in range(200):

        obs, reward, term, trunc, info = env.step(1)
        
        history_pnl.append(info['step_pnl'])
        history_reward.append(reward)
        history_penalty.append(info['penalty'])
        history_inv.append(info['inventory'])
        
        if term or trunc: break
        
    # --- VISUALIZATION ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot 1: Cumulative PnL vs Cumulative Reward
    ax1.plot(np.cumsum(history_pnl), label="Raw PnL (Profit)", color='green', linewidth=2)
    ax1.plot(np.cumsum(history_reward), label="Risk-Adjusted Reward", color='blue', linestyle='--')
    ax1.set_title("PnL vs. Reward (The Risk Penalty Gap)")
    ax1.set_ylabel("Value")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Inventory vs Penalty Size
    ax2_twin = ax2.twinx()
    ax2.plot(history_inv, label="Inventory Level", color='green', alpha=0.6)
    ax2_twin.plot(history_penalty, label="Risk Penalty", color='red')
    
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Inventory (Shares)", color='green')
    ax2_twin.set_ylabel("Penalty Value (Reward Deduction)", color='red')
    ax2.set_title("Mechanism Check: Inventory Penalty Growth")
    
    plt.tight_layout()
    plt.show()
    
    print("Validation Criteria:")
    print("1. Blue Line (Reward) < Green Line (PnL).")
    print("2. Red Line (Penalty) spikes as Inventory grows.")

if __name__ == "__main__":
    validate_reward_function()