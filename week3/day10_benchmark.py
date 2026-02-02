import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from day2 import TradingEnv
from requirements.market_maker_agent import MarketMakerAgent
from requirements.noise_agent import NoiseTrader
from requirements.momentum_agent import MomentumTrader

def calculate_sharpe(returns):
    if np.std(returns) == 0: return 0.0
    return np.mean(returns) / np.std(returns) * np.sqrt(252)

def calculate_max_drawdown(wealth_curve):
    peak = wealth_curve[0]
    max_dd = 0.0
    for value in wealth_curve:
        if value > peak:
            peak = value
        dd = (peak - value) / peak
        if dd > max_dd:
            max_dd = dd
    return -max_dd

def run_strategy(env, agent_type, model=None):
    """Runs a single episode for a specific strategy"""
    obs, _ = env.reset(seed=42)

    env.background_agents = []
    for i in range(20): env.background_agents.append(NoiseTrader(f"Noise_{i}", env.fv))
    for i in range(5): env.background_agents.append(MarketMakerAgent(f"MM_{i}"))

    mom_history = []
    wealth_history = []

    terminated = False
    step = 0

    while not terminated and step < 2000:
        action = 0 

        if agent_type == 'RL':
            action, _ = model.predict(obs, deterministic=True)

        elif agent_type == 'Random':
            action = env.action_space.sample()

        elif agent_type == 'BuyHold':
            if step == 0: action = 1 # Buy once
            else: action = 0 # Hold forever

        elif agent_type == 'Momentum':
            price = env.last_mid_price
            mom_history.append(price)
            if len(mom_history) > 5:
                # 5-step return
                ret = (price - mom_history[-5]) / mom_history[-5]
                if ret > 0.001: action = 1 # Buy
                elif ret < -0.001: action = 2 # Sell

        obs, reward, terminated, truncated, info = env.step(action)

        current_wealth = env.rl_cash + (env.rl_inventory * env.last_mid_price)

        wealth_history.append(current_wealth)
        step += 1

    return np.array(wealth_history)

def run_benchmark():
    print("--- DAY 10: THE ALPHA TEST ---")

    print("1. Training RL Agent with Golden Config...")
    env = TradingEnv()

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.00045,
        gamma=0.92,
        ent_coef=1e-5,
        batch_size=256,
        policy_kwargs={"net_arch": [64, 64]},
        verbose=0
    )
    model.learn(total_timesteps=20000)
    print("   Training Complete.")

    strategies = ['BuyHold', 'Random', 'Momentum', 'RL']
    results = {}
    curves = {}

    print("2. Running Simulations (Same Market Conditions)...")
    for strat in strategies:
        print(f"   Testing: {strat}...")
        curve = run_strategy(env, strat, model)
        curves[strat] = curve

        returns = np.diff(curve) / curve[:-1]
        if len(returns) == 0: returns = [0]

        sharpe = calculate_sharpe(returns)
        mdd = calculate_max_drawdown(curve)
        final_return = (curve[-1] - curve[0]) / curve[0]

        results[strat] = {
            'Sharpe': sharpe,
            'MaxDD': mdd,
            'Return': final_return
        }

    print("\n--- PERFORMANCE REPORT ---")
    df_res = pd.DataFrame(results).T
    print(df_res)

    plt.figure(figsize=(12, 6))
    for strat, curve in curves.items():
        if len(curve) > 0:
            norm_curve = curve / curve[0] * 100
            plt.plot(norm_curve, label=f"{strat}")

    plt.title("Equity Curves: RL vs Baselines")
    plt.ylabel("Portfolio Value (Normalized)")
    plt.xlabel("Steps")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    rl_sharpe = results['RL']['Sharpe']
    bh_sharpe = results['BuyHold']['Sharpe']

    print("\n--- FINAL VERDICT ---")
    if rl_sharpe > bh_sharpe and rl_sharpe > 0:
        print("SUCCESS: Alpha Detected. The agent beats the market.")
    elif rl_sharpe > results['Random']['Sharpe']:
        print("MIXED: Beats Random, but loses to Buy & Hold. Needs better reward function.")
    else:
        print("FAILURE: Agent failed to learn. Underperforms random noise.")

if __name__ == "__main__":
    run_benchmark()