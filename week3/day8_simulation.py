import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from day2 import TradingEnv
from requirements.market_maker_agent import MarketMakerAgent
from requirements.noise_agent import NoiseTrader
from requirements.momentum_agent import MomentumTrader
import os

def run_herding_simulation():
    print("--- DAY 8: HERDING DATA COLLECTION (CORRECTED) ---")

    env = TradingEnv()
    env.transaction_cost = 0.0001
    env.risk_aversion = 0.01
    env.inventory_penalty = 0.001

    model = None
    if os.path.exists("ppo_trading_agent_pro.zip"):
        model = PPO.load("ppo_trading_agent_pro")
        print("Loaded 'Pro' RL Agent.")
    else:
        print("Warning: RL Agent not found. Running passive.")

    env.reset(seed=101)
    env.background_agents = []
    trackable_agents = []

    print("Injecting 15 Momentum Traders to trigger herding...")
    for i in range(15):
        agent = MomentumTrader(f"Mom_{i}", lookback=10, panic_threshold=0.05)
        env.background_agents.append(agent)
        trackable_agents.append(agent)

    for i in range(15):
        agent = NoiseTrader(f"Noise_{i}", env.fv)
        env.background_agents.append(agent)

    for i in range(5):
        mm = MarketMakerAgent(f"MM_{i}")
        env.background_agents.append(mm)

    print(f"Tracking positions for {len(trackable_agents)} Momentum Agents.")

    position_history = []
    price_history = []

    obs, _ = env.reset(seed=101)

    print("Running 5,000 steps...")
    for step in range(5000):
        if model:
            action, _ = model.predict(obs, deterministic=False)
        else:
            action = 0

        obs, reward, terminated, truncated, info = env.step(action)

        snapshot = {'step': step}

        snapshot['RL_Agent'] = env.rl_inventory

        for agent in trackable_agents:
            snapshot[agent.id] = agent.inventory

        position_history.append(snapshot)
        price_history.append({'step': step, 'price': env.last_mid_price})

        if step % 1000 == 0:
            print(f"Step {step}...")

    print("Saving Data...")
    df_pos = pd.DataFrame(position_history)
    df_pos.to_csv("agent_positions.csv", index=False)

    df_price = pd.DataFrame(price_history)
    df_price.to_csv("herding_prices.csv", index=False)
    print("Done. Saved agent_positions.csv and herding_prices.csv")

if __name__ == "__main__":
    run_herding_simulation()