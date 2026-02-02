import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from day2 import TradingEnv
from requirements.market_maker_agent import MarketMakerAgent
from requirements.noise_agent import NoiseTrader
from requirements.matching_engine import Order
import os
import gc 

def run_market_simulation():
    print("--- PHASE II: MULTI-AGENT MARKET SIMULATION (OPTIMIZED) ---")

    env = TradingEnv()
    env.transaction_cost = 0.0001
    env.risk_aversion = 0.01
    env.inventory_penalty = 0.001

    model = None
    if os.path.exists("ppo_trading_agent_pro.zip"):
        model = PPO.load("ppo_trading_agent_pro")
        print("Loaded 'Pro' RL Agent.")
    else:
        print("Warning: Running without RL Agent.")

    env.reset(seed=101)
    env.background_agents = [] 
    for i in range(50):
        env.background_agents.append(NoiseTrader(f"Noise_{i}", env.fv))
    for i in range(10):
        env.background_agents.append(MarketMakerAgent(f"MM_{i}"))

    print(f"Total Agents: {len(env.background_agents)}")

    lob_file = "lob_data.csv"
    mid_file = "mid_prices.csv"

    with open(lob_file, "w") as f:
        f.write("step,price,vol,side\n")

    mid_prices = []

    lob_buffer = []

    print("Starting 5,000 step run with Batch Saving...")

    obs, _ = env.reset(seed=101)

    for step in range(5000):
        if model:
            action, _ = model.predict(obs, deterministic=False)
        else:
            action = 0

        obs, reward, terminated, truncated, info = env.step(action)

        for price, _, order in env.engine.bids:
            lob_buffer.append((step, -price, order.qty, 'Bid'))

        for price, _, order in env.engine.asks:
            lob_buffer.append((step, price, order.qty, 'Ask'))

        mid_prices.append({'step': step, 'price': env.last_mid_price})

        if step % 100 == 0:
            print(f"Step {step}: Mid {env.last_mid_price:.2f} | Flushing {len(lob_buffer)} rows...")

            df_chunk = pd.DataFrame(lob_buffer, columns=['step', 'price', 'vol', 'side'])

            df_chunk.to_csv(lob_file, mode='a', header=False, index=False)

            del df_chunk
            lob_buffer = []
            gc.collect() 

    if lob_buffer:
        df_chunk = pd.DataFrame(lob_buffer, columns=['step', 'price', 'vol', 'side'])
        df_chunk.to_csv(lob_file, mode='a', header=False, index=False)
        del df_chunk

    pd.DataFrame(mid_prices).to_csv(mid_file, index=False)

    print("Simulation Complete. Data saved incrementally to lob_data.csv")

if __name__ == "__main__":
    run_market_simulation()