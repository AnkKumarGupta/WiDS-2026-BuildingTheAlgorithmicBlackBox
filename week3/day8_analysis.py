import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from day2 import TradingEnv
from requirements.matching_engine import Order
from requirements.base_agent import Agent, OrderIntent

class HerdAgent(Agent):
    def __init__(self, id):
        super().__init__(id)
        self.last_price = 100.0
        self.inventory = 1000
        self.max_inventory = 100000

    def get_action(self, market_snapshot):
        curr_price = market_snapshot.get('mid_price')
        if not curr_price: return []

        ret = (curr_price - self.last_price) / self.last_price
        self.last_price = curr_price

        orders = []
        if ret < -0.005:
            orders.append(OrderIntent('Sell', price=None, qty=100))

        return orders

def run_simulation():

    env = TradingEnv()
    env.transaction_cost = 0.0
    env.reset(seed=42)

    env.background_agents = []
    trackable_agents = []

    print("Injecting 20 Herd Agents (Stockholders)...")
    for i in range(20):
        agent = HerdAgent(f"Herd_{i}")
        agent.inventory = 1000
        env.background_agents.append(agent)
        trackable_agents.append(agent)

    class LiquidityWall(Agent):
        def get_action(self, s):
            return [OrderIntent('Buy', price=95.0, qty=10000)]
    env.background_agents.append(LiquidityWall("Wall"))

    pos_data = []
    price_data = []

    print("Running Simulation...")
    for step in range(300):

        if step == 150:
            print("\nCRASHING PRICE")
            env.last_mid_price = 90.0

        env.step(0)

        for trade in env.engine.trades:
            if trade.timestamp == step or step == 151:
                for ag in trackable_agents:
                    if ag.id == trade.seller_id:
                        ag.inventory -= trade.qty
                    if ag.id == trade.buyer_id:
                        ag.inventory += trade.qty

        row = {'step': step}
        for ag in trackable_agents:
            row[ag.id] = ag.inventory
        pos_data.append(row)
        price_data.append({'step': step, 'price': env.last_mid_price})

    print("Generating Proof...")
    df_pos = pd.DataFrame(pos_data).set_index('step')
    df_price = pd.DataFrame(price_data).set_index('step')

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Price
    ax1.plot(df_price.index, df_price['price'], color='black')
    ax1.axvline(x=150, color='red', linestyle='--', label='Crash')
    ax1.set_title("Market Price")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Inventory
    for col in df_pos.columns[:5]:
        ax2.plot(df_pos.index, df_pos[col], alpha=0.6)
    ax2.set_title("Herd Inventories (MUST DROP from 1000)")
    ax2.grid(True, alpha=0.3)

    # Correlation
    window = 20
    corrs = []
    steps = df_pos.index
    for i in range(window, len(steps), 5):
        w = df_pos.iloc[i-window:i]
        # Check variance
        if w.std().sum() > 0:
            c = w.corr().mean().mean()
            corrs.append({'step': steps[i], 'corr': c})
        else:
            corrs.append({'step': steps[i], 'corr': 0})

    df_corr = pd.DataFrame(corrs)
    if not df_corr.empty:
        ax3.plot(df_corr['step'], df_corr['corr'], color='red')
    ax3.set_title("Herding Correlation")
    ax3.set_ylim(-0.1, 1.1)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation()