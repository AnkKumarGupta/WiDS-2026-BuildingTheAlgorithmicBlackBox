import numpy as np
import matplotlib.pyplot as plt
from day7 import Agent, OrderIntent, SimulationKernel, FairValueModel, NoiseTrader

class MarketMakerAgent(Agent):
    def __init__(self, agent_id, half_spread=0.5, skew_factor=0.1):
        super().__init__(agent_id)

        self.inventory = 0  
        self.cash = 0
        
        self.half_spread = half_spread
        self.skew_factor = skew_factor 
        self.current_orders = [] 

    def get_action(self, market_snapshot):
        mid = market_snapshot.get('mid_price')
        if mid is None or np.isnan(mid):
             mid = 100.0 

        reservation_price = mid - (self.inventory * self.skew_factor)

        bid_price = round(reservation_price - self.half_spread, 2)
        ask_price = round(reservation_price + self.half_spread, 2)

        actions = []
        bid_intent = OrderIntent('Limit', 'Buy', bid_price, 10)
        ask_intent = OrderIntent('Limit', 'Sell', ask_price, 10)
        
        return [bid_intent, ask_intent]

    def notify_fill(self, side, price, qty):
        if side == 'Buy':
            self.inventory += qty
            self.cash -= price * qty
        else:
            self.inventory -= qty
            self.cash += price * qty

class MultiActionKernel(SimulationKernel):
    def noise_trader_arrival(self):
        agent = NoiseTrader("ZITrader", self.fv_model)
        action = agent.get_action({})
        
        actions = [action] if action else []
        
        for act in actions:
            self.history_orders.append((self.time, act.price))

        next_delay = np.random.exponential(1.0/5.0) # 5 orders/sec
        self.schedule(next_delay, self.noise_trader_arrival)

def run_mm_experiment():
    sim = MultiActionKernel()
    sim.fv_model = FairValueModel(start_price=100.0, volatility=0.05)
    
    mm = MarketMakerAgent("MM_Hero", half_spread=0.20, skew_factor=0.05)
    
    history_mid = []
    history_mm_quotes = [] 
    history_inventory = []

    def mm_update_cycle():
        current_val = sim.fv_model.current_value
        snapshot = {'mid_price': current_val}
        
        actions = mm.get_action(snapshot)
        
        bid = next(a for a in actions if a.side == 'Buy').price
        ask = next(a for a in actions if a.side == 'Sell').price
        
        history_mm_quotes.append((sim.time, bid, ask))
        history_mid.append((sim.time, current_val))
        history_inventory.append((sim.time, mm.inventory))
        
        if np.random.random() < 0.3:
            if np.random.random() < 0.5:
                mm.notify_fill('Buy', bid, 10) 
            else:
                mm.notify_fill('Sell', ask, 10)

        sim.schedule(1.0, mm_update_cycle)

    sim.schedule(0, mm_update_cycle)
    sim.run(duration=100)

    times, mids = zip(*history_mid)
    q_times, bids, asks = zip(*history_mm_quotes)
    inv_times, invs = zip(*history_inventory)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(times, mids, label="Fair Value", color='black', alpha=0.5)
    ax1.plot(q_times, bids, label="MM Bid", color='green', linestyle='--')
    ax1.plot(q_times, asks, label="MM Ask", color='red', linestyle='--')
    ax1.set_title("Market Maker Quoting Behavior (Skewing)")
    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(inv_times, invs, label="MM Inventory", color='purple')
    ax2.axhline(0, color='black', linestyle=':', alpha=0.5)
    ax2.set_title("Inventory Management")
    ax2.set_ylabel("Net Position (Qty)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_mm_experiment()