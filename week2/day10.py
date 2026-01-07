import heapq
import itertools
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import collections


class OrderIntent:
    def __init__(self, action_type, side, price, qty):
        self.action_type = action_type
        self.side = side
        self.price = price
        self.qty = qty

class Agent(ABC):
    def __init__(self, agent_id):
        self.id = agent_id
        self.inventory = 0
        self.cash = 0
    
    @abstractmethod
    def get_action(self, market_snapshot): pass

    def notify_fill(self, side, price, qty):
        if side == 'Buy':
            self.inventory += qty
            self.cash -= price * qty
        else:
            self.inventory -= qty
            self.cash += price * qty

class SimulationKernel:
    def __init__(self):
        self.time = 0.0
        self.events = [] 
        self.history_mid = []
        self.history_spread = []
        self.current_mid = 100.0 # Proxy for L1 Mid
        self.current_spread = 0.0 # Proxy for L1 Spread

    def schedule(self, delay, func):
        heapq.heappush(self.events, (self.time + delay, id(func), func))

    def run(self, duration=100):
        while self.events and self.time < duration:
            t, _, func = heapq.heappop(self.events)
            self.time = t
            func()

    def record_metrics(self, mid, spread):
        self.current_mid = mid
        self.current_spread = spread
        self.history_mid.append((self.time, mid))
        self.history_spread.append((self.time, spread))


class FairValueModel:
    def __init__(self, start_price=100.0, volatility=0.1):
        self.current_value = start_price
        self.volatility = volatility
    
    def step(self):
        self.current_value += np.random.normal(0, self.volatility)
        return self.current_value

class NoiseTrader(Agent):
    def __init__(self, agent_id, fv_model):
        super().__init__(agent_id)
        self.fv_model = fv_model

    def get_action(self, snapshot):
        valuation = self.fv_model.current_value + np.random.normal(0, 0.5)
        side = 'Buy' if random.random() < 0.5 else 'Sell'
        
        spread = 0.50 
        price = valuation - (spread/2) if side == 'Buy' else valuation + (spread/2)
        return [OrderIntent('Limit', side, round(price, 2), 10)]

class MarketMakerAgent(Agent):
    def __init__(self, agent_id, half_spread=0.10, skew_factor=0.1):
        super().__init__(agent_id)
        self.half_spread = half_spread
        self.skew_factor = skew_factor

    def get_action(self, snapshot):
        mid = snapshot.get('mid_price', 100.0)
        # Skew: Lower price if Long, Raise if Short
        reservation = mid - (self.inventory * self.skew_factor)
        
        bid = round(reservation - self.half_spread, 2)
        ask = round(reservation + self.half_spread, 2)
        
        return [
            OrderIntent('Limit', 'Buy', bid, 10),
            OrderIntent('Limit', 'Sell', ask, 10)
        ]

class MomentumAgent(Agent):
    def __init__(self, agent_id, window=20):
        super().__init__(agent_id)
        self.history = collections.deque(maxlen=window)
        self.window = window

    def get_action(self, snapshot):
        mid = snapshot.get('mid_price', 100.0)
        self.history.append(mid)
        
        if len(self.history) < self.window: return []

        sma = sum(self.history) / len(self.history)
        deviation = mid - sma
        
        if deviation > 0.2:
            return [OrderIntent('Limit', 'Buy', mid + 0.5, 20)] # Cross spread
        elif deviation < -0.2:
            return [OrderIntent('Limit', 'Sell', mid - 0.5, 20)]
        return []


def run_scenario(scenario_name, agents, duration=200):
    print(f"Running Scenario: {scenario_name}...")
    np.random.seed(42) 
    random.seed(42)
    
    sim = SimulationKernel()
    fv = FairValueModel(100.0, 0.05)
    
    
    def market_step():
        true_val = fv.step()
        
        bids = []
        asks = []
        snapshot = {'mid_price': sim.current_mid}
        
        momentum_pressure = 0
        
        for agent in agents:
            actions = agent.get_action(snapshot)
            for action in actions:
                if action.side == 'Buy': bids.append(action.price)
                if action.side == 'Sell': asks.append(action.price)
                
                if isinstance(agent, MomentumAgent):
                    if action.side == 'Buy': momentum_pressure += 0.1
                    if action.side == 'Sell': momentum_pressure -= 0.1

        if bids and asks:
            best_bid = max(bids)
            best_ask = min(asks)
            
            fv.current_value += momentum_pressure
            
            if best_bid >= best_ask:
                exec_price = (best_bid + best_ask) / 2
                spread = 0.0 
                
                for agent in agents:
                    if isinstance(agent, MarketMakerAgent):
                        agent.notify_fill('Buy' if np.random.random() < 0.5 else 'Sell', exec_price, 10)
            else:
                exec_price = (best_bid + best_ask) / 2
                spread = best_ask - best_bid
        else:
            exec_price = sim.current_mid
            spread = 1.0 
            
        sim.record_metrics(exec_price, spread)
        sim.schedule(1.0, market_step)

    sim.schedule(0, market_step)
    sim.run(duration)
    return sim.history_mid, sim.history_spread


fv_model = FairValueModel()
noise_traders = [NoiseTrader(f"Noise_{i}", fv_model) for i in range(10)]

res_a_mid, res_a_spread = run_scenario("A (Noise Only)", noise_traders)

mms = [MarketMakerAgent(f"MM_{i}") for i in range(3)]
res_b_mid, res_b_spread = run_scenario("B (Noise + MM)", noise_traders + mms)

momos = [MomentumAgent(f"Momo_{i}") for i in range(5)]
res_c_mid, res_c_spread = run_scenario("C (Noise + Momo)", noise_traders + momos)


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

times_a, prices_a = zip(*res_a_mid)
times_b, prices_b = zip(*res_b_mid)
times_c, prices_c = zip(*res_c_mid)

ax1.plot(times_a, prices_a, label="Scenario A: Noise Only", color='gray', alpha=0.6)
ax1.plot(times_b, prices_b, label="Scenario B: Noise + MM", color='green', linewidth=2)
ax1.plot(times_c, prices_c, label="Scenario C: Noise + Momentum", color='red', linestyle='--')
ax1.set_title("Emergent Price Behavior by Agent Mix")
ax1.set_ylabel("Price")
ax1.legend()
ax1.grid(True, alpha=0.3)

_, spread_a = zip(*res_a_spread)
_, spread_b = zip(*res_b_spread)
_, spread_c = zip(*res_c_spread)

def smooth(data, w=5): return pd.Series(data).rolling(w).mean()

ax2.plot(times_a, smooth(spread_a), label="Scenario A Spread", color='gray', alpha=0.6)
ax2.plot(times_b, smooth(spread_b), label="Scenario B Spread", color='green')
ax2.plot(times_c, smooth(spread_c), label="Scenario C Spread", color='red', linestyle='--')
ax2.set_title("Market Liquidity (Spread) Evolution")
ax2.set_ylabel("Bid-Ask Spread")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

def calc_vol(data): return np.std(np.diff(data))
print("\n=== MARKET REPORT METRICS ===")
print(f"Scenario A (Noise): Avg Spread={np.mean(spread_a):.4f} | Volatility={calc_vol(prices_a):.4f}")
print(f"Scenario B (+ MM):  Avg Spread={np.mean(spread_b):.4f} | Volatility={calc_vol(prices_b):.4f}")
print(f"Scenario C (+ Mo):  Avg Spread={np.mean(spread_c):.4f} | Volatility={calc_vol(prices_c):.4f}")