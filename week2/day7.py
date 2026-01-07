import random
import heapq
import itertools
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class OrderIntent:
    def __init__(self, action_type, side, price, qty):
        self.action_type = action_type
        self.side = side
        self.price = price
        self.qty = qty
    def __repr__(self):
        return f"{self.action_type} {self.side} {self.qty} @ {self.price}"

class Agent(ABC):
    def __init__(self, agent_id):
        self.id = agent_id
    @abstractmethod
    def get_action(self, market_snapshot): pass

class FairValueModel:
    def __init__(self, start_price=100.0, volatility=0.1):
        self.current_value = start_price
        self.volatility = volatility
    
    def step(self):
        shock = np.random.normal(0, self.volatility)
        self.current_value += shock
        return self.current_value

class NoiseTrader(Agent):
    def __init__(self, agent_id, fair_value_model, aggression=0.5):
        super().__init__(agent_id)
        self.fv_model = fair_value_model
        self.aggression = aggression 

    def get_action(self, market_snapshot):
        my_valuation = self.fv_model.current_value + np.random.normal(0, 0.05)
        
        side = 'Buy' if random.random() < 0.5 else 'Sell'
        
        spread_guess = 0.10
        if side == 'Buy':
            price = my_valuation - (spread_guess * (1 - self.aggression))
        else:
            price = my_valuation + (spread_guess * (1 - self.aggression))
            
        qty = random.randint(1, 100)
        
        return OrderIntent('Limit', side, round(price, 2), qty)

class SimulationKernel:
    def __init__(self):
        self.time = 0.0
        self.events = [] 
        self.fv_model = FairValueModel(start_price=100.0)
        self.history_fv = []
        self.history_orders = []

    def schedule(self, delay, func):
        heapq.heappush(self.events, (self.time + delay, id(func), func))

    def run(self, duration=60):
        print(f"--- Simulation Start (Duration: {duration}s) ---")
        
        self.schedule(0, self.update_fair_value)
        
        self.schedule(0, self.noise_trader_arrival)
        
        while self.events and self.time < duration:
            t, _, func = heapq.heappop(self.events)
            self.time = t
            func()
            
    def update_fair_value(self):
        new_fv = self.fv_model.step()
        self.history_fv.append((self.time, new_fv))
        
        self.schedule(1.0, self.update_fair_value)

    def noise_trader_arrival(self):
        agent = NoiseTrader("ZITrader", self.fv_model)
        
        snapshot = {} 
        action = agent.get_action(snapshot)
        self.history_orders.append((self.time, action.price))
        
        next_delay = random.expovariate(5.0) 
        self.schedule(next_delay, self.noise_trader_arrival)

if __name__ == "__main__":
    sim = SimulationKernel()
    sim.run(duration=100)
    
    fv_times, fv_prices = zip(*sim.history_fv)
    order_times, order_prices = zip(*sim.history_orders)
    
    plt.figure(figsize=(12, 6))
    plt.plot(fv_times, fv_prices, label="Fair Value (Brownian Motion)", color='black', linewidth=2)
    plt.scatter(order_times, order_prices, label="Noise Trader Orders", color='purple', alpha=0.3, s=10)
    
    plt.title("Day 7: The 'Cloud' of Noise Traders tracking Fair Value")
    plt.xlabel("Time (s)")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()