import collections
import numpy as np
import matplotlib.pyplot as plt
from day7 import Agent, OrderIntent, SimulationKernel, FairValueModel, NoiseTrader

class MomentumAgent(Agent):
    def __init__(self, agent_id, window_size=20, aggression=1.0):
        super().__init__(agent_id)
        self.window_size = window_size
        self.aggression = aggression
        self.price_history = collections.deque(maxlen=window_size)

    def get_action(self, market_snapshot):
        mid = market_snapshot.get('mid_price')
        if mid is None or np.isnan(mid):
            return None 
        
        self.price_history.append(mid)

        if len(self.price_history) < self.window_size:
            return None 

        sma = sum(self.price_history) / len(self.price_history)
        
        deviation = mid - sma
        threshold = 0.05 

        side = None
        if deviation > threshold:
            side = 'Buy'  
        elif deviation < -threshold:
            side = 'Sell' 
        
        if side:
            price_offset = 0.5 * self.aggression
            price = mid + price_offset if side == 'Buy' else mid - price_offset
            qty = 10 
            
            return OrderIntent('Limit', side, round(price, 2), qty)
        
        return None

def run_momentum_experiment():
    sim = SimulationKernel()
    
    sim.fv_model = FairValueModel(start_price=100.0, volatility=0.1)
    
    for i in range(2):
        sim.schedule(0, lambda: sim.noise_trader_arrival()) 

    momentum_agents = []
    for i in range(10):
        window = np.random.randint(10, 30)
        ma = MomentumAgent(f"Momo_{i}", window_size=window)
        momentum_agents.append(ma)

    history_price = []
    history_sma = [] 

    def trigger_momentum_agents():
        market_price = sim.fv_model.current_value + np.random.normal(0, 0.1)
        snapshot = {'mid_price': market_price}
        
        history_price.append((sim.time, market_price))

        for ma in momentum_agents:
            action = ma.get_action(snapshot)
            if action:
                impact = 0.05 * (1 if action.side == 'Buy' else -1)
                sim.fv_model.current_value += impact

        if len(momentum_agents[0].price_history) == momentum_agents[0].window_size:
            sma = sum(momentum_agents[0].price_history) / momentum_agents[0].window_size
            history_sma.append((sim.time, sma))
        
        sim.schedule(1.0, trigger_momentum_agents)

    sim.schedule(1.0, trigger_momentum_agents)
    
    sim.run(duration=200)

    times, prices = zip(*history_price)
    sma_times, sma_values = zip(*history_sma) if history_sma else ([], [])

    plt.figure(figsize=(12, 6))
    plt.plot(times, prices, label="Market Price", color='black', alpha=0.6)
    plt.plot(sma_times, sma_values, label="Momentum SMA (Agent 0)", color='blue', linestyle='--')
    
    plt.title("The 'Pump and Dump' Feedback Loop")
    plt.xlabel("Time (s)")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_momentum_experiment()