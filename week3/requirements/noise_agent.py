import random
import numpy as np
from .base_agent import Agent, OrderIntent

class NoiseTrader(Agent):
    def __init__(self, agent_id, fair_value_model):
        super().__init__(agent_id)
        self.fv = fair_value_model

    def get_action(self, snapshot):
        val = self.fv.current_value + np.random.normal(0, 0.5)
        
        side = 'Buy' if random.random() < 0.5 else 'Sell'
        qty = random.randint(1, 10)
        
        if random.random() < 0.5:
            spread = 0.20
            price = val - (spread/2) if side == 'Buy' else val + (spread/2)
            return [OrderIntent(side, round(price, 2), qty, 'Limit')]
        else:
            return [OrderIntent(side, None, qty, 'Market')]