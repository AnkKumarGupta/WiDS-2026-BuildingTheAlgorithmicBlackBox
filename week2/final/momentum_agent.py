from collections import deque
from base_agent import Agent, OrderIntent

class MomentumAgent(Agent):
    def __init__(self, agent_id, window=50):
        super().__init__(agent_id)
        self.history = deque(maxlen=window)
        self.window = window

    def get_action(self, snapshot):
        mid = snapshot.get('mid_price')
        if mid is None: return []
        
        self.history.append(mid)
        
        if len(self.history) < self.window: return []
        
        sma = sum(self.history) / len(self.history)
        
        if mid > sma:
            return [OrderIntent('Buy', None, 10, 'Market')]
        elif mid < sma:
            return [OrderIntent('Sell', None, 10, 'Market')]
        
        return []