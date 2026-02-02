from .base_agent import Agent, OrderIntent

class MarketMakerAgent(Agent):
    def __init__(self, agent_id, half_spread=0.05, skew_factor=0.01):
        super().__init__(agent_id)
        self.half_spread = half_spread
        self.skew_factor = skew_factor

    def get_action(self, snapshot):
        mid = snapshot.get('mid_price')
        if mid is None: return []

        reservation = mid - (self.inventory * self.skew_factor)
        
        bid = round(reservation - self.half_spread, 2)
        ask = round(reservation + self.half_spread, 2)
        
        return [
            OrderIntent('Buy', bid, 10, 'Limit'),
            OrderIntent('Sell', ask, 10, 'Limit')
        ]