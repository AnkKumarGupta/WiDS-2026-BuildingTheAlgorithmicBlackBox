# from collections import deque
# from .base_agent import Agent, OrderIntent

# class MomentumAgent(Agent):
#     def __init__(self, agent_id, window=50):
#         super().__init__(agent_id)
#         self.history = deque(maxlen=window)
#         self.window = window

#     def get_action(self, snapshot):
#         mid = snapshot.get('mid_price')
#         if mid is None: return []
        
#         self.history.append(mid)
        
#         if len(self.history) < self.window: return []
        
#         sma = sum(self.history) / len(self.history)
        
#         if mid > sma:
#             return [OrderIntent('Buy', None, 10, 'Market')]
#         elif mid < sma:
#             return [OrderIntent('Sell', None, 10, 'Market')]
        
#         return []



from .base_agent import Agent, OrderIntent

class MomentumTrader(Agent):
    def __init__(self, id, lookback=5, panic_threshold=0.0):
        super().__init__(id)
        self.lookback = lookback
        self.price_history = []
        self.panic_threshold = panic_threshold
        # FORCE HUGE LIMITS so they don't stop trading
        self.max_inventory = 100000 
        self.inventory_limit = 100000

    def get_action(self, market_snapshot):
        mid_price = market_snapshot.get('mid_price')
        if mid_price is None: return []
        
        self.price_history.append(mid_price)
        if len(self.price_history) > self.lookback:
            self.price_history.pop(0)
            
        if len(self.price_history) < self.lookback:
            return []

        start_price = self.price_history[0]
        end_price = self.price_history[-1]
        price_change = end_price - start_price
        
        orders = []
        
        # Hyper-sensitive logic
        if price_change < -0.0001: 
            # Sell 10 units
            orders.append(OrderIntent('Sell', price=None, qty=10))
            
        elif price_change > 0.0001:
            # Buy 10 units
            orders.append(OrderIntent('Buy', price=None, qty=10))
            
        return orders