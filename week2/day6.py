from abc import ABC, abstractmethod
import random
import itertools

class OrderIntent:
    def __init__(self, action_type, side, price, qty):
        self.action_type = action_type # 'Limit', 'Market', 'Cancel'
        self.side = side               # 'Buy', 'Sell'
        self.price = price             # Float or None
        self.qty = qty
    
    def __repr__(self):
        return f"[INTENT] {self.action_type} {self.side} {self.qty} @ {self.price}"

class Agent(ABC):
    def __init__(self, agent_id, initial_cash=100000, initial_inventory=0):
        self.id = agent_id
        self.cash = initial_cash
        self.inventory = initial_inventory
        self.active_orders = {} 

    @abstractmethod
    def get_action(self, market_snapshot):
        pass

    def notify_fill(self, side, price, qty):
        if side == 'Buy':
            self.inventory += qty
            self.cash -= price * qty
        else:
            self.inventory -= qty
            self.cash += price * qty
        print(f"Agent {self.id} Filled: {side} {qty} @ {price}")

class RandomAgent(Agent):
    def __init__(self, agent_id, activity_rate=0.1):
        super().__init__(agent_id)
        self.activity_rate = activity_rate 
    def get_action(self, market_snapshot):
        if random.random() > self.activity_rate:
            return None 

        side = 'Buy' if random.random() < 0.5 else 'Sell'
        

        mid = market_snapshot.get('mid_price', 100.0)
        if mid is None or (isinstance(mid, float) and mid != mid): # Check for NaN
            mid = 100.0

        valuation = random.gauss(mid, 0.5)
        
        price = round(valuation, 2)
        qty = random.randint(1, 10)

        return OrderIntent('Limit', side, price, qty)

if __name__ == "__main__":
    bot = RandomAgent("Bot_007", activity_rate=1.0) # 100% active for test

    snapshot = {
        'best_bid': 99.50,
        'best_ask': 100.50,
        'mid_price': 100.00,
        'last_trade': 100.00
    }

    print(f"--- Asking RandomAgent {bot.id} for actions ---")
    for i in range(5):
        action = bot.get_action(snapshot)
        print(f"Tick {i}: {action}")

        if action and action.side == 'Buy':
            bot.notify_fill('Buy', action.price, action.qty)
    
    print(f"\nFinal State -> Cash: ${bot.cash:.2f}, Inventory: {bot.inventory}")