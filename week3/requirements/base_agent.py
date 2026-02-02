from abc import ABC, abstractmethod

class OrderIntent:
    def __init__(self, side, price, qty, action_type='Limit'):
        self.side = side
        self.price = price
        self.qty = qty
        self.action_type = action_type # 'Limit', 'Market', 'Cancel'

class Agent(ABC):
    def __init__(self, agent_id):
        self.id = agent_id
        self.cash = 100000
        self.inventory = 0
        
    @abstractmethod
    def get_action(self, snapshot):
        pass

    def notify_fill(self, side, price, qty):
        if side == 'Buy':
            self.inventory += qty
            self.cash -= price * qty
        else:
            self.inventory -= qty
            self.cash += price * qty