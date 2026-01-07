import heapq
import random
import numpy as np
import matplotlib.pyplot as plt

class Order:
    def __init__(self, agent_id, side, price, qty, timestamp):
        self.agent_id, self.side, self.price, self.qty, self.timestamp = agent_id, side, price, qty, timestamp

    def __repr__(self):
        return f"[{self.side}] {self.qty} @ ${self.price:.2f}"

    def __lt__(self, other):
        return self.timestamp < other.timestamp

class Trade:
    def __init__(self, price, qty, timestamp, buyer_id, seller_id):
        self.price, self.qty, self.timestamp, self.buyer, self.seller = price, qty, timestamp, buyer_id, seller_id

class OrderBook:
    def __init__(self, tick_size=0.01):
        self.tick_size = tick_size
        self.bids = [] 
        self.asks = [] 
        self.trades = []

    def format_price(self, price):
        return round(price / self.tick_size) * self.tick_size

    def add_order(self, order):
        order.price = self.format_price(order.price)
        if order.side == 'Buy':
            self._match_buy(order)
        else:
            self._match_sell(order)

    def _match_buy(self, order):
        while self.asks and order.qty > 0 and order.price >= self.asks[0][0]:
            best_ask_price, _, ask_order = self.asks[0]
            qty = min(order.qty, ask_order.qty)
            self._execute_trade(order, ask_order, best_ask_price, qty)
            order.qty -= qty
            ask_order.qty -= qty
            if ask_order.qty == 0: heapq.heappop(self.asks)
        if order.qty > 0:
            heapq.heappush(self.bids, (-order.price, order.timestamp, order))

    def _match_sell(self, order):
        while self.bids and order.qty > 0 and order.price <= -self.bids[0][0]:
            best_bid = self.bids[0]
            price, _, bid_order = -best_bid[0], best_bid[1], best_bid[2]
            qty = min(order.qty, bid_order.qty)
            self._execute_trade(bid_order, order, price, qty)
            order.qty -= qty
            bid_order.qty -= qty
            if bid_order.qty == 0: heapq.heappop(self.bids)
        if order.qty > 0:
            heapq.heappush(self.asks, (order.price, order.timestamp, order))

    def _execute_trade(self, buyer, seller, price, qty):
        self.trades.append(Trade(price, qty, buyer.timestamp, buyer.agent_id, seller.agent_id))

    def get_l1_snapshot(self):
        best_bid = -self.bids[0][0] if self.bids else 0.0
        best_ask = self.asks[0][0] if self.asks else 0.0
        return {'best_bid': best_bid, 'best_ask': best_ask}

class MarketEnvironment:
    def __init__(self):
        self.book = OrderBook()
        self.clock = 0
        self.agents = []

    def reset(self):
        self.book = OrderBook()
        self.clock = 0
        self.agents = []
        return self.get_observation()

    def add_agent(self, agent):
        self.agents.append(agent)

    def step(self):
        self.clock += 1
        
        # Agents take actions
        obs = self.get_observation()
        
        for agent in self.agents:
            action = agent.get_action(obs)
            if action:
                self.book.add_order(action)
        
        return obs

    def get_observation(self):
        return {
            'time': self.clock,
            'l1': self.book.get_l1_snapshot(),
            'last_trade': self.book.trades[-1].price if self.book.trades else 100.0
        }

class MarketMaker:
    def __init__(self, agent_id, env):
        self.id = agent_id
        self.env = env

    def get_action(self, obs):
        mid = obs['last_trade']
        spread = 0.5
        if random.random() < 0.5:
            return Order(self.id, 'Buy', mid - spread/2, 10, self.env.clock)
        else:
            return Order(self.id, 'Sell', mid + spread/2, 10, self.env.clock)

env = MarketEnvironment()
env.reset()

for i in range(20):
    env.add_agent(MarketMaker(f"MM_{i}", env))

price_history = []
for _ in range(200):
    state = env.step()
    price_history.append(state['last_trade'])

plt.plot(price_history)
plt.title("Day 5 Integration: Market Maker Simulation")
plt.xlabel("Time Step")
plt.ylabel("Price")
plt.show()