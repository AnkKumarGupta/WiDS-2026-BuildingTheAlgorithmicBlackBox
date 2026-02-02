import heapq
import random
import matplotlib.pyplot as plt
import numpy as np


class Order:
    def __init__(self, agent_id, side, price, qty, timestamp):
        self.agent_id = agent_id
        self.side = side 
        self.price = price
        self.qty = qty
        self.timestamp = timestamp

    def __repr__(self):
        return f"[T{self.timestamp}] Agent {self.agent_id} {self.side} {self.qty} @ {self.price:.2f}"

class Trade:
    def __init__(self, price, qty, timestamp, buyer_id, seller_id):
        self.price = price
        self.qty = qty
        self.timestamp = timestamp
        self.buyer_id = buyer_id
        self.seller_id = seller_id


class OrderBook:
    def __init__(self, tick_size=0.01):
        self.tick_size = tick_size
        self.bids = [] 
        self.asks = []
        self.trades = [] 

    def add_order(self, order):
        order.price = round(order.price / self.tick_size) * self.tick_size
        
        if order.side == 'Buy':
            self._match_buy(order)
        else:
            self._match_sell(order)

    def _match_buy(self, order):
        while self.asks and order.qty > 0 and order.price >= self.asks[0][0]:
            best_ask_price, _, ask_order = self.asks[0]
            
            trade_qty = min(order.qty, ask_order.qty)
            self._execute_trade(order, ask_order, best_ask_price, trade_qty)
            
            order.qty -= trade_qty
            ask_order.qty -= trade_qty
            
            if ask_order.qty == 0:
                heapq.heappop(self.asks)
        
        if order.qty > 0:
            heapq.heappush(self.bids, (-order.price, order.timestamp, order))

    def _match_sell(self, order):
        while self.bids and order.qty > 0 and order.price <= -self.bids[0][0]:
            best_bid_price_neg, _, bid_order = self.bids[0]
            best_bid_price = -best_bid_price_neg
            
            trade_qty = min(order.qty, bid_order.qty)
            self._execute_trade(bid_order, order, best_bid_price, trade_qty)
            
            order.qty -= trade_qty
            bid_order.qty -= trade_qty
            
            if bid_order.qty == 0:
                heapq.heappop(self.bids)
                
        if order.qty > 0:
            heapq.heappush(self.asks, (order.price, order.timestamp, order))

    def _execute_trade(self, buy_order, sell_order, price, qty):
        t = Trade(price, qty, buy_order.timestamp, buy_order.agent_id, sell_order.agent_id)
        self.trades.append(t)

    def get_mid_price(self, default=100.0):
        best_bid = -self.bids[0][0] if self.bids else None
        best_ask = self.asks[0][0] if self.asks else None
        
        if best_bid and best_ask:
            return (best_bid + best_ask) / 2
        return best_bid if best_bid else (best_ask if best_ask else default)


class Agent:
    def __init__(self, agent_id, env):
        self.agent_id = agent_id
        self.env = env 

    def act(self):
        """ The 'Brain' of the agent. Overwritten by subclasses. """
        pass

class NoiseTrader(Agent):
    def __init__(self, agent_id, env, arrival_rate=0.1):
        super().__init__(agent_id, env)
        self.arrival_rate = arrival_rate

    def act(self):
        if random.random() < self.arrival_rate:
            side = 'Buy' if random.random() < 0.5 else 'Sell'
            current_price = self.env.order_book.get_mid_price()
            
            if side == 'Buy':
                price = current_price + random.uniform(0.01, 1.0)
            else:
                price = current_price - random.uniform(0.01, 1.0)
                
            order = Order(self.agent_id, side, price, qty=10, timestamp=self.env.current_tick)
            self.env.submit_order(order)

class MarketMaker(Agent):
    def __init__(self, agent_id, env, spread=0.10):
        super().__init__(agent_id, env)
        self.spread = spread

    def act(self):
        mid = self.env.order_book.get_mid_price()
        
        # Place Bid
        bid_price = mid - (self.spread / 2)
        bid_order = Order(self.agent_id, 'Buy', bid_price, 100, self.env.current_tick)
        self.env.submit_order(bid_order)
        
        # Place Ask
        ask_price = mid + (self.spread / 2)
        ask_order = Order(self.agent_id, 'Sell', ask_price, 100, self.env.current_tick)
        self.env.submit_order(ask_order)


class MarketEnvironment:
    def __init__(self, tick_size=0.01, max_ticks=1000):
        self.order_book = OrderBook(tick_size)
        self.current_tick = 0
        self.max_ticks = max_ticks
        self.agents = []
        self.price_history = []

    def add_agent(self, agent):
        self.agents.append(agent)

    def submit_order(self, order):
        self.order_book.add_order(order)

    def run_simulation(self):
        print(f"--- SIMULATION STARTED (Max Ticks: {self.max_ticks}) ---")
        
        for t in range(self.max_ticks):
            self.current_tick = t
            
            # 1. Agents Act
            for agent in self.agents:
                agent.act()
                
            # 2. Record State
            last_price = self.order_book.trades[-1].price if self.order_book.trades else 100.0
            self.price_history.append(last_price)

        print("--- SIMULATION COMPLETE ---")


SIM_TICKS = 500
TICK_SIZE = 0.05

market = MarketEnvironment(tick_size=TICK_SIZE, max_ticks=SIM_TICKS)

mm = MarketMaker(agent_id="MM_01", env=market, spread=0.20)
market.add_agent(mm)

for i in range(10):
    trader = NoiseTrader(agent_id=f"Noise_{i}", env=market, arrival_rate=0.3)
    market.add_agent(trader)

market.run_simulation()

plt.figure(figsize=(12, 6))
plt.plot(market.price_history, label='Market Price', color='blue', linewidth=1)
plt.title(f'Simulated Dummy Market: {len(market.agents)} Agents over {SIM_TICKS} Ticks')
plt.xlabel('Time (Ticks)')
plt.ylabel('Price ($)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

print(f"Total Trades: {len(market.order_book.trades)}")
print(f"Final Price: ${market.price_history[-1]:.2f}")