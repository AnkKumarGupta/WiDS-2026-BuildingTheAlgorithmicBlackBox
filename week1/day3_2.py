import heapq
import random
import matplotlib.pyplot as plt

class SimulationOrderBook:
    def __init__(self):
        self.bids = [] # Max-Heap (stored as negative prices)
        self.asks = [] # Min-Heap
        self.spread_history = []
        
    def add_limit_order(self, side, price, qty):
        if side == 'Buy':
            heapq.heappush(self.bids, -price)
        else:
            heapq.heappush(self.asks, price)
            
    def get_best_bid_ask(self):
        best_bid = -self.bids[0] if self.bids else None
        best_ask = self.asks[0] if self.asks else None
        return best_bid, best_ask

    def clean_crossed_book(self):
        while self.bids and self.asks and (-self.bids[0] >= self.asks[0]):
            heapq.heappop(self.bids)
            heapq.heappop(self.asks)

    def capture_spread(self):
        bb, ba = self.get_best_bid_ask()
        if bb and ba:
            return ba - bb
        return None


# --- PARAMETERS ---
NUM_TRADERS = 1000
TRUE_VALUE = 100.00
NOISE_LEVEL = 2.0 

sim_book = SimulationOrderBook()
spreads = []

print("Simulating 1000 Traders...")
for i in range(NUM_TRADERS):
    side = 'Buy' if random.random() < 0.5 else 'Sell'
    
    valuation = random.gauss(TRUE_VALUE, NOISE_LEVEL)
    valuation = random.gauss(TRUE_VALUE, NOISE_LEVEL)
    
    # Buyers bid slightly below their valuation (seeking profit)
    # Sellers ask slightly above their valuation
    if side == 'Buy':
        price = round(valuation - random.uniform(0, 0.5), 2)
        sim_book.add_limit_order('Buy', price, 10)
    else:
        price = round(valuation + random.uniform(0, 0.5), 2)
        sim_book.add_limit_order('Sell', price, 10)
    

    sim_book.clean_crossed_book()
    
    spread = sim_book.capture_spread()
    if spread:
        spreads.append(spread)

# --- VISUALIZATION: SPREAD CONVERGENCE ---
plt.figure(figsize=(12, 5))
plt.plot(spreads, color='blue', linewidth=0.8)
plt.title(f"Spread Convergence over {NUM_TRADERS} Orders")
plt.xlabel("Trader Number (Time)")
plt.ylabel("Bid-Ask Spread ($)")
plt.axhline(y=0, color='black', linestyle='--')
plt.grid(True, alpha=0.3)
plt.show()

print(f"Final Spread: ${spreads[-1]:.2f}")