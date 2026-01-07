import pandas as pd
import time

class Order:
    def __init__(self, order_id, side, price, quantity, order_type):
        self.order_id = order_id
        self.side = side # 'Buy' or 'Sell'
        self.price = price # None for Market Orders
        self.quantity = quantity
        self.timestamp = time.time()
        self.order_type = order_type

    def __repr__(self):
        return f"[{self.side} {self.order_type}] {self.quantity} @ {self.price if self.price else 'MKT'}"

class OrderBook:
    def __init__(self):
        # Bids: sorted by Price DESC, then Time ASC
        self.bids = [] 
        # Asks: sorted by Price ASC, then Time ASC
        self.asks = []
        self.trade_log = []

    def add_limit_order(self, order):
        if order.side == 'Buy':
            self.bids.append(order)
            # Sorted: Highest price first, then earliest time
            self.bids.sort(key=lambda x: (-x.price, x.timestamp))
        else:
            self.asks.append(order)
            # Sorted: Lowest price first, then earliest time
            self.asks.sort(key=lambda x: (x.price, x.timestamp))
        print(f"--> LIMIT ORDER PLACED: {order}")

    def execute_market_order(self, order):
        print(f"\n--> MARKET ORDER INBOUND: {order}")
        qty_needed = order.quantity
        
        # Selecting the opposite book
        best_queue = self.asks if order.side == 'Buy' else self.bids
        
        while qty_needed > 0 and best_queue:
            # Looking at the best available order
            best_match = best_queue[0]
            
            trade_qty = min(qty_needed, best_match.quantity)
            price = best_match.price
            
            self.trade_log.append(f"Trade! {trade_qty} @ ${price} (Taker: {order.side})")
            print(f"   MATCHED {trade_qty} shares @ ${price}")
            
            qty_needed -= trade_qty
            best_match.quantity -= trade_qty
            
            if best_match.quantity == 0:
                best_queue.pop(0)
                
        if qty_needed > 0:
            print(f"   PARTIAL FILL. {qty_needed} unfulfilled (No Liquidity).")
        else:
            print("   FULL FILL.")

    def display_book(self):
        print("\n" + "="*40)
        print("      CURRENT ORDER BOOK STATE")
        print("="*40)
        
        # Displaying top 5 Asks (reversed so highest price is at top)
        print("ASKS (Sellers)")
        for order in reversed(self.asks[:5]): 
            print(f"   ${order.price:.2f}  |  Qty: {order.quantity}")
            
        print("-" * 20 + f" Spread: {self.get_spread():.2f} " + "-" * 20)

        # Displaying top 5 Bids
        for order in self.bids[:5]:
            print(f"   ${order.price:.2f}  |  Qty: {order.quantity}")
        print("BIDS (Buyers)")
        print("="*40 + "\n")

    def get_spread(self):
        if self.bids and self.asks:
            return self.asks[0].price - self.bids[0].price
        return 0.0

# --- SIMULATION START ---
lob = OrderBook()

# 1. PHASE 1: POPULATING THE BOOK (15 Limit Orders)
print("--- PHASE 1: INJECTING 15 LIMIT ORDERS ---")
limit_orders = [
    # Bids
    Order(1, 'Buy', 149.95, 100, 'Limit'),
    Order(2, 'Buy', 149.90, 50, 'Limit'),
    Order(3, 'Buy', 150.00, 200, 'Limit'), # Best Bid
    Order(4, 'Buy', 149.85, 100, 'Limit'),
    Order(5, 'Buy', 150.00, 50, 'Limit'),  # Same price as #3, but later (Time priority)
    Order(6, 'Buy', 149.80, 300, 'Limit'),
    Order(7, 'Buy', 149.75, 100, 'Limit'),

    # Asks
    Order(8, 'Sell', 150.05, 100, 'Limit'), # Best Ask
    Order(9, 'Sell', 150.10, 150, 'Limit'),
    Order(10, 'Sell', 150.05, 50, 'Limit'), # Same price as #8, later time
    Order(11, 'Sell', 150.20, 200, 'Limit'),
    Order(12, 'Sell', 150.15, 100, 'Limit'),
    Order(13, 'Sell', 150.30, 50, 'Limit'),
    Order(14, 'Sell', 150.25, 100, 'Limit'),
    Order(15, 'Sell', 150.10, 25, 'Limit')
]

for o in limit_orders:
    lob.add_limit_order(o)

# Showing the book before trading
lob.display_book()

# 2. PHASE 2: EXECUTE TRADES (10 Market Orders)
print("--- PHASE 2: EXECUTING 10 MARKET ORDERS ---")
market_orders = [
    Order(101, 'Buy', None, 50, 'Market'),   # Should hit Best Ask ($150.05)
    Order(102, 'Sell', None, 100, 'Market'), # Should hit Best Bid ($150.00)
    Order(103, 'Buy', None, 200, 'Market'),  
    Order(104, 'Sell', None, 20, 'Market'),
    Order(105, 'Buy', None, 10, 'Market'),
    Order(106, 'Sell', None, 300, 'Market'),
    Order(107, 'Buy', None, 50, 'Market'),
    Order(108, 'Buy', None, 50, 'Market'),
    Order(109, 'Sell', None, 10, 'Market'),
    Order(110, 'Buy', None, 100, 'Market')
]

for mo in market_orders:
    lob.execute_market_order(mo)
    time.sleep(0.01) 

# 3. FINAL STATE
lob.display_book()

print("--- TRADE LOG ---")
for t in lob.trade_log:
    print(t)