import heapq

class Order:
    def __init__(self, order_id, side, price, qty, timestamp):
        self.id = order_id
        self.side = side     
        self.price = price
        self.qty = qty
        self.timestamp = timestamp  

    def __repr__(self):
        return f"[{self.timestamp}] {self.side} {self.qty} @ {self.price}"

class MatchingEngine:
    def __init__(self):
        self.bids = []
        self.asks = []
        self.trades = []  

    def process_order(self, order):
        if order.side == 'Buy':
            self._match_buy(order)
        else:
            self._match_sell(order)

    def _match_buy(self, order):
        while self.asks and order.qty > 0 and order.price >= self.asks[0][0]:
            best_ask_tuple = self.asks[0]
            best_ask_price, _, best_ask_order = best_ask_tuple

            trade_qty = min(order.qty, best_ask_order.qty)
            
            self._execute_trade(order.id, best_ask_order.id, best_ask_price, trade_qty)

            order.qty -= trade_qty
            best_ask_order.qty -= trade_qty

            if best_ask_order.qty == 0:
                heapq.heappop(self.asks)

        if order.qty > 0:
            heapq.heappush(self.bids, (-order.price, order.timestamp, order))

    def _match_sell(self, order):
        # While there are Bids and Sell Price <= Best Bid Price
        while self.bids and order.qty > 0 and order.price <= -self.bids[0][0]:
            best_bid_tuple = self.bids[0]
            neg_bid_price, _, best_bid_order = best_bid_tuple
            best_bid_price = -neg_bid_price

            trade_qty = min(order.qty, best_bid_order.qty)
            
            self._execute_trade(best_bid_order.id, order.id, best_bid_price, trade_qty)

            order.qty -= trade_qty
            best_bid_order.qty -= trade_qty

            if best_bid_order.qty == 0:
                heapq.heappop(self.bids)

        if order.qty > 0:
            heapq.heappush(self.asks, (order.price, order.timestamp, order))

    def _execute_trade(self, buy_id, sell_id, price, qty):
        self.trades.append({
            'buy_id': buy_id,
            'sell_id': sell_id,
            'price': price,
            'qty': qty
        })

    def print_depth(self):
        print("\n--- ORDER BOOK DEPTH (Top 5) ---")
        top_asks = heapq.nsmallest(5, self.asks)
        top_bids = heapq.nsmallest(5, self.bids)
        
        print("ASKS (Sellers)")
        for price, ts, o in reversed(top_asks):
            print(f"  ${price:.2f} | {o.qty}")
            
        print("-" * 20)

        # Displaying Bids
        for neg_price, ts, o in top_bids:
            print(f"  ${-neg_price:.2f} | {o.qty}")
        print("BIDS (Buyers)")
        print("--------------------------------")


def run_simulation_run(run_id, order_stream):
    print(f"\n[Run {run_id}] Starting Simulation...")
    engine = MatchingEngine()
    
    # Processing the stream
    for raw_order in order_stream:
        # Unpack: (id, side, price, qty, timestamp)
        o = Order(*raw_order)
        engine.process_order(o)
        
    engine.print_depth()
    print(f"[Run {run_id}] Total Trades Executed: {len(engine.trades)}")
    return engine.trades

# 1. Defining a deterministic stream of orders
# Format: (ID, Side, Price, Qty, Tick)
order_stream_data = [
    (1, 'Buy', 100.0, 10, 1),
    (2, 'Sell', 102.0, 5, 2),
    (3, 'Buy', 101.0, 20, 3),   # Improving Bid
    (4, 'Sell', 99.0, 15, 4),   # Aggressive Sell (Crosses book!)
    (5, 'Buy', 99.0, 10, 5),
    (6, 'Sell', 101.0, 30, 6)
]

# 2. Run Twice to verify Determinism
trades_run_1 = run_simulation_run(1, order_stream_data)
trades_run_2 = run_simulation_run(2, order_stream_data)

# 3. Validation
print("\n--- DETERMINISTIC REPLAY CHECK ---")
if trades_run_1 == trades_run_2:
    print("SUCCESS: Both runs produced identical trade logs.")
    print("Sample Trade:", trades_run_1[0])
else:
    print("FAILURE: Simulation is non-deterministic!")