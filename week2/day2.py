import heapq
import itertools

class Order:
    _id_counter = itertools.count()

    def __init__(self, side, price, qty, owner_id):
        self.id = next(self._id_counter) 
        self.side = side      
        self.price = price    
        self.qty = qty
        self.owner_id = owner_id

    def __repr__(self):
        p_str = f"${self.price:.2f}" if self.price else "MKT"
        return f"[{self.id}] {self.side} {self.qty} @ {p_str}"

class Trade:
    def __init__(self, price, qty, buyer, seller):
        self.price = price
        self.qty = qty
        self.buyer_id = buyer
        self.seller_id = seller

    def __repr__(self):
        return f"TRADE: {self.qty} @ ${self.price:.2f} ({self.buyer_id} <-> {self.seller_id})"

class MatchingEngine:
    def __init__(self):
        # Asks: Min-Heap (Price, SeqID, Order) -> Lowest price first
        self.asks = [] 
        # Bids: Min-Heap (-Price, SeqID, Order) -> Highest price first
        self.bids = [] 
        self.trades = []

    def process_order(self, order):
        """
        Logic used here:
        1. Attempting to match immediately (Aggressive / Crossing Spread).
        2. If quantity remains & it's a Limit Order, adding to book (Passive).
        """
        print(f"--> PROCESSING: {order}")
        
        if order.side == 'Buy':
            self._match_buy(order)
        else:
            self._match_sell(order)

    def _match_buy(self, order):
        while self.asks and order.qty > 0:

            best_ask_price, _, best_ask_order = self.asks[0]

            if order.price is not None and order.price < best_ask_price:
                break # As spread is not crossed. 

            trade_qty = min(order.qty, best_ask_order.qty)
            self._execute_trade(
                price=best_ask_price, 
                qty=trade_qty,
                buyer=order,
                seller=best_ask_order
            )

            order.qty -= trade_qty
            best_ask_order.qty -= trade_qty

            if best_ask_order.qty == 0:
                heapq.heappop(self.asks)

        if order.qty > 0 and order.price is not None:
            entry = (-order.price, order.id, order)
            heapq.heappush(self.bids, entry)
            print(f"    ... Rested remaining {order.qty} on BID book.")

    def _match_sell(self, order):
        while self.bids and order.qty > 0:
            neg_bid_price, _, best_bid_order = self.bids[0]
            best_bid_price = -neg_bid_price


            if order.price is not None and order.price > best_bid_price:
                break 

            trade_qty = min(order.qty, best_bid_order.qty)
            self._execute_trade(
                price=best_bid_price,
                qty=trade_qty,
                buyer=best_bid_order,
                seller=order
            )

            order.qty -= trade_qty
            best_bid_order.qty -= trade_qty

            if best_bid_order.qty == 0:
                heapq.heappop(self.bids)

        if order.qty > 0 and order.price is not None:
            entry = (order.price, order.id, order)
            heapq.heappush(self.asks, entry)
            print(f"    ... Rested remaining {order.qty} on ASK book.")

    def _execute_trade(self, price, qty, buyer, seller):
        t = Trade(price, qty, buyer.owner_id, seller.owner_id)
        self.trades.append(t)
        print(f"    MATCH! {t}")

    def print_book(self):
        print("\n=== BOOK STATE ===")
        print("ASKS (Sellers):")
        for p, _, o in sorted(self.asks):
            print(f"  ${p:.2f} x {o.qty} ({o.owner_id})")
        print("BIDS (Buyers):")
        for p, _, o in sorted(self.bids):
            print(f"  ${-p:.2f} x {o.qty} ({o.owner_id})")
        print("==================\n")

# --- VALIDATION TEST SCRIPT ---
engine = MatchingEngine()

print("--- STEP 1: Building the Ask Ladder ---")
asks = [
    Order('Sell', 101.00, 10, 'Seller_A'),
    Order('Sell', 102.00, 20, 'Seller_B'),
    Order('Sell', 103.00, 30, 'Seller_C')
]
for a in asks:
    engine.process_order(a)

engine.print_book()

print("--- STEP 2: The Massive Market Buy ---")
market_buy = Order('Buy', None, 60, 'Big_Buyer')
engine.process_order(market_buy)

print("--- STEP 3: Final Verification ---")
engine.print_book()


assert len(engine.trades) == 3, f"Failed! Expected 3 trades, got {len(engine.trades)}"

assert engine.trades[0].price == 101.00
assert engine.trades[1].price == 102.00
assert engine.trades[2].price == 103.00

assert len(engine.asks) == 0, "Failed! Ask book should be empty."

print("\n>>> TEST PASSED: ENGINE LOGIC LOOKS GOOD <<<")