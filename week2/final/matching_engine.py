import heapq
import itertools
from collections import deque

class Order:
    _id_counter = itertools.count()
    def __init__(self, side, price, qty, owner_id, timestamp):
        self.id = next(self._id_counter)
        self.side = side
        self.price = price 
        self.qty = qty
        self.owner_id = owner_id
        self.timestamp = timestamp

    def __lt__(self, other):
        return self.id < other.id

class Trade:
    def __init__(self, price, qty, timestamp, buyer, seller):
        self.price = price
        self.qty = qty
        self.timestamp = timestamp
        self.buyer_id = buyer
        self.seller_id = seller

class MatchingEngine:
    def __init__(self):
        self.asks = [] 
        self.bids = [] 
        self.trades = []

    def process(self, order):
        if order.price is not None and order.price < 0: raise ValueError("Negative Price")
        if order.qty <= 0: raise ValueError("Non-positive Quantity")

        if order.side == 'Buy': self._match_buy(order)
        else: self._match_sell(order)

    def _match_buy(self, order):
        while self.asks and order.qty > 0:
            best_ask = self.asks[0]
            price, _, ask_order = best_ask[0], best_ask[1], best_ask[2]
            
            if order.price is not None and order.price < price: break 

            qty = min(order.qty, ask_order.qty)
            self._execute_trade(price, qty, order.timestamp, order, ask_order)
            
            order.qty -= qty
            ask_order.qty -= qty
            if ask_order.qty == 0: heapq.heappop(self.asks)

        if order.qty > 0 and order.price is not None:
            heapq.heappush(self.bids, (-order.price, order.id, order))

    def _match_sell(self, order):
        while self.bids and order.qty > 0:
            best_bid = self.bids[0]
            price, _, bid_order = -best_bid[0], best_bid[1], best_bid[2]

            if order.price is not None and order.price > price: break 

            qty = min(order.qty, bid_order.qty)
            self._execute_trade(price, qty, order.timestamp, bid_order, order)

            order.qty -= qty
            bid_order.qty -= qty
            if bid_order.qty == 0: heapq.heappop(self.bids)

        if order.qty > 0 and order.price is not None:
            heapq.heappush(self.asks, (order.price, order.id, order))

    def _execute_trade(self, price, qty, timestamp, buyer, seller):
        self.trades.append(Trade(price, qty, timestamp, buyer.owner_id, seller.owner_id))

    def get_l1_snapshot(self):
        best_bid = -self.bids[0][0] if self.bids else None
        best_ask = self.asks[0][0] if self.asks else None
        return best_bid, best_ask

def run_integrity_test():
    print("Running Matching Engine Integrity Test...")
    eng = MatchingEngine()
    
    asks = [(101, 10), (102, 20), (103, 30)]
    for p, q in asks:
        eng.process(Order('Sell', p, q, 'Seller', 0))
        
    eng.process(Order('Buy', None, 60, 'Buyer', 0))
    
    assert len(eng.trades) == 3, f"Fail: Expected 3 trades, got {len(eng.trades)}"
    
    assert eng.trades[0].price == 101
    assert eng.trades[1].price == 102
    assert eng.trades[2].price == 103
    
    assert len(eng.asks) == 0, "Fail: Ask book should be empty"
    
    print("PASS: Matching Engine Integrity Verified.")

if __name__ == "__main__":
    run_integrity_test()