import heapq
import time

class Order:
    def __init__(self, agent_id, side, price, qty, timestamp):
        self.agent_id = agent_id
        self.side = side      # 'Buy' or 'Sell'
        self.price = price
        self.qty = qty
        self.timestamp = timestamp

    def __lt__(self, other):
        return self.timestamp < other.timestamp

    def __repr__(self):
        return f"[{self.side}] {self.qty} @ ${self.price:.2f}"

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
            self._execute_trade(order.agent_id, ask_order.agent_id, best_ask_price, trade_qty)
            
            order.qty -= trade_qty
            ask_order.qty -= trade_qty
            
            if ask_order.qty == 0:
                heapq.heappop(self.asks)
        
        if order.qty > 0:
            heapq.heappush(self.bids, (-order.price, order.timestamp, order))

    def _match_sell(self, order):
        while self.bids and order.qty > 0 and order.price <= -self.bids[0][0]:
            best_bid_neg, _, bid_order = self.bids[0]
            best_bid_price = -best_bid_neg
            
            trade_qty = min(order.qty, bid_order.qty)
            self._execute_trade(bid_order.agent_id, order.agent_id, best_bid_price, trade_qty)
            
            order.qty -= trade_qty
            bid_order.qty -= trade_qty
            
            if bid_order.qty == 0:
                heapq.heappop(self.bids)

        if order.qty > 0:
            heapq.heappush(self.asks, (order.price, order.timestamp, order))

    def _execute_trade(self, buyer, seller, price, qty):
        t = Trade(price, qty, time.time(), buyer, seller)
        self.trades.append(t)

if __name__ == "__main__":
    book = OrderBook()
    
    print("--- Placing Sell Orders ---")
    book.add_order(Order("Seller_A", "Sell", 101.00, 10, 1))
    book.add_order(Order("Seller_B", "Sell", 102.00, 10, 2))
    
    print("--- Placing Buy Order ---")
    book.add_order(Order("Buyer_1", "Buy", 102.00, 15, 3))
    
    print(f"Trades Executed: {len(book.trades)}")
    for t in book.trades:
        print(f"Trade: {t.qty} shares @ ${t.price:.2f} (Buyer: {t.buyer_id}, Seller: {t.seller_id})")