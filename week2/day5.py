import heapq
import itertools
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from matplotlib.backends.backend_pdf import PdfPages
import warnings

warnings.filterwarnings("ignore")


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
        best_bid = -self.bids[0][0] if self.bids else np.nan
        best_ask = self.asks[0][0] if self.asks else np.nan
        return best_bid, best_ask

class MarketRecorder:
    def __init__(self):
        self.tape = []
        self.snapshots = []

    def record_trade(self, trade):
        self.tape.append({
            'timestamp': trade.timestamp,
            'price': trade.price,
            'qty': trade.qty
        })

    def record_snapshot(self, timestamp, engine):
        bb, ba = engine.get_l1_snapshot()
        self.snapshots.append({
            'timestamp': timestamp,
            'best_bid': bb,
            'best_ask': ba,
            'spread': ba - bb if (not np.isnan(ba) and not np.isnan(bb)) else np.nan,
            'mid_price': (ba + bb)/2 if (not np.isnan(ba) and not np.isnan(bb)) else np.nan
        })

class SimulationKernel:
    def __init__(self):
        self.engine = MatchingEngine()
        self.recorder = MarketRecorder()
        self.clock = 0.0
        self.events = [] 
        self.order_count = 0

    def schedule(self, delay, func):
        heapq.heappush(self.events, (self.clock + delay, self.order_count, func))
        self.order_count += 1

    def run(self, max_orders=1000):
        print(f"--- STARTING SIMULATION ({max_orders} Orders) ---")
        
        self.engine.process(Order('Buy', 99.0, 100, 'MM', 0))
        self.engine.process(Order('Sell', 101.0, 100, 'MM', 0))
        
        while self.events:
            t, _, func = heapq.heappop(self.events)
            self.clock = t
            func()
            
            if self.engine.trades:
                for trade in reversed(self.engine.trades):
                    if trade.timestamp == self.clock:
                        pass 
                    else:
                        break 

                last_trade = self.engine.trades[-1]
                if last_trade.timestamp == self.clock:
                    self.recorder.record_trade(last_trade)
            
            self.recorder.record_snapshot(self.clock, self.engine)

        print("--- SIMULATION COMPLETE ---")

def generate_scenario(kernel):
    """ Generates 1000 random orders with realistic distributions. """
    random.seed(42) 
    current_price = 100.0
    
    for i in range(1000):
        delay = random.expovariate(1.0)
        
        qty = int(random.lognormvariate(2, 0.5))
        
        current_price += random.gauss(0, 0.05)
        
        side = 'Buy' if random.random() < 0.5 else 'Sell'
        
        is_market = random.random() < 0.2 
        price = None
        if not is_market:
            offset = random.gauss(0, 0.5)
            price = round(current_price + (offset if side == 'Sell' else -offset), 2)
        
        def place_order(s=side, p=price, q=qty, o=f"Trader_{i}"):
            order = Order(s, p, q, o, kernel.clock)
            kernel.engine.process(order)
            
        kernel.schedule(delay, place_order)

def generate_report(recorder, filename="simulation_report.pdf"):
    print("Generating Report...")
    
    df_tape = pd.DataFrame(recorder.tape)
    df_l1 = pd.DataFrame(recorder.snapshots)
    
    if df_tape.empty:
        print("ERROR: No trades occurred. Report Generation Aborted.")
        return

    start_date = pd.Timestamp("2026-01-01 09:30:00")
    
    df_tape['datetime'] = start_date + pd.to_timedelta(df_tape['timestamp'], unit='s')
    df_tape.set_index('datetime', inplace=True)
    
    df_l1['datetime'] = start_date + pd.to_timedelta(df_l1['timestamp'], unit='s')
    df_l1.set_index('datetime', inplace=True)

    ohlc = df_tape['price'].resample('1min').ohlc()
    ohlc['volume'] = df_tape['qty'].resample('1min').sum()
    ohlc.dropna(inplace=True)

    print("Running Validation Checks...")
    neg_spread = df_l1[df_l1['spread'] < 0]
    if not neg_spread.empty:
        print(f"CRITICAL WARNING: Crossed book detected at {neg_spread.index[0]}")
    else:
        print("PASS: Spread invariant holds.")
        
    if any(ohlc['low'] > ohlc['high']):
        print("CRITICAL FAIL: Low > High in candlestick data.")
    else:
        print("PASS: OHLC Integrity verified.")

    with PdfPages(filename) as pdf:
        fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        
        up = ohlc[ohlc.close >= ohlc.open]
        down = ohlc[ohlc.close < ohlc.open]
        
        ax1.vlines(up.index, up.low, up.high, color='green', linewidth=1)
        ax1.vlines(down.index, down.low, down.high, color='red', linewidth=1)
        
        ax1.vlines(up.index, up.open, up.close, color='green', linewidth=3)
        ax1.vlines(down.index, down.open, down.close, color='red', linewidth=3)
        
        ax1.set_title("Market Simulation: Price Action (1-Min OHLC)")
        ax1.set_ylabel("Price")
        ax1.grid(True, alpha=0.3)
        
        ax2.bar(ohlc.index, ohlc.volume, color='blue', alpha=0.5)
        ax2.set_ylabel("Volume")
        ax2.set_xlabel("Simulation Time")
        
        plt.tight_layout()
        pdf.savefig(fig1)
        plt.close()

        fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(11, 8))
        
        ax3.plot(df_l1.index, df_l1['spread'], color='orange', linewidth=0.8)
        ax3.set_title("Bid-Ask Spread Over Time")
        ax3.set_ylabel("Spread ($)")
        ax3.grid(True, alpha=0.3)
        
        mid_1s = df_l1['mid_price'].resample('1s').last().ffill()
        log_ret = np.log(mid_1s / mid_1s.shift(1))
        rolling_vol = log_ret.rolling(window=60).std() * np.sqrt(3600*24) # Annualized-ish scaling
        
        ax4.plot(rolling_vol.index, rolling_vol, color='purple', linewidth=0.8)
        ax4.set_title("Realized Volatility (Rolling 60s)")
        ax4.set_ylabel("Volatility")
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig2)
        plt.close()
        
    print(f"SUCCESS: Report saved to {filename}")

if __name__ == "__main__":
    kernel = SimulationKernel()
    generate_scenario(kernel)
    kernel.run()
    generate_report(kernel.recorder)