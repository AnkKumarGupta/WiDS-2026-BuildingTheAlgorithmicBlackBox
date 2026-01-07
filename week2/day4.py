import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class MarketRecorder:
    def __init__(self):
        self.tape = []      
        self.l1_snapshots = [] 
        
    def record_trade(self, trade):
        self.tape.append({
            'timestamp': trade.timestamp,
            'price': trade.price,
            'qty': trade.qty,
            'buyer': trade.buyer_id,
            'seller': trade.seller_id
        })

    def record_snapshot(self, timestamp, book):
        best_bid = -book.bids[0][0] if book.bids else np.nan
        best_ask = book.asks[0][0] if book.asks else np.nan
        
        if not np.isnan(best_bid) and not np.isnan(best_ask):
            mid = (best_bid + best_ask) / 2.0
            spread = best_ask - best_bid
        else:
            mid = np.nan
            spread = np.nan
            
        self.l1_snapshots.append({
            'timestamp': timestamp,
            'best_bid': best_bid,
            'best_ask': best_ask,
            'spread': spread,
            'mid_price': mid
        })

    def get_tape_df(self):
        return pd.DataFrame(self.tape)

    def get_l1_df(self):
        return pd.DataFrame(self.l1_snapshots)

class AnalyticsEngine:
    def __init__(self, recorder):
        self.tape_df = recorder.get_tape_df()
        self.l1_df = recorder.get_l1_df()
        
    def calculate_vwap(self):
        if self.tape_df.empty: return 0.0
        
        v = self.tape_df['qty']
        p = self.tape_df['price']
        return (p * v).sum() / v.sum()

    def calculate_volatility(self, window_seconds=1):
        """ 
        1. Resampling Mid-Price to 1-second bars.
        2. Calculating Log Returns.
        3. Calculating Std Dev of Returns.
        """
        if self.l1_df.empty: return 0.0
        
        df = self.l1_df.copy()
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('datetime', inplace=True)
        
        resampled = df['mid_price'].resample('1s').last().ffill()
        
        log_returns = np.log(resampled / resampled.shift(1))
        
        return log_returns.std()

    def generate_report(self):
        print("\n--- MARKET ANALYTICS REPORT ---")
        vwap = self.calculate_vwap()
        vol = self.calculate_volatility()
        avg_spread = self.l1_df['spread'].mean()
        
        print(f"1. Total Volume:    {self.tape_df['qty'].sum() if not self.tape_df.empty else 0}")
        print(f"2. VWAP:            ${vwap:.2f}")
        print(f"3. Avg Spread:      ${avg_spread:.4f}")
        print(f"4. Volatility (1s): {vol:.6f}")
        
        if not self.tape_df.empty:
            low = self.tape_df['price'].min()
            high = self.tape_df['price'].max()
            if not (low <= vwap <= high):
                print("WARNING: VWAP is outside price range! (Math Error)")
            else:
                print("CHECK PASS: VWAP is consistent.")

recorder = MarketRecorder()

print("Simulating Data Stream...")
current_price = 100.00
for t in range(0, 3600): 
    current_price += np.random.normal(0, 0.05)
    
    class MockBook:
        bids = [(- (current_price - 0.02),)] 
        asks = [(current_price + 0.02,)]    
    
    recorder.record_snapshot(timestamp=t, book=MockBook())
    
    if np.random.random() < 0.1: 
        trade_price = current_price + np.random.choice([-0.02, 0.02])
        
        class MockTrade:
            timestamp = t
            price = trade_price
            qty = np.random.randint(1, 100)
            buyer_id = "Buyer"
            seller_id = "Seller"
            
        recorder.record_trade(MockTrade())

analytics = AnalyticsEngine(recorder)
analytics.generate_report()

plt.figure(figsize=(10, 5))
plt.plot(analytics.l1_df['timestamp'], analytics.l1_df['mid_price'], label='Mid Price')
if not analytics.tape_df.empty:
    plt.scatter(analytics.tape_df['timestamp'], analytics.tape_df['price'], 
                color='red', s=10, alpha=0.5, label='Trades')
plt.title("Day 4: Reconstructed Market History")
plt.legend()
plt.show()