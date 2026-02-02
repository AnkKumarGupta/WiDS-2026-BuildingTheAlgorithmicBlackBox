import matplotlib.pyplot as plt
import numpy as np


def generate_synthetic_depth(center_price=100, volatility=2.0, num_orders=500):
    
    bid_prices = center_price - np.abs(np.random.normal(0, volatility, num_orders))
    ask_prices = center_price + np.abs(np.random.normal(0, volatility, num_orders))

    bid_qtys = np.random.randint(1, 100, num_orders)
    ask_qtys = np.random.randint(1, 100, num_orders)

    bids_sorted = sorted(zip(bid_prices, bid_qtys), key=lambda x: -x[0]) 
    asks_sorted = sorted(zip(ask_prices, ask_qtys), key=lambda x: x[0])
    
    return bids_sorted, asks_sorted

def plot_depth_chart(bids, asks):
    b_prices, b_qtys = zip(*bids)
    a_prices, a_qtys = zip(*asks)

    b_cum_vol = np.cumsum(b_qtys)
    a_cum_vol = np.cumsum(a_qtys)
    
    plt.figure(figsize=(12, 6))
    
    plt.fill_between(b_prices, b_cum_vol, step="pre", color="green", alpha=0.4, label="Bids (Buyers)")
    
    plt.fill_between(a_prices, a_cum_vol, step="pre", color="red", alpha=0.4, label="Asks (Sellers)")
    
    plt.title("Market Depth Chart (L2 Liquidity Snapshot)")
    plt.xlabel("Price")
    plt.ylabel("Cumulative Quantity (Liquidity)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

bids, asks = generate_synthetic_depth(center_price=150.00)
plot_depth_chart(bids, asks)