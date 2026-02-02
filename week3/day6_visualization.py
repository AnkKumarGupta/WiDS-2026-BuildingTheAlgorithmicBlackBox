import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_lob_heatmap():
    print("Loading Data...")
    try:
        df_lob = pd.read_csv("lob_data.csv")
        df_mids = pd.read_csv("mid_prices.csv")
    except FileNotFoundError:
        print("Error: Run day6_simulation.py first.")
        return

    print("Processing Heatmap (This may take a moment)...")

    df_lob['price_bin'] = df_lob['price'].round(1)

    # Pivot for Bids (Green) and Asks (Red)
    pivot_vol = df_lob.pivot_table(index='price_bin', columns='step', values='vol', aggfunc='sum', fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 8))

    # X: Steps, Y: Prices
    X = pivot_vol.columns
    Y = pivot_vol.index
    Z = pivot_vol.values

    c = ax.pcolormesh(X, Y, Z, cmap='inferno', shading='auto', vmin=0, vmax=np.percentile(Z, 95))
    fig.colorbar(c, label='Resting Liquidity (Volume)')

    ax.plot(df_mids['step'], df_mids['price'], color='cyan', linewidth=1, label='Mid Price', alpha=0.8)

    ax.set_title("Limit Order Book Heatmap (Liquidity Walls)")
    ax.set_xlabel("Simulation Steps")
    ax.set_ylabel("Price Level")
    ax.legend()

    mid_mean = df_mids['price'].mean()
    ax.set_ylim(mid_mean - 5, mid_mean + 5)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_lob_heatmap()