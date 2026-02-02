import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.graphics.tsaplots import plot_acf

def analyze_stylized_facts():
    print("--- DAY 7: STYLIZED FACTS VALIDATION ---")

    try:
        df = pd.read_csv("mid_prices.csv")
    except FileNotFoundError:
        print("Error: mid_prices.csv not found. Run Day 6 simulation first.")
        return

    # r_t = ln(P_t) - ln(P_{t-1})
    df['log_ret'] = np.log(df['price'] / df['price'].shift(1))
    df = df.dropna() 

    print("Checking Fact #1: Volatility Clustering...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot A: Returns over Time
    ax1.plot(df['log_ret'], alpha=0.7, color='blue')
    ax1.set_title("Log Returns over Time (Visual Inspection)")
    ax1.set_ylabel("Log Return")
    ax1.grid(True, alpha=0.3)

    # Plot B: Autocorrelation of Absolute Returns
    abs_rets = np.abs(df['log_ret'])
    plot_acf(abs_rets, lags=50, ax=ax2, title="Autocorrelation of Absolute Returns (|r_t|)")
    ax2.set_xlabel("Lag")

    plt.tight_layout()
    plt.show()

    print("\nChecking Fact #2: Fat Tails...")

    returns = df['log_ret'].values

    # Statistics
    mu, std = stats.norm.fit(returns)
    kurtosis = stats.kurtosis(returns) # Excess kurtosis (Normal = 0 in scipy)

    print(f"Mean: {mu:.6f}")
    print(f"Std Dev: {std:.6f}")
    print(f"Excess Kurtosis: {kurtosis:.4f} (Target > 0)")

    if kurtosis > 0.5:
        print("VERDICT: Leptokurtic (Fat Tails Detected) ✅")
    else:
        print("VERDICT: Platykurtic/Normal (Thin Tails) ❌")

    # Visualization: Histogram vs Normal Distribution
    fig, ax = plt.subplots(figsize=(10, 6))

    # 1. Empirical Histogram
    count, bins, ignored = ax.hist(returns, bins=100, density=True, alpha=0.6, color='blue', label='Simulation Data')

    # 2. Gaussian Overlay
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=2, label=f'Normal Dist ($\sigma$={std:.4f})')

    # 3. Log Scale 
    ax.set_yscale('log')
    ax.set_ylim(bottom=0.1) 

    ax.set_title(f"Return Distribution (Log Scale)\nExcess Kurtosis = {kurtosis:.2f}")
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    plt.show()

if __name__ == "__main__":
    analyze_stylized_facts()