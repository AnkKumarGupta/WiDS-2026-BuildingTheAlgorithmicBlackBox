# Citadel: Building an Algorithmic Black Box


This project is part of the **WiDS 2026**.

---

## Project Timeline & Modules

### **Week 0: Quantitative Foundations**
*Focus: Financial Data Analysis & Stochastic Modeling* <br>
Before building the market, I had to understand the data it generates.
- **Data Analysis:** Processed historical `AAPL` data using Pandas to visualize price action, moving averages, and volatility clustering.
- **Statistical Rigor:** Calculated key metrics like annualized volatility and Z-scores to quantify "Fat Tail" risks (events that happen more often than a Normal Distribution predicts).
- **Stochastic Calculus:** Built a Monte Carlo simulator using **Geometric Brownian Motion (GBM)** to model future price paths and visualize the "Cone of Uncertainty."

### **Week 1: The Engine (Market Microstructure)**
*Focus: System Architecture & Matching Logic* <br>
Here, I built the core infrastructure of the exchange.
- **The Matching Engine:** Implemented a **Price-Time Priority (FIFO)** matching algorithm. This is the standard logic used by major exchanges (NYSE, Nasdaq).
- **Order Book Data Structures:** Moved from simple lists to **Min/Max Heaps** (`heapq`) to optimize order insertion and retrieval time from $O(N)$ to $O(\log N)$.
- **Architecture:** Designed a modular system separating the `MarketEnvironment` (the exchange) from the `Agents` (the traders), using a Gym-compatible API (Observation $\to$ Action).

### **Week 2: The Ecosystem (Agent-Based Modeling)**
*Focus: Emergent Behavior & Market Dynamics* <br>
A market is only as alive as its participants. I populated the exchange with a "Zoo" of heterogeneous agents to observe how macro-level patterns emerge from micro-level rules.
- **Discrete Event Simulation:** Built a scheduler to handle **Latency** and asynchronous order arrivals (Poisson process), ensuring the simulation runs on "Event Time," not wall-clock time.
- **The Agents:**
    - **Noise Traders:** Zero-Intelligence agents acting randomly to create baseline liquidity.
    - **Momentum Traders:** Trend followers who buy when prices rise, demonstrating how feedback loops create bubbles and crashes.
    - **Market Makers:** Agents managing inventory risk (Avellaneda-Stoikov logic) to provide two-sided quotes and narrow the spread.
- **Analytics Pipeline:** Built a "Tape" to record every trade and an L1 Snapshot engine to track spreads and volatility.
- **Final Result:** Successfully reproduced complex market phenomena (like flash crashes and mean reversion) purely through agent interaction.

---

## Used Technical Stack
- **Python 3.12**
- **Used Core Libraries:** `numpy`, `pandas`, `heapq` (standard lib), `collections`
- **Visualization:** `matplotlib`, `mplfinance`
- **Concepts Applied:** Object-Oriented Programming (Polymorphism), Discrete Event Simulation, Big O Optimization.