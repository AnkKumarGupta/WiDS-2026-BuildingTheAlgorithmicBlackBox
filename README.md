# AI in High-Frequency Trading: Building an algorithmic black-box
**Winter in Data Science (WiDS) 2025 | Final Project**

## Project Overview
This project tries to explore and answer: **Can a Deep RL agent discover profitable trading strategies in a realistic, high-friction simulation?**

Unlike standard backtesters that assume infinite liquidity, I built a "Digital Twin" of a financial exchange from scratch. It features a realistic Limit Order Book (LOB), network latency simulation, and adversarial agents that fight back.

### The "Black Box" Dashboard
I built an interactive dashboard to visualize the agent's decision-making process step-by-step.
**[Launch Interactive Dashboard (HTML)](week3/outputs/dashboard_deterministic.html)**

---

## 🛠 Phase I: The Mechanism (Building the Engine)
*Week 0-1*

The first challenge was building a Matching Engine that respects **Price-Time Priority**. Existing libraries were too high-level, so I implemented the core data structures manually:
* **Dual-Heap Order Book:** Max-Heap for Bids, Min-Heap for Asks to ensure $O(1)$ access to best prices.
* **Event-Driven Kernel:** A `SimulationKernel` that manages a global clock, allowing me to simulate millisecond-level latency between order submission and execution.

**Validation:**
I reconstructed the market tape from the simulation logs to verify that the engine produced valid OHLC candles and Volume profiles.
![Engine Validation](outputs/wids_week2_day4.png)
*Fig 1: Reconstructed price history validating the matching logic.*

---

## Phase II: The Simulation (Creating Chaos)
*Week 2*

A market is defined by its participants. I engineered a "Zoo" of algorithmic agents to generate organic volatility and liquidity.

### 1. The Stabilizers (Market Makers)
I built Market Maker (MM) agents that quote both sides of the book. To make them realistic, I implemented **Inventory Skewing**: if an MM holds too much inventory, they lower their prices to encourage selling.
![MM Skewing](outputs/wids_week2_day9.png)
*Observation: You can see the MM (Green/Red lines) actively adjusting quotes around the fair value (Grey) to manage risk.*

### 2. The Instability (Momentum Traders)
To introduce risk, I added Momentum agents that buy when prices rise. This created **"Pump and Dump"** feedback loops, proving the simulator could replicate market bubbles.
![Momentum Loop](outputs/wids_week2_day8.png)

### 3. Emergent Behavior
By mixing Noise Traders, MMs, and Momentum agents, I observed emergent market properties. **Scenario B (Noise + MM)** showed the tightest spreads, quantitatively proving how liquidity providers stabilize markets.
![Emergent Behavior](outputs/wids_week2_day10.png)

---

## Phase III: The Predator (Reinforcement Learning)
*Week 3*

With a living market, I trained a **Proximal Policy Optimization (PPO)** agent to trade against these bots.

### Experiment 1: The "God Mode" Stress Test
Before training, I tested the market's fragility by forcing a Flash Crash at Step 150.
![Herding Crash](outputs/image_bbedd4.png)
* **Observation:** The moment the price crashed, the Momentum agents (purple line) synchronized and dumped their inventory instantly.
* **Metric:** The Herding Correlation (red line) spiked to `1.0`. This confirmed my environment supports **Endogenous Risk** (risk arising from agent interaction).

### Experiment 2: Hyperparameter Tuning
I used **Optuna** to tune the agent's brain.
![Optuna Results](outputs/image_bb812d.png)
* **Key Learning:** The most critical parameter was **Gamma (Discount Factor)**.
* **Interpretation:** A lower Gamma ($\gamma \approx 0.92$) performed best. In HFT, the distant future is noise; the agent had to focus on immediate order book imbalances to survive.

---

## Final Results: The "Null" Victory

I benchmarked the trained RL agent against standard strategies over 2,000 steps.

| Strategy | Sharpe Ratio | Max Drawdown | Verdict |
| :--- | :--- | :--- | :--- |
| **Random Agent** | -5.41 | -0.72% | **Bust.** Bled money due to spread costs. |
| **Buy & Hold** | 0.65 | 0.00% | **Beta.** Captured market drift. |
| **RL Agent** | **0.00** | **0.00%** | **Survival.** |

![Benchmarks](image_bb0d32.png)

### Conclusion & Interpretation
At first glance, a Sharpe Ratio of 0.0 looks like a failure. However, in the context of this simulation, it is a **success**.

1.  **The Cost of Business:** The simulation imposes a transaction cost of 1 basis point per trade. The **Random Agent** proves that trading without a clear edge guarantees bankruptcy.
2.  **Intelligence is Restraint:** My RL agent learned a sophisticated lesson: **Risk Aversion**. It realized that the predictive signal in the noise was often weaker than the cost of trading, so it chose capital preservation over gambling.
3.  **Future Work:** To generate positive Alpha, the agent likely needs Level 2 Order Flow data (OFI) as a state input, rather than just Level 1 snapshots.

---

## Tech Stack
* **Core:** Python 3.10, NumPy, Pandas
* **RL:** Stable-Baselines3 (PPO), Gymnasium
* **Analysis:** Plotly, Matplotlib, Optuna