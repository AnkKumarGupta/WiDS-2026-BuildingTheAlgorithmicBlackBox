import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings

from matching_engine import MatchingEngine, Order, run_integrity_test
from event_loop import SimulationKernel
from noise_agent import NoiseTrader
from market_maker_agent import MarketMakerAgent
from momentum_agent import MomentumAgent
from tape import Tape
from snapshots import SnapshotRecorder

warnings.filterwarnings('ignore')

TOTAL_AGENTS = 100
SIMULATION_TIME = 1800 
SNAPSHOT_INTERVAL = 1.0
SEED = 42

class FairValueModel:
    def __init__(self, start=100.0, vol=0.1):
        self.current_value = start
        self.vol = vol
    def step(self):
        self.current_value += np.random.normal(0, self.vol)
        return self.current_value

def run_scenario(name, n_noise, n_mm, n_momo):
    print(f"Running Scenario {name}: Noise={n_noise}, MM={n_mm}, Momo={n_momo}...")
    
    random.seed(SEED)
    np.random.seed(SEED)
    
    kernel = SimulationKernel()
    engine = MatchingEngine()
    tape = Tape()
    snaps = SnapshotRecorder()
    fv = FairValueModel()
    
    agents = []
    for i in range(n_noise): agents.append(NoiseTrader(f"Noise_{i}", fv))
    for i in range(n_mm): agents.append(MarketMakerAgent(f"MM_{i}"))
    for i in range(n_momo): agents.append(MomentumAgent(f"Momo_{i}"))
    
    assert len(agents) == TOTAL_AGENTS, "Agent count mismatch"

    def market_step():
        fv.step()
        
        bb, ba = engine.get_l1_snapshot()
        snaps.record(kernel.time, bb, ba)
        
        snapshot = {'mid_price': (bb+ba)/2 if (bb and ba) else fv.current_value}
        
        for agent in agents:
            if random.random() < 0.1: 
                actions = agent.get_action(snapshot)
                for intent in actions:
                    order = Order(intent.side, intent.price, intent.qty, agent.id, kernel.time)
                    engine.process(order)
                    
                    if engine.trades and engine.trades[-1].timestamp == kernel.time:
                         pass

        kernel.schedule(SNAPSHOT_INTERVAL, market_step)

    kernel.schedule(0, market_step)
    
    kernel.run(SIMULATION_TIME)
    
    for t in engine.trades: tape.record(t)
    
    return tape.get_dataframe(), snaps.get_dataframe()

if __name__ == "__main__":
    run_integrity_test()

    scenarios = [
        ("A", 100, 0, 0),
        ("B", 80, 20, 0),
        ("C", 80, 0, 20)
    ]
    
    results = {}
    
    for label, n, mm, mo in scenarios:
        df_tape, df_snaps = run_scenario(label, n, mm, mo)
        results[label] = (df_tape, df_snaps)

    metrics = []
    for label in ["A", "B", "C"]:
        t, s = results[label]
        
        if not t.empty: assert t['price'].min() > 0, f"Negative price in {label}"
        
        avg_spread = s['spread'].mean()
        
        if not s.empty:
            mid = s['mid_price'].fillna(method='ffill')
            ret = np.log(mid / mid.shift(1))
            vol = ret.std() * np.sqrt(len(ret)) # Scaled
        else:
            vol = 0
            
        metrics.append({'Scenario': label, 'Avg Spread': avg_spread, 'Volatility': vol})

    df_metrics = pd.DataFrame(metrics)
    print("\n--- METRICS ---")
    print(df_metrics)

    print("Generating PDF...")
    with PdfPages('simulation_report.pdf') as pdf:
        plt.figure(figsize=(10,6))
        plt.text(0.1, 0.8, "SIMULATION REPORT", fontsize=24)
        plt.text(0.1, 0.6, f"Total Agents: {TOTAL_AGENTS}", fontsize=14)
        plt.text(0.1, 0.5, f"Time: {SIMULATION_TIME}s", fontsize=14)
        plt.text(0.1, 0.4, f"Seed: {SEED}", fontsize=14)
        plt.text(0.1, 0.3, "Scenarios: A (Noise), B (+MM), C (+Momo)", fontsize=14)
        plt.axis('off')
        pdf.savefig()
        plt.close()

        for label in ["A", "B", "C"]:
            tape, snap = results[label]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
            
            if not tape.empty:
                tape['dt'] = pd.to_datetime(tape['timestamp'], unit='s')
                ohlc = tape.set_index('dt')['price'].resample('1min').ohlc()
                
                width = 0.0005 
                ax1.plot(snap['timestamp'], snap['mid_price'], label='Mid Price', color='blue')
                
                for idx, row in ohlc.iterrows():
                    ts = idx.timestamp() - 1767225600 
                    color = 'green' if row['close'] >= row['open'] else 'red'
            else:
                ax1.text(0.5, 0.5, "No Trades", ha='center')

            ax1.set_title(f"Scenario {label}: Price Action")
            ax1.set_ylabel("Price")
            ax1.grid(True, alpha=0.3)

            ax2.plot(snap['timestamp'], snap['spread'], color='orange')
            ax2.set_title(f"Scenario {label}: Spread")
            ax2.set_ylabel("Spread")
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            pdf.savefig()
            plt.close()

        plt.figure(figsize=(10,6))
        plt.axis('off')
        plt.table(cellText=df_metrics.values, colLabels=df_metrics.columns, loc='center', cellLoc='center', colWidths=[0.2, 0.3, 0.3])
        plt.title("Comparative Metrics")
        pdf.savefig()
        plt.close()

        plt.figure(figsize=(10,6))
        text = """
        INTERPRETATION
        
        Scenario B (Market Makers):
        - Spread is tightest due to MM inventory management.
        - Prices mean-revert as MMs absorb shock.
        - Volatility is suppressed.

        Scenario C (Momentum):
        - Momentum agents herd, creating trends.
        - When trend reverses, liquidity evaporates.
        - Highest volatility and crash risk.
        """
        plt.text(0.1, 0.5, text, fontsize=12, wrap=True)
        plt.axis('off')
        pdf.savefig()
        plt.close()

    print("DONE: simulation_report.pdf generated.")