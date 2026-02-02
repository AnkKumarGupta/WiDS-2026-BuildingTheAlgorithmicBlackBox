import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

from requirements.matching_engine import MatchingEngine, Order
from requirements.event_loop import SimulationKernel
from requirements.noise_agent import NoiseTrader
from requirements.market_maker_agent import MarketMakerAgent

class TradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, config=None):
        super(TradingEnv, self).__init__()
        
        self.max_steps = 1000        # Episode length
        self.step_size = 10.0        # Simulation seconds per RL step
        self.max_inventory = 100     # Normalization factor
        self.trade_qty = 10          # Fixed trade size
        self.transaction_cost = 0.0001 # Cost per trade (approx spread/fees)
        
        # --- Action Space (Discrete) ---
        # 0 = Hold
        # 1 = Buy (Market Order)
        # 2 = Sell (Market Order)
        self.action_space = spaces.Discrete(3)

        # --- Observation Space (Continuous) ---
        # [Log_Return, Spread, Vol_Imbalance, Norm_Inventory]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, 0, 0, -1], dtype=np.float32), 
            high=np.array([np.inf, np.inf, 1, 1], dtype=np.float32), 
            dtype=np.float32
        )

        self.kernel = None
        self.engine = None
        self.background_agents = []
        self.current_step = 0
        self.last_mid_price = 100.0
        self.last_net_worth = 100000.0

        self.risk_aversion = 0.01     # Penalty for volatility
        self.inventory_penalty = 0.001 # Penalty for holding large positions
        
        self.pnl_history = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.kernel = SimulationKernel()
        self.engine = MatchingEngine()
        
        self.kernel.engine = self.engine
        
        self.background_agents = []
        
        class SimpleFV:
            def __init__(self): self.current_value = 100.0
            def step(self): self.current_value += np.random.normal(0, 0.05)
            
        self.fv = SimpleFV()
        
        for i in range(10):
            self.background_agents.append(NoiseTrader(f"Noise_{i}", self.fv))
        for i in range(2):
            self.background_agents.append(MarketMakerAgent(f"MM_{i}"))

        self._run_background_simulation(duration=60.0)
        
        self.rl_inventory = 0
        self.rl_cash = 100000.0
        self.current_step = 0
        
        best_bid, best_ask = self.engine.get_l1_snapshot()
        self.last_mid_price = (best_bid + best_ask) / 2.0 if (best_bid and best_ask) else 100.0
        self.last_net_worth = self._calculate_net_worth(self.last_mid_price)

        self.pnl_history = []
        return self._get_observation(), {}

    def step(self, action):
        trade_occurred = False
        
        if action == 1: 
            order = Order('Buy', None, self.trade_qty, 'RL_Agent', self.kernel.time)
            self.engine.process(order)
            _, best_ask = self.engine.get_l1_snapshot()
            exec_price = best_ask if best_ask else self.last_mid_price
            self.rl_inventory += self.trade_qty
            self.rl_cash -= exec_price * self.trade_qty
            self.rl_cash -= (exec_price * self.trade_qty) * self.transaction_cost 
            trade_occurred = True

        elif action == 2: 
            order = Order('Sell', None, self.trade_qty, 'RL_Agent', self.kernel.time)
            self.engine.process(order)
            best_bid, _ = self.engine.get_l1_snapshot()
            exec_price = best_bid if best_bid else self.last_mid_price
            self.rl_inventory -= self.trade_qty
            self.rl_cash += exec_price * self.trade_qty
            self.rl_cash -= (exec_price * self.trade_qty) * self.transaction_cost
            trade_occurred = True

        self._run_background_simulation(duration=self.step_size)

        best_bid, best_ask = self.engine.get_l1_snapshot()
        
        if best_bid is None or best_ask is None:
            mid_price = self.last_mid_price
        else:
            mid_price = (best_bid + best_ask) / 2.0

        
        current_net_worth = self._calculate_net_worth(mid_price)
        step_pnl = current_net_worth - self.last_net_worth
        
        self.pnl_history.append(step_pnl)
        if len(self.pnl_history) > 50: 
            self.pnl_history.pop(0) 
            
        volatility = np.std(self.pnl_history) if len(self.pnl_history) > 10 else 0.0
        
        norm_inv = self.rl_inventory / self.max_inventory
        inventory_risk = norm_inv ** 2
        
        penalty_val = (self.risk_aversion * volatility) + (self.inventory_penalty * inventory_risk)
        reward = step_pnl - penalty_val

        self.last_net_worth = current_net_worth
        self.last_mid_price = mid_price
        self.current_step += 1
        
        terminated = False
        truncated = False
        if self.current_step >= self.max_steps: truncated = True
        if current_net_worth < 0: terminated = True; reward -= 1000

        info = {
            'net_worth': current_net_worth,
            'step_pnl': step_pnl,
            'reward': reward,
            'penalty': penalty_val,
            'inventory': self.rl_inventory
        }

        return self._get_observation(), reward, terminated, truncated, info

    
    def _run_background_simulation(self, duration):
        end_time = self.kernel.time + duration
        
        dt = 1.0 
        while self.kernel.time < end_time:
            self.kernel.time += dt
            self.fv.step()
            
            snapshot = {'mid_price': self.last_mid_price} 
            for agent in self.background_agents:
                if random.random() < 0.2: 
                    actions = agent.get_action(snapshot)
                    for intent in actions:
                        o = Order(intent.side, intent.price, intent.qty, agent.id, self.kernel.time)
                        self.engine.process(o)
            
    def _calculate_net_worth(self, price):
        return self.rl_cash + (self.rl_inventory * price)

    def _get_observation(self):
        best_bid, best_ask = self.engine.get_l1_snapshot()
        
        if best_bid and best_ask:
            mid = (best_bid + best_ask) / 2.0
        else:
            mid = self.last_mid_price
            
        log_ret = np.log(mid / self.last_mid_price) if self.last_mid_price > 0 else 0
        
        spread = (best_ask - best_bid) / mid if (best_bid and best_ask) else 0
        
        # Volume Imbalance (Simplified - using Book Depth count)
        b_vol = len(self.engine.bids)
        a_vol = len(self.engine.asks)
        total_vol = b_vol + a_vol
        imbalance = b_vol / total_vol if total_vol > 0 else 0.5
        
        norm_inv = self.rl_inventory / self.max_inventory
        
        obs = np.array([log_ret, spread, imbalance, norm_inv], dtype=np.float32)
        
        if np.isnan(obs).any():
            obs = np.nan_to_num(obs)
            
        return obs