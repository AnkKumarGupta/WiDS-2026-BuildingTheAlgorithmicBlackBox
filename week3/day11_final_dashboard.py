import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from stable_baselines3 import PPO
from day2 import TradingEnv
from requirements.market_maker_agent import MarketMakerAgent
from requirements.noise_agent import NoiseTrader
from requirements.momentum_agent import MomentumTrader

def run_simulation(env, agent_type, model=None):
    """Runs simulation and returns a DataFrame of detailed logs."""
    obs, _ = env.reset(seed=42)

    env.background_agents = []
    for i in range(20): env.background_agents.append(NoiseTrader(f"Noise_{i}", env.fv))
    for i in range(5): env.background_agents.append(MarketMakerAgent(f"MM_{i}"))

    logs = []
    mom_history = []
    terminated = False
    step = 0

    initial_wealth = env.rl_cash

    while not terminated and step < 2000:
        action = 0

        if agent_type == 'RL':
            action, _ = model.predict(obs, deterministic=True)
        elif agent_type == 'Momentum':
            price = env.last_mid_price
            mom_history.append(price)
            if len(mom_history) > 5:
                ret = (price - mom_history[-5]) / mom_history[-5]
                if ret > 0.001: action = 1 # Buy
                elif ret < -0.001: action = 2 # Sell
        elif agent_type == 'BuyHold':
            action = 1 if step == 0 else 0
        elif agent_type == 'Random':
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        mid_price = env.last_mid_price
        wealth = env.rl_cash + (env.rl_inventory * mid_price)

        logs.append({
            'Step': step,
            'Price': mid_price,
            'Action': action,
            'Wealth': wealth,
            'Strategy': agent_type,
            'Inventory': env.rl_inventory
        })
        step += 1

    df = pd.DataFrame(logs)
    df['Step_PnL'] = df['Wealth'].diff().fillna(0)
    df['Cum_PnL'] = (df['Wealth'] - initial_wealth) / initial_wealth * 100 # % Return
    return df

def generate_dashboard():
    print("--- DAY 11: INTERACTIVE DASHBOARD ---")

    env = TradingEnv()

    print("1. Loading RL Agent...")
    model = PPO(
        "MlpPolicy", env,
        learning_rate=0.00045, gamma=0.92, ent_coef=1e-5,
        batch_size=256, policy_kwargs={"net_arch": [64, 64]}, verbose=0
    )
    model.learn(total_timesteps=10000)

    print("2. Generating Data for Dashboard...")
    df_rl = run_simulation(env, 'RL', model)
    df_mom = run_simulation(env, 'Momentum')
    df_bh = run_simulation(env, 'BuyHold')

    print("3. Building Plotly Interface...")

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.4, 0.4, 0.2],
        subplot_titles=("Portfolio Performance (Wealth)", "Market Price & RL Agent Actions", "RL Agent Step PnL Distribution")
    )

    # --- CHART 1: WEALTH CURVES ---
    fig.add_trace(go.Scatter(x=df_rl['Step'], y=df_rl['Wealth'], name='RL Agent', line=dict(color='red', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_mom['Step'], y=df_mom['Wealth'], name='Momentum', line=dict(color='green', dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_bh['Step'], y=df_bh['Wealth'], name='Buy & Hold', line=dict(color='blue', dash='dash')), row=1, col=1)

    # --- CHART 2: PRICE & ACTIONS ---
    fig.add_trace(go.Scatter(x=df_rl['Step'], y=df_rl['Price'], name='Market Price', line=dict(color='black', width=1)), row=2, col=1)

    buys = df_rl[df_rl['Action'] == 1]
    fig.add_trace(go.Scatter(
        x=buys['Step'], y=buys['Price'], mode='markers', name='RL Buy',
        marker=dict(symbol='triangle-up', color='green', size=10)
    ), row=2, col=1)

    sells = df_rl[df_rl['Action'] == 2]
    fig.add_trace(go.Scatter(
        x=sells['Step'], y=sells['Price'], mode='markers', name='RL Sell',
        marker=dict(symbol='triangle-down', color='red', size=10)
    ), row=2, col=1)

    mom_buys = df_mom[df_mom['Action'] == 1]
    fig.add_trace(go.Scatter(
        x=mom_buys['Step'], y=mom_buys['Price'], mode='markers', name='Mom Buy (Ref)',
        marker=dict(symbol='triangle-up-open', color='lightgreen', size=8),
        visible='legendonly' 
    ), row=2, col=1)

    # --- CHART 3: PnL HISTOGRAM ---
    fig.add_trace(go.Histogram(
        x=df_rl['Step_PnL'], nbinsx=50, name='RL Step PnL', marker_color='purple'
    ), row=3, col=1)

    fig.update_layout(
        title="Black Box Report: AI Trading Agent Analysis",
        template="plotly_white",
        height=900,
        hovermode="x unified"
    )

    fig.write_html("dashboard.html")
    print("Dashboard saved to 'dashboard.html'.")

    try:
        fig.show()
    except:
        print("Could not render inline. Download the HTML file.")

if __name__ == "__main__":
    generate_dashboard()