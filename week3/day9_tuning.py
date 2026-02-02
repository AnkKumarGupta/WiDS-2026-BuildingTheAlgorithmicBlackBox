import optuna
import gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from day2 import TradingEnv
from requirements.market_maker_agent import MarketMakerAgent
from requirements.noise_agent import NoiseTrader

def make_env():
    env = TradingEnv()
    env.reset(seed=42)
    env.background_agents = []
    for i in range(20):
        env.background_agents.append(NoiseTrader(f"Noise_{i}", env.fv))
    for i in range(5):
        env.background_agents.append(MarketMakerAgent(f"MM_{i}"))
    return env

def objective(trial):
 
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)

    gamma = trial.suggest_float("gamma", 0.9, 0.9999)

    ent_coef = trial.suggest_float("ent_coef", 1e-8, 0.1, log=True)

    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium"])
    if net_arch_type == "small":
        net_arch = [64, 64]
    else:
        net_arch = [128, 128]

    env = make_vec_env(make_env, n_envs=1)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        ent_coef=ent_coef,
        batch_size=batch_size,
        policy_kwargs={"net_arch": net_arch},
        verbose=0  
    )

    try:
        model.learn(total_timesteps=10000)
    except Exception as e:
        print(f"Trial failed: {e}")
        return -99999

    eval_env = make_env()
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5)

    return mean_reward

def run_tuning():
    print("--- HYPERPARAMETER TUNING (OPTUNA) ---")

    study = optuna.create_study(direction="maximize")

    print("Starting optimization (20 Trials)... ")
    study.optimize(objective, n_trials=20)

    print("\n--- TUNING COMPLETE ---")
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best Value (Reward): {study.best_value:.4f}")
    print("Best Params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    df = study.trials_dataframe()
    df.to_csv("optuna_results.csv")
    print("Saved results to optuna_results.csv")

    return study

if __name__ == "__main__":
    study = run_tuning()

    try:
        from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plot_optimization_history(study)
        plt.title("Optimization History")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))
        plot_param_importances(study)
        plt.title("Hyperparameter Importance")
        plt.tight_layout()
        plt.show()

    except ImportError:
        print("Visualization skipped. Check optuna_results.csv")