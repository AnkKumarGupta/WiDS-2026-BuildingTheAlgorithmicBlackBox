from day2 import TradingEnv
import numpy as np

def test_environment():
    print("--- Testing TradingEnv v1 ---")
    
    env = TradingEnv()
    obs, info = env.reset(seed=42)
    
    print(f"Initial Observation: {obs}")
    print(f"Obs Shape: {obs.shape}")
    assert obs.shape == (4,), "Observation shape mismatch"
    
    done = False
    total_reward = 0
    steps = 0
    
    while not done:
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        steps += 1
        done = terminated or truncated
        
        if steps % 100 == 0:
            print(f"Step {steps}: Act={action}, Reward={reward:.2f}, NetWorth={info['net_worth']:.2f}, Price={info['price']:.2f}")

    print("\n--- Test Complete ---")
    print(f"Total Steps: {steps}")
    print(f"Final Net Worth: {info['net_worth']:.2f}")
    
    if np.isnan(obs).any():
        print("FAIL: NaNs detected in observation!")
    elif info['net_worth'] == 100000.0:
        print("WARNING: Net Worth didn't change (Environment might be frozen)")
    else:
        print("PASS: Environment appears stable.")

if __name__ == "__main__":
    test_environment()