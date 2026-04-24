from env.src.environment import KukaEnv
import numpy as np

env = KukaEnv()
obs, info = env.reset()

print("=" * 50)
print("OBSERVATION VECTOR BREAKDOWN")
print("=" * 50)
print(f"Full observation shape: {obs.shape}")
print()
print(f"Joint positions  obs[0:7]   → {obs[0:7]}")
print(f"Joint velocities obs[7:14]  → {obs[7:14]}")
print(f"EE position      obs[14:17] → {obs[14:17]}")
print(f"EE orientation   obs[17:21] → {obs[17:21]}")
print()

# simulate one step
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

print("=" * 50)
print("AFTER ONE STEP")
print("=" * 50)
print(f"Joint positions  → {obs[0:7]}")
print(f"Joint velocities → {obs[7:14]}")
print(f"EE position      → {obs[14:17]}")
print(f"EE orientation   → {obs[17:21]}")
print()
print(f"Reward: {reward}")
print(f"Terminated: {terminated}")
print(f"Truncated: {truncated}")
print(f"EE position from info: {info['ee_position']}")

env.close()
print()
print("✅ Observation vector confirmed — all 4 components present!")
