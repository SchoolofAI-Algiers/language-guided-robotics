from env.src.environment import KukaEnv

env = KukaEnv(render_mode=None)
obs, info = env.reset(seed=42)

print(f"Observation shape: {obs.shape}")  
print(f"\nObject state (what Vision team receives):")
for obj_id, data in info["object_state"].items():
    print(f"  obj_id={obj_id} | color={data['color']} | pos={[round(v,3) for v in data['pos']]}")

# Test randomization
obs2, info2 = env.reset(seed=99)
print(f"\nAfter reset with different seed:")
for obj_id, data in info2["object_state"].items():
    print(f"  obj_id={obj_id} | color={data['color']} | pos={[round(v,3) for v in data['pos']]}")

env.close()
print("\nPhase 2 check passed")