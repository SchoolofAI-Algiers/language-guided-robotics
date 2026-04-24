"""
Interface script for the Vision/ML team.
Shows how to get RGB frames + object states from the environment.
"""
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from env.src.environment import KukaEnv

env = KukaEnv(render_mode="rgb_array")
obs, info = env.reset(seed=99)

frame = env.render()
object_state = info["object_state"]

print(f"Frame shape:  {frame.shape}")
print(f"Frame dtype:  {frame.dtype}")
print(f"Num objects:  {len(object_state)}")
print(f"\nObject states (pass obj_id to get_boxes() and get_state()):")
for obj_id, data in object_state.items():
    print(f"  obj_id={obj_id} | color={data['color']} | pos={[round(v,3) for v in data['pos']]}")

# ── Show the frame ─────────────────────────────────────────────
plt.imshow(frame)
plt.title("Robotics → Vision interface | objects visible in frame")
plt.axis("off")
plt.tight_layout()
plt.show()

env.close()