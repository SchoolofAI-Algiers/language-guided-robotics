import numpy as np
import time
import matplotlib
matplotlib.use("TkAgg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pybullet as p
from env.src.environment import KukaEnv
from env.src.config import RENDER_WIDTH, RENDER_HEIGHT, END_EFFECTOR_LINK_INDEX, JOINT_LOWER_LIMITS, JOINT_UPPER_LIMITS

env = KukaEnv(render_mode="rgb_array")
obs, info = env.reset(seed=42)

first_obj_id = list(info["object_state"].keys())[0]
target_pos = info["object_state"][first_obj_id]["pos"]
target_color = info["object_state"][first_obj_id]["color"]
print(f"Targeting: {target_color} object at {target_pos}")

plt.ion()
fig, ax = plt.subplots(figsize=(8, 6))
img_plot = ax.imshow(np.zeros((RENDER_HEIGHT, RENDER_WIDTH, 3), dtype=np.uint8))
ax.set_axis_off()
fig.tight_layout()

min_dist_seen = 999

for step in range(300):
    joint_angles = p.calculateInverseKinematics(
        env._kuka_id,
        END_EFFECTOR_LINK_INDEX,
        target_pos,
        physicsClientId=env._physics_client_id,
    )
    joint_angles = np.array(joint_angles[:7], dtype=np.float32)
    midpoint = (JOINT_UPPER_LIMITS + JOINT_LOWER_LIMITS) / 2.0
    half_range = (JOINT_UPPER_LIMITS - JOINT_LOWER_LIMITS) / 2.0
    action = np.clip((joint_angles - midpoint) / half_range, -1.0, 1.0)

    obs, _, _, _, info = env.step(action)

    # get real distance between EE and object
    ee_pos = np.array(info["ee_position"])
    obj_pos, _ = p.getBasePositionAndOrientation(
        first_obj_id,
        physicsClientId=env._physics_client_id,
    )
    dist = np.linalg.norm(ee_pos - np.array(obj_pos))

    if dist < min_dist_seen:
        min_dist_seen = dist

    rgb = env.render()
    img_plot.set_data(rgb)
    ax.set_title(
        f"Step {step} | "
        f"Dist to object: {dist:.4f}m | "
        f"Min so far: {min_dist_seen:.4f}m | "
        f"Gripper: {'HOLDING ✅' if info['gripper_state'] == 1.0 else 'empty'}"
    )
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    time.sleep(1/30)

print(f"\nClosest the EE got to object: {min_dist_seen:.4f}m")
print(f"Current GRASP_DISTANCE is: 0.15m")
if min_dist_seen > 0.05:
    print(f"❌ Never got close enough — increase GRASP_DISTANCE to at least {min_dist_seen + 0.02:.2f}")
else:
    print("✅ Got close enough — grasp should have triggered!")

plt.ioff()
plt.show()
env.close()