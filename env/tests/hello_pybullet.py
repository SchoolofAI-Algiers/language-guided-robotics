import pybullet as p
import pybullet_data
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from env.src.config import GraphicalMode

# setup simulation
physicsClient = p.connect(GraphicalMode.DIRECT.value)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")
startPos = [0, 0, 1]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
boxId = p.loadURDF("r2d2.urdf", startPos, startOrientation)

# camera settings
cam_distance = 3
cam_yaw = 50
cam_pitch = -35
cam_target = [0, 0, 0.5]
width, height = 640, 480

# setup live display
plt.ion()
fig, ax = plt.subplots(figsize=(8, 6))
img_array = np.zeros((height, width, 3), dtype=np.uint8)
img_plot = ax.imshow(img_array)
ax.set_axis_off()
ax.set_title("PyBullet - R2D2 Drop Simulation")
fig.tight_layout()

view_matrix = p.computeViewMatrixFromYawPitchRoll(
    cameraTargetPosition=cam_target,
    distance=cam_distance,
    yaw=cam_yaw,
    pitch=cam_pitch,
    roll=0,
    upAxisIndex=2,
)
proj_matrix = p.computeProjectionMatrixFOV(
    fov=60, aspect=width / height, nearVal=0.1, farVal=100
)

try:
    for i in range(2400):  # 10 seconds at 240Hz
        p.stepSimulation()

        # render every 8th step (~30 fps display)
        if i % 8 == 0:
            _, _, rgb, _, _ = p.getCameraImage(
                width, height, view_matrix, proj_matrix,
                renderer=p.ER_TINY_RENDERER,
            )
            rgb_array = np.array(rgb, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
            img_plot.set_data(rgb_array)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    pos, orn = p.getBasePositionAndOrientation(boxId)
    print(f"Final position: {pos}")
    print(f"Final orientation: {orn}")
    print("Close the window to exit.")
    plt.ioff()
    plt.show()
finally:
    p.disconnect()
