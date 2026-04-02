"""
env_render.py — Step 1: KukaEnv Scene Rendering
================================================
Renders RGB, depth, and segmentation frames from the PyBullet KUKA IIWA environment.
"""

import numpy as np
import pybullet as p
import pybullet_data

WIDTH, HEIGHT = 224, 224

# Fixed overhead-side camera (consistent with RL policy camera)
VIEW_MATRIX = None  # initialised in connect_env()
PROJ_MATRIX = None


def connect_env():
    """Connect PyBullet in DIRECT mode and load the KukaEnv scene."""
    global VIEW_MATRIX, PROJ_MATRIX

    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # Plane + Kuka arm
    plane_id = p.loadURDF("plane.urdf")
    kuka_id  = p.loadURDF("kuka_iiwa/model.urdf", basePosition=[0, 0, 0], useFixedBase=True)

    # Three coloured blocks (red, green, blue)
    block_ids = []
    block_configs = [
        ([0.5,  0.0, 0.02], [1, 0, 0, 1], "red"),
        ([0.5,  0.2, 0.02], [0, 1, 0, 1], "green"),
        ([0.5, -0.2, 0.02], [0, 0, 1, 1], "blue"),
    ]
    for pos, colour, name in block_configs:
        bid = p.loadURDF("cube_small.urdf", basePosition=pos, globalScaling=0.05)
        p.changeVisualShape(bid, -1, rgbaColor=colour)
        block_ids.append(bid)

    # Warm-up simulation
    for _ in range(100):
        p.stepSimulation()

    VIEW_MATRIX = p.computeViewMatrix(
        cameraEyePosition=[1.5, 0, 1.2],
        cameraTargetPosition=[0.5, 0, 0],
        cameraUpVector=[0, 0, 1],
    )
    PROJ_MATRIX = p.computeProjectionMatrixFOV(
        fov=60, aspect=1.0, nearVal=0.1, farVal=10.0
    )

    return kuka_id, block_ids


def render_scene():
    """
    Capture one frame from the fixed camera.

    Returns:
        rgb   : (H, W, 3) uint8
        depth : (H, W)    float32
        seg   : (H, W)    int32
    """
    _, _, rgb_raw, depth_raw, seg_raw = p.getCameraImage(
        width=WIDTH, height=HEIGHT,
        viewMatrix=VIEW_MATRIX,
        projectionMatrix=PROJ_MATRIX,
        renderer=p.ER_TINY_RENDERER,
    )

    rgb   = np.array(rgb_raw,   dtype=np.uint8  )[:, :, :3]
    depth = np.array(depth_raw, dtype=np.float32)
    seg   = np.array(seg_raw,   dtype=np.int32  )
    return rgb, depth, seg
