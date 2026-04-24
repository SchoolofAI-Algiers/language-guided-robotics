import pybullet as p
import pybullet_data
import numpy as np

from config import WIDTH, HEIGHT


def connect_pybullet():
    """Connect to PyBullet in DIRECT (headless) mode and load a default scene."""
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    p.loadURDF("plane.urdf")
    p.loadURDF("r2d2.urdf",    basePosition=[ 0.0,  0.0, 1.0])
    p.loadURDF("cube.urdf",    basePosition=[ 0.5,  0.0, 0.5], globalScaling=0.5)
    p.loadURDF("sphere2.urdf", basePosition=[-0.5,  0.0, 0.5], globalScaling=0.5)

    for _ in range(100):
        p.stepSimulation()


def render_scene() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Render the current PyBullet scene from a fixed camera.

    Returns
    -------
    rgb   : np.ndarray  shape (H, W, 3)  uint8
    depth : np.ndarray  shape (H, W)     float32
    seg   : np.ndarray  shape (H, W)     int32   — object IDs per pixel
    """
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[2, 2, 2],
        cameraTargetPosition=[0, 0, 0],
        cameraUpVector=[0, 0, 1],
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=1.0, nearVal=0.1, farVal=10.0
    )

    _, _, rgb_raw, depth_raw, seg_raw = p.getCameraImage(
        width=WIDTH, height=HEIGHT,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_TINY_RENDERER,
    )

    rgb   = np.array(rgb_raw,   dtype=np.uint8  )[:, :, :3]
    depth = np.array(depth_raw, dtype=np.float32)
    seg   = np.array(seg_raw,   dtype=np.int32  )

    return rgb, depth, seg


def disconnect_pybullet():
    p.disconnect()