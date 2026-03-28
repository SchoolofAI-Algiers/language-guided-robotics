from enum import Enum

import pybullet as p
import numpy as np

# pybullet
class GraphicalMode(Enum):
    GUI = p.GUI # for graphical version
    DIRECT = p.DIRECT # for non-graphical version
    
class RenderMode(Enum):
    HUMAN = "human"
    RGB_ARRAY = "rgb_array"
    
# environment
NUM_JOINTS = 7
END_EFFECTOR_LINK_INDEX = 6
MAX_FORCE = 300.0
SIM_TIMESTEP = 1.0 / 240.0
SIM_STEPS_PER_ACTION = 8  # 240 Hz sim / 8 = 30 Hz control
MAX_EPISODE_STEPS = 500

JOINT_LOWER_LIMITS = np.array(
    [-2.9671, -2.0944, -2.9671, -2.0944, -2.9671, -2.0944, -3.0543],
    dtype=np.float32,
)
JOINT_UPPER_LIMITS = np.array(
    [2.9671, 2.0944, 2.9671, 2.0944, 2.9671, 2.0944, 3.0543],
    dtype=np.float32,
)
HOME_POSITION = np.zeros(7, dtype=np.float32)

# maximum joint velocity (rad/s) — used to bound the observation space
MAX_JOINT_VELOCITY = 10.0

# workspace bounds (meters)
WORKSPACE_LOW = np.array([-1.5, -1.5, 0.0], dtype=np.float32)
WORKSPACE_HIGH = np.array([1.5, 1.5, 1.5], dtype=np.float32)

# render
CAM_DISTANCE = 1.5
CAM_YAW = 50
CAM_PITCH = -35
CAM_TARGET = [0, 0, 0.5]

RENDER_FPS = 30
RENDER_WIDTH = 640
RENDER_HEIGHT = 480