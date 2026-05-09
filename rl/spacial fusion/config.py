from enum import Enum
import pybullet as p
import numpy as np

class GraphicalMode(Enum):
    GUI = p.GUI
    DIRECT = p.DIRECT

class RenderMode(Enum):
    HUMAN = "human"
    RGB_ARRAY = "rgb_array"

NUM_JOINTS = 7
END_EFFECTOR_LINK_INDEX = 6
MAX_FORCE = 300.0
SIM_TIMESTEP = 1.0 / 240.0
SIM_STEPS_PER_ACTION = 4   # reduced from 8 for faster training
MAX_EPISODE_STEPS = 500

JOINT_LOWER_LIMITS = np.array([-2.9671, -2.0944, -2.9671, -2.0944, -2.9671, -2.0944, -3.0543], dtype=np.float32)
JOINT_UPPER_LIMITS = np.array([2.9671,  2.0944,  2.9671,  2.0944,  2.9671,  2.0944,  3.0543], dtype=np.float32)
HOME_POSITION = np.zeros(7, dtype=np.float32)
MAX_JOINT_VELOCITY = 10.0

WORKSPACE_LOW  = np.array([-1.5, -1.5, 0.0], dtype=np.float32)
WORKSPACE_HIGH = np.array([1.5,  1.5,  1.5], dtype=np.float32)

# Reduced resolution — SigLIP/ResNet native size, saves rendering time
CAM_DISTANCE = 1.5
CAM_YAW = 50
CAM_PITCH = -35
CAM_TARGET = [0, 0, 0.5]
RENDER_FPS = 30
RENDER_WIDTH = 224
RENDER_HEIGHT = 224

NUM_OBJECTS = 4
OBJECT_COLORS = {
    "red":    [1.0, 0.0, 0.0, 1.0],
    "green":  [0.0, 1.0, 0.0, 1.0],
    "blue":   [0.0, 0.0, 1.0, 1.0],
    "yellow": [1.0, 1.0, 0.0, 1.0],
}
OBJECT_SHAPES   = ["box", "sphere", "cylinder"]
OBJECT_SIZE_MIN = 0.02
OBJECT_SIZE_MAX = 0.05
TABLE_POSITION      = [0.5, 0.0, 0.2]
TABLE_HALF_EXTENTS  = [0.4, 0.4, 0.05]
TABLE_SURFACE_Z     = 0.42
SPAWN_RANGE         = 0.22
