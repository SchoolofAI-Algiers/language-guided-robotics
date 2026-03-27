from enum import Enum

import pybullet as p

class GraphicalMode(Enum):
    GUI = p.GUI # for graphical version
    DIRECT = p.DIRECT # for non-graphical version