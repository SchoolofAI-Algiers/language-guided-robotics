import torch

# Image dimensions
WIDTH  = 224
HEIGHT = 224

# Detection
MAX_OBJECTS      = 5
FEATURES_PER_BOX = 5   # x1_norm, y1_norm, x2_norm, y2_norm, mean_depth

# Observation tensor
VIS_DIM  = 256                              # FPN backbone output
DET_DIM  = MAX_OBJECTS * FEATURES_PER_BOX  # 25
OBS_DIM  = VIS_DIM + DET_DIM               # 281

# RL
ACTION_DIM = 7   # 7-DOF arm

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")