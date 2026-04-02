"""
physics_state.py — Step 2: Physics State Vector (9,)
======================================================
Extracts the 7 joint angles + EE (x, y) from the KUKA arm and returns
a normalised (9,) float32 tensor ready for concatenation.

Architecture decision (Selma's review):
  - 7 DOF joint angles + EE position (x, y) = 9 values
  - All values normalised to [-1, 1]
  - EE z is omitted (partially redundant with joint configuration)
"""

import numpy as np
import pybullet as p
import torch

KUKA_DOF = 7  # Only real joints (0–6)

# Workspace bounds for EE normalisation
WS_LO = np.array([-0.8, -0.8, 0.0])
WS_HI = np.array([ 0.8,  0.8, 1.2])


def get_physics_state(kuka_id: int, device: str = "cpu") -> torch.Tensor:
    """
    Build the (9,) physics state vector for the given Kuka robot.

    Args:
        kuka_id : PyBullet body id of the KUKA arm.
        device  : Torch device string ("cpu" or "cuda").

    Returns:
        physics_t : torch.Tensor of shape (9,), float32, values in [-1, 1].
    """
    joint_angles = []

    for j in range(KUKA_DOF):
        info  = p.getJointInfo(kuka_id, j)
        state = p.getJointState(kuka_id, j)

        angle = state[0]
        lower, upper = info[8], info[9]

        if upper > lower:
            norm = 2.0 * (angle - lower) / (upper - lower) - 1.0
        else:
            norm = 0.0

        joint_angles.append(float(np.clip(norm, -1.0, 1.0)))

    assert len(joint_angles) == 7, f"Expected 7 joints, got {len(joint_angles)}"

    # End-effector position
    ee_state = p.getLinkState(kuka_id, KUKA_DOF - 1)
    ee_pos   = np.array(ee_state[0])  # (x, y, z)

    ee_norm = 2.0 * (ee_pos - WS_LO) / (WS_HI - WS_LO) - 1.0
    ee_norm = np.clip(ee_norm, -1.0, 1.0)
    ee_xy   = ee_norm[:2]  # only (x, y)

    # 7 joints + 2 EE = 9
    physics_state = np.array(joint_angles + ee_xy.tolist(), dtype=np.float32)
    physics_t     = torch.tensor(physics_state, dtype=torch.float32).to(device)

    assert physics_t.shape[0] == 9, \
        f"Physics state shape error: got {physics_t.shape}, expected (9,)"

    return physics_t
