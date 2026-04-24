"""
observation.py — Steps 5 & 6: Physics Dropout + Final Observation Tensor
==========================================================================
Combines visual features, detection vector, and physics state into the
(546,) observation tensor consumed by the RL policy.

Physics dropout strategy (mitigates shortcut learning):
  - At each rollout START, sample mask = Bernoulli(0.7)
      mask = 1.0  with probability 0.70  → physics included
      mask = 0.0  with probability 0.30  → physics zeroed out
  - The mask is FIXED for the entire rollout (not per step).
  - At eval time, mask = 1.0 always (no dropout).

Observation layout:
  [vis_feat (512) | det_vec (25) | physics_state * mask (9)] → (546,)
"""

import numpy as np
import torch

PHYSICS_DROPOUT_RATE = 0.30

# Component dimensions
VIS_DIM     = 512
DET_DIM     = 25
PHYSICS_DIM = 9
OBS_DIM     = VIS_DIM + DET_DIM + PHYSICS_DIM  # 546


def sample_physics_mask(dropout_rate: float = PHYSICS_DROPOUT_RATE) -> float:
    """
    Sample a rollout-level physics mask.

    Returns:
        1.0 (physics on)  with probability (1 - dropout_rate)
        0.0 (physics off) with probability dropout_rate
    """
    return 0.0 if np.random.rand() < dropout_rate else 1.0


def build_observation(
    vis_feat: torch.Tensor,
    det_t:    torch.Tensor,
    physics_t: torch.Tensor,
    mask: float = 1.0,
) -> torch.Tensor:
    """
    Concatenate components into the (546,) observation tensor.

    Args:
        vis_feat  : (512,) — L2-normalised ResNet18 features.
        det_t     : (25,)  — L2-normalised detection + depth vector.
        physics_t : (9,)   — normalised joint angles + EE (x, y).
        mask      : float  — 1.0 = include physics, 0.0 = zero out.

    Returns:
        obs : (546,) float32 tensor on the same device as vis_feat.
    """
    physics_masked = physics_t * mask
    obs = torch.cat([vis_feat, det_t, physics_masked], dim=0)
    assert obs.shape[0] == OBS_DIM, \
        f"Observation shape mismatch: got {obs.shape[0]}, expected {OBS_DIM}"
    return obs
