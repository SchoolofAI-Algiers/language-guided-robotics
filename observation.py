import torch
import numpy as np

from config import DEVICE
from boxes    import seg_to_boxes, build_detection_vector
from backbone import extract_visual_features


def build_observation(
    rgb:      np.ndarray,
    depth:    np.ndarray,
    seg:      np.ndarray,
    backbone: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Assemble the full (281,) observation tensor from one rendered frame.

    Steps
    -----
    1. Extract visual features via FPN backbone   → (256,)
    2. Extract bounding boxes from seg mask        → list of dicts
    3. Build normalised detection vector           → (25,)
    4. Concatenate                                 → (281,)

    Parameters
    ----------
    rgb      : (H, W, 3)  uint8
    depth    : (H, W)     float32
    seg      : (H, W)     int32
    backbone : loaded FPN backbone (from backbone.load_backbone())

    Returns
    -------
    observation : torch.Tensor  shape (281,)  on DEVICE
    vis_feat    : torch.Tensor  shape (256,)  — visual component
    det_vec     : torch.Tensor  shape (25,)   — detection component
    """
    # Step 1 — visual features
    vis_feat = extract_visual_features(rgb, backbone)

    # Step 2 & 3 — bounding boxes → detection vector
    boxes   = seg_to_boxes(seg, depth)
    det_vec = build_detection_vector(boxes, seg, depth)

    # Step 4 — concatenate
    observation = torch.cat([vis_feat, det_vec], dim=0)   # (281,)

    return observation, vis_feat, det_vec