"""
detection.py — Step 4: Bounding Boxes from Segmentation
=========================================================
PyBullet's segmentation mask provides exact per-object bounding boxes for free.

Output: fixed (25,) detection vector compatible with the RL policy input.
  Layout: [x1_norm, y1_norm, x2_norm, y2_norm, mean_depth] × MAX_OBJECTS
"""

import numpy as np
import torch
import torch.nn.functional as F

MAX_OBJECTS       = 5
FEATURES_PER_BOX  = 5   # x1_norm, y1_norm, x2_norm, y2_norm, mean_depth
DETECTION_DIM     = MAX_OBJECTS * FEATURES_PER_BOX  # 25


def seg_to_boxes(seg_map: np.ndarray, depth_map: np.ndarray, min_pixels: int = 30):
    """
    Convert a PyBullet segmentation map to a list of bounding-box dicts.

    Args:
        seg_map    : (H, W) int32 — object id per pixel (0 = background).
        depth_map  : (H, W) float32 — linearised depth values.
        min_pixels : Minimum pixel count to keep an object.

    Returns:
        List of dicts with keys: object_id, box [x1,y1,x2,y2], mean_depth.
    """
    H, W  = seg_map.shape
    boxes = []
    for obj_id in np.unique(seg_map):
        if obj_id <= 0:
            continue
        mask = (seg_map == obj_id)
        if mask.sum() < min_pixels:
            continue
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        boxes.append({
            "object_id":  int(obj_id),
            "box":        [int(cols.min()), int(rows.min()),
                           int(cols.max()), int(rows.max())],
            "mean_depth": float(depth_map[mask].mean()),
        })
    return boxes


def build_detection_vector(
    boxes,
    seg_map: np.ndarray,
    depth_map: np.ndarray,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Pack bounding boxes into a fixed-length (25,) tensor and L2-normalise it.

    Args:
        boxes     : Output of seg_to_boxes().
        seg_map   : (H, W) int32 — used only for shape.
        depth_map : (H, W) float32 — used only for shape (already in boxes).
        device    : Torch device string.

    Returns:
        det_t : (25,) float32 tensor, L2-normalised.
    """
    H, W = seg_map.shape
    vec  = np.zeros(DETECTION_DIM, dtype=np.float32)

    for i, obj in enumerate(boxes[:MAX_OBJECTS]):
        x1, y1, x2, y2 = obj["box"]
        idx  = i * FEATURES_PER_BOX
        vec[idx : idx + 5] = [x1 / W, y1 / H, x2 / W, y2 / H, obj["mean_depth"]]

    det_t = torch.tensor(vec).to(device)
    return F.normalize(det_t, dim=0)
