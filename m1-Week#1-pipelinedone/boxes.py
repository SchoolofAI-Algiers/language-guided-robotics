import numpy as np
import torch
import torch.nn.functional as F

from config import MAX_OBJECTS, FEATURES_PER_BOX, DEVICE


def seg_to_boxes(
    seg_map:   np.ndarray,
    depth_map: np.ndarray,
    min_pixels: int = 50,
) -> list[dict]:
    """
    Extract axis-aligned bounding boxes from a PyBullet segmentation mask.

    Each object in the mask becomes one dict:
        {
            "object_id":  int,
            "box":        [x1, y1, x2, y2],   # pixel coords
            "mean_depth": float,
        }

    Parameters
    ----------
    seg_map    : (H, W) int32   — PyBullet segmentation IDs
    depth_map  : (H, W) float32 — PyBullet depth buffer
    min_pixels : ignore objects with fewer pixels than this

    Returns
    -------
    list of box dicts, sorted by object_id
    """
    H, W = seg_map.shape
    boxes = []

    for obj_id in np.unique(seg_map):
        if obj_id <= 0:              # background / plane
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
    boxes:     list[dict],
    seg_map:   np.ndarray,
    depth_map: np.ndarray,
    max_objects:      int = MAX_OBJECTS,
    features_per_box: int = FEATURES_PER_BOX,
) -> torch.Tensor:
    """
    Pack bounding boxes into a fixed-size normalised tensor.

    Layout per slot: [x1_norm, y1_norm, x2_norm, y2_norm, mean_depth]
    Unused slots are zero-padded.

    Returns
    -------
    torch.Tensor  shape (max_objects * features_per_box,)  on DEVICE, L2-normalised
    """
    H, W = seg_map.shape
    vec  = np.zeros(max_objects * features_per_box, dtype=np.float32)

    for i, obj in enumerate(boxes[:max_objects]):
        x1, y1, x2, y2 = obj["box"]
        idx = i * features_per_box
        vec[idx : idx + 5] = [x1 / W, y1 / H, x2 / W, y2 / H, obj["mean_depth"]]

    det_t = torch.tensor(vec).to(DEVICE)
    det_t = F.normalize(det_t, dim=0)
    return det_t