"""
pipeline.py — End-to-End M3 Vision Week 2 Pipeline
====================================================
High-level convenience function that runs all steps in one call.

Usage
-----
    from pipeline import build_pipeline, run_step

    kuka_id, backbone = build_pipeline(device="cpu")
    obs = run_step(kuka_id, backbone, train=True, device="cpu")
"""

import torch

from src.env_render      import connect_env, render_scene
from src.physics_state   import get_physics_state
from src.visual_features import build_backbone, extract_features
from src.detection       import seg_to_boxes, build_detection_vector
from src.observation     import sample_physics_mask, build_observation, OBS_DIM


def build_pipeline(device: str = "cpu"):
    """
    Initialise the PyBullet environment and the ResNet18 backbone.

    Returns:
        kuka_id  : int       — PyBullet body id of the KUKA arm.
        backbone : nn.Module — Pretrained ResNet18 (eval mode).
    """
    kuka_id, _ = connect_env()
    backbone   = build_backbone(device=device)
    return kuka_id, backbone


def run_step(
    kuka_id:  int,
    backbone: torch.nn.Module,
    train:    bool  = True,
    mask:     float = None,   # None → sample automatically
    device:   str   = "cpu",
) -> torch.Tensor:
    """
    Execute one observation-building step (one env frame).

    Args:
        kuka_id  : PyBullet body id of the KUKA arm.
        backbone : ResNet18 backbone.
        train    : If True, apply augmentation and physics dropout.
        mask     : Override the physics mask (useful for eval, set to 1.0).
        device   : Torch device string.

    Returns:
        obs : (546,) float32 observation tensor.
    """
    # Step 1 — render
    rgb, depth, seg = render_scene()

    # Step 2 — physics state
    physics_t = get_physics_state(kuka_id, device=device)

    # Step 3 — visual features
    vis_feat = extract_features(rgb, backbone, device=device, augment=train)

    # Step 4 — detection vector
    boxes = seg_to_boxes(seg, depth)
    det_t = build_detection_vector(boxes, seg, depth, device=device)

    # Step 5 — physics dropout mask
    if mask is None:
        mask = sample_physics_mask() if train else 1.0

    # Step 6 — concatenate
    obs = build_observation(vis_feat, det_t, physics_t, mask=mask)
    return obs
