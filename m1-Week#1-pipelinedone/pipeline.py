import torch

from render      import connect_pybullet, render_scene, disconnect_pybullet
from backbone    import load_backbone
from observation import build_observation


def run_pipeline(verbose: bool = True) -> torch.Tensor:
    """

    1. Connect PyBullet and render a scene
    2. Load the FPN backbone
    3. Build the (281,) observation tensor
    4. Disconnect PyBullet

    Parameters
    ----------
    verbose : print shape/stats at each step

    Returns
    -------
    observation : torch.Tensor  shape (281,)
    """
    # --- render ---
    connect_pybullet()
    rgb, depth, seg = render_scene()
    disconnect_pybullet()

    if verbose:
        print(f"[render]  rgb={rgb.shape}  depth={depth.shape}  seg={seg.shape}")

    # --- backbone ---
    backbone = load_backbone()

    if verbose:
        print("[backbone] FPN loaded")

    # --- observation ---
    observation, vis_feat, det_vec = build_observation(rgb, depth, seg, backbone)

    if verbose:
        print(f"[obs] vis_feat={vis_feat.shape}  det_vec={det_vec.shape}")
        print(f"[obs] observation={observation.shape}")
        print(f"[obs] min={observation.min():.4f}  "
              f"max={observation.max():.4f}  "
              f"mean={observation.mean():.4f}")

    return observation