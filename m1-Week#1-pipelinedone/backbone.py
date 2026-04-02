import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np

from config import DEVICE


# Build transform once at import time — no re-allocation per frame
_to_tensor = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),   # HWC uint8 → CHW float32 in [0, 1]
])


def load_backbone() -> torch.nn.Module:
    """
    Load the Faster R-CNN FPN backbone (ImageNet-pretrained), frozen for inference.

    Returns
    -------
    backbone : torch.nn.Module  on DEVICE, eval mode
    """
    backbone = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    ).backbone.to(DEVICE).eval()

    # Freeze all parameters — we are feature-extracting, not fine-tuning
    for param in backbone.parameters():
        param.requires_grad_(False)

    return backbone


def extract_visual_features(
    rgb:      np.ndarray,
    backbone: torch.nn.Module,
) -> torch.Tensor:
    """
    Run one forward pass through the FPN backbone and return a global embedding.

    The "0" feature map (finest scale, 256 channels) is global-average-pooled
    and L2-normalised to produce a (256,) vector.

    Parameters
    ----------
    rgb      : (H, W, 3) uint8 numpy array
    backbone : output of load_backbone()

    Returns
    -------
    torch.Tensor  shape (256,)  on DEVICE, L2-normalised
    """
    frame_t = _to_tensor(rgb).unsqueeze(0).to(DEVICE)   # (1, 3, H, W)

    with torch.no_grad():
        feature_maps = backbone(frame_t)   # dict: "0", "1", "2", "3", "pool"

    vis_feat = feature_maps["0"]                     # (1, 256, H', W')
    vis_feat = vis_feat.mean(dim=[2, 3]).squeeze(0)  # (256,)
    vis_feat = F.normalize(vis_feat, dim=0)

    return vis_feat