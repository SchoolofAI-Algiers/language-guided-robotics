"""
visual_features.py — Step 3: Improved Visual Features (ResNet18 + Augmentation)
================================================================================
Architecture locked (Selma's decision): ResNet18 → (512,) global average pool.

Anti-shortcut augmentation applied to every frame:
  - Random colour jitter  (brightness / contrast / saturation / hue)
  - Random resized crop   (forces scale & position invariance)

At eval time a milder version is used; training uses the full augmentation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T


# ── Transforms ───────────────────────────────────────────────────────────────

INFERENCE_AUGMENT = T.Compose([
    T.ToPILImage(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.05),
    T.RandomResizedCrop(size=(224, 224), scale=(0.85, 1.0)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

CLEAN_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_backbone(device: str = "cpu") -> nn.Module:
    """
    Load pretrained ResNet18, remove the classification head,
    and return the backbone in eval mode.

    Returns:
        resnet : nn.Module outputting (B, 512) feature vectors.
    """
    resnet = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.DEFAULT
    )
    resnet.fc = nn.Identity()  # output is (512,) after global avg pool
    return resnet.to(device).eval()


def extract_features(
    rgb: np.ndarray,
    backbone: nn.Module,
    device: str = "cpu",
    augment: bool = True,
) -> torch.Tensor:
    """
    Extract normalised (512,) visual feature vector from an RGB frame.

    Args:
        rgb       : (H, W, 3) uint8 numpy array.
        backbone  : ResNet18 backbone (from build_backbone()).
        device    : Torch device string.
        augment   : If True, apply anti-shortcut augmentation (training mode).

    Returns:
        vis_feat : (512,) float32 tensor, L2-normalised.
    """
    transform = INFERENCE_AUGMENT if augment else CLEAN_TRANSFORM
    frame     = transform(rgb).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    with torch.no_grad():
        feat = backbone(frame).squeeze(0)  # (512,)

    return F.normalize(feat, dim=0)
