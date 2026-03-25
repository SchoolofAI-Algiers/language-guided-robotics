import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────
# DEVICE SETUP
# ─────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Vision] Device: {DEVICE}")

# ══════════════════════════════════════════════════════════════
# SECTION 1 — CNN FEATURE EXTRACTOR
# Milestone: given a rendered frame, output a fixed-size
# feature vector representing the scene.
# ══════════════════════════════════════════════════════════════

def build_resnet_extractor():
    """
    Load ResNet18 pretrained on ImageNet.
    Strip the final FC classifier head so the network
    outputs a 512-dim spatial feature vector per crop.

    Returns:
        extractor  : torch.nn.Sequential — (N, 3, H, W) → (N, 512)
        preprocess : torchvision transform pipeline
    """
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(DEVICE)
    # Remove AvgPool + FC; keep everything up to the last conv block
    extractor = torch.nn.Sequential(*list(resnet.children())[:-1]).to(DEVICE)
    extractor.eval()

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return extractor, preprocess


# Build once at module level — shared by all functions below
_EXTRACTOR, _PREPROCESS = build_resnet_extractor()
print("[Vision] ResNet18 extractor ready — output: 512-dim per crop")


def resnet_features(crops_list, crop_size=64):
    """
    Extract L2-normalised 512-dim feature vectors for a list of image crops.
    All crops are processed in a single batched forward pass for efficiency.

    Args:
        crops_list : list of uint8 NumPy arrays (H×W×3, BGR or RGB)
        crop_size  : resize target before feeding to ResNet (default 64)

    Returns:
        torch.Tensor of shape (N, 512), float32, L2-normalised (unit sphere)
    """
    tensors = []
    for crop in crops_list:
        crop_resized = cv2.resize(crop, (crop_size, crop_size),
                                  interpolation=cv2.INTER_LINEAR)
        tensors.append(_PREPROCESS(crop_resized))

    batch = torch.stack(tensors).to(DEVICE)          # (N, 3, crop_size, crop_size)

    with torch.no_grad():
        out = _EXTRACTOR(batch)                       # (N, 512, 1, 1)

    out = out.squeeze(-1).squeeze(-1)                 # (N, 512)
    return F.normalize(out, dim=1)                    # unit-sphere normalised


# ══════════════════════════════════════════════════════════════
# SECTION 2 — PHYSICS STATE EXTRACTOR
# Pulls ground-truth position, velocity, orientation from PyBullet.
# ══════════════════════════════════════════════════════════════

def get_state(object_ids, sim):
    """
    Query PyBullet for per-object physics state.

    For each object returns a 9-dim tensor:
        [x, y, z,  vx, vy, vz,  roll, pitch, yaw]
    Each vector is normalised so no single axis dominates training.

    Args:
        object_ids : list of int — PyBullet body IDs
        sim        : pybullet module handle

    Returns:
        dict  {obj_id: torch.Tensor(9,)}
    """
    features = {}
    for obj_id in object_ids:
        pos, orn = sim.getBasePositionAndOrientation(obj_id)
        vel, _   = sim.getBaseVelocity(obj_id)
        euler    = sim.getEulerFromQuaternion(orn)

        raw = torch.tensor(
            list(pos) + list(vel) + list(euler),
            dtype=torch.float32                       # (9,)
        )
        features[obj_id] = raw / (raw.abs().max() + 1e-6)  # safe normalisation
    return features


# ══════════════════════════════════════════════════════════════
# SECTION 3 — COMBINED VISUAL FEATURES (main deliverable)
# ══════════════════════════════════════════════════════════════

def visual_features(frame, boxes, sim):
    """
    Week 1–2 Vision/ML track deliverable.

    Extracts two complementary feature representations per visible object:

    visual — WHAT the object looks like (colour, texture, shape)
              ResNet18 crop features, one batched forward pass, 512-dim each

    state  — WHERE the object is (position, velocity, orientation)
              PyBullet ground truth, 9-dim each

    Args:
        frame  : uint8 ndarray (H, W, 3) — from capture()
        boxes  : dict {obj_id: box_dict}  — from get_boxes()
        sim    : pybullet module handle

    Returns:
        (visual, state) tuple
            visual : {obj_id: Tensor(512,)} — L2-normalised ResNet18 features
            state  : {obj_id: Tensor(9,)}   — normalised physics state
    """
    obj_ids, crops = [], []

    for obj_id, box in boxes.items():
        crop = frame[box["y_min"]:box["y_max"],
                     box["x_min"]:box["x_max"]]
        if crop.size == 0:
            continue
        crops.append(crop)
        obj_ids.append(obj_id)

    if not obj_ids:
        return {}, {}

    # Single batched forward pass for all visible objects
    feats  = resnet_features(crops)                   # (N, 512)
    visual = {oid: feats[k] for k, oid in enumerate(obj_ids)}
    state  = get_state(obj_ids, sim)

    return visual, state
