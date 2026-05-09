import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Beta Vision] Device: {DEVICE}")


def build_4channel_resnet():
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    old_conv = resnet.conv1
    new_conv = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        new_conv.weight[:, :3] = old_conv.weight
        new_conv.weight[:, 3] = 0.0
    resnet.conv1 = new_conv
    extractor = nn.Sequential(*list(resnet.children())[:-1]).to(DEVICE)
    extractor.eval()
    print("[Beta Vision] 4-channel ResNet18 ready — output: 512-dim")
    return extractor


_EXTRACTOR = build_4channel_resnet()

_PREPROCESS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.0],
                         std=[0.229, 0.224, 0.225, 1.0]),
])


def build_instance_mask(seg_map, object_ids):
    mask = np.zeros(seg_map.shape, dtype=np.float32)
    for idx, obj_id in enumerate(object_ids):
        mask[seg_map == obj_id] = (idx + 1) / len(object_ids)
    return mask[:, :, np.newaxis]


def beta_features(rgb_frame, seg_map, object_ids):
    mask = build_instance_mask(seg_map, object_ids)
    stacked = np.concatenate([rgb_frame, mask * 255.0], axis=2).astype(np.uint8)
    tensor = _PREPROCESS(stacked).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = _EXTRACTOR(tensor)
    out = out.squeeze(-1).squeeze(-1)
    return F.normalize(out, dim=1).squeeze(0)