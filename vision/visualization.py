import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for plain .py scripts
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# OUTPUT DIRECTORY — all PNGs saved here
# ─────────────────────────────────────────────────────────────
OUT_DIR = "vision_output"
os.makedirs(OUT_DIR, exist_ok=True)
print(f"[Vision] Output folder: {os.path.abspath(OUT_DIR)}")

# ══════════════════════════════════════════════════════════════
# SECTION 4 — PYBULLET SCENE HELPERS (Constants)
# ══════════════════════════════════════════════════════════════

def get_color_for_name(name):
    """
    Generate a deterministic random (R, G, B) tuple from a name string.
    Ensures unique instances (e.g., 'cube1', 'cube2') get different colors.
    """
    import hashlib
    hash_object = hashlib.md5(name.encode())
    hex_hash = hash_object.hexdigest()
    
    # Take first 6 chars for RGB
    r = int(hex_hash[0:2], 16)
    g = int(hex_hash[2:4], 16)
    b = int(hex_hash[4:6], 16)
    return (r, g, b)

# ══════════════════════════════════════════════════════════════
# SECTION 5 — SCENE DESCRIPTION (text summary)
# ══════════════════════════════════════════════════════════════

def _box_area(box):
    return (box["x_max"] - box["x_min"]) * (box["y_max"] - box["y_min"])

def _iou(b1, b2):
    ix1 = max(b1["x_min"], b2["x_min"]); iy1 = max(b1["y_min"], b2["y_min"])
    ix2 = min(b1["x_max"], b2["x_max"]); iy2 = min(b1["y_max"], b2["y_max"])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = _box_area(b1) + _box_area(b2) - inter
    return inter / union if union > 0 else 0.0

def describe_scene(object_ids, boxes, names, frame_shape, spawned_count):
    """
    Build a one-line human-readable scene description.
    Used as the subtitle in visualisation panels.
    """
    H, W       = frame_shape[:2]
    visible    = list(boxes.keys())
    hidden     = spawned_count - len(visible)

    # Overlap analysis
    blist = list(boxes.values())
    max_iou = 0.0
    for i in range(len(blist)):
        for j in range(i + 1, len(blist)):
            max_iou = max(max_iou, _iou(blist[i], blist[j]))

    overlap = ("no overlap"     if max_iou == 0    else
               "slight overlap" if max_iou < 0.15  else
               "some overlap"   if max_iou < 0.40  else
               "heavy overlap")

    # Spatial spread
    if len(visible) > 1:
        centres = np.array([
            [(b["x_min"]+b["x_max"])/2/W, (b["y_min"]+b["y_max"])/2/H]
            for b in boxes.values()])
        s = centres.std()
        spread = ("tightly clustered" if s < 0.10 else
                  "moderately spread" if s < 0.20 else
                  "spread out")
    else:
        spread = "centered"

    vis_str = f"{len(visible)} visible" + (f" ({hidden} hidden)" if hidden else "")
    parts   = [f"{spawned_count} spawned | {vis_str}", spread, overlap]
    return " | ".join(parts)


# ══════════════════════════════════════════════════════════════
# SECTION 6 — VISUALISATION (Week 1–2 milestone: label objects)
# Three complementary views per scene:
#   A) Bounding boxes + labels drawn on the raw RGB frame
#   B) Segmentation colour map (each object a unique hue)
#   C) Feature vector heatmap (512-dim ResNet output per object)
# ══════════════════════════════════════════════════════════════

def _get_obj_color(name):
    """Return an (R, G, B) tuple (0–255) for a given object name."""
    return get_color_for_name(name)


def draw_bounding_boxes(frame, boxes, names):
    """
    Draw axis-aligned bounding boxes with class labels on a copy of frame.

    Args:
        frame : uint8 ndarray (H, W, 3) — RGB
        boxes : {obj_id: box_dict}
        names : {obj_id: str}

    Returns:
        annotated : uint8 ndarray (H, W, 3) — RGB
    """
    annotated = frame.copy()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

    for obj_id, box in boxes.items():
        label = names.get(obj_id, f"obj{obj_id}")
        r, g, b = _get_obj_color(label)
        color_bgr = (b, g, r)

        x1, y1, x2, y2 = box["x_min"], box["y_min"], box["x_max"], box["y_max"]

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, thickness=2)

        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(annotated,
                      (x1, y1 - th - baseline - 4),
                      (x1 + tw + 4, y1),
                      color_bgr, -1)
        cv2.putText(annotated, label,
                    (x1 + 2, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                    cv2.LINE_AA)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.circle(annotated, (cx, cy), 3, color_bgr, -1)

    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


def draw_segmentation_mask(seg, object_ids, names, frame_shape):
    """
    Build a pure segmentation mask image — black background, each object
    filled with its class colour and labelled at its centroid.

    Args:
        seg         : int32 ndarray (H, W) — PyBullet segmentation map
        object_ids  : list of int body IDs to colour
        names       : {obj_id: str}
        frame_shape : (H, W, 3) tuple

    Returns:
        mask_img : uint8 ndarray (H, W, 3)
    """
    H, W = frame_shape[:2]
    mask_img = np.zeros((H, W, 3), dtype=np.uint8)

    for obj_id in object_ids:
        mask = (seg == obj_id)
        if not mask.any():
            continue

        label = names.get(obj_id, f"obj{obj_id}")
        r, g, b = _get_obj_color(label)
        mask_img[mask] = [r, g, b]

        rows, cols = np.where(mask)
        cy, cx = int(rows.mean()), int(cols.mean())
        mask_bgr = cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR)
        cv2.putText(mask_bgr, label, (cx - 10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
                    cv2.LINE_AA)
        mask_img = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)

    return mask_img


# ══════════════════════════════════════════════════════════════
# SECTION 7 — BUILD + VISUALISE SCENES
# ══════════════════════════════════════════════════════════════

def visualize_scene(scene, save_path=None):
    """
    Render and SAVE a three-panel figure for one scene.
    Uses matplotlib Agg backend so it works in plain .py scripts
    (no display server / tkinter required).

      Panel 1 — raw RGB frame + bounding boxes + labels
      Panel 2 — segmentation colour map
      Panel 3 — ResNet18 feature heatmap per object

    Args:
        scene     : scene dict (from build_scene)
        save_path : str — PNG file path. Auto-generated if None.
    """
    frame        = scene["frame"]
    boxes        = scene["boxes"]
    names        = scene["names"]
    seg          = scene["seg"]
    feats_visual = scene["feats_visual"]
    title        = scene["title"]
    subtitle     = scene["subtitle"]

    bbox_img = draw_bounding_boxes(frame, boxes, names)
    seg_img  = draw_segmentation_mask(seg, list(boxes.keys()), names, frame.shape)

    fig = plt.figure(figsize=(15, 5.5), facecolor="#0d0d1a")
    fig.suptitle(f"{title}\n{subtitle}",
                 color="#e0e0ff", fontsize=11, fontweight="bold", y=1.01)

    # Panel 1 — bounding boxes
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(bbox_img)
    ax1.set_title("Bounding Boxes + Labels", color="#aaaacc", fontsize=9)
    ax1.axis("off")

    # Panel 2 — segmentation
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(seg_img)
    ax2.set_title("Segmentation Mask", color="#aaaacc", fontsize=9)
    ax2.axis("off")

    # Panel 3 — feature heatmap
    ax3 = fig.add_subplot(1, 3, 3)
    obj_ids = list(feats_visual.keys())
    if obj_ids:
        n_dims = 64
        heat_data = np.stack([feats_visual[oid].cpu().numpy()[:n_dims]
                               for oid in obj_ids])         # (N, 64)
        im = ax3.imshow(heat_data, aspect="auto",
                        cmap="RdYlGn", vmin=-1, vmax=1)
        ax3.set_yticks(range(len(obj_ids)))
        ax3.set_yticklabels([names.get(oid, f"obj{oid}") for oid in obj_ids],
                             fontsize=8, color="#ccccff")
        ax3.set_xlabel("Feature dim (first 64 of 512)", color="#aaaacc", fontsize=8)
        ax3.set_title("ResNet18 Feature Vectors", color="#aaaacc", fontsize=9)
        ax3.tick_params(colors="#aaaacc")
        for spine in ax3.spines.values():
            spine.set_color("#333355")
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor("#0d0d1a")
        for spine in ax.spines.values():
            spine.set_color("#333355")

    plt.tight_layout()

    # ── Save to disk ─────────────────────────────────────────
    if save_path is None:
        safe_title = title.replace(" ", "_").replace("—", "-")
        save_path  = os.path.join(OUT_DIR, f"{safe_title}.png")

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"[Vision] Saved: {save_path}")
    plt.close(fig)
