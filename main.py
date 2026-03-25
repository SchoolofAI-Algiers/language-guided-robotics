# ============================================================
# Vision Pipeline — SOAI Labs 2026
# Track: Vision / ML  |  Weeks 1–2 Milestones
#
# Milestones covered:
#    Set up torchvision + OpenCV pipeline
#    Render PyBullet scene screenshot
#    Run CNN feature extractor (ResNet18)
#    Identify + label objects via bounding boxes (segmentation)
#    Output fixed-size feature vector per object
#    Visualize scenes: bounding boxes, labels, segmentation overlay
#
# ============================================================

import os
import numpy as np
import pybullet as p
import pybullet_data
import cv2

# Import refactored modules
import vision_pipeline
import visualization

# ══════════════════════════════════════════════════════════════
# SECTION 4 — PYBULLET SCENE HELPERS
# ══════════════════════════════════════════════════════════════

def paint(obj_id, name, sim):
    """Apply a consistent RGBA colour to a PyBullet body."""
    r, g, b = visualization.get_color_for_name(name)
    rgba = [r / 255.0, g / 255.0, b / 255.0, 1.0]
    sim.changeVisualShape(obj_id, -1, rgbaColor=rgba)


def capture(sim, width=224, height=224, eye=(2, 2, 2)):
    """
    Render a top-perspective scene image from PyBullet.

    FIX: getCameraImage returns a flat 1-D RGBA buffer.
    Must reshape to (H, W, 4) before slicing channels.

    Returns:
        frame : uint8 ndarray (H, W, 3) — RGB image
        seg   : int32 ndarray (H, W)    — per-pixel object IDs
    """
    vm  = sim.computeViewMatrix(eye, [0, 0, 0], [0, 0, 1])
    pm  = sim.computeProjectionMatrixFOV(60, 1.0, 0.1, 10.0)
    _, _, rgba, _, seg = sim.getCameraImage(width, height, vm, pm)

    #  Reshape flat 1-D buffers → (H, W, 4) / (H, W) before use
    frame = np.array(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
    seg   = np.array(seg, dtype=np.int32).reshape(height, width)
    return frame, seg


def get_boxes(object_ids, seg):
    """
    Derive axis-aligned bounding boxes from the segmentation mask.

    Args:
        object_ids : list of int PyBullet body IDs
        seg        : int32 ndarray (H, W) — segmentation map from getCameraImage

    Returns:
        dict {obj_id: {"x_min", "x_max", "y_min", "y_max"}}
        Only visible objects (any pixel == obj_id) are included.
    """
    boxes = {}
    for obj_id in object_ids:
        mask = (seg == obj_id)
        if mask.any():
            rows = np.where(mask.any(axis=1))[0]
            cols = np.where(mask.any(axis=0))[0]
            boxes[obj_id] = {
                "x_min": int(cols.min()), "x_max": int(cols.max()),
                "y_min": int(rows.min()), "y_max": int(rows.max()),
            }
    return boxes


def reset_scene(sim):
    """Reset PyBullet, reload plane, set gravity."""
    sim.resetSimulation()
    sim.setAdditionalSearchPath(pybullet_data.getDataPath())
    sim.setGravity(0, 0, -9.8)
    sim.loadURDF("plane.urdf")


def settle(sim, steps=80):
    """Step the simulation until objects come to rest."""
    for _ in range(steps):
        sim.stepSimulation()


def build_scene(title, spawned_count, ids, names, boxes, frame, seg, sim):
    """
    Assemble one complete scene dictionary.

    Returns:
        dict with keys: title, subtitle, frame, seg, names, boxes,
                        feats_visual, feats_state, n_objects
    """
    feats_visual, feats_state = vision_pipeline.visual_features(frame, boxes, sim)
    subtitle = visualization.describe_scene(ids, boxes, names, frame.shape, spawned_count)

    return dict(
        title        = title,
        subtitle     = subtitle,
        frame        = frame,
        seg          = seg,
        names        = names,
        boxes        = boxes,
        feats_visual = feats_visual,
        feats_state  = feats_state,
        n_objects    = spawned_count,
    )


# ══════════════════════════════════════════════════════════════
# SECTION 8 — MAIN: spawn 5 scenes, visualise + save each
# ══════════════════════════════════════════════════════════════

def main():
    sim = p
    sim.connect(sim.DIRECT)
    sim.setAdditionalSearchPath(pybullet_data.getDataPath())

    scenes = []

    # ── Level 1 — single object ──────────────────────────────
    reset_scene(sim)
    cube = sim.loadURDF("cube.urdf", basePosition=[0, 0, 0.5])
    paint(cube, "cube", sim)
    settle(sim)
    frame, seg = capture(sim, eye=[2, 2, 2])
    ids, names = [cube], {cube: "cube"}
    boxes = get_boxes(ids, seg)
    scenes.append(build_scene("Level_1-single_object", 1,
        ids, names, boxes, frame, seg, sim))

    # ── Level 2 — three objects ──────────────────────────────
    reset_scene(sim)
    cube   = sim.loadURDF("cube.urdf",       basePosition=[-0.6, 0.0, 0.5])
    sphere = sim.loadURDF("sphere2.urdf",    basePosition=[ 0.6, 0.0, 0.5])
    duck   = sim.loadURDF("duck_vhacd.urdf", basePosition=[ 0.0, 0.6, 0.5],
                           globalScaling=0.5)
    paint(cube, "cube", sim); paint(sphere, "sphere", sim); paint(duck, "duck", sim)
    settle(sim)
    frame, seg = capture(sim, eye=[2.5, 2.5, 2.5])
    ids   = [cube, sphere, duck]
    names = {cube: "cube", sphere: "sphere", duck: "duck"}
    boxes = get_boxes(ids, seg)
    scenes.append(build_scene("Level_2-three_objects", 3,
        ids, names, boxes, frame, seg, sim))

    # ── Level 3 — four objects ───────────────────────────────
    reset_scene(sim)
    o1 = sim.loadURDF("cube.urdf",        basePosition=[-0.3,  0.3, 0.5])
    o2 = sim.loadURDF("sphere2.urdf",     basePosition=[ 0.3,  0.3, 0.5])
    o3 = sim.loadURDF("duck_vhacd.urdf",  basePosition=[-0.3, -0.3, 0.5],
                       globalScaling=0.5)
    o4 = sim.loadURDF("teddy_vhacd.urdf", basePosition=[ 0.3, -0.3, 0.5],
                       globalScaling=0.5)
    paint(o1, "cube", sim); paint(o2, "sphere", sim)
    paint(o3, "duck", sim); paint(o4, "teddy", sim)
    settle(sim)
    frame, seg = capture(sim, eye=[2.2, 2.2, 2.2])
    ids   = [o1, o2, o3, o4]
    names = {o1: "cube", o2: "sphere", o3: "duck", o4: "teddy"}
    boxes = get_boxes(ids, seg)
    scenes.append(build_scene("Level_3-four_objects", 4,
        ids, names, boxes, frame, seg, sim))

    # ── Level 4 — six objects, stacked, low camera ───────────
    reset_scene(sim)
    positions4 = [[-0.5,  0.5, 0.5], [ 0.5,  0.5, 0.5],
                  [-0.5, -0.5, 0.5], [ 0.5, -0.5, 0.5],
                  [ 0.0,  0.0, 0.5], [ 0.0,  0.0, 1.5]]
    urdfs4 = ["cube.urdf", "sphere2.urdf", "duck_vhacd.urdf",
              "teddy_vhacd.urdf", "cube.urdf", "sphere2.urdf"]
    
    # Needs explicit names for deterministic coloring
    lvl4_names = ["cube1", "sphere1", "duck", "teddy", "cube2", "sphere2"]
    ids4 = []
    
    for i, (urdf, pos) in enumerate(zip(urdfs4, positions4)):
        scale = 0.5 if "vhacd" in urdf else 1.0
        oid   = sim.loadURDF(urdf, basePosition=pos, globalScaling=scale)
        ids4.append(oid)
        paint(oid, lvl4_names[i], sim)
    
    settle(sim, steps=120)
    frame, seg = capture(sim, eye=[2.8, 2.0, 1.8])
    names4 = {oid: lbl for oid, lbl in zip(ids4, lvl4_names)}
    boxes = get_boxes(ids4, seg)
    scenes.append(build_scene("Level_4-six_objects", 6,
        ids4, names4, boxes, frame, seg, sim))

    # ── Level 5 — eight objects, random sizes, max clutter ───
    reset_scene(sim)
    np.random.seed(7)
    urdfs5  = ["cube.urdf", "sphere2.urdf", "duck_vhacd.urdf", "teddy_vhacd.urdf",
               "cube.urdf", "sphere2.urdf", "duck_vhacd.urdf", "teddy_vhacd.urdf"]
    labels5 = ["cube1", "sphere1", "duck1", "teddy1",
               "cube2", "sphere2", "duck2", "teddy2"]
    ids5 = []
    for i, urdf in enumerate(urdfs5):
        pos = [np.random.uniform(-0.7, 0.7),
               np.random.uniform(-0.7, 0.7),
               np.random.uniform(0.3,  0.9)]
        oid = sim.loadURDF(urdf, basePosition=pos,
                            globalScaling=np.random.uniform(0.3, 0.7))
        ids5.append(oid)
        paint(oid, labels5[i], sim)
    settle(sim, steps=150)
    frame, seg = capture(sim, eye=[1.8, 1.8, 1.6])
    names5 = {oid: lbl for oid, lbl in zip(ids5, labels5)}
    boxes  = get_boxes(ids5, seg)
    scenes.append(build_scene("Level_5-eight_objects_max_clutter", 8,
        ids5, names5, boxes, frame, seg, sim))

    sim.disconnect()

    # ── Visualise + save all scenes ──────────────────────────
    print(f"\n[Vision] Rendering and saving {len(scenes)} scenes...\n")
    for i, scene in enumerate(scenes):
        print(f"  Scene {i+1}: {scene['title']} | {scene['subtitle']}")
        n_vis = len(scene["feats_visual"])
        if n_vis > 0:
            sample_id  = next(iter(scene["feats_visual"]))
            vec_shape  = tuple(scene["feats_visual"][sample_id].shape)
            print(f"           visual feat shape: {vec_shape} × {n_vis} objects")
        visualization.visualize_scene(scene)

    print(f"\n[Vision] Week 1-2 milestones complete!")
    print(f"   All images saved to: {os.path.abspath(visualization.OUT_DIR)}/")
    print(f"   Fixed-size feature vectors (512-dim) extracted per object")
    print(f"   Bounding boxes + segmentation masks visualised for all scenes")
    print(f"   feats_visual and feats_state ready to hand to RL team")


if __name__ == "__main__":
    main()
