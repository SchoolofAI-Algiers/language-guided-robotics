# M3 Vision — Week 1

PyBullet scene → fixed-size observation tensor for the RL team.

---

## What this does

Renders a PyBullet simulation frame and converts it into a `(281,)` tensor that plugs directly into the PPO policy:

```
RGB (224×224×3)  ──► FPN backbone ──► visual features  (256,)  ─┐
depth + seg mask ──► bounding boxes ──► detection vector  (25,)  ─┴──► observation (281,)
```

- **Visual features (256)** — global-average-pool of the FPN `"0"` feature map from a pretrained Faster R-CNN backbone.
- **Detection vector (25)** — up to 5 objects × 5 features each: `[x1_norm, y1_norm, x2_norm, y2_norm, mean_depth]`. Bounding boxes come free from the PyBullet segmentation mask — no detector needed.
- Both components are L2-normalised before concatenation.

---

## Project structure

```
maria-vision/
├── config.py        # all constants — WIDTH, HEIGHT, MAX_OBJECTS, DEVICE, …
├── render.py        # PyBullet scene setup and camera rendering
├── boxes.py         # seg_to_boxes() and build_detection_vector()
├── backbone.py      # load_backbone() and extract_visual_features()
├── observation.py   # build_observation() — assembles the (281,) tensor
├── pipeline.py      # run_pipeline() — wires everything end-to-end
├── main.py          # entry point + Week 2 dummy policy check
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10+ recommended. CUDA optional — falls back to CPU automatically.

---

## Run

```bash
python main.py
```

Expected output:

```
==================================================
M3 Vision — Week 1 Pipeline
==================================================
[render]  rgb=(224, 224, 3)  depth=(224, 224)  seg=(224, 224)
[backbone] FPN loaded
[obs] vis_feat=torch.Size([256])  det_vec=torch.Size([25])
[obs] observation=torch.Size([281])

WEEK 2 CHECK PASSED
  torch.Size([281])  →  dummy policy  →  torch.Size([1, 7])
```

---

## Import in your own code

```python
from backbone    import load_backbone
from observation import build_observation

backbone = load_backbone()   # call once at startup

# per frame:
observation, vis_feat, det_vec = build_observation(rgb, depth, seg, backbone)
# observation.shape → (281,)  ready for the PPO policy
```

Or run the full pipeline including rendering:

```python
from pipeline import run_pipeline

observation = run_pipeline(verbose=False)
```

---

## Key constants (`config.py`)

| Name | Value | Notes |
|---|---|---|
| `WIDTH / HEIGHT` | 224 | render resolution |
| `MAX_OBJECTS` | 5 | max detected objects per frame |
| `FEATURES_PER_BOX` | 5 | x1, y1, x2, y2, mean_depth |
| `VIS_DIM` | 256 | FPN backbone output |
| `DET_DIM` | 25 | MAX_OBJECTS × FEATURES_PER_BOX |
| `OBS_DIM` | 281 | VIS_DIM + DET_DIM — RL policy input |
| `ACTION_DIM` | 7 | 7-DOF arm |

---

## Week 2 handoff

The `week2_check()` in `main.py` is a placeholder — swap it with the real PPO policy:

```python
# main.py
action = real_ppo_policy(observation.unsqueeze(0))
```

The observation tensor contract is fixed: `(281,)` float32, L2-normalised, on `DEVICE`.
