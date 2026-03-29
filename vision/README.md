# Vision Pipeline — SOAI Labs 2026

**Track:** Vision / ML | **Milestone:** Weeks 1-2 | **Status:** [done]

An end-to-end vision pipeline designed for a reinforcement learning environment. It renders PyBullet physics scenes, detects and segments objects, extracts deep CNN spatial feature vectors, pulls physical state ground-truths, and generates annotated visualizations.

---

## What It Does

This pipeline bridges the gap between raw PyBullet pixel output and structured data ready for Reinforcement Learning (RL). For every scene it builds, the pipeline produces two critical representations:

- **`feats_visual`** — _What the object looks like:_ A fixed-size, 512-dimensional L2-normalized feature vector per object, extracted via a headless, pretrained `ResNet18`.
- **`feats_state`** — _Where the object is in space:_ A 9-dimensional normalized physics vector (`[x, y, z, vx, vy, vz, roll, pitch, yaw]`) queried directly from PyBullet's physics engine.

Finally, to assist with debugging and verification, the pipeline generates and saves a high-quality 3-panel PNG per scene showing bounding boxes, distinct semantic segmentation masks, and a statistical feature heatmap.

---

## Inputs / Outputs

**Inputs (What it takes in)**

- **PyBullet Simulation**:
  - **RGB Frames**: `(H, W, 3)` `uint8` images rendered from the dynamic camera.
  - **Segmentation Masks**: `(H, W)` `int32` maps where every pixel value precisely flags an object instance ID.
  - **Physics States**: Real-time position, velocity, and orientation (Euler angles) for every spawned object.
- **Pretrained Weights**: `torchvision.models.resnet18` (ImageNet default).

**Outputs (What it returns or saves)**

- **Returns**: Python dictionaries containing `feats_visual` (512-dim tensors) and `feats_state` (9-dim tensors) keyed by Python variables, perfectly formatted to be concatenated and fed to the RL agent.
- **Saves**: Annotated PNG visualizations exported to the `vision_output/` directory.

---

## Project Structure

Following the structural refactoring, the pipeline is split into clean, modular files:

```text
language-guided-robotics/
├── main.py               # Main entry point (Simulation, level layout, execution)
├── vision_pipeline.py    # Torch/CV code (ResNet18 extraction, PyBullet state parsing)
├── visualization.py      # Plotting code (Matplotlib, OpenCV BBox drawing, hashing)
├── README_VISION.md      # This file
├── README.md             # Top-level project goal description
└── vision_output/        # Auto-created on first run (holds saved .png scenes)
```

---

## Setup & Requirements

Python 3.10+ is recommended. No dedicated GPU is required — the pipeline gracefully falls back to CPU if necessary. If a CUDA GPU is available, the PyTorch tensors and ResNet convolutions will be heavily accelerated automatically.

```bash
# It is recommended to use a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install standard dependencies
pip install pybullet torch torchvision opencv-python matplotlib numpy
```

---

## Running the Pipeline

Execute the main controller script:

```bash
cd language-guided-robotics
python main.py
```

No external graphical window or display server is needed — Matplotlib's `Agg` headless backend is utilized so it seamlessly executes on SSH remote servers or standard `.py` interfaces. Output runs are pushed directly to `vision_output/`.

**Expected console output:**

```text
[Vision] Device: cpu (or cuda)
[Vision] ResNet18 extractor ready - output: 512-dim per crop
[Vision] Output folder: C:\...\vision_output

[Vision] Rendering and saving 5 scenes...
  Scene 1: Level_1-single_object | 1 spawned | 1 visible | centered | no overlap
           visual feat shape: (512,) x 1 objects
[Vision] Saved: vision_output\Level_1-single_object.png
  ...
```

---

## The 5 Demo Scenes

The main script runs five escalating difficulty scenarios designed to test the CNN extractor and segmentation maps:

| Level | Object Count                 | Scene Challenge Focus                                         |
| :---: | :--------------------------- | :------------------------------------------------------------ |
| **1** | **1** (Cube)                 | Baseline verification, single object detection.               |
| **2** | **3** (Cube, Sphere, Duck)   | Multi-class feature variance and detection.                   |
| **3** | **4** (All Classes)          | Full coverage. Tests spatial spread properties.               |
| **4** | **6** (Stacked, Camera drop) | Heavy occlusion, overlap detection, challenging camera angle. |
| **5** | **8** (Random Size/Pos)      | Maximum bounding box clutter and size normalization test.     |

---

## RL Team: Gymnasium Observation Tensors

The vision model has been structurally integrated into the `KukaEnv` Gymnasium environment (`robotics/env/src/environment.py`).

To help avoid network shape errors during the definition of your actor-critic architectures, here are the explicit spaces depending on the `obs_mode` flag passed during environment initialization:

### Mode 1: `obs_mode="visual_state"` (or `"visual"`)

- **Space Type:** `gymnasium.spaces.Box(low=..., high=..., shape=(533,), dtype=np.float32)`
- **Tensor Structure (533 dimensions)**:
  - `[0 : 512]`: **Vision Features** - 1D array of ResNet18 outputs. Explicitly L2-normalized bounds `[-1.0, 1.0]`. Eliminates neural-network activation blowups.
  - `[512 : 533]`: **Physics State** - 21-dim physics representation (`joint_pos(7)`, `joint_vel(7)`, `ee_pos(3)`, `ee_orn(4)`).
- **Usage**: Ideal standard operating mode. Your MLP feature extractor network simply needs to accept `input_dim=533`.

### Mode 2: `obs_mode="visual_only"`

- **Space Type:** `gymnasium.spaces.Box(low=..., high=..., shape=(512,), dtype=np.float32)`
- **Tensor Structure (512 dimensions)**:
  - `[0 : 512]`: **Vision Features** - 1D array of ResNet18 outputs, L2-normalized bounds `[-1.0, 1.0]`.
- **Usage**: Allows the RL brain to solve tasks using pure vision without "cheating" by directly reading joint encoder outputs.

### Mode 3: `obs_mode="pixels"` (For CNN Ablation)

- **Space Type:** `gymnasium.spaces.Dict`
- **Structure**:
  - `obs["pixels"]`: `Box(shape=(H, W, 3), dtype=np.uint8)` — Raw rendered RGB representation.
  - `obs["state"]`: `Box(shape=(21,), dtype=np.float32)` — Physics representation.
- **Usage**: Use this mode when testing purely-pixel based RL algorithms (e.g. Dreamer) or custom CNN feature extractors.

### Mode 4: `obs_mode="state"` (Robot-Only Ablation)

- **Space Type:** `gymnasium.spaces.Box(shape=(21,), dtype=np.float32)`
- **Tensor Structure**: `[ joint_pos(7), joint_vel(7), ee_pos(3), ee_orn(4) ]`
- **Usage**: Pure physics state without rendered visual tensors. Ideal for checking mechanical solving capacity before introducing visual variance.

To verify your dummy agent shapes, a script has been set up at `robotics/env/tests/test_dummy_policy.py`.

---

## Visualization Output Guide

Each saved PNG in `vision_output/` contains three descriptive panels:

1.  **Bounding Boxes + Labels** (Left)
    Axis-aligned boxes precisely engineered from PyBullet's internal pixel segmentation map. Real-time class labels correspond to the instance, mapped beautifully onto the RGB frame dynamically with OpenCV.
2.  **Segmentation Mask** (Center)
    Each object's pixel area is extracted perfectly and flooded with its deterministically hashed instance color on a stark black background. Complete pure masks with zero arbitrary edge blending.
3.  **ResNet18 Feature Heatmap** (Right)
    The first 64 of 512 embedded feature dimensions displayed dynamically via a visual color strip per object instance (`RdYlGn` colormap, range `[-1, 1]`). This acts as an active sanity-check to assure different objects visually fire distinctly in the spatial feature space.

---

## Feature Structure Details (RL Hand-off)

The feature variables returned by `visual_features()`:

| Feature Variable       | Tensor Shape | Semantic Description                                       |
| :--------------------- | :----------- | :--------------------------------------------------------- |
| `feats_visual[obj_id]` | `[512]`      | Batched, strictly L2-normalized deep extraction profile.   |
| `feats_state[obj_id]`  | `[9]`        | Per-vector max-abs normalized physical constraint profile. |

To construct a flat state-observation tensor suitable for passing down the RL team's `.step()` bounds:

```python
import torch

# Concat visual context (512) and state context (9) for one final observation
obs = torch.cat([feats_visual[obj_id], feats_state[obj_id]])  # Shape: (521,)
```

### Important Dev Notes:

- **Colors are automatically generated!** Bounding box & segment mask coloring strictly uses deterministic MD5 hashing of the instance `name` tag. An object named `cube1` will persistently share an exact color, independent of order, while natively standing apart from `cube2`.
- `feats_state` natively maps inside rough scalar bounds `[-1, 1]`, normalizing without crushing rotational physics definitions.
- The `ResNet` block instance caches within module memory upon initialization and performs efficient singular batched forward passes to conserve memory execution. Do not invoke `resnet` initializations per-frame iteratively.

---

## Phase 2: Gymnasium Integration & Observation Tensors

In **Phase 2**, our primary goal was to take the standalone visual feature extractors built in Phase 1 and seamlessly wire them directly into the Reinforcement Learning loop (the `Gymnasium` environment).

Here is exactly what we achieved in Phase 2:

### 1. The Vision-to-Environment Bridge

We integrated the `resnet_features` pipeline right into the core Gymnasium environment (`KukaEnv` in `robotics/env/src/environment.py`).

- Now, the vision model natively listens to the PyBullet physics engine loop.
- On every single `env.step()` and `env.reset()`, the environment automatically renders the camera image, passes it through the ResNet18 CNN, normalizes the features, and tightly packages them into the observation array ready for the agent to use.

### 2. Observation Space Normalization (RL Handshake)

Actor-Critic networks in RL (like PPO or SAC) can destabilize and "blow up" if the incoming observations are completely unscaled. To completely prevent this:

- The raw 512-dimensional ResNet features are immediately passed through an **L2 Normalization**, definitively crushing their values into a safe `[-1.0, 1.0]` bound.
- Physical robot joint positions and velocities were mapped to their real absolute limits.
- Gymnasium's `observation_space` natively reads these bounds, meaning standard libraries (like _Stable-Baselines3_) can instantly build flawlessly scaled network policies upon instantiation.

### 3. Comprehensive Ablation Modes

To help the RL team research how the agent performs under different perceptual constraints, we built out specific toggles allowing exact mixtures of Semantics (Visuals), Proprioception (Joints), and Ground Geometry (PyBullet Targets). You configure it just by passing an argument: `env = KukaEnv(obs_mode="...")`.

- **`visual_joints_statepybullet` (542-dim array):** The full multimodal setup. Concatenates 512-dim CNN crop features (semantics) + 21-dim arm state (proprioception) + 9-dim object PyBullet state (ground-truth depth/position). Solves "3D from 2D" sample inefficiencies!
- **`visual_joints` (533-dim array):** Concatenates 512 CNN features with 21 internal physics arm joint values.
- **`visual_statepybullet` (521-dim array):** Concatenates 512 CNN features directly with target 9-dim object physics states (forces arm to guess its proprioception).
- **`visual_only` (512-dim array):** Strips away all physics arrays. Forces the AI to solve tasks by relying purely on its camera.
- **`pixels` (Gymnasium Dict):** Strips the ResNet entirely and passes a raw RGB camera matrix directly into the observation for purely pixel-based RL testing (e.g., _DreamerV3_).
- **`state` (21-dim array):** The classic control setup. The agent acts entirely blind, using only its joint sensors.

### 4. Dummy Policy Verification

To guarantee there will be absolutely zero crash-inducing shape errors when handed off to the RL researchers, Phase 2 deployed `test_dummy_policy.py`.

- It instantiates a mini `torch.nn.Sequential` multi-layer perceptron (MLP).
- It routes our live multimodal observation tensors (the vision array + PyBullet outputs) deep into the PyTorch operations.
- **Result:** It scientifically guarantees that dimensionalities stack correctly, and tensors flow endlessly through step transitions without throwing shape mismatch exceptions.
