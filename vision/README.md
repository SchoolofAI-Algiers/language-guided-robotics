# Vision Pipeline — SOAI Labs 2026

**Track:** Vision / ML | **Milestone:** Weeks 1-2 | **Status:** [done]

An end-to-end vision pipeline designed for a reinforcement learning environment. It renders PyBullet physics scenes, detects and segments objects, extracts deep CNN spatial feature vectors, pulls physical state ground-truths, and generates annotated visualizations.

---

## 1. What It Does

This pipeline bridges the gap between raw PyBullet pixel output and structured data ready for Reinforcement Learning (RL). For every scene it builds, the pipeline produces two critical representations:

- **eats_visual** — *What the object looks like:* A fixed-size, 512-dimensional L2-normalized feature vector per object, extracted via a headless, pretrained ResNet18.
- **eats_state** — *Where the object is in space:* A 9-dimensional normalized physics vector ([x, y, z, vx, vy, vz, roll, pitch, yaw]) queried directly from PyBullet's physics engine.

---

## 2. Project Structure

Following the structural refactoring across Phase 1 and Phase 2, the project is split into clean, modular components:

`	ext
language-guided-robotics/
├── core/
│   └── multimodal_wrapper.py # Phase 2: Late-fusion Gymnasium Observation Wrapper
├── robotics/
│   └── env/                  # Phase 2: Pure PyBullet physics environment
├── tests/
│   └── test_dummy_policy.py  # Phase 2: Shape and dimension integration testing
├── vision/
│   ├── main.py               # Phase 1: Main entry point (Simulation, level layout)
│   ├── vision_pipeline.py    # Torch/CV code (ResNet18 extraction)
│   ├── visualization.py      # Plotting code
│   └── README.md             # This file
└── vision_output/            # Auto-created on first run (saved .png scenes)
`

---

## 3. Setup & Requirements

Python 3.10+ is recommended. No dedicated GPU is required — the pipeline gracefully falls back to CPU if necessary. If a CUDA GPU is available, PyTorch will be heavily accelerated automatically.

`ash
# Strongly recommended: use a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install strictly standard dependencies
pip install pybullet torch torchvision opencv-python matplotlib numpy
`

---

## 4. Phase 1: Standalone Vision Pipeline

To verify bounding boxes, raw ResNet capabilities, and semantic masks, you can run the standalone vision tests without touching the robotics loop.

### Running the Pipeline

Execute the main controller script:

`ash
# From the project root
cd vision
python main.py
`

No graphical display server is needed — Matplotlib's Agg headless backend is utilized dynamically so it seamlessly executes on SSH remote servers. Output runs are pushed directly to ision_output/.

### The 5 Demo Scenes

The standalone script runs 5 escalating difficulty scenarios designed to test the CNN extractor and segmentation masks:

| Level | Object Count | Scene Challenge Focus |
| :---: | :--- | :--- |
| **1** | **1** (Cube) | Baseline verification, single object detection. |
| **2** | **3** (Cube, Sphere, Duck) | Multi-class feature variance and detection. |
| **3** | **4** (All Classes) | Full coverage. Tests spatial spread properties. |
| **4** | **6** (Stacked, Camera drop) | Heavy occlusion, overlap detection, challenging camera angle. |
| **5** | **8** (Random Size/Pos) | Maximum bounding box clutter and size normalization test. |

### Visualization Output Guide

Each saved PNG in ision_output/ contains three descriptive panels:

1. **Bounding Boxes + Labels** (Left) — Axis-aligned boxes precisely engineered from PyBullet's internal pixel segmentation map.
2. **Segmentation Mask** (Center) — Each object's pixel area is extracted perfectly and flooded with its deterministically hashed instance color on a stark black background.
3. **ResNet18 Feature Heatmap** (Right) — The first 64 of 512 embedded feature dimensions displayed dynamically via a visual color strip per object instance (RdYlGn colormap, range [-1, 1]). 

---

## 5. Phase 2: Gymnasium Integration & Observation Tensors

In **Phase 2**, our primary goal was taking the isolated visual standalone features built in Phase 1 and seamlessly wiring them straight into the Reinforcement Learning loop via a powerful **Late-Fusion Architecture**.

### The Vision-to-Environment Bridge (Architectural Decoupling)

To prevent "Dependency Hell" for upcoming Language Embeddings, we radically separated domains using standard Gymnasium Wrappers:

- **
obotics/env/src/environment.py (Pure Physics):** We purged all external computer vision logic out of the core wrapper. It purely returns a fast dictionary array: {"pixels": (H,W,3), "state": (21,), "object_state": (9,)}.
- **core/multimodal_wrapper.py (The Glue / Middleware):** Initializes a MultimodalObservationWrapper(gym.ObservationWrapper) that physically pulls both domains together. It takes the PyBullet rendering and dynamically extracts the ResNet representations, performing heavy matrix concatenations safely on top.

### Observation Modes (RL Handshake)

Actor-Critic networks in RL (like PPO or SAC) can destabilize and "blow up" if the incoming observations are completely unscaled. All ResNet18 outputs strictly enforce L2-normalization mapped to [-1.0, 1.0]. 
You can customize the specific tensor shape your RL agent receives simply by configuring obs_mode when wrapping the environment:

`python
from robotics.env.src.environment import KukaEnv
from core.multimodal_wrapper import MultimodalObservationWrapper

base_env = KukaEnv(render_mode="rgb_array")
env = MultimodalObservationWrapper(base_env, obs_mode="visual_joints_statepybullet")
`

Below are the 6 available obs_mode ablations:

1. **isual_joints_statepybullet** (Full Multimodal)
   - **Space:** Box(shape=(542,), float32)
   - **Structure:** [0:512] Visual Semantics + [512:533] 21-dim Arm mechanics + [533:542] 9-dim Object geometry targets.
   - **Usage:** Standard mode. Solves "3D-from-2D" inefficiencies. MLP accepts 542 input dimensions.
2. **isual_joints**
   - **Space:** Box(shape=(533,), float32)
   - **Structure:** [0:512] Visual Semantics + [512:533] 21-dim Arm mechanics.
3. **isual_statepybullet** 
   - **Space:** Box(shape=(521,), float32)
   - **Structure:** [0:512] Visual Semantics + [512:521] 9-dim Object geometry targets. Forces arm entirely blind to joint positions.
4. **isual_only**
   - **Space:** Box(shape=(512,), float32)
   - **Usage:** Forces agent to solve purely on visual semantics without any numeric physics advantages.
5. **pixels** (CNN Custom Ablation)
   - **Space:** gymnasium.spaces.Dict mapping exactly to raw Box matrices.
   - **Usage:** Used primarily for purely-pixel based RL algorithms (e.g. Dreamer) where researchers formulate their own proprietary CNN feature extractor.
6. **state**
   - **Space:** Box(shape=(21,), float32)
   - **Usage:** Pure robotic 21-dim proprioception physics state. Classic control. The agent acts entirely blind to visuals.

### Dummy Policy Verification
To guarantee there will be absolutely zero crash-inducing shape errors when handed off to RL researchers, we deployed 	ests/test_dummy_policy.py. It instantiates a lightweight PyTorch MLP model and scientifically guarantees tensors flow seamlessly in the unified structure without mismatch exceptions.

---

## 6. Message to the RL Team (Modality Dominance)

 **Important Note regarding the Shortcuts Problem:**
When training your Actor-Critic algorithms, be very careful about which obs_mode you use. 

If you feed the RL model the full isual_joints_statepybullet array (542-dim), it will suffer from **Modality Dominance**. The agent acts like a lazy student—it will quickly realize it can completely ignore the complex 512-dimension visual CNN block because the exact [X, Y, Z] target coordinates are handed to it for free in the object_state block at the end of the array. It will zero-out the vision weights entirely.

**Recommended Training Progression:**
1. **Phase 1 (Motor Control Debugging):** Use isual_joints_statepybullet (542-dim). Let the agent "cheat" using the 9-dim object geometry just to prove your PPO reward function and robotic arm mapping work.
2. **Phase 2 (The Real Task):** Use isual_joints (533-dim) or deploy an *Asymmetric Actor-Critic (AAC)*. You must strip out the 9-dim object geometry array for the Actor so it is literally forced to learn spatial geometry strictly from the ResNet's vision output!

---

## 7. Important Dev Notes

*   **Mock Variables in Development:** Currently, PyBullet target blocks haven't started spawning in the main environment loop yet, so object_state dynamically evaluates to 
p.zeros(9). Once the physical blocks exist, inject get_state natively in environment.py.
*   **Colors are automatically generated!** Bounding box & segment mask coloring strictly uses deterministic MD5 hashing of the instance 
ame tag. An object named cube1 will persistently share an exact color, independent of order, while natively standing apart from cube2.
*   eats_state natively maps inside rough scalar bounds [-1, 1], normalizing without crushing rotational physics definitions.
*   The ResNet block instance caches within module memory upon initialization and performs efficient singular batched forward passes to conserve memory execution. Do not invoke 
esnet initializations per-frame iteratively.
