# Inter-Module Interfaces — Language-Guided Robotics
**SOAI Labs 2026 · Project Progress Guide**
**Last updated: Status: Phase 1 complete for Robotics, Vision, Demo**

> This document is the **contract between all tracks**.
> If you change a shape, format, or field name — update this file and notify all tracks first.


## Current Team Status

| Track | Members | Phase 1 | Phase 2 |
|---|---|---|---|
| Robotics | M1–M2 | ✅ Complete | 🔄 In progress |
| Vision / ML | M3–M4 | ✅ Complete | 🔄 In progress |
| NLP | M5–M6 | 🔄 In progress | ⏳ Not started |
| RL | M7–M8 | 🔄 In progress | ⏳ Not started |
| Demo / Systems | M9–M10 | ✅ Complete | 🔄 In progress |

---

## Data Flow Overview

```
User Instruction (string)
        │
        ▼
   ┌─────────┐
   │   NLP   │  sentence-transformers
   └─────────┘
        │  embedding: (384,)
        │  command_type: str
        │  target: str
        ▼
   ┌─────────┐
   │   CV    │  ResNet18 + PyBullet segmentation
   └─────────┘
        │  feats_visual: {obj_id: (512,)}
        │  feats_state:  {obj_id: (9,)}
        │  target_object: dict
        ▼
   ┌─────────┐
   │   RL    │  PPO policy
   └─────────┘
        │  joint_angles: (7,)  radians
        │  pose: str
        ▼
   ┌─────────────┐
   │  Robotics   │  KukaEnv PyBullet
   └─────────────┘
        │  observation: (21,)
        │  rgb_frame: (480, 640, 3)
        ▼
   ┌─────────────┐
   │    Demo     │  Flask + React
   └─────────────┘
```

---

## Module 1 — Robotics (M1–M2)

**File:** `robotics/env/src/environment.py`
**Class:** `KukaEnv(gym.Env)`

### Phase 1 — ✅ Complete
```
✅ PyBullet installed, simulation confirmed
✅ Kuka IIWA URDF loaded, arm stable
✅ KukaEnv: env.step() / env.reset() Gymnasium wrapper
✅ Action space:      (7,)  float32  [-1, 1]  normalized
✅ Observation space: (21,) float32
     obs[0:7]    joint_positions   radians
     obs[7:14]   joint_velocities  rad/s
     obs[14:17]  ee_position       meters (x, y, z)
     obs[17:21]  ee_orientation    quaternion
✅ RGB frame: env.render() → (480, 640, 3) uint8
✅ 9 verification tests passing
✅ Wave + random sample policies
✅ Reset deterministic with seed
```

### Phase 2 — 🔄 In progress (W3-4)
```
❌ Multi-object scene: at least 3 objects, randomized positions on reset()
❌ Object property randomization: color, size variations
❌ Object state exposed in observation:
     expected format: {obj_id: {"pos": [x,y,z], "color": str}}
❌ Secondary contribution: help Vision team render consistent scene images

End-of-week-4 check:
   env supports episode randomization + exposes full object state dict
```

### Spaces & Config
```
Action:             (7,)  float32  [-1.0, 1.0]  normalized
Observation:        (21,) float32
RGB Frame:          (480, 640, 3) uint8
MAX_EPISODE_STEPS:  500
SIM_TIMESTEP:       1/240 s
CONTROL_HZ:         30  (SIM_STEPS_PER_ACTION = 8)
HOME_POSITION:      [0, 0, 0, 0, 0, 0, 0]  radians

Joint Limits (radians):
  Joint:    J1       J2       J3       J4       J5       J6       J7
  Lower:  -2.9671  -2.0944  -2.9671  -2.0944  -2.9671  -2.0944  -3.0543
  Upper:   2.9671   2.0944   2.9671   2.0944   2.9671   2.0944   3.0543
```

---

## Module 2 — Vision / ML (M3–M4)

**File:** `vision/vision_pipeline.py`

### Phase 1 — ✅ Complete
```
✅ torchvision + OpenCV pipeline running
✅ PyBullet scene rendering (headless)
✅ ResNet18 CNN feature extractor
✅ Bounding boxes + segmentation masks
✅ 5 escalating difficulty scenes tested
✅ 3-panel PNG visualization

visual_features(frame, boxes, sim) returns:
   feats_visual:  {obj_id: Tensor(512,)}  L2-normalized ResNet18 features
   feats_state:   {obj_id: Tensor(9,)}    [x,y,z, vx,vy,vz, roll,pitch,yaw]

Combined RL tensor per object:
   obs = torch.cat([feats_visual[obj_id], feats_state[obj_id]])  # (521,)
```

### Phase 2 — 🔄 In progress (W3-4)
```
❌ Integrate CNN extractor with Gymnasium observation pipeline
❌ Features normalized + compatible with RL agent input shape
❌ Observation tensor: visual features + object state → dummy policy (no shape errors)
❌ Ablation: raw pixels vs CNN features
❌ Secondary contribution: help RL team understand observation tensor structure

End-of-week-4 check:
   obs tensor combining visual features + object state flows into
   dummy policy without shape errors

⚠️  BLOCKED ON: Robotics delivering multi-object scene
    Vision needs PyBullet object IDs to run get_boxes() and get_state()
```

### CNN Model Details
```
Architecture:   ResNet18 (ImageNet pretrained)
Head removed:   AvgPool + FC stripped → outputs (N, 512)
Crop size:      64x64 per object
Device:         CUDA if available, else CPU
Normalization:  ImageNet mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]
```

---

## Module 3 — NLP (M5–M6)

**File:** `nlp/` *(Phase 1 in progress)*
**Model:** `all-MiniLM-L6-v2` via sentence-transformers

### Phase 1 — 🔄 In progress (W1-2)
```
❌ sentence-transformers installed, all-MiniLM-L6-v2 loaded
❌ Encode 20 sample instructions, inspect embedding vectors
❌ Initial instruction dataset: 100 instructions across 5 command types
❌ t-SNE / PCA embedding visualization

End-of-week-2 check:
   given any instruction string → return (384,) embedding in under 10ms
```

### Phase 2 — ⏳ Not started (W3-4)
```
❌ Expand dataset to 500 instructions across 10 command types
❌ Cosine similarity between paraphrases > 0.85
❌ Embedding cache for fast lookup during training
❌ Secondary contribution: provide embeddings to RL for first training run

End-of-week-4 check:
   500-instruction dataset ready, embeddings cached and loadable in one line
```

### Output Interface (for RL and Demo)
```
embedding:      Tensor  (384,)   float32   L2-normalized
command_type:   str     one of: ["pick", "place", "reach", "home"]
target:         str     e.g. "red block", "left zone", "home position"
confidence:     float   [0.0, 1.0]
latency:        < 10ms per instruction

Interface for RL policy input:
   policy_input = torch.cat([nlp_embedding, visual_obs])  # (905,)
```

---

## Module 4 — RL (M7–M8)

**File:** `rl/` *(Phase 1 not started)*
**Algorithm:** PPO via Stable-Baselines3

### Phase 1 — 🔄 In progress  (W1-2)
```
❌ Read OpenAI Spinning Up intro (MDPs, policy gradients)
❌ Run SB3 PPO on CartPole — watch it learn
❌ Understand: rollout → advantage estimation → policy update
❌ Study reward function design for pick-and-place

End-of-week-2 check:
   able to explain PPO to a teammate and why it's stable
```

### Phase 2 — ❌ Not started (W3-4)
```
❌ Language-conditioned policy network:
     fuse nlp_embedding (384,) + visual_obs (521,)
❌ First PPO training run on 3 command types:
     "pick red block", "reach green object", "move to table"
❌ Training curves: reward shows upward trend by end of W4
❌ Secondary contribution: review NLP dataset for diversity and edge cases
❌ Target: > 40% success on at least 1 command type

End-of-week-4 check:
   agent achieves > 40% success on at least 1 command type after training
```

### Policy Interface
```
Input (full, Phase 3):
   (905,) = nlp_embedding (384,) + feats_visual (512,) + feats_state (9,)

Input (minimal, Phase 2 training):
   (405,) = nlp_embedding (384,) + joint_obs (21,)

Output:
   (7,)  float32  [-1.0, 1.0]  normalized joint targets
   → passed directly to KukaEnv.step(action)

Checkpoint format:
   path:     rl/checkpoints/
   filename: ppo_v{version}_{command_type}_{episodes}ep.zip
```

---

## Module 5 — Demo / Systems (M9–M10)

**File:** `demo/`

### Phase 1 — ✅ Complete
```
✅ Flask backend running on port 5000
✅ POST /api/instruction — request/response loop confirmed
✅ React frontend scaffold
✅ Frontend wired to Flask (USE_BACKEND toggle)
✅ CORS configured
```

### Phase 2 — 🔄 In progress (W3-4)
```
✅ Live PyBullet stream: GET /api/stream → real MJPEG at 30fps
✅ Real joint angles from KukaEnv in /api/instruction response
✅ Shared env instance between stream and instruction routes
❌ Experiment logging setup (TensorBoard) — ready for RL track
❌ Secondary Vision contribution: scene-level CNN feature in /api/instruction

End-of-week-4 check:
   live PyBullet frames in browser ✅
   training metrics in shared dashboard ❌
```

### API Endpoints

#### POST /api/instruction
```
Request:  { "instruction": str }

Current response (Phase 2):
{
  "received":        str,
  "status":          "ok",
  "phase":           2,
  "joint_angles":    [{"joint": int, "angle": float}] x7   ← real, degrees
  "ee_position":     [x, y, z]                              ← real, meters
  "ee_orientation":  [qx, qy, qz, qw],
  "nlp":             null   ← stub, replaced Phase 3
  "cv":              null   ← stub, replaced Phase 3
  "rl":              null   ← stub, replaced Phase 3
}

Target response (Phase 3):
{
  "nlp": { "command_type": str, "target": str, "confidence": float, "latency_ms": int },
  "cv":  { "target_object": {"label": str, "x": float, "y": float, "z": float}, "latency_ms": int },
  "rl":  { "joint_angles": [...] x7, "pose": str, "success_prob": float, "latency_ms": int }
}
```

#### GET /api/stream
```
Response:  multipart/x-mixed-replace  MJPEG
Frame:     JPEG  640x480  RGB  30fps
Source:    KukaEnv.render() ✅
```

#### GET /api/status — Phase 3
```
{ "nlp": "ready"|"error", "cv": "ready"|"error", "rl": "ready"|"error", "robotics": "ready"|"error" }
```

---

## Phase 2 Remaining Checklist (W3-4)

```
Robotics M1-M2:
[ ] Multi-object scene with randomized positions on reset
[ ] Color and size randomization
[ ] Object state dict exposed in observation
[ ] Help Vision render consistent scene images

Vision M3-M4:
[ ] CNN features integrated with Gymnasium obs pipeline
[ ] Ablation: raw pixels vs CNN features
[ ] Obs tensor into dummy policy without shape errors
[ ] Help RL understand tensor structure
BLOCKED: waiting on Robotics multi-object scene

NLP M5-M6:
[ ] Complete Phase 1 (encoder + 100 instruction dataset)
[ ] Expand to 500 instructions across 10 command types
[ ] Embedding cache
[ ] Cosine similarity validation > 0.85

RL M7-M8:
[ ] Complete Phase 1 (read SB3, run CartPole PPO, understand loop)
[ ] Language-conditioned policy network
[ ] First training run on 3 command types
[ ] > 40% success on 1 command type

Demo M9-M10:
[ ] TensorBoard logging wired and ready for RL
[ ] Secondary Vision contribution
```

---

## Phase 3 Checklist (W5-6) — Mixed Teams

```
Mixed Team A (M1, M3, M5, M7, M9):
[ ] NLP embedding correctly conditions RL policy
[ ] Ablation A: with vs without language conditioning
[ ] Train on 5 additional command types (total 8+)
[ ] Document integration decisions in repo

Mixed Team B (M2, M4, M6, M8, M10):
[ ] CNN features flowing cleanly into PPO policy
[ ] Ablation B: CNN features vs raw object state
[ ] Extend to remaining command types (total 10-15)
[ ] Demo pipeline in sync with latest trained models

Everyone:
[ ] Generalization tests: unseen colors, sizes, positions
[ ] Target: ≥ 70% generalization success rate
[ ] Identify and document 2-3 failure modes
[ ] pipeline_runner.py replaces simulator.js stubs
[ ] Arm moves in response to instruction (not wave policy)
```

---

## Known Blockers

| Blocker | Affects | Owner | Phase |
|---|---|---|---|
| Multi-object scene in KukaEnv | Vision, RL, Demo | Robotics M1-M2 | Phase 2 |
| NLP Phase 1 not complete | RL, Demo | NLP M5-M6 | Phase 1 |
| RL not started | Integration | RL M7-M8 | Phase 1 |
| TensorBoard logging | RL training visibility | Demo M9-M10 | Phase 2 |

---

*For questions ping @smotrishna on Discord or open an issue on GitHub.*