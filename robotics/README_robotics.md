# Robotics Track — Phase 1 & 2
**SOAI Labs 2026 | Language-Guided Robotics**
**Track:** Robotics | **Supervisor:** Selma Khelili

---

## What This Module Does

This is the **simulation environment** — the physics world where the robot lives. Every other track depends on it:

- **RL team** trains the agent by calling `reset()` and `step()`
- **Vision team** gets camera frames from `render()` and object positions from `info["object_state"]`
- **NLP team** will pass language embeddings into the policy running inside this environment
- **Demo team** streams frames to the browser

---

## Project Structure

```
env/
├── src/
│   ├── config.py           ← all constants (joint limits, sim params, camera, objects)
│   ├── environment.py      ← KukaEnv: the Gymnasium-compatible robot world
│   └── simulation.py       ← visual demo runner
└── tests/
    ├── environment.py      ←  automated verification tests
    ├── hello_pybullet.py   ← PyBullet sanity check
    └── vision_interface.py ← handoff script for the Vision/ML team
```

---

## How to Run

```bash
conda create -n robotics python=3.10
conda activate robotics
pip install pybullet gymnasium numpy matplotlib
pip install -e .

python env/src/simulation.py          # watch the arm move
python tests/environment.py           # run all  tests
python tests/vision_interface.py      # Vision team handoff check
```

---

## Phase 1 — Environment Setup
> ✅ Complete

### The Robot
A **Kuka IIWA** — a real 7-joint industrial arm simulated in **PyBullet**. The physics engine handles gravity, collisions, and joint dynamics automatically. The simulation runs at 240 Hz physics with 30 Hz control rate.

### Action Space
The agent sends 7 joint targets, normalized to `[-1, 1]`. Internally these get scaled to real joint angles in radians. Normalization keeps all joints on the same scale for the neural network.

### Observation Space
A 21-dimensional vector returned after every step:

| Slice | Content | Unit |
|-------|---------|------|
| `obs[0:7]` | Joint positions | radians |
| `obs[7:14]` | Joint velocities | rad/s |
| `obs[14:17]` | End-effector position (x, y, z) | meters |
| `obs[17:21]` | End-effector orientation (quaternion) | — |

### Camera
`env.render()` returns an RGB frame of shape `(480, 640, 3)` from a fixed camera angle looking down at the robot and table.

### Verification
9 automated tests cover: space shapes, reset behavior, step output, action clipping, rendering, Gymnasium compliance, seed determinism, and episode truncation. All passing.

---

## Phase 2 — Multi-Object Scene
> 🔄 In progress 

### What Was Added

**Table** — a static flat surface placed in front of the robot. All objects rest on it.

**Colored objects** — 4 objects spawned on the table on every `reset()` with:
- Random positions within the table area
- Random sizes (2cm – 5cm)
- Random shapes: box, sphere, cylinder (picked independently per object each reset)
- Colors: red, green, blue, yellow

**Anti-overlap check** — before placing each object, the code ensures a minimum 10cm distance from all already-placed objects. Prevents objects from being pushed off the table by physics collisions.

**Object state in `info`** — every `reset()` and `step()` returns:
```
info["object_state"] = {
    obj_id: {"pos": [x, y, z], "color": "red",    "shape": "sphere"},
    obj_id: {"pos": [x, y, z], "color": "green",  "shape": "box"},
    obj_id: {"pos": [x, y, z], "color": "blue",   "shape": "cylinder"},
}
```
The Vision team passes these `obj_id` values directly to their `get_boxes()` and `get_state()` functions.

**Vision interface script** — `tests/vision_interface.py` shows the Vision team exactly how to consume our environment output.



---

## Interface Contract

This is a **promise to every other track** — regardless of what changes inside our code, these outputs will always stay the same format. Other teams don't need to read our implementation, they just need to know what comes out.

```
env.reset(seed)  →  obs (21,), info
env.step(action) →  obs (21,), reward, done, truncated, info
env.render()     →  frame (480, 640, 3) uint8

info always contains:
  - info["ee_position"]   → end-effector [x, y, z]
  - info["object_state"]  → {obj_id: {"pos": [...], "color": str, "shape": str}}
```


---

*Language-Guided Robotics | SOAI Labs 2026 | Team Robotics | Supervisor: Selma Khelili*
