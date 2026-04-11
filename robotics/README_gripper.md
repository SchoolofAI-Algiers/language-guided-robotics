# Robotics W4 - Constraint-Based Gripper
**SOAI Labs 2026 | Language-Guided Robotics**
**Author:** Takoua

## What this does
Adds a magnetic constraint-based gripper (Option C) to the KukaEnv.
When the end-effector gets within 0.15m of any object, it snaps to it
using PyBullet createConstraint — no URDF needed, no finger physics.

## How to run

```bash
conda activate robotics
cd robotics
python -m env.tests.grasp_visual_test
```

## What was added
- try_grasp checks distance every step, snaps object if close enough
- release removes constraint to drop object
- gripper_state added to observation now 22 dims instead of 21
- gripper_state and grasped_object exposed in info dict

## Inputs / Outputs
- Observation: now (22,) instead of (21,) — last value is gripper state
- info gripper_state returns 0.0 or 1.0
- info grasped_object returns obj_id of held object or None

## Status
[done]