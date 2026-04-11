\# Robotics W4 - Constraint-Based Gripper

\*\*SOAI Labs 2026 | Language-Guided Robotics\*\*

\*\*Author:\*\* Takoua



\## What this does

Adds a magnetic constraint-based gripper (Option C) to the KukaEnv.

When the end-effector gets within 0.15m of any object, it snaps to it

using PyBullet's createConstraint — no URDF needed, no finger physics.



\## How to run

```bash

conda activate robotics

cd robotics

python -m env.tests.grasp\_visual\_test

```



\## What was added

\- `\_try\_grasp()` — checks distance every step, snaps object if close enough

\- `\_release()` — removes constraint to drop object

\- `gripper\_state` added to observation (22,) — 0.0 empty, 1.0 holding

\- `gripper\_state` and `grasped\_object` exposed in info dict



\## Inputs / Outputs

\- Observation: now (22,) instead of (21,) — last value is gripper state

\- info\["gripper\_state"] → 0.0 or 1.0

\- info\["grasped\_object"] → obj\_id of held object or None



\## Status

\[done]



