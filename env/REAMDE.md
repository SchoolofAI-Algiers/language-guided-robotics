# Kuka IIWA 7-DOF Environment

Gymnasium-compatible wrapper around a PyBullet Kuka IIWA robot arm.

## Spaces

| | Shape | Range | Contents |
|---|---|---|---|
| **Action** | `(7,)` | `[-1, 1]` | Normalized joint position targets, scaled to real joint limits internally |
| **Observation** | `(21,)` | finite | Joint positions (7), joint velocities (7), end-effector XYZ (3), end-effector quaternion (4) |

## Episode

- **Reset**: arm returns to home position (all joints = 0)
- **Truncation**: after 500 steps (~16.7s at 30 Hz control)
- **Termination**: none (TODO: once the task is defined)
- **Reward**: `0.0` (TODO: once the task is defined)

## Run

```bash
# tests
uv run python -m env.tests.environment

# live simulation (matplotlib window)
uv run python -m env.tests.simulation          # smooth wave policy
uv run python -m env.tests.simulation --random  # random policy
```

## Files

```
env/
├── src/
│   ├── config.py         # constants: joint limits, sim params, render settings
│   └── environment.py    # KukaEnv(gymnasium.Env)
└── tests/
    ├── environment.py    # 9 verification tests
    ├── simulation.py     # live matplotlib rendering with sample policies
    └── hello_pybullet.py # sanity check on pybullet setup
```
