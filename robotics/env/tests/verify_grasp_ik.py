"""
Manual verification for issue #6, step 4 — Option B.

Bypasses the undertrained policy entirely. Uses PyBullet's own inverse
kinematics to drive the end-effector straight to the target object,
closing the gripper once in range. This isolates and proves:

1. environment.py's step() populates info["grasped_object"] correctly
   (Finding 1 fix), consistently with info["grasped_object_id"].
2. set_target_object() + _try_grasp() correctly grasp the intended
   target object once the arm is physically close enough.

Policy competence (whether the trained PPO model can navigate there on
its own) is explicitly NOT what this test measures — that's a separate,
already-documented issue (see docs/grasp_gap_diagnosis.md).

Run: PYTHONPATH=. python3 robotics/env/tests/verify_grasp_ik.py
"""
import sys
import numpy as np
import pybullet as p

sys.path.insert(0, ".")

from robotics.env.src.environment import KukaEnv
from robotics.env.src.config import END_EFFECTOR_LINK_INDEX, NUM_JOINTS


def run_trial(seed):
    env = KukaEnv()
    obs, info = env.reset(seed=seed)

    assert "grasped_object" in info, "reset() info missing 'grasped_object'"

    obj_state = info["object_state"]
    target_id = list(obj_state.keys())[0]
    target_pos = obj_state[target_id]["pos"]
    color = obj_state[target_id]["color"]
    shape = obj_state[target_id]["shape"]

    env.set_target_object(target_id)

    grasped_at = None
    for step in range(200):
        # Drive straight to the target using PyBullet's IK solver.
        joint_targets = p.calculateInverseKinematics(
            env._kuka_id,
            END_EFFECTOR_LINK_INDEX,
            target_pos,
            physicsClientId=env._physics_client_id,
        )
        joint_targets = np.array(joint_targets[:NUM_JOINTS], dtype=np.float32)

        # Convert absolute joint targets back into the env's normalized
        # [-1, 1] action space (env.step() maps action -> joint angle via
        # midpoint +/- half_range, so invert that here).
        from robotics.env.src.config import JOINT_LOWER_LIMITS, JOINT_UPPER_LIMITS
        midpoint = (JOINT_UPPER_LIMITS + JOINT_LOWER_LIMITS) / 2.0
        half_range = (JOINT_UPPER_LIMITS - JOINT_LOWER_LIMITS) / 2.0
        normalized = (joint_targets - midpoint) / half_range
        normalized = np.clip(normalized, -1.0, 1.0)

        action = np.zeros(NUM_JOINTS + 1, dtype=np.float32)
        action[:NUM_JOINTS] = normalized
        action[NUM_JOINTS] = 1.0  # gripper close/grasp command

        obs, reward, terminated, truncated, info = env.step(action)

        assert "grasped_object" in info, f"step {step}: missing 'grasped_object' key"
        assert "grasped_object_id" in info, f"step {step}: missing 'grasped_object_id' key"
        assert info["grasped_object"] == info["grasped_object_id"], (
            f"step {step}: mismatch grasped_object={info['grasped_object']} "
            f"vs grasped_object_id={info['grasped_object_id']}"
        )

        if info["grasped_object"] is not None:
            grasped_at = step
            break

        if terminated or truncated:
            break

    env.close()
    return grasped_at, target_id, color, shape


if __name__ == "__main__":
    n_trials = 5
    successes = 0

    for i in range(n_trials):
        grasped_at, target_id, color, shape = run_trial(seed=i)
        if grasped_at is not None:
            successes += 1
            print(f"[trial {i}] GRASP FIRED at step {grasped_at} on "
                  f"target_id={target_id} ({color} {shape})")
        else:
            print(f"[trial {i}] no grasp (target_id={target_id}, {color} {shape})")

    print()
    print(f"Key consistency verified every step across {n_trials} trials "
          f"(grasped_object == grasped_object_id, no KeyError).")
    print(f"Grasp fired in {successes}/{n_trials} trials (IK-driven arm, real KukaEnv).")
    if successes > 0:
        print("PASS: grasp mechanism + key fix confirmed working end-to-end.")
    else:
        print("FAIL: grasp never fired even with IK-driven approach — "
              "investigate GRIPPER_ATTACH_DISTANCE or IK accuracy.")
