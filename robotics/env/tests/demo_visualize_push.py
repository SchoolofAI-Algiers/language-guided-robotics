"""
Visual GUI demo: watch the arm push objects and see whether they slide
smoothly or launch. Opens a PyBullet GUI window.

Run: PYTHONPATH=. python3 robotics/env/tests/demo_visualize_push.py
"""
import time
import numpy as np
import pybullet as p
import pybullet_data

from robotics.env.src.environment import KukaEnv
from robotics.env.src.config import (
    NUM_JOINTS, END_EFFECTOR_LINK_INDEX, JOINT_LOWER_LIMITS, JOINT_UPPER_LIMITS,
    MAX_JOINT_STEP_RADIANS,
)

MIDPOINT = (JOINT_UPPER_LIMITS + JOINT_LOWER_LIMITS) / 2.0
HALF_RANGE = (JOINT_UPPER_LIMITS - JOINT_LOWER_LIMITS) / 2.0


def get_jp(env):
    return np.array([p.getJointState(env._kuka_id, j, physicsClientId=env._physics_client_id)[0]
                      for j in range(NUM_JOINTS)])


def solve_ik(env, target):
    current = get_jp(env).tolist()
    orn = p.getQuaternionFromEuler([np.pi, 0, 0])
    jt = p.calculateInverseKinematics(
        env._kuka_id, END_EFFECTOR_LINK_INDEX, target, orn,
        lowerLimits=JOINT_LOWER_LIMITS.tolist(), upperLimits=JOINT_UPPER_LIMITS.tolist(),
        jointRanges=(JOINT_UPPER_LIMITS - JOINT_LOWER_LIMITS).tolist(), restPoses=current,
        maxNumIterations=100, physicsClientId=env._physics_client_id)
    return np.array(jt[:NUM_JOINTS])


def act(jt, g):
    n = np.clip((jt - MIDPOINT) / HALF_RANGE, -1, 1)
    return np.concatenate([n, [g]]).astype(np.float32)


def run_visual_case(seed, obj_index, label):
    print(f"\n=== {label} (seed={seed}, obj_index={obj_index}) ===")
    print(f"Current MAX_JOINT_STEP_RADIANS = {MAX_JOINT_STEP_RADIANS}")

    env = KukaEnv()
    env._physics_client_id = p.connect(p.GUI)  # use GUI instead of DIRECT
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=50, cameraPitch=-35,
                                  cameraTargetPosition=[0.5, 0, 0.3])
    obs, info = env.reset(seed=seed)
    obj_state = info["object_state"]
    obj_id = list(obj_state.keys())[obj_index]
    obj_pos = np.array(obj_state[obj_id]["pos"])
    shape = obj_state[obj_id]["shape"]
    print(f"Object: {shape} at {np.round(obj_pos, 3)}. Watch the GUI window.")
    time.sleep(2)

    approach_target = [obj_pos[0] - 0.03, obj_pos[1], obj_pos[2]]
    approach_action = act(solve_ik(env, approach_target), -1.0)
    for i in range(300):
        obs, r, term, trunc, info = env.step(approach_action)
        time.sleep(1.0 / 60.0)
        ee = np.array(info["ee_position"])
        if np.linalg.norm(ee - approach_target) < 0.02:
            break

    push_target = [obj_pos[0] + 0.15, obj_pos[1], obj_pos[2]]
    push_action = act(solve_ik(env, push_target), -1.0)
    print("Pushing now...")
    for i in range(80):
        obs, r, term, trunc, info = env.step(push_action)
        time.sleep(1.0 / 60.0)

    final_pos, _ = p.getBasePositionAndOrientation(obj_id, physicsClientId=env._physics_client_id)
    print(f"Final object position: {np.round(final_pos, 3)}  "
          f"(started at {np.round(obj_pos, 3)})")
    print("Closing in 3 seconds...")
    time.sleep(3)
    p.disconnect(physicsClientId=env._physics_client_id)


if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    obj_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    run_visual_case(seed, obj_index, "push demo")
