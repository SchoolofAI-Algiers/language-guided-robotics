"""
Visual GUI demo: cycles through every instruction/task type
(reach, pick, lift, place, lower, push, pull) on a fresh object each time,
in a single continuous PyBullet GUI window.

Run: PYTHONPATH=. python3 robotics/env/tests/demo_all_tasks.py
"""
import time
import numpy as np
import pybullet as p
import pybullet_data

import robotics.env.src.config as _config
_config.NUM_OBJECTS = 1  # demo-only override: spawn just the target object,
                          # not all 4. Does NOT touch the real config.py file
                          # or affect the actual training environment.

from robotics.env.src.environment import KukaEnv
from robotics.env.src.config import (
    NUM_JOINTS, END_EFFECTOR_LINK_INDEX, JOINT_LOWER_LIMITS, JOINT_UPPER_LIMITS,
)

MIDPOINT = (JOINT_UPPER_LIMITS + JOINT_LOWER_LIMITS) / 2.0
HALF_RANGE = (JOINT_UPPER_LIMITS - JOINT_LOWER_LIMITS) / 2.0
DT = 1.0 / 60.0


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


def approach_to(env, target, gripper_cmd, max_steps=500, tol=0.015, trace_obj_id=None):
    """Solve IK ONCE toward a static target, then hold that action constant.
    We deliberately do NOT re-solve IK mid-motion here: doing so from a
    shifting joint configuration can cause calculateInverseKinematics to
    jump to a different, equally-valid elbow/wrist solution for the same
    Cartesian target (an "elbow flip"). That flip is what caused the
    object to swing/overshoot during LIFT when we re-solved every 5 steps.
    A single solve + capped joint motion converges smoothly instead.

    If trace_obj_id is given, prints ee height and (if tracked) object
    height every 20 steps.
    """
    action = act(solve_ik(env, target), gripper_cmd)
    info = None
    for i in range(max_steps):
        obs, r, term, trunc, info = env.step(action)
        time.sleep(DT)
        ee = np.array(info["ee_position"])
        if trace_obj_id is not None and i % 20 == 0:
            obj_state = info.get("object_state", {})
            obj_z = obj_state.get(trace_obj_id, {}).get("pos", [0, 0, -1])[2]
            print(f"    step {i}: ee_z={ee[2]:.4f}  obj_z={obj_z:.4f}  "
                  f"target_z={target[2]:.4f}  grasped={info.get('grasped_object')}")
        if np.linalg.norm(ee - np.array(target)) < tol:
            return info
    dist = np.linalg.norm(np.array(info["ee_position"]) - np.array(target))
    print(f"  [warning] did not reach target within {max_steps} steps (dist={dist:.4f})")
    return info


def push_through(env, target, gripper_cmd, max_steps=100):
    """Single IK solve, hold constant action -- validated push/pull pattern."""
    action = act(solve_ik(env, target), gripper_cmd)
    info = None
    for i in range(max_steps):
        obs, r, term, trunc, info = env.step(action)
        time.sleep(DT)
    return info


def label(text):
    print(f"\n>>> {text}")


def demo_reach(env, seed):
    label("REACH: move the end-effector toward the object (no grasp)")
    obs, info = env.reset(seed=seed)
    env.set_target_object(list(info["object_state"].keys())[0])
    obj_pos = np.array(info["object_state"][list(info["object_state"].keys())[0]]["pos"])
    approach_to(env, [obj_pos[0], obj_pos[1], obj_pos[2] + 0.05], gripper_cmd=-1.0)
    print("Reach complete.")


def demo_pick(env, seed):
    label("PICK: approach + close gripper on the object")
    obs, info = env.reset(seed=seed)
    obj_id = list(info["object_state"].keys())[0]
    env.set_target_object(obj_id)
    obj_pos = np.array(info["object_state"][obj_id]["pos"])
    info = approach_to(env, [obj_pos[0], obj_pos[1], obj_pos[2] + 0.005], gripper_cmd=1.0,
                        trace_obj_id=obj_id)
    print(f"Grasped: {info['grasped_object'] == obj_id}")


def demo_lift(env, seed):
    label("LIFT: grasp then raise the object")
    obs, info = env.reset(seed=seed)
    obj_id = list(info["object_state"].keys())[0]
    env.set_target_object(obj_id)
    obj_pos = np.array(info["object_state"][obj_id]["pos"])
    approach_to(env, [obj_pos[0], obj_pos[1], obj_pos[2] + 0.005], gripper_cmd=1.0,
                trace_obj_id=obj_id)
    info = approach_to(env, [obj_pos[0], obj_pos[1], obj_pos[2] + 0.10], gripper_cmd=1.0,
                        trace_obj_id=obj_id)
    final_pos = info["object_state"][obj_id]["pos"]
    print(f"Lift height gained: {final_pos[2] - obj_pos[2]:.3f} m")


def demo_place(env, seed):
    label("PLACE: grasp, carry sideways, release, let it settle")
    obs, info = env.reset(seed=seed)
    obj_id = list(info["object_state"].keys())[0]
    env.set_target_object(obj_id)
    obj_pos = np.array(info["object_state"][obj_id]["pos"])
    approach_to(env, [obj_pos[0], obj_pos[1], obj_pos[2] + 0.005], gripper_cmd=1.0)
    approach_to(env, [obj_pos[0] + 0.15, obj_pos[1] + 0.1, obj_pos[2] + 0.10], gripper_cmd=1.0)
    info = push_through(env, [obj_pos[0] + 0.15, obj_pos[1] + 0.1, obj_pos[2] + 0.10], gripper_cmd=-1.0, max_steps=40)
    print("Released. Letting it settle...")
    for _ in range(60):
        env.step(act(get_jp(env), -1.0))
        time.sleep(DT)


def demo_lower(env, seed):
    label("LOWER: grasp, lift, then bring back down without releasing")
    obs, info = env.reset(seed=seed)
    obj_id = list(info["object_state"].keys())[0]
    env.set_target_object(obj_id)
    obj_pos = np.array(info["object_state"][obj_id]["pos"])
    approach_to(env, [obj_pos[0], obj_pos[1], obj_pos[2] + 0.005], gripper_cmd=1.0)
    approach_to(env, [obj_pos[0], obj_pos[1], obj_pos[2] + 0.10], gripper_cmd=1.0)
    approach_to(env, [obj_pos[0], obj_pos[1], obj_pos[2] + 0.02], gripper_cmd=1.0)
    print("Lowered back down.")


def demo_push(env, seed):
    label("PUSH: approach on the base-facing side, drive AWAY from the base")
    obs, info = env.reset(seed=seed)
    obj_id = list(info["object_state"].keys())[0]
    env.set_target_object(obj_id)
    obj_pos = np.array(info["object_state"][obj_id]["pos"])
    # Direction away from the robot base (origin), in the xy plane.
    radial = obj_pos[:2] / (np.linalg.norm(obj_pos[:2]) + 1e-6)
    approach_xy = obj_pos[:2] - radial * 0.03  # stand on the base-facing side
    approach_to(env, [approach_xy[0], approach_xy[1], obj_pos[2]], gripper_cmd=-1.0, max_steps=400)
    push_xy = obj_pos[:2] + radial * 0.30  # drive further away from the base
    info = push_through(env, [push_xy[0], push_xy[1], obj_pos[2]], gripper_cmd=-1.0, max_steps=140)
    final_pos = info["object_state"][obj_id]["pos"]
    disp = np.linalg.norm(np.array(final_pos) - obj_pos)
    print(f"Pushed displacement: {disp:.3f} m")


def demo_pull(env, seed):
    label("PULL: approach on the far side from the base, drive TOWARD the base")
    obs, info = env.reset(seed=seed)
    obj_id = list(info["object_state"].keys())[0]
    env.set_target_object(obj_id)
    obj_pos = np.array(info["object_state"][obj_id]["pos"])
    radial = obj_pos[:2] / (np.linalg.norm(obj_pos[:2]) + 1e-6)
    approach_xy = obj_pos[:2] + radial * 0.03  # stand on the far side from the base
    approach_to(env, [approach_xy[0], approach_xy[1], obj_pos[2]], gripper_cmd=-1.0, max_steps=400)
    pull_xy = obj_pos[:2] - radial * 0.30  # drive toward the base
    info = push_through(env, [pull_xy[0], pull_xy[1], obj_pos[2]], gripper_cmd=-1.0, max_steps=140)
    final_pos = info["object_state"][obj_id]["pos"]
    disp = np.linalg.norm(np.array(final_pos) - obj_pos)
    print(f"Pulled displacement: {disp:.3f} m")


if __name__ == "__main__":
    env = KukaEnv()
    env._physics_client_id = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=50, cameraPitch=-35,
                                  cameraTargetPosition=[0.5, 0, 0.3])

    tasks = [
        (demo_reach, 1),
        (demo_pick, 7),
        (demo_lift, 42),
        (demo_place, 99),
        (demo_lower, 1),
        (demo_push, 1),
        (demo_pull, 7),
    ]

    for fn, seed in tasks:
        fn(env, seed)
        time.sleep(1.5)

    print("\nAll task demos complete. Closing in 5 seconds...")
    time.sleep(5)
    p.disconnect(physicsClientId=env._physics_client_id)
