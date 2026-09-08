"""
Test for: arm contact launches light objects instead of sliding them.

Unlike the earlier scratch diagnostic, this does NOT re-solve IK every step.
It solves IK once for a point past the object, then holds that single joint
target constant and lets the env's own per-step motion cap (see
MAX_JOINT_STEP_RADIANS in config.py) govern how fast the arm actually moves.
This mirrors how a trained policy behaves -- it outputs an action based on
the current observation, it doesn't re-run inverse kinematics every step.

Run: PYTHONPATH=. python3 robotics/env/tests/test_push_contact_stability.py
"""
import numpy as np
import pybullet as p

from robotics.env.src.environment import KukaEnv
from robotics.env.src.config import (
    NUM_JOINTS, END_EFFECTOR_LINK_INDEX, JOINT_LOWER_LIMITS, JOINT_UPPER_LIMITS,
)

MIDPOINT = (JOINT_UPPER_LIMITS + JOINT_LOWER_LIMITS) / 2.0
HALF_RANGE = (JOINT_UPPER_LIMITS - JOINT_LOWER_LIMITS) / 2.0

LAUNCH_SPEED_THRESHOLD = 0.5   # m/s -- above this, call it a launch, not a slide
STEPS = 80


def get_joint_positions(env):
    return np.array([
        p.getJointState(env._kuka_id, j, physicsClientId=env._physics_client_id)[0]
        for j in range(NUM_JOINTS)
    ])


def solve_ik_once(env, target_pos):
    target_orn = p.getQuaternionFromEuler([np.pi, 0, 0])
    current = get_joint_positions(env).tolist()
    joint_poses = p.calculateInverseKinematics(
        env._kuka_id, END_EFFECTOR_LINK_INDEX, target_pos, target_orn,
        lowerLimits=JOINT_LOWER_LIMITS.tolist(), upperLimits=JOINT_UPPER_LIMITS.tolist(),
        jointRanges=(JOINT_UPPER_LIMITS - JOINT_LOWER_LIMITS).tolist(), restPoses=current,
        maxNumIterations=100, physicsClientId=env._physics_client_id,
    )
    return np.array(joint_poses[:NUM_JOINTS])


def joint_target_to_action(joint_target, gripper_cmd):
    normalized = np.clip((joint_target - MIDPOINT) / HALF_RANGE, -1.0, 1.0)
    return np.concatenate([normalized, [gripper_cmd]]).astype(np.float32)


def run_case(seed, obj_index, verbose=True):
    env = KukaEnv()
    obs, info = env.reset(seed=seed)
    obj_state = info["object_state"]
    obj_id = list(obj_state.keys())[obj_index]
    obj_pos = np.array(obj_state[obj_id]["pos"])
    shape = obj_state[obj_id]["shape"]

    # Approach phase: get the end-effector close to the object first (this
    # part isn't what we're testing, so give it a generous, fixed budget --
    # capped motion the whole way, matching real env.step() behavior).
    approach_target = [obj_pos[0] - 0.03, obj_pos[1], obj_pos[2]]
    approach_joint_target = solve_ik_once(env, approach_target)
    approach_action = joint_target_to_action(approach_joint_target, gripper_cmd=-1.0)
    ee_pos = None
    for i in range(300):
        obs, r, term, trunc, info = env.step(approach_action)
        ee_pos = np.array(info["ee_position"])
        if np.linalg.norm(ee_pos - np.array(approach_target)) < 0.02:
            break
    approach_dist = float(np.linalg.norm(ee_pos - np.array(approach_target))) if ee_pos is not None else -1.0

    # Push phase: fixed action toward a point past the object.
    push_target = [obj_pos[0] + 0.20, obj_pos[1], obj_pos[2]]
    joint_target = solve_ik_once(env, push_target)
    action = joint_target_to_action(joint_target, gripper_cmd=-1.0)

    max_speed_seen = 0.0
    for i in range(STEPS):
        obs, r, term, trunc, info = env.step(action)
        vel, _ = p.getBaseVelocity(obj_id, physicsClientId=env._physics_client_id)
        speed = float(np.linalg.norm(vel))
        max_speed_seen = max(max_speed_seen, speed)

    final_pos, _ = p.getBasePositionAndOrientation(obj_id, physicsClientId=env._physics_client_id)
    displacement = float(np.linalg.norm(np.array(final_pos) - obj_pos))
    launched = max_speed_seen > LAUNCH_SPEED_THRESHOLD

    if verbose:
        status = "LAUNCH" if launched else "ok"
        reached = "reached" if approach_dist < 0.03 else f"DID NOT REACH (dist={approach_dist:.3f})"
        print(f"  seed={seed} idx={obj_index} shape={shape:8s} approach={reached:22s} "
              f"max_speed={max_speed_seen:.3f} displacement={displacement:.4f}  [{status}]")

    env.close()
    return launched, max_speed_seen, displacement


if __name__ == "__main__":
    cases = [(1, 0), (7, 1), (99, 0), (99, 2), (42, 0), (1, 1), (1, 2), (7, 0), (7, 2), (42, 1), (42, 2)]
    print(f"Running {len(cases)} push cases (fixed action, no per-step IK re-solve)...")
    results = [run_case(seed, idx) for seed, idx in cases]
    n_launched = sum(1 for launched, _, _ in results if launched)
    print(f"\n{n_launched}/{len(cases)} cases launched (speed > {LAUNCH_SPEED_THRESHOLD} m/s)")
