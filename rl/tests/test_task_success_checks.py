"""
Manual/scripted verification for Issue #8 task-specific success checks.

Not a pytest suite of the "assert on network output" kind -- these are
scripted, IK-driven episodes (same approach as
robotics/env/tests/verify_grasp_ik.py) that drive the arm directly, bypassing
the undertrained policy, so a failure here means the reward_shaping.py logic
itself is wrong, not that the policy is bad.

Run directly:
    PYTHONPATH=. python3 rl/tests/test_task_success_checks.py
"""
import numpy as np
import pybullet as p

from robotics.env.src.environment import KukaEnv
from robotics.env.src.config import (
    NUM_JOINTS, END_EFFECTOR_LINK_INDEX, JOINT_LOWER_LIMITS,
    JOINT_UPPER_LIMITS, GRIPPER_ATTACH_DISTANCE,
)
from rl.reward_shaping import RewardShapingWrapper

SEED = 42

MIDPOINT = (JOINT_UPPER_LIMITS + JOINT_LOWER_LIMITS) / 2.0
HALF_RANGE = (JOINT_UPPER_LIMITS - JOINT_LOWER_LIMITS) / 2.0


def joint_targets_to_action(joint_targets, gripper_cmd):
    """Convert absolute joint targets (radians) + gripper command into the
    normalized [-1, 1] 8-dim action KukaEnv.step() expects."""
    normalized = (np.array(joint_targets) - MIDPOINT) / HALF_RANGE
    normalized = np.clip(normalized, -1.0, 1.0)
    return np.concatenate([normalized, [gripper_cmd]]).astype(np.float32)


def get_joint_positions(env):
    return np.array([
        p.getJointState(env._kuka_id, j, physicsClientId=env._physics_client_id)[0]
        for j in range(NUM_JOINTS)
    ])


def solve_ik(env, target_pos, target_orn=None):
    """Solve IK biased toward the arm's CURRENT joint configuration via
    restPoses. Without this, calculateInverseKinematics can jump to a
    different (but equally valid) elbow configuration for a nearby
    cartesian point, causing capped joint-space motion to chase a target
    that keeps switching solutions instead of converging smoothly."""
    if target_orn is None:
        target_orn = p.getQuaternionFromEuler([np.pi, 0, 0])
    current = get_joint_positions(env).tolist()
    lower = JOINT_LOWER_LIMITS.tolist()
    upper = JOINT_UPPER_LIMITS.tolist()
    ranges = (JOINT_UPPER_LIMITS - JOINT_LOWER_LIMITS).tolist()
    joint_poses = p.calculateInverseKinematics(
        env._kuka_id, END_EFFECTOR_LINK_INDEX, target_pos, target_orn,
        lowerLimits=lower, upperLimits=upper, jointRanges=ranges, restPoses=current,
        maxNumIterations=100,
        physicsClientId=env._physics_client_id,
    )
    return np.array(joint_poses[:NUM_JOINTS])


def drive_to(wrapped_env, env, target_pos, gripper_cmd, steps=40, max_joint_step=None):
    """Step the wrapped env toward target_pos for `steps` calls, holding
    gripper_cmd constant. Re-solves IK each step (tracks well as the ee gets
    close to a fixed cartesian target -- this is what worked in earlier runs).
    If max_joint_step is set, per-step joint motion is additionally capped so
    the arm eases in near an object instead of applying full MAX_FORCE torque
    in one step -- use this only for the final approach near an object, not
    for free-space travel over a large distance (where it would just make the
    arm too slow to arrive within the step budget).
    Returns the last (obs, reward, terminated, truncated, info) tuple."""
    result = None
    for i in range(steps):
        ik_target = solve_ik(env, target_pos)
        if max_joint_step is not None:
            current = get_joint_positions(env)
            delta = np.clip(ik_target - current, -max_joint_step, max_joint_step)
            ik_target = current + delta
        action = joint_targets_to_action(ik_target, gripper_cmd)
        result = wrapped_env.step(action)
        if result[2]:  # terminated
            break
    return result


def get_obj_pos(info, obj_id):
    return np.array(info["object_state"][obj_id]["pos"], dtype=np.float32)


def pick_object_by_radius(info, prefer="max"):
    """Choose the spawned object whose xy-distance from the robot base
    (origin) is largest ("max", good for PULL - room to move inward) or
    smallest ("min", good for PUSH - room to move outward)."""
    obj_state = info["object_state"]
    radii = {oid: float(np.linalg.norm(np.array(st["pos"][:2]))) for oid, st in obj_state.items()}
    return max(radii, key=radii.get) if prefer == "max" else min(radii, key=radii.get)


def run_push_or_pull(task_type, direction_sign, label):
    """Verify reward_shaping.py's PUSH/PULL success logic directly, instead
    of relying on the scripted IK arm to physically shove the object.

    The physical approach was abandoned: PyBullet's contact response between
    the arm's mesh collision geometry and a light (0.1kg) object under
    MAX_FORCE=300N position control is unstable -- contact routinely
    "explodes" the object to a random far-off position instead of sliding
    it, regardless of approach angle or step size. That's an environment
    physics-tuning issue (contact stiffness/damping/restitution), tracked
    separately -- it is NOT a reward_shaping.py bug, and chasing it further
    here doesn't tell us anything more about issue #8's actual logic.

    Instead: get the end-effector into real contact range of the object
    (so made_contact is set the same way it would be in a real episode),
    then directly displace the object via resetBasePositionAndOrientation
    in small increments -- simulating "the object got pushed/pulled" -- and
    confirm reward_shaping.py correctly computes directional_progress and
    fires success at the 25cm threshold, in the correct direction only.
    """
    print(f"\n=== {label} ===")
    env = KukaEnv()
    wrapped = RewardShapingWrapper(env)
    obs, info = wrapped.reset(seed=SEED)

    obj_id = pick_object_by_radius(info, prefer=("min" if task_type == "push" else "max"))
    wrapped._target_obj_id = obj_id
    env.set_target_object(obj_id)
    wrapped.set_instruction(f"{task_type} the object")
    wrapped._task_type = task_type

    obj_pos = get_obj_pos(info, obj_id)
    wrapped._initial_obj_xy = obj_pos[:2].copy()
    wrapped._prev_obj_pos = obj_pos.copy()
    radial = obj_pos[:2] / (np.linalg.norm(obj_pos[:2]) + 1e-6)
    print(f"  chosen obj_id={obj_id}  start pos={np.round(obj_pos,3)}")

    # Get the ee into real contact range (this part is fine -- IK reliably
    # converges to a static point; it's only continuous contact-through-motion
    # that's unstable). Hover just above the object, close enough to trigger
    # made_contact, without touching it and triggering the unstable response.
    approach_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.08]
    _, _, _, _, info = drive_to(wrapped, env, approach_pos, gripper_cmd=-1.0, steps=100)
    ee = np.array(info["ee_position"])
    print(f"  [approach] ee={np.round(ee,3)}  dist_to_obj={np.linalg.norm(ee-obj_pos):.4f}  "
          f"made_contact={info['made_contact']}")

    # Directly displace the object in small increments, simulating a
    # successful push/pull, and check reward_shaping.py's response each step.
    n_increments = 20
    total_displacement = 0.30  # slightly over threshold, correct direction
    result = None
    for i in range(1, n_increments + 1):
        new_xy = obj_pos[:2] + direction_sign * radial * (total_displacement * i / n_increments)
        new_pos = [new_xy[0], new_xy[1], obj_pos[2]]
        p.resetBasePositionAndOrientation(obj_id, new_pos, [0, 0, 0, 1],
                                           physicsClientId=env._physics_client_id)
        # Hold the arm roughly in place (small no-op-ish action) while the
        # object is "pushed", one env.step() per increment so
        # reward_shaping.step() re-reads the object's new position.
        joint_targets = solve_ik(env, approach_pos)
        action = joint_targets_to_action(joint_targets, gripper_cmd=-1.0)
        result = wrapped.step(action)
        if result[2]:  # terminated
            break

    obs, reward, terminated, truncated, info = result
    final_obj_pos = get_obj_pos(info, obj_id)
    displacement_xy = final_obj_pos[:2] - obj_pos[:2]
    radial_disp = float(np.dot(displacement_xy, radial * direction_sign))
    print(f"  obj_pos start={obj_pos[:2]}  end={final_obj_pos[:2]}")
    print(f"  directional_progress={radial_disp:.4f}  (threshold=0.25)")
    print(f"  is_success={info['is_success']}  made_contact={info['made_contact']}  "
          f"terminated={terminated}")
    assert info["is_success"], f"{label} FAILED to report success"
    env.close()


def run_push_or_pull_wrong_direction(task_type, direction_sign, label):
    """Negative test: displace the object 30cm in the WRONG direction for
    the given task type (e.g. toward the base for PUSH). Confirms
    directional gating actually rejects wrong-direction displacement,
    not just magnitude."""
    print(f"\n=== {label} (negative: wrong direction) ===")
    env = KukaEnv()
    wrapped = RewardShapingWrapper(env)
    obs, info = wrapped.reset(seed=SEED)

    obj_id = pick_object_by_radius(info, prefer=("min" if task_type == "push" else "max"))
    wrapped._target_obj_id = obj_id
    env.set_target_object(obj_id)
    wrapped.set_instruction(f"{task_type} the object")
    wrapped._task_type = task_type

    obj_pos = get_obj_pos(info, obj_id)
    wrapped._initial_obj_xy = obj_pos[:2].copy()
    wrapped._prev_obj_pos = obj_pos.copy()
    radial = obj_pos[:2] / (np.linalg.norm(obj_pos[:2]) + 1e-6)

    approach_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.08]
    drive_to(wrapped, env, approach_pos, gripper_cmd=-1.0, steps=100)

    # WRONG direction: opposite sign from what this task type requires.
    wrong_direction_sign = -direction_sign
    n_increments = 20
    total_displacement = 0.30
    result = None
    for i in range(1, n_increments + 1):
        new_xy = obj_pos[:2] + wrong_direction_sign * radial * (total_displacement * i / n_increments)
        new_pos = [new_xy[0], new_xy[1], obj_pos[2]]
        p.resetBasePositionAndOrientation(obj_id, new_pos, [0, 0, 0, 1],
                                           physicsClientId=env._physics_client_id)
        joint_targets = solve_ik(env, approach_pos)
        action = joint_targets_to_action(joint_targets, gripper_cmd=-1.0)
        result = wrapped.step(action)
        if result[2]:
            break

    obs, reward, terminated, truncated, info = result
    print(f"  is_success={info['is_success']}  terminated={terminated}")
    assert not info["is_success"], f"{label} INCORRECTLY reported success for wrong-direction displacement"
    env.close()


def run_lift():
    print("\n=== LIFT ===")
    env = KukaEnv()
    wrapped = RewardShapingWrapper(env)
    obs, info = wrapped.reset(seed=SEED)
    obj_id = wrapped._target_obj_id
    env.set_target_object(obj_id)
    wrapped.set_instruction("lift the object")
    wrapped._task_type = "lift"

    obj_pos = get_obj_pos(info, obj_id)
    above_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.01]
    print(f"  obj_pos={obj_pos}  target above_pos={above_pos}")
    for i in range(60):
        joint_targets = solve_ik(env, above_pos)
        action = joint_targets_to_action(joint_targets, gripper_cmd=1.0)
        obs, reward, terminated, truncated, info = wrapped.step(action)
        if i % 15 == 0 or i == 59:
            ee = info["ee_position"]
            dist = float(np.linalg.norm(np.array(ee) - obj_pos))
            print(f"    step {i}: ee={np.round(ee,3)}  dist_to_obj={dist:.4f}  "
                  f"grasped_object={info['grasped_object']}")

    lift_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.15]
    obs, reward, terminated, truncated, info = drive_to(
        wrapped, env, lift_pos, gripper_cmd=1.0, steps=60
    )

    print(f"  is_success={info['is_success']}  grasped_target={info['grasped_target']}  "
          f"terminated={terminated}")
    assert info["is_success"], "LIFT FAILED to report success"
    env.close()


def run_place():
    print("\n=== PLACE (positive) ===")
    env = KukaEnv()
    wrapped = RewardShapingWrapper(env)
    obs, info = wrapped.reset(seed=SEED)
    obj_id = wrapped._target_obj_id
    env.set_target_object(obj_id)
    wrapped.set_instruction("place the object on the side")
    wrapped._task_type = "place"

    obj_pos = get_obj_pos(info, obj_id)
    above_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.01]
    drive_to(wrapped, env, above_pos, gripper_cmd=1.0, steps=60)  # grasp

    move_pos = [obj_pos[0] + 0.15, obj_pos[1] + 0.1, obj_pos[2] + 0.05]
    drive_to(wrapped, env, move_pos, gripper_cmd=1.0, steps=60)  # carry

    obs, reward, terminated, truncated, info = drive_to(
        wrapped, env, move_pos, gripper_cmd=-1.0, steps=60
    )

    print(f"  is_success={info['is_success']}  settled_steps={info['settled_steps']}  "
          f"had_release_after_grasp={info['had_release_after_grasp']}  terminated={terminated}")
    assert info["is_success"], "PLACE (positive) FAILED to report success"
    env.close()


def run_place_negative():
    print("\n=== PLACE (negative: premature release mid-air) ===")
    env = KukaEnv()
    wrapped = RewardShapingWrapper(env)
    obs, info = wrapped.reset(seed=SEED)
    obj_id = wrapped._target_obj_id
    env.set_target_object(obj_id)
    wrapped.set_instruction("place the object on the side")
    wrapped._task_type = "place"

    obj_pos = get_obj_pos(info, obj_id)
    above_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.01]
    drive_to(wrapped, env, above_pos, gripper_cmd=1.0, steps=60)  # grasp

    high_pos = [obj_pos[0], obj_pos[1], obj_pos[2] + 0.25]
    drive_to(wrapped, env, high_pos, gripper_cmd=1.0, steps=60)  # carry up

    joint_targets = solve_ik(env, high_pos)
    action = joint_targets_to_action(joint_targets, gripper_cmd=-1.0)
    obs, reward, terminated, truncated, info = wrapped.step(action)

    print(f"  is_success={info['is_success']}  settled_steps={info['settled_steps']}  "
          f"had_release_after_grasp={info['had_release_after_grasp']}  terminated={terminated}")
    assert not info["is_success"], "PLACE (negative) INCORRECTLY reported success immediately after drop"
    env.close()


if __name__ == "__main__":
    run_push_or_pull("push", direction_sign=1.0, label="PUSH")
    run_push_or_pull("pull", direction_sign=-1.0, label="PULL")
    run_push_or_pull_wrong_direction("push", direction_sign=1.0, label="PUSH")
    run_push_or_pull_wrong_direction("pull", direction_sign=-1.0, label="PULL")
    run_lift()
    run_place()
    run_place_negative()
    print("\nAll #8 task success checks passed.")
