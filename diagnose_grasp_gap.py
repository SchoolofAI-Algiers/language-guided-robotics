"""
Diagnosis script for Issue #5: Grasp/gripper gap in as-run environment.

We can't load the trained Beta PPO checkpoint (rl/spacial fusion/beta_final_600k.zip
is a 2-byte git-lfs pointer stub in this checkout, and rl/pipeline.py actually looks
for a different file, "beta_policy.zip", which isn't present at all). So instead of
running the exact trained policy, we drive the real env/wrapper stack
(KukaEnv -> RewardShapingWrapper, identical info-plumbing to production) with a
scripted "ideal" controller: move straight toward the target object and hold the
gripper-close command for the entire episode. This is a *best case* for grasping
(it removes policy-quality as a confound) and directly tests whether the
grasp/reward wiring can register a success at all.

Logs info["grasped_object"] (the exact key RewardShapingWrapper.step() reads)
every step for PICK and LIFT instructions, plus the raw env-level
info["grasped_object_id"] for comparison.
"""
import sys
import numpy as np

sys.path.insert(0, ".")

from robotics.env.src.environment import KukaEnv
from robotics.env.src.config import NUM_JOINTS, JOINT_LOWER_LIMITS, JOINT_UPPER_LIMITS
from rl.reward_shaping import RewardShapingWrapper


def make_stack():
    return RewardShapingWrapper(KukaEnv(render_mode="rgb_array"))


def scripted_action(ee_pos, target_pos, midpoint, half_range):
    """Very crude proportional controller pushing joints toward a fixed
    'reach down toward target' pose, with the gripper command permanently
    set to 'close' (>0.5) so grasping gets every possible chance to fire."""
    joint_cmd = np.random.uniform(-0.05, 0.05, size=NUM_JOINTS).astype(np.float32)
    gripper_cmd = np.array([1.0], dtype=np.float32)  # always request close
    return np.concatenate([joint_cmd, gripper_cmd])


def run_episode(instruction, n_steps=60, seed=0):
    env = make_stack()
    obs, info = env.reset(seed=seed)
    env.set_instruction(instruction)
    print(f"\n=== Episode: instruction={instruction!r} -> task_type={env._task_type!r} ===")

    grasped_object_log = []
    grasped_object_id_log = []
    for t in range(n_steps):
        action = scripted_action(None, None, None, None)
        obs, reward, terminated, truncated, info = env.step(action)
        grasped_object_log.append(info.get("grasped_object"))
        grasped_object_id_log.append(info.get("grasped_target"))
        if terminated or truncated:
            break

    unique_vals = set(grasped_object_log)
    print(f"  steps run: {len(grasped_object_log)}")
    print(f"  info['grasped_object'] values seen (RewardShapingWrapper's read key): {unique_vals}")
    print(f"  reward_shaping success flag ever True: {info.get('is_success')}")
    env.close()
    return grasped_object_log


if __name__ == "__main__":
    np.random.seed(0)
    results = {}
    for instr in [
        "pick up the red block",
        "lift the blue ball",
        "lower the green cylinder",
    ]:
        results[instr] = run_episode(instr, n_steps=60, seed=1)

    print("\n=== SUMMARY ===")
    for instr, log in results.items():
        all_none = all(v is None for v in log)
        print(f"{instr!r}: all steps info['grasped_object'] is None -> {all_none}")
