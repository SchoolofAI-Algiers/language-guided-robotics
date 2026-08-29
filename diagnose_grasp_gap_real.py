"""
Diagnosis script for Issue #5 (REAL version) — Grasp/gripper gap in as-run environment.

Unlike the first pass, this drives the ACTUAL trained Beta PPO policy
(rl/spacial fusion/beta_policy.zip) through the ACTUAL production wrapper stack
used by rl/pipeline.py's run_episode():

    KukaEnv -> BetaLanguageConditionedWrapper -> RewardShapingWrapper

It replicates pipeline.run_episode()'s setup exactly:
  - picks a target object matching the instruction's color (or first object)
  - calls kuka_env.set_target_object(target_obj_id)  <- this is what's supposed
    to arm the gripper/grasp mechanism for that episode
  - sets the NLP embedding for the instruction via find_best_embedding()
  - builds the observation the same way pipeline.py does
  - runs model.predict(obs, deterministic=True) in a loop, exactly like production

At every step it logs:
  - info["grasped_object"]      <- the exact key RewardShapingWrapper.step() reads
  - info["grasped_object_id"]   <- the raw env-level truth (if present)
  - info["is_success"]
  - info.get("distance_to_target")

This directly answers the issue's literal request: run PICK/LIFT episodes with the
current trained Beta policy and confirm info["grasped_object"] is always None.
"""
import sys
import numpy as np

sys.path.insert(0, ".")

import rl.pipeline as pipeline


def run_real_episode(instruction_text, max_steps=150):
    pipeline._load()  # loads model + bg env + embeddings once

    bg_env = pipeline._bg_env
    model = pipeline._model

    kuka_env = bg_env.env.env  # RewardShaping -> BetaWrapper -> KukaEnv

    obs0, reset_info = bg_env.reset()
    obj_state = kuka_env._get_object_state()
    raw_obs = kuka_env._get_observation()
    ee_pos = raw_obs[14:17]

    embedding, matched = pipeline.find_best_embedding(instruction_text)
    target_colour = pipeline._parse_colour(instruction_text)

    target_obj_id = None
    if target_colour:
        for oid, st in obj_state.items():
            if st["color"] == target_colour:
                target_obj_id = oid
                break
    if target_obj_id is None and obj_state:
        target_obj_id = list(obj_state.keys())[0]

    grasp_armed = False
    if target_obj_id and target_obj_id in obj_state:
        kuka_env.set_target_object(target_obj_id)  # <- the real gripper-arming call
        bg_env._target_pos = np.array(obj_state[target_obj_id]["pos"], dtype=np.float32)
        bg_env._target_obj_id = target_obj_id
        bg_env._prev_dist = float(np.linalg.norm(ee_pos - bg_env._target_pos))
        grasp_armed = True

    bg_env.env.set_embedding(embedding)
    bg_env.env._inference_mode = True

    info0 = {"ee_position": ee_pos.tolist(), "object_state": obj_state}
    obs = bg_env.env.observation(raw_obs, info0)
    obs["nlp"] = embedding

    print(f"\n=== Episode: instruction={instruction_text!r} ===")
    print(f"  matched embedding instruction: {matched!r}")
    print(f"  target object: {target_obj_id} (color={target_colour}) | gripper armed via set_target_object: {grasp_armed}")

    grasped_object_log = []
    grasped_object_id_log = []
    is_success_log = []
    last_info = {}

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = bg_env.step(action)
        last_info = info

        grasped_object_log.append(info.get("grasped_object"))
        grasped_object_id_log.append(info.get("grasped_object_id"))
        is_success_log.append(info.get("is_success", False))

        if terminated or truncated:
            break

    bg_env.env._inference_mode = False

    steps_run = len(grasped_object_log)
    unique_grasped = set(grasped_object_log)
    unique_grasped_id = set(grasped_object_id_log)
    any_success = any(is_success_log)

    print(f"  steps run: {steps_run}")
    print(f"  info['grasped_object'] values seen (RewardShapingWrapper's read key): {unique_grasped}")
    print(f"  info['grasped_object_id'] values seen (raw env-level key): {unique_grasped_id}")
    print(f"  info['is_success'] ever True: {any_success}")
    print(f"  final distance_to_target: {last_info.get('distance_to_target')}")

    return {
        "instruction": instruction_text,
        "grasped_object_log": grasped_object_log,
        "grasped_object_id_log": grasped_object_id_log,
        "any_success": any_success,
    }


if __name__ == "__main__":
    np.random.seed(0)
    test_instructions = [
        "pick up the red cylinder",
        "lift the blue box",
        "grab the yellow box",
        "lower the green cylinder",
    ]

    results = {}
    for instr in test_instructions:
        results[instr] = run_real_episode(instr, max_steps=150)

    print("\n=== SUMMARY (real trained Beta policy) ===")
    for instr, res in results.items():
        all_none = all(v is None for v in res["grasped_object_log"])
        print(f"{instr!r}: all steps info['grasped_object'] is None -> {all_none} | any is_success -> {res['any_success']}")
