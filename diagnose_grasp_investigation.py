"""
Extended diagnosis for issue #5 write-up.

1. Does grasp ever actually fire at the raw KukaEnv level (grasped_object_id
   becoming non-None), independent of whether RewardShapingWrapper can see it?
   Runs longer episodes and logs every step where grasped_object_id is not None,
   with the exact step index and distance at that moment.

2. Force task_type to "lift" directly on the wrapper (bypassing the missing
   set_instruction() call) to see whether the LIFT branch's logic is even
   reachable/sane when task_type is set correctly, using the real trained policy.

3. For each episode, report whether termination was via truncation (timeout)
   vs an actual terminated=True some other way, to separate "policy never got
   close" from "episode just ran out of time".
"""
import sys
import numpy as np

sys.path.insert(0, ".")

import rl.pipeline as pipeline


def run_investigative_episode(instruction_text, max_steps=300, force_task_type=None):
    pipeline._load()
    bg_env = pipeline._bg_env
    model = pipeline._model
    kuka_env = bg_env.env.env

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

    if target_obj_id and target_obj_id in obj_state:
        kuka_env.set_target_object(target_obj_id)
        bg_env._target_pos = np.array(obj_state[target_obj_id]["pos"], dtype=np.float32)
        bg_env._target_obj_id = target_obj_id
        bg_env._prev_dist = float(np.linalg.norm(ee_pos - bg_env._target_pos))

    if force_task_type:
        bg_env._task_type = force_task_type
        bg_env._task_start_height = bg_env._target_pos[2]

    bg_env.env.set_embedding(embedding)
    bg_env.env._inference_mode = True

    info0 = {"ee_position": ee_pos.tolist(), "object_state": obj_state}
    obs = bg_env.env.observation(raw_obs, info0)
    obs["nlp"] = embedding

    print(f"\n=== Episode: instruction={instruction_text!r} | forced_task_type={force_task_type!r} ===")

    grasp_fire_events = []
    last_info = {}
    terminated_flag = False
    truncated_flag = False

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = bg_env.step(action)
        last_info = info

        gid = info.get("grasped_object_id")
        if gid is not None:
            grasp_fire_events.append((step, info.get("distance_to_target"), gid))

        if terminated:
            terminated_flag = True
        if truncated:
            truncated_flag = True

        if terminated or truncated:
            break

    bg_env.env._inference_mode = False

    steps_run = step + 1
    print(f"  steps run: {steps_run} | terminated={terminated_flag} truncated={truncated_flag}")
    print(f"  final distance_to_target: {last_info.get('distance_to_target')}")
    print(f"  final is_success: {last_info.get('is_success')}")
    print(f"  final task_type as seen by wrapper: {last_info.get('task_type')}")
    if grasp_fire_events:
        print(f"  ENV-LEVEL GRASP FIRED at {len(grasp_fire_events)} step(s):")
        for s, d, gid in grasp_fire_events[:10]:
            print(f"    step={s} distance_to_target={d} grasped_object_id={gid}")
    else:
        print("  env-level grasp (grasped_object_id) never fired at any step.")

    return {
        "instruction": instruction_text,
        "grasp_fire_events": grasp_fire_events,
        "terminated": terminated_flag,
        "truncated": truncated_flag,
        "final_info": last_info,
    }


if __name__ == "__main__":
    np.random.seed(0)

    print("\n#################### PART A: does env-level grasp ever fire? (longer episodes, repeated) ####################")
    for trial in range(5):
        run_investigative_episode("pick up the red cylinder", max_steps=300)

    print("\n#################### PART B: force task_type='lift' so the LIFT branch is reachable ####################")
    for trial in range(3):
        run_investigative_episode("lift the blue box", max_steps=300, force_task_type="lift")

    print("\n#################### PART C: force task_type='pick' so the PICK branch is reachable ####################")
    for trial in range(3):
        run_investigative_episode("grab the yellow box", max_steps=300, force_task_type="pick")
