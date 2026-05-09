reward_code = '''import numpy as np
import gymnasium as gym


class RewardShapingWrapper(gym.Wrapper):
    """
    Multi-task reward shaping with curriculum awareness.
    Different reward functions for reach, pick, push, pull, lift, place, lower.
    EXACT copy from spacial-fusion (3).ipynb Cell 7.
    """

    def __init__(self, env):
        super().__init__(env)
        self._target_pos = np.zeros(3, dtype=np.float32)
        self._prev_dist  = 0.0
        self._current_task_type = "reach"
        self._prev_grasped = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._current_task_type = info.get("task_type", "reach")
        obj_state = info.get("object_state", {})
        if obj_state:
            first_id  = list(obj_state.keys())[0]
            self._target_pos = np.array(obj_state[first_id]["pos"], dtype=np.float32)
        else:
            self._target_pos = np.array([0.5, 0.0, 0.42], dtype=np.float32)

        ee_pos = np.array(info.get("ee_position", [0.0, 0.0, 0.0]), dtype=np.float32)
        self._prev_dist = float(np.linalg.norm(ee_pos - self._target_pos))
        self._prev_grasped = False
        info["target_pos"] = self._target_pos.tolist()
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        self._current_task_type = info.get("task_type", "reach")
        ee_pos   = np.array(info.get("ee_position", [0.0, 0.0, 0.0]), dtype=np.float32)
        curr_dist = float(np.linalg.norm(ee_pos - self._target_pos))
        is_grasped = info.get("grasped_object_id") is not None
        obj_state = info.get("object_state", {})

        if self._current_task_type == "reach":
            reward = (self._prev_dist - curr_dist) * 0.5 - 0.001
            if curr_dist < 0.10:
                reward += 1.0
                terminated = True
        elif self._current_task_type == "pick":
            reach_reward = (self._prev_dist - curr_dist) * 0.3
            grasp_reward = 0.0
            if is_grasped and not self._prev_grasped:
                grasp_reward = 5.0
                terminated = True
            elif not is_grasped and self._prev_grasped:
                grasp_reward = -0.5
            reward = reach_reward + grasp_reward - 0.001
        elif self._current_task_type == "push":
            if obj_state and len(obj_state) > 0:
                first_id = list(obj_state.keys())[0]
                obj_pos = np.array(obj_state[first_id]["pos"][:2], dtype=np.float32)
                target_2d = self._target_pos[:2]
                push_dist = np.linalg.norm(obj_pos - target_2d)
                push_reward = (self._prev_dist - push_dist) * 0.3 - 0.001
                if push_dist > 0.15:
                    push_reward += 0.5
                    if push_dist > 0.25:
                        push_reward += 0.5
                        terminated = True
                reward = push_reward
            else:
                reward = -0.001
        elif self._current_task_type == "pull":
            if obj_state and len(obj_state) > 0:
                first_id = list(obj_state.keys())[0]
                obj_pos = np.array(obj_state[first_id]["pos"][:2], dtype=np.float32)
                start_pos = self._target_pos[:2]
                pull_dist = np.linalg.norm(obj_pos - start_pos)
                pull_reward = (self._prev_dist - pull_dist) * 0.3 - 0.001
                if pull_dist > 0.15:
                    pull_reward += 0.5
                    if pull_dist > 0.25:
                        pull_reward += 0.5
                        terminated = True
                reward = pull_reward
            else:
                reward = -0.001
        elif self._current_task_type == "lift":
            if is_grasped:
                grasped_obj = info.get("grasped_object_id")
                if obj_state and grasped_obj and grasped_obj in obj_state:
                    obj_z = obj_state[grasped_obj]["pos"][2]
                    lift_threshold_z = self._target_pos[2] + 0.1
                    lift_reward = max(0, (obj_z - self._target_pos[2]) * 2.0) - 0.001
                    if obj_z > lift_threshold_z:
                        lift_reward += 2.0
                        if obj_z > lift_threshold_z + 0.1:
                            lift_reward += 1.0
                            terminated = True
                    reward = lift_reward
                else:
                    reward = -0.001
            else:
                reach_reward = (self._prev_dist - curr_dist) * 0.3 - 0.001
                if is_grasped and not self._prev_grasped:
                    reach_reward += 2.0
                reward = reach_reward
        elif self._current_task_type == "place":
            if is_grasped:
                reward = (self._prev_dist - curr_dist) * 0.3 - 0.001
            else:
                if obj_state and len(obj_state) > 0:
                    first_id = list(obj_state.keys())[0]
                    obj_pos = np.array(obj_state[first_id]["pos"], dtype=np.float32)
                    place_dist = np.linalg.norm(obj_pos - self._target_pos)
                    reward = max(0, 1.0 - place_dist) - 0.001
                    if place_dist < 0.15:
                        reward += 2.0
                        terminated = True
                else:
                    reward = -0.001
        elif self._current_task_type == "lower":
            if is_grasped:
                grasped_obj = info.get("grasped_object_id")
                if obj_state and grasped_obj and grasped_obj in obj_state:
                    obj_z = obj_state[grasped_obj]["pos"][2]
                    lower_reward = max(0, (self._target_pos[2] - obj_z) * 2.0) - 0.001
                    if obj_z < self._target_pos[2] + 0.05:
                        lower_reward += 1.0
                        terminated = True
                    reward = lower_reward
                else:
                    reward = -0.001
            else:
                reward = (self._prev_dist - curr_dist) * 0.3 - 0.001
        else:
            reward = (self._prev_dist - curr_dist) * 0.5 - 0.001

        self._prev_dist = curr_dist
        self._prev_grasped = is_grasped
        info["target_pos"]           = self._target_pos.tolist()
        info["distance_to_target"]   = curr_dist
        info["is_grasped"]           = is_grasped
        info["task_type"]            = self._current_task_type
        return obs, reward, terminated, truncated, info

    def get_segmentation(self):
        return self.env.get_segmentation()

    @property
    def _object_ids(self):
        return self.env._object_ids
    
    @property
    def _object_colors(self):
        return self.env._object_colors
    
    @property
    def _object_shapes(self):
        return self.env._object_shapes

    @property
    def _physics_client_id(self):
        return self.env._physics_client_id
'''

with open(r"rl/reward_shaping.py", "w") as f:
    f.write(reward_code)
print("✓ reward_shaping.py synced with notebook")
