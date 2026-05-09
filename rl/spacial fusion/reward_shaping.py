import numpy as np
import gymnasium as gym


class RewardShapingWrapper(gym.Wrapper):
    SUCCESS_THRESHOLD = 0.05

    COLOR_KEYWORDS = {
        'red': 'red', 'blue': 'blue', 'green': 'green', 'yellow': 'yellow',
    }
    SHAPE_KEYWORDS = {
        'box': 'box', 'cube': 'box', 'block': 'box',
        'sphere': 'sphere', 'ball': 'sphere',
        'cylinder': 'cylinder', 'can': 'cylinder',
    }

    def __init__(self, env):
        super().__init__(env)
        self._target_pos    = np.zeros(3, dtype=np.float32)
        self._target_obj_id = None
        self._target_color  = None
        self._target_shape  = None
        self._prev_dist     = 0.0

    def set_instruction(self, text: str):
        """Parse instruction to extract target color/shape for object selection."""
        text = text.lower()
        self._target_color = None
        self._target_shape = None
        for kw, col in self.COLOR_KEYWORDS.items():
            if kw in text:
                self._target_color = col
                break
        for kw, shp in self.SHAPE_KEYWORDS.items():
            if kw in text:
                self._target_shape = shp
                break

    def _find_best_object(self, obj_state: dict):
        if not obj_state:
            return None
        if self._target_color is None and self._target_shape is None:
            return list(obj_state.keys())[0]
        best_id, best_score = list(obj_state.keys())[0], -1
        for oid, st in obj_state.items():
            score = 0
            if self._target_color and st.get("color") == self._target_color:
                score += 10
            if self._target_shape and st.get("shape") == self._target_shape:
                score += 5
            if score > best_score:
                best_score = score
                best_id    = oid
        return best_id

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obj_state = info.get("object_state", {})

        self._target_obj_id = self._find_best_object(obj_state)
        if self._target_obj_id and self._target_obj_id in obj_state:
            self._target_pos = np.array(obj_state[self._target_obj_id]["pos"], dtype=np.float32)
        else:
            self._target_pos = np.array([0.5, 0.0, 0.42], dtype=np.float32)

        ee_pos = np.array(info.get("ee_position", [0.0, 0.0, 0.0]), dtype=np.float32)
        self._prev_dist = float(np.linalg.norm(ee_pos - self._target_pos))
        info["target_pos"]    = self._target_pos.tolist()
        info["target_obj_id"] = self._target_obj_id
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        ee_pos    = np.array(info.get("ee_position", [0.0, 0.0, 0.0]), dtype=np.float32)
        curr_dist = float(np.linalg.norm(ee_pos - self._target_pos))
        reward    = (self._prev_dist - curr_dist) - 0.001
        self._prev_dist = curr_dist
        if curr_dist < self.SUCCESS_THRESHOLD:
            reward    += 1.0
            terminated = True
        info["target_pos"]         = self._target_pos.tolist()
        info["target_obj_id"]      = self._target_obj_id
        info["distance_to_target"] = curr_dist
        info["is_success"]         = curr_dist < self.SUCCESS_THRESHOLD
        return obs, reward, terminated, truncated, info

    # Forward PyBullet access through to KukaEnv
    def get_segmentation(self):
        return self.env.get_segmentation()

    @property
    def _object_ids(self):
        return self.env._object_ids

    @property
    def _physics_client_id(self):
        return self.env._physics_client_id