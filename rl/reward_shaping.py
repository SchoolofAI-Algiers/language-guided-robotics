import json
import os
import numpy as np
import gymnasium as gym
import joblib
import pandas as pd



class RewardShapingWrapper(gym.Wrapper):
    SUCCESS_THRESHOLD = 0.15

    COLOR_KEYWORDS = {
        'red': 'red', 'blue': 'blue', 'green': 'green', 'yellow': 'yellow',
    }
    SHAPE_KEYWORDS = {
        'box': 'box', 'cube': 'box', 'block': 'box',
        'sphere': 'sphere', 'ball': 'sphere',
        'cylinder': 'cylinder', 'can': 'cylinder',
    }

    _BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spacial fusion")
    _CLASSIFIER_PATH = os.path.join(_BASE_DIR, "task_classifier.joblib")
    _LABELS_PATH = os.path.join(_BASE_DIR, "task_classifier_labels.json")
    _CSV_PATH = os.path.join(_BASE_DIR, "nlp_instructions.csv")
    _EMBEDDINGS_PATH = os.path.join(_BASE_DIR, "embeddings.npy")

    _classifier = None
    _known_instruction_to_type = None
    _sentence_encoder = None

    @classmethod
    def _load_classifier_resources(cls):
        """Lazily load the SVM classifier and known-instruction lookup table.
        Loaded once at class level, shared across all wrapper instances."""
        if cls._classifier is not None:
            return
        cls._classifier = joblib.load(cls._CLASSIFIER_PATH)
        df = pd.read_csv(cls._CSV_PATH)
        cls._known_instruction_to_type = dict(zip(df["instruction"], df["type"]))

    def __init__(self, env):
        super().__init__(env)
        self._target_pos    = np.zeros(3, dtype=np.float32)
        self._target_obj_id = None
        self._target_color  = None
        self._target_shape  = None
        self._prev_dist     = 0.0
        self._task_type     = "reach"  # Default task type
        self._task_start_height = 0.0  # For tracking lift progress

    def set_instruction(self, text: str, embedding: np.ndarray = None):
        """Parse instruction to extract target color/shape and task type.

        Task type resolution order:
        1. Exact match against the known 340-instruction dataset -> use its
           ground-truth `type` column directly (100% reliable).
        2. Otherwise, if an embedding is provided, use the trained SVM
           classifier (rl/train_task_classifier.py) -- ~94.7% cross-val
           accuracy, chosen over nearest-neighbor lookup because the
           embedding space's silhouette score (~0.05) and frequent
           cross-class collisions make nearest-neighbor unreliable for
           task-type inference (see notebook.ipynb, Cell 9).
        3. Otherwise (no embedding available, e.g. some test paths), fall
           back to keyword matching as a last resort.
        """
        self._load_classifier_resources()

        original_text = text
        text = text.lower()
        self._target_color = None
        self._target_shape = None

        known_type = self._known_instruction_to_type.get(original_text) or \
            self._known_instruction_to_type.get(text)

        if known_type is not None:
            self._task_type = known_type
        elif embedding is not None:
            pred = self._classifier.predict(embedding.reshape(1, -1))
            self._task_type = str(pred[0])
        else:
            # Keyword fallback (kept for callers that don't pass an embedding).
            # Order matters: "lower" must be checked before "place" since
            # phrases like "set the box down" would otherwise match "set"
            # (place) before "down" (lower) gets a chance.
            if any(kw in text for kw in ["pick", "grasp", "grab", "get", "take"]):
                self._task_type = "pick"
            elif any(kw in text for kw in ["lift", "raise", "hoist"]):
                self._task_type = "lift"
            elif any(kw in text for kw in ["lower", "descend", "down"]):
                self._task_type = "lower"
            elif any(kw in text for kw in ["place", "put", "drop", "set"]):
                self._task_type = "place"
            elif "push" in text:
                self._task_type = "push"
            elif any(kw in text for kw in ["pull", "drag", "draw"]):
                self._task_type = "pull"
            elif any(kw in text for kw in ["approach", "go to", "go near", "head to", "head toward", "navigate", "move"]):
                self._task_type = "move"
            else:
                self._task_type = "reach"

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
            self._task_start_height = self._target_pos[2]  # Track starting height of object
        else:
            self._target_pos = np.array([0.5, 0.0, 0.42], dtype=np.float32)
            self._task_start_height = 0.42

        ee_pos = np.array(info.get("ee_position", [0.0, 0.0, 0.0]), dtype=np.float32)
        self._prev_dist = float(np.linalg.norm(ee_pos - self._target_pos))
        info["target_pos"]    = self._target_pos.tolist()
        info["target_obj_id"] = self._target_obj_id
        info["task_type"]     = self._task_type
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        ee_pos    = np.array(info.get("ee_position", [0.0, 0.0, 0.0]), dtype=np.float32)
        curr_dist = float(np.linalg.norm(ee_pos - self._target_pos))
        
        # Reward for moving closer (reduced scale to prevent overshooting)
        reward = (self._prev_dist - curr_dist) * 0.5 - 0.001
        self._prev_dist = curr_dist
        
        # MAJOR reward for grasping target object
        grasped_id = info.get("grasped_object")
        grasped_target = (grasped_id == self._target_obj_id and grasped_id is not None)
        if grasped_target:
            reward += 2.0  # Strong signal: you grabbed the right object!
        elif grasped_id is not None:
            reward += 0.5  # Still good to try grasping
        
        # Task-specific success conditions
        success = False
        if self._task_type == "reach":
            # REACH: just get close (0.15m)
            if curr_dist < self.SUCCESS_THRESHOLD:
                reward += 1.0
                success = True
                terminated = True
        
        elif self._task_type == "pick":
            # PICK: approach + grasp target object
            if grasped_target:
                reward += 5.0  # Huge reward for picking!
                success = True
                terminated = True
        
        elif self._task_type == "lift":
            # LIFT: grasp + lift object above starting height (5cm minimum)
            if grasped_target:
                obj_height = self._target_pos[2]
                lift_amount = obj_height - self._task_start_height
                if lift_amount > 0.05:  # Lifted 5cm+ above start
                    reward += 10.0  # Huge reward for lifting!
                    success = True
                    terminated = True
                else:
                    reward += 3.0  # Partial reward for grasping
            else:
                reward += (self._prev_dist - curr_dist) * 0.5  # Still reward approaching
        
        elif self._task_type in ["place", "push", "pull", "lower", "move"]:
            # These are more complex, use reach threshold for now
            if curr_dist < self.SUCCESS_THRESHOLD:
                reward += 1.0
                success = True
                terminated = True
        
        info["target_pos"]         = self._target_pos.tolist()
        info["target_obj_id"]      = self._target_obj_id
        info["task_type"]          = self._task_type
        info["distance_to_target"] = curr_dist
        info["is_success"]         = success
        info["grasped_target"]     = grasped_target
        return obs, reward, terminated, truncated, info

    # Forward PyBullet access through to KukaEnv
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
