import json
import os
import numpy as np
import gymnasium as gym
import joblib
import pandas as pd


class RewardShapingWrapper(gym.Wrapper):
    SUCCESS_THRESHOLD = 0.15

    # --- Issue #8: task-specific success thresholds ------------------------
    # Assumes the 8-dim explicit-gripper action space (7 joints + gripper
    # command) per the team's decision on #8 — release is policy-commanded,
    # not automatic, so PLACE's settle check protects against a premature/
    # accidental release rather than guarding against a tautological
    # auto-release trigger.
    PUSH_PULL_DISPLACEMENT_THRESHOLD = 0.25   # 25cm object displacement, correct direction
    CONTACT_DISTANCE_THRESHOLD       = 0.15   # ee-object distance counted as "made contact"
    SETTLE_VELOCITY_THRESHOLD        = 0.005  # m/step object movement below this = "at rest"
    SETTLE_STEPS_REQUIRED            = 10     # consecutive at-rest steps before PLACE can succeed
    ROBOT_BASE_XY = np.array([0.0, 0.0], dtype=np.float32)

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

        # --- Issue #8 per-episode tracking state ---
        self._initial_obj_xy       = np.zeros(2, dtype=np.float32)
        self._prev_obj_pos         = np.zeros(3, dtype=np.float32)
        self._made_contact         = False  # ee ever got within CONTACT_DISTANCE_THRESHOLD of target
        self._was_grasped_target   = False  # target has been grasped at least once this episode
        self._had_release_after_grasp = False  # grasped -> released transition has occurred
        self._settled_steps        = 0

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

        # --- Issue #8: reset per-episode tracking state ---
        self._initial_obj_xy = self._target_pos[:2].copy()
        self._prev_obj_pos   = self._target_pos.copy()
        self._made_contact = False
        self._was_grasped_target = False
        self._had_release_after_grasp = False
        self._settled_steps = 0

        info["target_pos"]    = self._target_pos.tolist()
        info["target_obj_id"] = self._target_obj_id
        info["task_type"]     = self._task_type
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        # --- Issue #8 fix: refresh target position from live sim state.
        # Previously self._target_pos was frozen at the value captured in
        # reset() and never updated here, which silently broke LIFT (its
        # height check compared self._target_pos[2] to itself, so
        # lift_amount was always 0) and would have broken PUSH/PULL/PLACE
        # the same way, since none of them can work off a stale position.
        obj_state = info.get("object_state", {})
        if self._target_obj_id is not None and self._target_obj_id in obj_state:
            self._target_pos = np.array(obj_state[self._target_obj_id]["pos"], dtype=np.float32)

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

        # --- Issue #8: shared per-step bookkeeping used by PUSH/PULL/PLACE ---
        if curr_dist < self.CONTACT_DISTANCE_THRESHOLD:
            self._made_contact = True

        if grasped_target:
            self._was_grasped_target = True
        if self._was_grasped_target and not grasped_target:
            # Sticky: once a grasp->release transition happens, stays true
            # for the rest of the episode so we can evaluate the settle.
            self._had_release_after_grasp = True

        obj_step_delta = float(np.linalg.norm(self._target_pos - self._prev_obj_pos))
        self._prev_obj_pos = self._target_pos.copy()
        if self._had_release_after_grasp:
            if obj_step_delta < self.SETTLE_VELOCITY_THRESHOLD:
                self._settled_steps += 1
            else:
                self._settled_steps = 0  # still moving/falling - reset the streak

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

        elif self._task_type in ("push", "pull"):
            # PUSH/PULL: success = 25cm+ displacement of the object from its
            # start position, in the correct radial direction relative to
            # the robot base (push = away from base, pull = toward base),
            # gated on the ee having actually made contact with the object
            # at some point (so incidental drift doesn't count). No grasp
            # dependency, per the issue spec.
            displacement_xy = self._target_pos[:2] - self._initial_obj_xy
            radial_dir = self._initial_obj_xy - self.ROBOT_BASE_XY
            radial_norm = np.linalg.norm(radial_dir)
            radial_unit = radial_dir / radial_norm if radial_norm > 1e-6 else np.array([1.0, 0.0], dtype=np.float32)
            radial_disp = float(np.dot(displacement_xy, radial_unit))

            if self._task_type == "push":
                directional_progress = radial_disp        # positive = away from base
            else:  # pull
                directional_progress = -radial_disp        # positive = toward base

            reward += max(0.0, directional_progress) * 2.0  # shaping toward the goal direction

            if self._made_contact and directional_progress >= self.PUSH_PULL_DISPLACEMENT_THRESHOLD:
                reward += 5.0
                success = True
                terminated = True

        elif self._task_type == "place":
            # PLACE: object must be grasped, then released (policy-commanded,
            # explicit gripper action), then settle for SETTLE_STEPS_REQUIRED
            # consecutive steps with negligible movement, and no longer be
            # grasped. Checking the settled outcome (not just the release
            # event) protects against a premature/accidental release while
            # the object is still mid-fall or rolling.
            # NOTE: this does not yet validate *where* it was placed (e.g.
            # "on the left side") -- the CSV's zone phrases aren't parsed
            # anywhere in the pipeline yet. Tracked as a follow-up.
            if self._had_release_after_grasp and not grasped_target:
                if self._settled_steps >= self.SETTLE_STEPS_REQUIRED:
                    reward += 5.0
                    success = True
                    terminated = True
            elif grasped_target:
                reward += 1.0  # partial credit for grasping en route to placing

        elif self._task_type in ("lower", "move"):
            # Not in scope for issue #8 (issue explicitly covers push/pull/
            # place/lift as "the 4 task types"). Left as the reach-proximity
            # placeholder; tracked as a likely follow-up issue.
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
        # Exposed for debugging/tests of the #8 logic
        info["had_release_after_grasp"] = self._had_release_after_grasp
        info["settled_steps"]           = self._settled_steps
        info["made_contact"]            = self._made_contact
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