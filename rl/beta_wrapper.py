import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import pybullet as p

from rl.beta_vision import beta_features, DEVICE

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_NLP_DIR  = os.path.join(_BASE_DIR, "spacial fusion")


class BetaLanguageConditionedWrapper(gym.ObservationWrapper):
    """
    Strategy Beta wrapper — matches training exactly.
    Stack: KukaEnv -> BetaLanguageConditionedWrapper -> RewardShapingWrapper
    observation:
        vision: (521,) — 4-channel ResNet18 (512,) + physics state (9,)
        nlp:    (384,) — sentence-transformer embedding
    """

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Dict({
            "vision": spaces.Box(low=-np.inf, high=np.inf, shape=(521,), dtype=np.float32),
            "nlp":    spaces.Box(low=-1.0,    high=1.0,    shape=(384,), dtype=np.float32),
        })

        npy_path = os.path.join(_NLP_DIR, "embeddings.npy")
        csv_path = os.path.join(_NLP_DIR, "nlp_instructions.csv")

        if os.path.exists(npy_path):
            self.embeddings = np.load(npy_path).astype(np.float32)
            self.instructions_df = pd.read_csv(csv_path) if os.path.exists(csv_path) else None
            print(f"[Beta Wrapper] Loaded {len(self.embeddings)} NLP embeddings")
        else:
            print(f"[Beta Wrapper] WARNING: embeddings not found at {npy_path}, using random")
            self.embeddings = np.random.uniform(-1, 1, size=(340, 384)).astype(np.float32)
            self.instructions_df = None

        self.num_instructions  = len(self.embeddings)
        self.current_embedding = np.zeros(384, dtype=np.float32)
        self._use_physics_dropout = True
        self._dropout_rate = 0.30
        self._dropout_active = False
        self._inference_mode = False  # Disable dropout during inference

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        idx = np.random.randint(0, self.num_instructions)
        self.current_embedding = self.embeddings[idx]
        if self.instructions_df is not None:
            info["current_instruction"] = self.instructions_df.iloc[idx]["instruction"]
        self._dropout_active = self._use_physics_dropout and (np.random.random() < self._dropout_rate)
        return self.observation(obs, info), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs, info), reward, terminated, truncated, info

    def set_embedding(self, embedding: np.ndarray):
        self.current_embedding = embedding.astype(np.float32)

    def observation(self, obs, info=None):
        frame, seg = self.env.get_segmentation()
        object_ids = self.env._object_ids

        vis_feat    = beta_features(frame, seg, object_ids)
        vis_feat_np = vis_feat.cpu().numpy()

        physics = np.zeros(9, dtype=np.float32)
        # Skip physics dropout during inference
        if info and (not self._dropout_active or self._inference_mode):
            obj_state = info.get("object_state", {})
            if obj_state:
                first_obj_id = list(obj_state.keys())[0]
                pos = obj_state[first_obj_id]["pos"]
                vel, ang = p.getBaseVelocity(first_obj_id,
                                             physicsClientId=self.env._physics_client_id)
                orn = p.getBasePositionAndOrientation(
                    first_obj_id, physicsClientId=self.env._physics_client_id)[1]
                euler = p.getEulerFromQuaternion(orn)
                raw = np.array(list(pos) + list(vel) + list(euler), dtype=np.float32)
                physics = raw / (np.abs(raw).max() + 1e-6)

        vision_tensor = np.concatenate([vis_feat_np, physics], axis=0)  # (521,)
        return {"vision": vision_tensor, "nlp": self.current_embedding}

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
