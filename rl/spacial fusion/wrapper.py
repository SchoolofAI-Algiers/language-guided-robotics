
import os
import sys
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import torch

sys.path.insert(0, "/kaggle/working")

from rl.beta_vision import beta_features, DEVICE


class BetaLanguageConditionedWrapper(gym.ObservationWrapper):
    """
    Strategy Beta wrapper.
    Transforms KukaEnv observations into:
        vision: (512,)  — 4-channel ResNet18 features (RGB + instance mask)
        nlp:    (384,)  — instruction embedding from all-MiniLM-L6-v2
    
    Key difference from Alpha: vision features include spatial bbox info
    baked INTO the CNN input — no separate detection vector the agent can exploit.
    """

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Dict({
            "vision": spaces.Box(low=-np.inf, high=np.inf, shape=(521,), dtype=np.float32),
            "nlp":    spaces.Box(low=-1.0,    high=1.0,    shape=(384,), dtype=np.float32),
        })
        # 521 = 512 (4-channel ResNet) + 9 (physics state)

        # Load NLP embeddings
        nlp_path = "/kaggle/working/nlp"
        npy_path = os.path.join(nlp_path, "embeddings.npy")
        csv_path = os.path.join(nlp_path, "nlp_instructions.csv")

        if os.path.exists(npy_path):
            self.embeddings = np.load(npy_path).astype(np.float32)
            self.instructions_df = pd.read_csv(csv_path) if os.path.exists(csv_path) else None
            print(f"[Beta Wrapper] Loaded {len(self.embeddings)} NLP embeddings")
        else:
            print("[Beta Wrapper] WARNING: NLP embeddings not found, using random")
            self.embeddings = np.random.uniform(-1, 1, size=(350, 384)).astype(np.float32)

        self.num_instructions  = len(self.embeddings)
        self.current_embedding = np.zeros(384, dtype=np.float32)
        self._use_physics_dropout = True
        self._dropout_rate = 0.30
        self._dropout_active = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Sample random instruction for this episode
        idx = np.random.randint(0, self.num_instructions)
        self.current_embedding = self.embeddings[idx]

        if self.instructions_df is not None:
            info["current_instruction"] = self.instructions_df.iloc[idx]["instruction"]

        # Physics dropout — sample once per episode, fixed for all steps
        self._dropout_active = self._use_physics_dropout and (np.random.random() < self._dropout_rate)

        return self.observation(obs, info), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs, info), reward, terminated, truncated, info

    def observation(self, obs, info=None):
        # Get frame + segmentation map together
        frame, seg = self.env.get_segmentation()
        object_ids = self.env._object_ids

        # Build 4-channel ResNet features (spatial fusion)
        vis_feat = beta_features(frame, seg, object_ids)       # (512,)
        vis_feat_np = vis_feat.cpu().numpy()

        # Physics state — [x,y,z,vx,vy,vz,roll,pitch,yaw] of first object
        physics = np.zeros(9, dtype=np.float32)
        if info and not self._dropout_active:
            obj_state = info.get("object_state", {})
            if obj_state:
                first_obj_id = list(obj_state.keys())[0]
                pos = obj_state[first_obj_id]["pos"]
                # Get velocity and orientation from PyBullet directly
                import pybullet as p
                vel, ang = p.getBaseVelocity(first_obj_id, physicsClientId=self.env._physics_client_id)
                orn = p.getBasePositionAndOrientation(first_obj_id, physicsClientId=self.env._physics_client_id)[1]
                euler = p.getEulerFromQuaternion(orn)
                raw = np.array(list(pos) + list(vel) + list(euler), dtype=np.float32)
                physics = raw / (np.abs(raw).max() + 1e-6)

        vision_tensor = np.concatenate([vis_feat_np, physics], axis=0)  # (521,)

        return {
            "vision": vision_tensor,
            "nlp":    self.current_embedding,
        }
