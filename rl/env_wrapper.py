import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import torch

from rl.common.wrapper import KukaWrapper
from vision.vision_pipeline import resnet_features

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_NLP_DIR  = os.path.join(_BASE_DIR, "spacial fusion")


class LanguageConditionedWrapper(KukaWrapper, gym.ObservationWrapper):
    """
    Strategy Alpha wrapper.
    Transforms raw PyBullet output into {"vision": (521,), "nlp": (384,)}.

    Updated during #9: originally loaded from a dataset folder
    ("Elbatoul-NLP-W1-instruction-embeddings", 125 instructions) that no
    longer exists in the repo. Migrated to the same current, real dataset
    Beta uses (rl/spacial fusion/, 340 instructions, 384-dim embeddings) --
    Alpha's observation format is otherwise identical to Beta's.
    """

    def __init__(self, env):
        super().__init__(env)
        self._init_common()  # sets up _inference_mode

        self.observation_space = spaces.Dict({
            'vision': spaces.Box(low=-np.inf, high=np.inf, shape=(521,), dtype=np.float32),
            'nlp':    spaces.Box(low=-1.0,    high=1.0,    shape=(384,), dtype=np.float32)
        })

        npy_path = os.path.join(_NLP_DIR, "embeddings.npy")
        csv_path = os.path.join(_NLP_DIR, "nlp_instructions.csv")

        if os.path.exists(npy_path):
            self.embeddings = np.load(npy_path).astype(np.float32)
            self.instructions_df = pd.read_csv(csv_path) if os.path.exists(csv_path) else None
            print(f"[Alpha Wrapper] Loaded {len(self.embeddings)} NLP embeddings")
        else:
            print(f"[Alpha Wrapper] WARNING: embeddings not found at {npy_path}, using random")
            self.embeddings = np.random.uniform(-1, 1, size=(340, 384)).astype(np.float32)
            self.instructions_df = None

        self.num_instructions = len(self.embeddings)
        self.current_embedding = np.zeros(384, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        idx = np.random.randint(0, self.num_instructions)
        self.current_embedding = self.embeddings[idx]
        if self.instructions_df is not None:
            info['current_instruction'] = self.instructions_df.iloc[idx]['instruction']

        return self.observation(obs, info), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs, info), reward, terminated, truncated, info

    def observation(self, obs, info=None):
        frame, _ = self.env.get_segmentation()  # Alpha only needs the RGB frame, not the seg mask

        with torch.no_grad():
            features = resnet_features([frame], crop_size=64)
            feat_np = features.cpu().numpy().flatten()

        physics = np.zeros(9, dtype=np.float32)
        if info:
            object_state = info.get('object_state', {})
            if object_state:
                first_obj_id = list(object_state.keys())[0]
                physics[:3] = object_state[first_obj_id].get('pos', [0, 0, 0])

        vision_tensor = np.concatenate([feat_np, physics], axis=0).astype(np.float32)

        return {
            'vision': vision_tensor,
            'nlp':    self.current_embedding
        }