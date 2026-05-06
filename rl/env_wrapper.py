import os
import sys
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import torch

# Ensure the project root is in path, so imports from 'vision' or 'robotics' work smoothly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Vision Pipeline safely as per requirement to align with Phase 1 deliveries
try:
    from vision.vision_pipeline import resnet_features
    VISION_AVAILABLE = True
except ImportError as e:
    VISION_AVAILABLE = False
    print(f"[RL Wrapper] WARNING: Could not import Vision pipeline. Details: {e}")

class LanguageConditionedWrapper(gym.ObservationWrapper):
    """
    Dynamically wraps the Robotics `KukaEnv` to export Phase 3 compliant formats.
    Transforms raw PyBullet dicts `{"pixels", "state", "object_state"}` into 
    the RL-required shape `{"vision": (521,), "nlp": (384,)}`.
    
    Dynamically loads NLP datasets (embeddings.npy and nlp_instructions.csv).
    """

    def __init__(self, env):
        super().__init__(env)
        
        # Override the original PyBullet observation space with the Multi-Modal expected layout
        self.observation_space = spaces.Dict({
            'vision': spaces.Box(low=-np.inf, high=np.inf, shape=(521,), dtype=np.float32),
            'nlp': spaces.Box(low=-1.0, high=1.0, shape=(384,), dtype=np.float32)
        })

        self.nlp_base_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', 'nlp'
        ))
        
        # Load NLP Embeddings Matrix and metadata
        self._load_nlp_dataset()
        
        self.current_instruction_idx = 0
        self.current_embedding = np.zeros(384, dtype=np.float32)
        self.physics_dropout_rate = 0.30

    def _load_nlp_dataset(self):
        npy_path = os.path.join(self.nlp_base_path, 'embeddings.npy')
        csv_path = os.path.join(self.nlp_base_path, 'nlp_instructions.csv')
        
        if os.path.exists(npy_path) and os.path.exists(csv_path):
            self.embeddings = np.load(npy_path).astype(np.float32)
            self.instructions_df = pd.read_csv(csv_path)
            self.num_instructions = len(self.embeddings)
            print(f"[RL Wrapper] Loaded {self.num_instructions} NLP instructions natively from dataset.")
        else:
            print("[RL Wrapper] WARNING: NLP dataset files not found. Using mocked embeddings.")
            self.embeddings = np.random.uniform(-1, 1, size=(125, 384)).astype(np.float32)
            self.num_instructions = 125

    def reset(self, **kwargs):
        """
        Intervene on environment reset to sample a new randomized instruction goal.
        """
        # We call the underlying environment reset
        obs, info = self.env.reset(**kwargs)
        
        # Sample a random language conditioning target from the loaded embeddings
        self.current_instruction_idx = np.random.randint(0, self.num_instructions)
        self.current_embedding = self.embeddings[self.current_instruction_idx]
        self._dropout_physics = np.random.random() < self.physics_dropout_rate
        
        if hasattr(self, 'instructions_df'):
            inst_text = self.instructions_df.iloc[self.current_instruction_idx]['instruction']
            info['current_instruction'] = inst_text
        
        # Format the rest via our custom observation() hook
        return self.observation(obs), info

    def observation(self, obs):
        """
        Intercept the step() raw dict output and transform it natively into {"vision", "nlp"}.
        """
        raw_pixels = obs['pixels']           # shape: (H, W, 3) expected from PyBullet
        object_state = obs['object_state']   # shape: (9,) provided as mock zero or true query by phase 2
        
        vision_tensor = np.zeros(521, dtype=np.float32)
        
        if VISION_AVAILABLE:
            # The Vision module takes a list of uint8 crops. Since phase 2 isn't ready with bounding boxes,
            # we provide the entire uncropped screen image. The vision resnet internally resizes this to 64x64.
            # This aligns our wrapper structurally with the M3-M4 deliverables.
            with torch.no_grad():
                features = resnet_features([raw_pixels], crop_size=64) # Returns (1, 512) tensor
                feat_np = features.cpu().numpy().flatten()
            
            # Form final: Concatenate CNN feature (512,) + Object State (9,)
            vision_tensor = np.concatenate([feat_np, object_state], axis=0).astype(np.float32)
        else:
            # Fallback random initialization matching size requirements for PPO network sanity
            vision_tensor = np.concatenate([
                np.random.randn(512), 
                object_state
            ], axis=0).astype(np.float32)

        if getattr(self, '_dropout_physics', False):
            vision_tensor[512:521] = 0.0
            
        return {
            'vision': vision_tensor,
            'nlp': self.current_embedding
        }
