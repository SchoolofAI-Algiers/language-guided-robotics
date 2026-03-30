import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
import os

# Add project root to sys.path to allow importing from vision module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))
try:
    from vision.vision_pipeline import resnet_features
except ImportError:
    print("[Warning] Could not import vision_pipeline. Visual modes will fail.")


class MultimodalObservationWrapper(gym.ObservationWrapper):
    """
    Gymnasium ObservationWrapper that handles Vision ResNet18 extraction
    and Multimodal state concatenation. 
    Keeps the core KukaEnv pure and physics-focused.
    """
    def __init__(self, env, obs_mode="visual_joints_statepybullet"):
        super().__init__(env)
        self.obs_mode = obs_mode
        self._setup_spaces()

    def _setup_spaces(self):
        # We need the underlying spaces to bound our tensors
        base_state_space = self.env.observation_space["state"]
        state_low = base_state_space.low
        state_high = base_state_space.high

        base_obj_space = self.env.observation_space["object_state"]
        obj_low = base_obj_space.low
        obj_high = base_obj_space.high

        feat_low = np.full(512, -1.0, dtype=np.float32)
        feat_high = np.full(512, 1.0, dtype=np.float32)

        if self.obs_mode == "state":
            # Pure physics state (21 dimensions)
            self.observation_space = spaces.Box(
                low=state_low, high=state_high, shape=(21,), dtype=np.float32
            )
        elif self.obs_mode == "visual_only":
            # Only Visual features (512 dims)
            self.observation_space = spaces.Box(
                low=feat_low, high=feat_high, shape=(512,), dtype=np.float32
            )
        elif self.obs_mode == "visual_joints" or self.obs_mode == "visual_state" or self.obs_mode == "visual":
            obs_low = np.concatenate([feat_low, state_low])
            obs_high = np.concatenate([feat_high, state_high])
            self.observation_space = spaces.Box(
                low=obs_low, high=obs_high, shape=(533,), dtype=np.float32
            )
        elif self.obs_mode == "visual_statepybullet":
            obs_low = np.concatenate([feat_low, obj_low])
            obs_high = np.concatenate([feat_high, obj_high])
            self.observation_space = spaces.Box(
                low=obs_low, high=obs_high, shape=(521,), dtype=np.float32
            )
        elif self.obs_mode == "visual_joints_statepybullet":
            obs_low = np.concatenate([feat_low, state_low, obj_low])
            obs_high = np.concatenate([feat_high, state_high, obj_high])
            self.observation_space = spaces.Box(
                low=obs_low, high=obs_high, shape=(542,), dtype=np.float32
            )
        elif self.obs_mode == "pixels":
            # Just pass through the structured dict (or prune object state if needed, but we can pass it entirely)
            import copy
            dict_space = copy.deepcopy(self.env.observation_space)
            self.observation_space = dict_space
        else:
            raise ValueError(f"Unknown obs_mode: {self.obs_mode}")

    def observation(self, obs):
        """
        Takes the raw Dict observation from KukaEnv and extracts the
        concatenated arrays based on `obs_mode`.
        """
        physics_state = obs["state"]
        object_state = obs["object_state"]
        img = obs["pixels"]

        if self.obs_mode == "state":
            return physics_state

        elif self.obs_mode == "pixels":
            return obs

        # For all visual tensor logic:
        import torch
        with torch.no_grad():
            vision_tensor = resnet_features([img]).cpu().numpy()[0]
        
        if self.obs_mode == "visual_only":
            return vision_tensor
        elif self.obs_mode == "visual_joints" or self.obs_mode == "visual_state" or self.obs_mode == "visual":
            return np.concatenate([vision_tensor, physics_state])
        elif self.obs_mode == "visual_statepybullet":
            return np.concatenate([vision_tensor, object_state])
        elif self.obs_mode == "visual_joints_statepybullet":
            return np.concatenate([vision_tensor, physics_state, object_state])
        else:
            return obs
