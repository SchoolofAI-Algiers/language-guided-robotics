import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import sys
import os

# Ensure vision pipeline is found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from vision.vision_pipeline import resnet_features
except ModuleNotFoundError:
    print("[Core] Warning: vision.vision_pipeline not found. Ensure vision module exists.")
    resnet_features = None

class MultimodalObservationWrapper(gym.ObservationWrapper):
    """
    Root-level Wrapper that fuses pure PyBullet physics (Robotics) with ResNet (Vision).
    Follows late-fusion decoupling for RL.
    """
    def __init__(self, env, obs_mode="visual_joints_statepybullet"):
        super().__init__(env)
        self.obs_mode = obs_mode
        
        # We assume the base environment is 'state' (21-dim) or 'pixels' (Box space of RGB image)
        # We will redefine the observation space to 542 dimensions.
        feat_low = np.full(512, -1.0, dtype=np.float32)
        feat_high = np.full(512, 1.0, dtype=np.float32)

        state_low = np.full(21, -1.0, dtype=np.float32) # Arbitrary bounded proxy
        state_high = np.full(21, 1.0, dtype=np.float32) 
        
        obj_low = np.full(9, -10.0, dtype=np.float32)
        obj_high = np.full(9, 10.0, dtype=np.float32)

        if hasattr(env, 'observation_space'):
            if isinstance(env.observation_space, spaces.Dict) and 'state' in env.observation_space.spaces:
                state_low = env.observation_space['state'].low
                state_high = env.observation_space['state'].high
                obj_low = env.observation_space['object_state'].low
                obj_high = env.observation_space['object_state'].high
            elif isinstance(env.observation_space, spaces.Box) and env.observation_space.shape == (21,):
                state_low = env.observation_space.low
                state_high = env.observation_space.high
        
        # Define observation space based on mode
        if self.obs_mode == "state":
            self.observation_space = spaces.Box(low=state_low, high=state_high, dtype=np.float32)
        elif self.obs_mode == "pixels":
            # Will just return the unwrapped env's dictionary
            self.observation_space = env.observation_space
        elif self.obs_mode == "visual_only":
            self.observation_space = spaces.Box(low=feat_low, high=feat_high, dtype=np.float32)
        elif self.obs_mode == "visual_joints":
            # 512 + 21 = 533
            self.observation_space = spaces.Box(
                low=np.concatenate([feat_low, state_low]),
                high=np.concatenate([feat_high, state_high]),
                dtype=np.float32
            )
        elif self.obs_mode == "visual_statepybullet":
            # 512 + 9 = 521
            self.observation_space = spaces.Box(
                low=np.concatenate([feat_low, obj_low]),
                high=np.concatenate([feat_high, obj_high]),
                dtype=np.float32
            )
        elif self.obs_mode == "joints_statepybullet":
            # 21 + 9 = 30
            self.observation_space = spaces.Box(
                low=np.concatenate([state_low, obj_low]),
                high=np.concatenate([state_high, obj_high]),
                dtype=np.float32
            )
        elif self.obs_mode == "visual_joints_statepybullet":
            # 512 + 21 + 9 = 542
            self.observation_space = spaces.Box(
                low=np.concatenate([feat_low, state_low, obj_low]),
                high=np.concatenate([feat_high, state_high, obj_high]),
                dtype=np.float32
            )
        else:
            raise ValueError(f"Unknown obs_mode: {self.obs_mode}")

    def observation(self, observation):
        """
        Extracts features dynamically from the environment instance.
        """
        if self.obs_mode == "pixels":
            return observation
            
        # 1. Physics State (21-dim)
        if isinstance(observation, dict) and "state" in observation:
            physics_state = observation["state"]
        elif isinstance(observation, np.ndarray) and observation.shape == (21,):
            physics_state = observation
        else:
            physics_state = self.env.unwrapped._get_observation()
            if isinstance(physics_state, dict):
                physics_state = physics_state.get('state', np.zeros(21))

        if self.obs_mode == "state":
            return physics_state
            
        # 2. Extract Visuals directly from the camera
        with torch.no_grad():
            img = self.env.unwrapped._get_camera_image()
            if resnet_features:
                vision_tensor = resnet_features([img]).cpu().numpy()[0]
            else:
                vision_tensor = np.zeros(512, dtype=np.float32)

        if self.obs_mode == "visual_only":
            return vision_tensor
            
        if self.obs_mode == "visual_joints":
            return np.concatenate([vision_tensor, physics_state])

        # 3. Object Ground Truth Target [x,y,z, vx,vy,vz, r,p,y]
        if isinstance(observation, dict) and "object_state" in observation:
            object_state = observation["object_state"]
        else:
            obs_dict = self.env.unwrapped._get_observation()
            object_state = obs_dict.get('object_state', np.zeros(9, dtype=np.float32)) if isinstance(obs_dict, dict) else np.zeros(9, dtype=np.float32)

        if self.obs_mode == "visual_statepybullet":
            return np.concatenate([vision_tensor, object_state])

        if self.obs_mode == "joints_statepybullet":
            return np.concatenate([physics_state, object_state])

        # 4. Integrate all! (visual_joints_statepybullet)
        return np.concatenate([vision_tensor, physics_state, object_state])
