# test_gamma_dry_run.py  — run from project root
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from rl.gamma_feature_extractor import GammaFeatureExtractor

# Simulate the observation space Gamma produces
obs_space = spaces.Dict({
    'vision': spaces.Box(low=-np.inf, high=np.inf, shape=(768,), dtype=np.float32),
    'nlp':    spaces.Box(low=-1.0,    high=1.0,    shape=(768,), dtype=np.float32)
})

extractor = GammaFeatureExtractor(obs_space, features_dim=256)

# Simulate a batch of 4 observations (like PPO rollout)
dummy_obs = {
    'vision': torch.randn(4, 768),
    'nlp':    torch.randn(4, 768),
}

output = extractor(dummy_obs)
assert output.shape == (4, 256), f"wrong shape: {output.shape}"
print(f"✅ GammaFeatureExtractor output: {output.shape}")
print(f"✅ vision branch: Linear(768→256)")
print(f"✅ nlp branch:    Linear(768→256)")
print(f"✅ fusion output: (4, 256)")
print("\nCheckpoint 1 DONE — hand gamma_feature_extractor.py and gamma_env_wrapper.py to Selma for training.")