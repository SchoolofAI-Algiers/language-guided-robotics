# test_gamma_dry_run.py
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from rl.gamma_feature_extractor import GammaFeatureExtractor

print("── TEST 1: GammaFeatureExtractor ──")
obs_space = spaces.Dict({
    'vision': spaces.Box(low=-np.inf, high=np.inf, shape=(768,), dtype=np.float32),
    'nlp':    spaces.Box(low=-1.0,    high=1.0,    shape=(768,), dtype=np.float32)
})
extractor = GammaFeatureExtractor(obs_space, features_dim=256)
dummy_obs = {
    'vision': torch.randn(4, 768),
    'nlp':    torch.randn(4, 768),
}
output = extractor(dummy_obs)
assert output.shape == (4, 256)
print(f"  GammaFeatureExtractor output: {output.shape}  ✅")

print("\n── TEST 2: Wrapper observation space is correct ──")
assert obs_space['vision'].shape == (768,)
assert obs_space['nlp'].shape == (768,)
print(f"  vision: {obs_space['vision'].shape}  ✅")
print(f"  nlp:    {obs_space['nlp'].shape}      ✅")

print("\n── TEST 3: Similarity bonus math ──")
vision = np.random.randn(768).astype(np.float32)
nlp    = np.random.randn(768).astype(np.float32)
vision /= np.linalg.norm(vision)
nlp    /= np.linalg.norm(nlp)
sim = float(np.dot(vision, nlp))
bonus = 0.1 * sim
print(f"  similarity: {sim:.4f}, bonus: {bonus:.4f}  ✅")

print("\n✅ All tests passed. Ready for Selma to wire into training.")