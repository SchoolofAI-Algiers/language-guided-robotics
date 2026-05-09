import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class BetaFeatureExtractor(BaseFeaturesExtractor):
    """
    Matches the checkpoint architecture (with LayerNorm).
    Vision input: (521,) = 4-channel ResNet18 (512,) + physics state (9,)
    NLP input:    (384,)
    Must match exactly what was used during training.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        vision_dim = observation_space.spaces["vision"].shape[0]  # 521
        nlp_dim    = observation_space.spaces["nlp"].shape[0]     # 384

        super().__init__(observation_space, features_dim)

        hidden = features_dim

        self.vision_branch = nn.Sequential(
            nn.Linear(vision_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.nlp_branch = nn.Sequential(
            nn.Linear(nlp_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 2, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        v = self.vision_branch(observations["vision"])
        n = self.nlp_branch(observations["nlp"])
        return self.fusion(torch.cat([v, n], dim=1))
