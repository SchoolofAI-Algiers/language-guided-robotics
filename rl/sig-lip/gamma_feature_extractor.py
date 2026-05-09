import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GammaFeatureExtractor(BaseFeaturesExtractor):
    """
    Gamma version of LanguageConditionedFeatureExtractor.
    Both vision and nlp branches now take (768,) SigLIP features.
    Architecture is otherwise identical to Chouaib's — same fusion layer,
    same hidden_dim, so PPO hyperparameters don't need to change.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        vision_key = 'vision'
        nlp_key    = 'nlp'

        vision_dim = observation_space.spaces[vision_key].shape[0]  # 768
        nlp_dim    = observation_space.spaces[nlp_key].shape[0]     # 768

        super().__init__(observation_space, features_dim)

        hidden = features_dim

        self.vision_key = vision_key
        self.nlp_key    = nlp_key

        self.vision_branch = nn.Sequential(
            nn.Linear(vision_dim, hidden),   # 768 → 256
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.nlp_branch = nn.Sequential(
            nn.Linear(nlp_dim, hidden),      # 768 → 256
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 2, features_dim),  # 512 → 256
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        v = self.vision_branch(observations[self.vision_key])  # (batch, 256)
        n = self.nlp_branch(observations[self.nlp_key])        # (batch, 256)
        return self.fusion(torch.cat([v, n], dim=1))           # (batch, 256)