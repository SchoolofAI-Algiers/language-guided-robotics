import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class LanguageConditionedFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom Feature Extractor for language-conditioned policy.
    
    This extractor assumes the observation space is a `gym.spaces.Dict` with at least
    two keys:
      - 'vision': The visual/state features of the environment (e.g., shape (521,))
      - 'nlp': The instruction embeddings (e.g., shape (384,))
      
    It concatenates both inputs to form a joint representation.
    """
    
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 512):
        """
        :param observation_space: The Dict observation space.
        :param features_dim: The dimension of the final extracted features vector.
        """
        # Resolve observation keys — support both the canonical names and legacy variants.
        self.vision_key = 'vision' if 'vision' in observation_space.spaces else 'visual_obs'
        self.nlp_key   = 'nlp'    if 'nlp'    in observation_space.spaces else 'nlp_embedding'

        if self.vision_key not in observation_space.spaces:
            raise KeyError(f"Expected 'vision' or 'visual_obs' key in observation space, got: {list(observation_space.spaces.keys())}")
        if self.nlp_key not in observation_space.spaces:
            raise KeyError(f"Expected 'nlp' or 'nlp_embedding' key in observation space, got: {list(observation_space.spaces.keys())}")

        vision_dim = observation_space.spaces[self.vision_key].shape[0]   # 521
        nlp_dim    = observation_space.spaces[self.nlp_key].shape[0]       # 384
        total_concat_dim = vision_dim + nlp_dim                            # 905
        
        # Initialize the base class with the appropriate output dimension
        super().__init__(observation_space, features_dim)
        
        # Two-branch architecture: each modality gets its own projection head,
        # then we fuse and pass through a shared layer.  LayerNorm stabilises
        # training when input magnitudes differ between vision and NLP.
        hidden = features_dim  # reuse features_dim as hidden width

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

    def forward(self, observations: dict) -> torch.Tensor:
        """
        Forward pass to process the Dictionary of observations.
        
        :param observations: A dictionary containing torch Tensors for 'vision' and 'nlp'.
        :return: A joint feature tensor of shape (batch_size, features_dim).
        """
        vision_obs = observations[self.vision_key]   # (batch, 521)
        nlp_obs    = observations[self.nlp_key]        # (batch, 384)

        # Process each modality through its own branch, then fuse
        v = self.vision_branch(vision_obs)              # (batch, hidden)
        n = self.nlp_branch(nlp_obs)                    # (batch, hidden)
        joint = torch.cat([v, n], dim=1)                # (batch, hidden*2)
        return self.fusion(joint)                        # (batch, features_dim)
