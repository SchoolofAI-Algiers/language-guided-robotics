# gamma_feature_extractor.py
# Gamma feature extractor — adapts (768,) SigLIP features for PPO
# Drop-in replacement for Alpha/Beta feature extractor
# RL team: plug this into your PPO wrapper, replace Linear(512→hidden) with this

import torch
import torch.nn as nn
import numpy as np


class GammaFeatureExtractor(nn.Module):
    """
    Takes {"vision": (768,), "nlp": (768,)} from gamma_pipeline.gamma_encode()
    and produces a fused feature vector for PPO's policy network.

    Hidden dim should match whatever Alpha/Beta used — ask RL team.
    Default 256 is a safe starting point.
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.vision_branch = nn.Linear(768, hidden_dim)
        self.nlp_branch    = nn.Linear(768, hidden_dim)
        # Fusion: same structure as Alpha/Beta so PPO wrapper needs zero changes
        self.fusion        = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        self.output_dim = hidden_dim

    def forward(self, obs: dict) -> torch.Tensor:
        """
        Args:
            obs: dict with keys "vision" and "nlp"
                 values: torch.Tensor (batch, 768) or numpy (768,)
        Returns:
            fused: torch.Tensor (batch, hidden_dim)
        """
        vision = obs["vision"]
        nlp    = obs["nlp"]

        # Handle numpy input (single step during rollout)
        if isinstance(vision, np.ndarray):
            vision = torch.from_numpy(vision).float()
        if isinstance(nlp, np.ndarray):
            nlp = torch.from_numpy(nlp).float()

        # Add batch dim if needed
        if vision.dim() == 1:
            vision = vision.unsqueeze(0)
        if nlp.dim() == 1:
            nlp = nlp.unsqueeze(0)

        v = self.vision_branch(vision)   # (batch, hidden_dim)
        t = self.nlp_branch(nlp)         # (batch, hidden_dim)

        fused = self.fusion(torch.cat([v, t], dim=-1))  # (batch, hidden_dim)
        return fused


class GammaRLWrapper:
    """
    Wraps gamma_pipeline + GammaFeatureExtractor into one object
    the RL training loop calls each step.

    Usage:
        wrapper = GammaRLWrapper(model, processor, device, embedding_cache)
        obs = wrapper.encode(instruction, image)
        # obs["vision"], obs["nlp"], obs["similarity"] ready for PPO
        features = wrapper.extract_features(obs)  # (1, hidden_dim) tensor for policy net
    """

    def __init__(self, siglip_model, processor, device,
                 embedding_cache: dict = None, hidden_dim: int = 256):
        from gamma_pipeline import gamma_encode
        self.gamma_encode    = gamma_encode
        self.model           = siglip_model
        self.processor       = processor
        self.device          = device
        self.embedding_cache = embedding_cache
        self.extractor       = GammaFeatureExtractor(hidden_dim=hidden_dim)
        self.extractor.eval()

    def encode(self, instruction: str, image) -> dict:
        """Run SigLIP → returns {"vision":(768,), "nlp":(768,), "similarity":float}"""
        return self.gamma_encode(
            instruction, image,
            self.model, self.processor, self.device,
            self.embedding_cache
        )

    def extract_features(self, obs: dict) -> torch.Tensor:
        """Run feature extractor → (1, hidden_dim) tensor for policy network."""
        with torch.no_grad():
            return self.extractor(obs)

    def similarity_bonus(self, obs: dict, scale: float = 0.1) -> float:
        """Optional reward bonus: reward += wrapper.similarity_bonus(obs)"""
        return scale * obs["similarity"]


# ─────────────────────────────────────────
# VERIFICATION — run this to confirm checkpoint 1
# ─────────────────────────────────────────

def verify_checkpoint_1():
    print("── CHECKPOINT 1: Feature extractor for (768,) inputs ──")

    extractor = GammaFeatureExtractor(hidden_dim=256)

    # Simulate what gamma_encode() returns
    dummy_obs = {
        "vision": np.random.randn(768).astype(np.float32),
        "nlp":    np.random.randn(768).astype(np.float32),
        "similarity": 0.42
    }

    output = extractor(dummy_obs)

    assert output.shape == (1, 256), f"wrong output shape: {output.shape}"
    print(f"  vision branch: Linear(768 → 256)  ✅")
    print(f"  nlp branch:    Linear(768 → 256)  ✅")
    print(f"  fusion output: {output.shape}      ✅")

    # Confirm it accepts batched input too
    batched_obs = {
        "vision": torch.randn(8, 768),
        "nlp":    torch.randn(8, 768),
    }
    batched_out = extractor(batched_obs)
    assert batched_out.shape == (8, 256), f"wrong batched shape: {batched_out.shape}"
    print(f"  batched input (8, 768) → {batched_out.shape}  ✅")

    # Confirm similarity bonus
    bonus = 0.1 * dummy_obs["similarity"]
    print(f"  similarity bonus (scale=0.1): {bonus:.4f}  ✅")

    print("\n✅ Checkpoint 1 done — hand gamma_feature_extractor.py to RL team")
    print("   They need to:")
    print("   - replace their Linear(512→hidden) with GammaFeatureExtractor(hidden_dim=?)")
    print("   - confirm hidden_dim matches their PPO network")
    print("   - wire GammaRLWrapper into the training loop")


if __name__ == "__main__":
    verify_checkpoint_1()