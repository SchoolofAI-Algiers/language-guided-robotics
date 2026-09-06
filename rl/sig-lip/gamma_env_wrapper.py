import os
import sys
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rl.common.wrapper import KukaWrapper

try:
    from gamma_pipeline import load_siglip, encode_image
    SIGLIP_AVAILABLE = True
except ImportError as e:
    SIGLIP_AVAILABLE = False
    print(f"[Gamma Wrapper] WARNING: Could not import gamma_pipeline. Details: {e}")


class GammaLanguageConditionedWrapper(KukaWrapper, gym.Wrapper):
    """
    Gamma wrapper for KukaEnv.
    KukaEnv returns a flat (21,) obs vector + info dict.
    This wrapper calls env.render() to get the RGB frame,
    runs SigLIP on it, and returns {"vision": (768,), "nlp": (768,)}.
    Uses gym.Wrapper (not ObservationWrapper) because we need render() separately.

    No physics-state concatenation and no dropout here by design — Gamma has
    no physics input at all (pure vision + language), so there's nothing to
    apply dropout to. See issue #9's note: this is intentional, not a gap.

    NLP/vision dependencies (gamma_pipeline.py, embeddings_gamma_768.npy,
    nlp_instructions.csv) are loaded from this same folder (rl/sig-lip/),
    where they actually live — not from the top-level nlp/ folder, which
    turned out to be a separate git submodule, not a regular folder in
    this repo.
    """

    def __init__(self, env):
        super().__init__(env)
        self._init_common()  # sets up _inference_mode (unused here, kept for consistency)

        self.observation_space = spaces.Dict({
            'vision': spaces.Box(low=-np.inf, high=np.inf, shape=(768,), dtype=np.float32),
            'nlp':    spaces.Box(low=-1.0,    high=1.0,    shape=(768,), dtype=np.float32)
        })

        if SIGLIP_AVAILABLE:
            self.siglip_model, self.siglip_processor, self.siglip_device = load_siglip()
        else:
            self.siglip_model = self.siglip_processor = self.siglip_device = None

        self.nlp_base_path = os.path.dirname(os.path.abspath(__file__))
        self._load_nlp_dataset()

        self.current_embedding = np.zeros(768, dtype=np.float32)

    def _load_nlp_dataset(self):
        npy_path = os.path.join(self.nlp_base_path, 'embeddings_gamma_768.npy')
        csv_path = os.path.join(self.nlp_base_path, 'nlp_instructions.csv')

        if os.path.exists(npy_path) and os.path.exists(csv_path):
            self.embeddings = np.load(npy_path).astype(np.float32)
            self.instructions_df = pd.read_csv(csv_path)
            self.num_instructions = len(self.embeddings)
            print(f"[Gamma Wrapper] Loaded {self.num_instructions} SigLIP embeddings (768-dim)")
        else:
            print("[Gamma Wrapper] WARNING: embeddings_gamma_768.npy not found. Using mock.")
            self.embeddings = np.random.uniform(-1, 1, size=(340, 768)).astype(np.float32)
            self.num_instructions = 340
            self.instructions_df = None

    def _get_vision_features(self):
        """Call env.render() and encode the frame with SigLIP."""
        frame = self.env.render()  # (480, 640, 3) uint8

        if SIGLIP_AVAILABLE and self.siglip_model is not None:
            return encode_image(
                frame,
                self.siglip_model,
                self.siglip_processor,
                self.siglip_device
            ).astype(np.float32)  # (768,)
        else:
            v = np.random.randn(768).astype(np.float32)
            return v / np.linalg.norm(v)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        idx = np.random.randint(0, self.num_instructions)
        self.current_embedding = self.embeddings[idx]

        if self.instructions_df is not None:
            info['current_instruction'] = self.instructions_df.iloc[idx]['instruction']

        vision_feat = self._get_vision_features()

        return {
            'vision': vision_feat,
            'nlp':    self.current_embedding
        }, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        vision_feat = self._get_vision_features()

        return {
            'vision': vision_feat,
            'nlp':    self.current_embedding
        }, reward, terminated, truncated, info