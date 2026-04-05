import gymnasium as gym
import numpy as np
from gymnasium import spaces
from rl.env_wrapper import LanguageConditionedWrapper
from rl.reward_shaping import RewardShapingWrapper


class DummyKukaEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Dict({
            'pixels':       spaces.Box(0, 255, shape=(64, 64, 3), dtype=np.uint8),
            'state':        spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
            'object_state': spaces.Box(-np.inf, np.inf, shape=(9,), dtype=np.float32),
        })
        self.action_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)

    def reset(self, **kwargs):
        obs = {
            'pixels':       np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
            'state':        np.random.randn(7).astype(np.float32),
            'object_state': np.random.randn(9).astype(np.float32),
        }
        return obs, {}

    def step(self, action):
        obs = {
            'pixels':       np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
            'state':        np.random.randn(7).astype(np.float32),
            'object_state': np.random.randn(9).astype(np.float32),
        }
        return obs, 0.0, False, False, {}
    

raw     = DummyKukaEnv()
wrapped = LanguageConditionedWrapper(raw)
shaped  = RewardShapingWrapper(wrapped)

obs, info = shaped.reset()
print("vision shape:", obs['vision'].shape)
print("nlp shape:",    obs['nlp'].shape)
print("instruction:", info.get('current_instruction', 'not set'))

obs, reward, _, _, _ = shaped.step(shaped.action_space.sample())
print("reward:", reward)