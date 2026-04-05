import gymnasium as gym
import numpy as np

class RewardShapingWrapper(gym.Wrapper):
    """
    Adds dense reward shaping on top of the sparse KukaEnv reward.
    Dense = the agent gets small rewards for progress, not just for finishing.
    """

    def __init__(self, env):
        super().__init__(env)
        self.prev_distance = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Measure starting distance to target so we can track progress
        self.prev_distance = self._get_distance(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Bonus: reward the agent for getting closer to the target
        current_distance = self._get_distance(obs)
        if self.prev_distance is not None:
            progress = self.prev_distance - current_distance
            reward += progress * 2.0   # scale factor — tune this

        self.prev_distance = current_distance

        # Small penalty each step to encourage finishing fast
        reward -= 0.001

        return obs, reward, terminated, truncated, info

    def _get_distance(self, obs):
        # The vision obs contains object state in the last 9 values.
        # The first 3 of those are typically the target object's x,y,z position.
        # You'll want to confirm this with the Robotics team.
        object_state = obs['vision'][-9:]
        target_pos = object_state[:3]   # rough — adjust based on KukaEnv docs
        end_effector = object_state[3:6]
        return float(np.linalg.norm(target_pos - end_effector))