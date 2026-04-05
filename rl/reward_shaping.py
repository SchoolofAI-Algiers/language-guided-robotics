"""
Reward Shaping Wrapper — lives in rl/ so it doesn't touch the Robotics codebase.

Wraps any environment whose `info` dict contains `ee_position` (list[float], len 3)
and adds dense potential-based reward shaping for a random reaching target.

Reward design:
    r_t = (dist_{t-1} - dist_t)          # potential-based shaping (closer → positive)
          - 0.001                          # small time penalty (encourages efficiency)
          + 1.0 * 1{success}              # sparse bonus on reaching within threshold
"""

import numpy as np
import gymnasium as gym


class RewardShapingWrapper(gym.Wrapper):
    """
    Adds dense reward shaping on top of an environment that exposes
    `info['ee_position']` (3D end-effector position).

    A random reachable target is sampled at each reset inside `workspace_low / workspace_high`.
    """

    SUCCESS_THRESHOLD = 0.05  # 5 cm

    def __init__(
        self,
        env: gym.Env,
        workspace_low: np.ndarray = np.array([-1.0, -1.0, 0.1]),
        workspace_high: np.ndarray = np.array([1.0, 1.0, 1.2]),
    ):
        super().__init__(env)
        self._ws_low = workspace_low.astype(np.float32)
        self._ws_high = workspace_high.astype(np.float32)
        self._target_pos = np.zeros(3, dtype=np.float32)
        self._prev_dist: float = 0.0

    # --------------------------------------------------------------------- #
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Sample a random target within the (slightly tighter) workspace bounds
        self._target_pos = self.np_random.uniform(
            low=self._ws_low, high=self._ws_high
        ).astype(np.float32)

        # Initialise potential from the starting end-effector position
        ee_pos = np.asarray(info.get("ee_position", [0.0, 0.0, 0.0]), dtype=np.float32)
        self._prev_dist = float(np.linalg.norm(ee_pos - self._target_pos))

        info["target_pos"] = self._target_pos.tolist()
        return obs, info

    # --------------------------------------------------------------------- #
    def step(self, action):
        obs, _reward, terminated, truncated, info = self.env.step(action)

        # Retrieve current end-effector position from info
        ee_pos = np.asarray(info.get("ee_position", [0.0, 0.0, 0.0]), dtype=np.float32)
        curr_dist = float(np.linalg.norm(ee_pos - self._target_pos))

        # Potential-based shaping: positive when getting closer
        shaping = self._prev_dist - curr_dist
        self._prev_dist = curr_dist

        # Dense reward = shaping - small time cost
        reward = shaping - 0.001

        # Sparse success bonus
        if curr_dist < self.SUCCESS_THRESHOLD:
            reward += 1.0
            terminated = True

        # Enrich info for TensorBoard custom logging
        info["target_pos"] = self._target_pos.tolist()
        info["distance_to_target"] = curr_dist
        info["is_success"] = curr_dist < self.SUCCESS_THRESHOLD

        return obs, reward, terminated, truncated, info
