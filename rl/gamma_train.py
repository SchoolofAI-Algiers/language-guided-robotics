import os
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl.gamma_feature_extractor import GammaFeatureExtractor
from rl.gamma_env_wrapper import GammaLanguageConditionedWrapper
from rl.reward_shaping import RewardShapingWrapper  # reuse Chouaib's unchanged
from robotics.env.src.environment import KukaEnv


def make_env():
    raw     = KukaEnv(render_mode="rgb_array")
    wrapped = GammaLanguageConditionedWrapper(raw)
    shaped  = RewardShapingWrapper(wrapped)
    return Monitor(shaped)


def main():
    env = DummyVecEnv([make_env])  # single env first — confirm it runs before scaling up

    policy_kwargs = dict(
        features_extractor_class=GammaFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )

    log_dir = "./logs/gamma/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("./checkpoints/gamma/", exist_ok=True)

    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=1024,
        ent_coef=0.01,
        clip_range=0.2,
        n_epochs=10,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path='./checkpoints/gamma/',
        name_prefix='gamma_ppo'
    )

    print("Starting Gamma training. TensorBoard: tensorboard --logdir=./logs/gamma")
    try:
        model.learn(total_timesteps=200000, callback=[checkpoint_callback])
        model.save("checkpoints/gamma/final_gamma_ppo")
        print("Done — checkpoints/gamma/final_gamma_ppo.zip")
    except KeyboardInterrupt:
        model.save("checkpoints/gamma/interrupted_gamma_ppo")


if __name__ == "__main__":
    main()