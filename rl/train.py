import os
import sys
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# Ensure the project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl.feature_extractor import LanguageConditionedFeatureExtractor
from robotics.env.src.environment import KukaEnv
from rl.env_wrapper import LanguageConditionedWrapper
from rl.reward_shaping import RewardShapingWrapper

# ==========================================
# Training Script Setup
# ==========================================
def make_env(render_mode="rgb_array"):
    """Build the full env pipeline: KukaEnv → ObservationWrapper → RewardShaping → Monitor."""
    raw = KukaEnv(render_mode=render_mode)
    shaped = RewardShapingWrapper(raw)
    obs_wrapped = LanguageConditionedWrapper(shaped)
    return Monitor(obs_wrapped)


def make_vec_env(num_envs=4):
    """Creates a vectorized environment across multiple CPU cores for faster FPS."""
    def make_env_fn(rank):
        def _init():
            env = make_env()
            return env
        return _init
    return SubprocVecEnv([make_env_fn(i) for i in range(num_envs)])


def main():
    # 1. Initialize environment using multi-processing to parallelize PyBullet
    print(f"[RL Train] Loading Multi-Process PyBullet Kuka Environment...")
    env = make_vec_env(num_envs=4)

    # 2. Configure our custom MultiInputPolicy to use our custom Feature Extractor
    policy_kwargs = dict(
        features_extractor_class=LanguageConditionedFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )

    # 3. Initialize the PPO Algorithm with TensorBoard logging
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)

    print("[RL Train] Initializing PPO Model with MultiInputPolicy and custom Feature Extractor...")
    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=1024,           # Collect enough steps per rollout
        ent_coef=0.01,          # Encourage exploration early on
        clip_range=0.2,         # Standard PPO clipping
        n_epochs=10,            # PPO update epochs per rollout
    )

    # 4. Callbacks: periodic checkpoints + automatic evaluation
    os.makedirs("./checkpoints/", exist_ok=True)
    # 4. Save Checkpoints Periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=max(5000 // 4, 1), # Adjusted for 4 parallel envs
        save_path='./checkpoints/',
        name_prefix='ppo'
    )
    
    # Setup Eval callback
    eval_env = DummyVecEnv([lambda: make_env()])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./checkpoints/best/',
        log_path='./logs/eval/',
        eval_freq=max(5000 // 4, 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1,
    )
    callbacks = [checkpoint_callback, eval_callback]

    # 5. Iteratively Train Strategy (Start the training)
    print("Starting training layout. You can visualize on Tensorboard:")
    print("Run `tensorboard --logdir=./logs`")

    try:
        # Train with more steps natively
        model.learn(total_timesteps=50000, callback=callbacks)
        print("Initial training complete!")
        model.save("checkpoints/final_ppo_policy")
        print("Model saved → checkpoints/final_ppo_policy.zip")
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        model.save("checkpoints/interrupted_ppo_policy")


if __name__ == "__main__":
    main()
