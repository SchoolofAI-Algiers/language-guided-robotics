import os
import sys
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

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
    obs_wrapped = LanguageConditionedWrapper(raw)
    shaped = RewardShapingWrapper(obs_wrapped)
    return Monitor(shaped)  # Monitor logs ep_len, ep_rew for TensorBoard


def main():
    # 1. Create training environment
    print("[RL Train] Loading PyBullet Kuka Environment...")
    env = make_env()

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
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path='./checkpoints/',
        name_prefix='saycan_ppo'
    )

    # Separate eval environment (must NOT share state with training env)
    eval_env = make_env()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./checkpoints/best/',
        log_path='./logs/eval/',
        eval_freq=5_000,        # evaluate every 5k training steps
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1,
    )

    callbacks = [checkpoint_callback, eval_callback]

    # 5. Train
    print("Starting training. Monitor progress with TensorBoard:")
    print("  tensorboard --logdir=./logs")
    print("  http://localhost:6006")

    try:
        model.learn(
            total_timesteps=200_000,
            callback=callbacks,
        )
        print("Training complete!")
        model.save("checkpoints/final_saycan_policy")
        print("Model saved → checkpoints/final_saycan_policy.zip")
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        model.save("checkpoints/interrupted_saycan_policy")


if __name__ == "__main__":
    main()
