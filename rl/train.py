import argparse
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

def make_env(render_mode="rgb_array"):
    """Build the full env pipeline: KukaEnv → ObservationWrapper → RewardShaping → Monitor."""
    raw = KukaEnv(render_mode=render_mode)
    obs_wrapped = LanguageConditionedWrapper(raw)
    try:
        shaped = RewardShapingWrapper(obs_wrapped)
    except ImportError:
        shaped = obs_wrapped
        print("[Warning] RewardShapingWrapper not found, falling back to unshaped environment.")
    return Monitor(shaped)

def make_vec_env(num_envs=4, render_mode="rgb_array"):
    def make_env_fn(rank):
        def _init():
            return make_env(render_mode=render_mode)
        return _init
    return SubprocVecEnv([make_env_fn(i) for i in range(num_envs)])

def parse_args():
    parser = argparse.ArgumentParser(description="RL training for Language-Guided Robotics")
    parser.add_argument("--steps", type=int, default=200000, help="Total training timesteps")
    parser.add_argument("--log-dir", type=str, default="./logs", help="TensorBoard log directory")
    parser.add_argument("--ckpt-dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--ckpt-interval", type=int, default=10000, help="Checkpoint interval in timesteps")
    parser.add_argument("--run-name", type=str, default="Alpha_ResNet18_Run1", help="TensorBoard run name")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--device", type=str, default="cuda" if ("torch" in globals() and __import__('torch').cuda.is_available()) else "cpu", help="Device for PPO (cuda or cpu)")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    print(f"[RL Train] Initializing {args.num_envs} parallel environments...")
    env = make_vec_env(num_envs=args.num_envs)
    policy_kwargs = dict(
        features_extractor_class=LanguageConditionedFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )
    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=args.log_dir,
        learning_rate=3e-4,
        n_steps=1024,
        ent_coef=0.01,
        clip_range=0.2,
        n_epochs=10,
        device=args.device,
    )
    save_freq = max(args.ckpt_interval // args.num_envs, 1)
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=args.ckpt_dir, name_prefix='ppo')
    eval_env = DummyVecEnv([lambda: make_env()])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.ckpt_dir, 'best'),
        log_path=os.path.join(args.log_dir, 'eval'),
        eval_freq=save_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1,
    )
    callbacks = [checkpoint_callback, eval_callback]
    print("[RL Train] Starting training. Run name:", args.run_name)
    print(f"TensorBoard: tensorboard --logdir={args.log_dir}")
    try:
        model.learn(total_timesteps=args.steps, callback=callbacks)
        print("Training finished.")
        model.save(os.path.join(args.ckpt_dir, 'final_ppo_policy'))
    except KeyboardInterrupt:
        print("\nInterrupted, saving checkpoint...")
        model.save(os.path.join(args.ckpt_dir, 'interrupted_ppo_policy'))

if __name__ == "__main__":
    main()
