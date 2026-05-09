import os
import sys
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl.gamma_feature_extractor import GammaFeatureExtractor
from rl.gamma_env_wrapper import GammaLanguageConditionedWrapper
from rl.reward_shaping import RewardShapingWrapper
from robotics.env.src.environment import KukaEnv

# ── Tunable knobs ─────────────────────────────────────────────────────────────
NUM_ENVS        = 4
TOTAL_TIMESTEPS = 300_000    # reduced — matches Beta budget, comparable results
N_STEPS         = 512        # reduced from 2048 — faster updates, more frequent logging
FEATURES_DIM    = 256        # reduced from 512 — SigLIP already rich, 256 is enough
LEARNING_RATE   = 3e-4       # back to Beta's LR — linear decay was too aggressive
ENT_COEF        = 0.02       # slightly above Beta's 0.01, below your 0.05
CLIP_RANGE      = 0.2        # back to standard — 0.1 was too tight
N_EPOCHS        = 5          # keep reduced, good call
CHECKPOINT_FREQ = 10_000     # save more frequently so crashes don't lose everything
RESUME_PATH     = None       # set to e.g. "checkpoints/gamma/gamma_ppo_200000_steps"
# ─────────────────────────────────────────────────────────────────────────────


def make_env():
    def _init():
        raw     = KukaEnv(render_mode="rgb_array")
        wrapped = GammaLanguageConditionedWrapper(raw)
        shaped  = RewardShapingWrapper(wrapped)
        return Monitor(shaped)
    return _init   # SubprocVecEnv needs a callable factory, not the env itself


def linear_schedule(initial_value: float):
    def schedule(progress_remaining: float) -> float:
        return initial_value * progress_remaining
    return schedule


def main():
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device count:   {torch.cuda.device_count()}")

    # ── SubprocVecEnv: true parallelism, each env in its own process ──────────
    # Much faster than DummyVecEnv for slow envs like SigLIP-wrapped KukaEnv
    env = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)])

    policy_kwargs = dict(
        features_extractor_class=GammaFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=FEATURES_DIM),
    )

    log_dir = "./logs/gamma/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("./checkpoints/gamma/best/", exist_ok=True)

    if RESUME_PATH and os.path.exists(RESUME_PATH + ".zip"):
        # Resume from checkpoint instead of starting over after a crash
        print(f"Resuming from {RESUME_PATH}")
        model = PPO.load(RESUME_PATH, env=env, tensorboard_log=log_dir)
        # Reapply any kwargs not saved in the checkpoint
        model.ent_coef  = ENT_COEF
        model.clip_range = CLIP_RANGE
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=LEARNING_RATE,   # fixed LR — simpler, works well
            n_steps=N_STEPS,
            batch_size=64,                 # explicit batch size same as Beta
            ent_coef=ENT_COEF,
            clip_range=CLIP_RANGE,
            n_epochs=N_EPOCHS,
            device="cuda",                 # explicit — don't let SB3 guess
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ // NUM_ENVS,  # save_freq is per-env steps
        save_path='./checkpoints/gamma/',
        name_prefix='gamma_ppo',
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    eval_env = SubprocVecEnv([make_env()])  # single env, same type as train
    eval_cb  = EvalCallback(
        eval_env,
        best_model_save_path="./checkpoints/gamma/best/",
        log_path=log_dir,
        eval_freq=max(10_000 // NUM_ENVS, 1),
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    samples_per_update = NUM_ENVS * N_STEPS
    print(
        f"Gamma training — {NUM_ENVS} envs × {N_STEPS} steps = "
        f"{samples_per_update:,} samples/update\n"
        f"{TOTAL_TIMESTEPS:,} total steps = "
        f"{TOTAL_TIMESTEPS // samples_per_update} updates\n"
        f"TensorBoard: tensorboard --logdir={log_dir}"
    )

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_cb, eval_cb],
            reset_num_timesteps=RESUME_PATH is None,  # don't reset step count on resume
        )
        model.save("checkpoints/gamma/final_gamma_ppo")
        print("Done — checkpoints/gamma/final_gamma_ppo.zip")
    except KeyboardInterrupt:
        model.save("checkpoints/gamma/interrupted_gamma_ppo")
        print("Interrupted — checkpoint saved.")


if __name__ == "__main__":
    main()