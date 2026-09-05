"""
Beta inference pipeline — SOAI Labs 2026
Runs episode on the background env (same scene the stream shows).
Does NOT reset on instruction — keeps current scene.
"""

import os
import numpy as np
from stable_baselines3 import PPO

from robotics.env.src.environment import KukaEnv
from rl.beta_wrapper import BetaLanguageConditionedWrapper
from rl.reward_shaping import RewardShapingWrapper
from rl.beta_feature_extractor import BetaFeatureExtractor

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_CKPT_ZIP = os.path.join(_BASE_DIR, "..", "checkpoints", "paper-v1", "beta_final_600k.zip")
_NLP_NPY  = os.path.join(_BASE_DIR, "spacial fusion", "embeddings.npy")
_NLP_CSV  = os.path.join(_BASE_DIR, "spacial fusion", "nlp_instructions.csv")

_model        = None
_bg_env       = None
_embeddings   = None
_instructions = None


def _make_env():
    """Training stack: KukaEnv -> BetaWrapper -> RewardShaping"""
    return RewardShapingWrapper(
        BetaLanguageConditionedWrapper(
            KukaEnv(render_mode="rgb_array")
        )
    )


def _load():
    global _model, _bg_env, _embeddings, _instructions
    if _model is not None:
        return

    print("[Pipeline] Building background env...")
    _bg_env = _make_env()
    _bg_env.reset()

    print(f"[Pipeline] Loading PPO checkpoint from {_CKPT_ZIP} ...")
    if not os.path.exists(_CKPT_ZIP):
        raise FileNotFoundError(f"Checkpoint zip not found at {_CKPT_ZIP}")

    _model = PPO.load(
        _CKPT_ZIP,
        env=_bg_env,
        custom_objects={
            "features_extractor_class":  BetaFeatureExtractor,
            "features_extractor_kwargs": {"features_dim": 256},
        }
    )
    print("[Pipeline] PPO model loaded ✓")

    _embeddings = np.load(_NLP_NPY).astype(np.float32)
    import pandas as pd
    _instructions = (
        pd.read_csv(_NLP_CSV)["instruction"].tolist()
        if os.path.exists(_NLP_CSV) else []
    )
    print(f"[Pipeline] {len(_embeddings)} embeddings loaded ✓")


def find_best_embedding(instruction_text: str):
    _load()
    try:
        from sentence_transformers import SentenceTransformer
        st  = SentenceTransformer("all-MiniLM-L6-v2")
        qry = st.encode([instruction_text], normalize_embeddings=True)[0].astype(np.float32)
        sims    = _embeddings @ qry
        idx     = int(np.argmax(sims))
        matched = _instructions[idx] if _instructions and idx < len(_instructions) else "unknown"
        return _embeddings[idx], matched
    except Exception:
        idx = np.random.randint(0, len(_embeddings))
        return _embeddings[idx], _instructions[idx] if _instructions else "unknown"


def _parse_colour(text: str) -> str:
    for c in ['red', 'green', 'blue', 'yellow']:
        if c in text.lower():
            return c
    return None


def _parse_command(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ['pick', 'grab', 'take', 'get', 'grasp']): return 'pick'
    if any(w in t for w in ['lift', 'raise', 'hoist']): return 'lift'
    if any(w in t for w in ['lower', 'descend', 'down']): return 'lower'
    if any(w in t for w in ['place', 'put', 'drop', 'set']): return 'place'
    if 'push' in t:  return 'push'
    if any(w in t for w in ['pull', 'drag', 'draw']): return 'pull'
    if any(w in t for w in ['approach', 'go to', 'go near', 'head to', 'head toward', 'navigate', 'move']): return 'move'
    return 'move'


def run_episode(instruction_text: str, max_steps: int = 500) -> dict:
    _load()

    import demo.backend.routes.stream as _stream
    import pybullet as p

    embedding, matched = find_best_embedding(instruction_text)
    print(f"[Pipeline] Matched: {matched}")

    # Critical fix (issue #7 / Finding 2): without this call, RewardShapingWrapper's
    # _task_type never leaves its default "reach", regardless of instruction verb.
    # Embedding is passed so set_instruction() can use the trained SVM classifier
    # for instructions that don't exactly match the known 340-instruction dataset.
    _bg_env.set_instruction(instruction_text, embedding=embedding)
    print(f"[Pipeline] task_type: {_bg_env._task_type}")

    target_colour    = _parse_colour(instruction_text)
    frames           = []
    ee_trajectory    = []
    success          = False
    step             = 0
    distance         = float('inf')
    detected_objects = []

    _stream.pause_stream()

    try:
        with _stream.get_stream_lock():
            kuka_env = _bg_env.env.env  # RewardShaping -> BetaWrapper -> KukaEnv

            obj_state = kuka_env._get_object_state()
            raw_obs   = kuka_env._get_observation()
            ee_pos    = raw_obs[14:17]

            # ── Find target object ─────────────────────────────────
            target_obj_id = None
            if target_colour:
                for oid, st in obj_state.items():
                    if st["color"] == target_colour:
                        target_obj_id = oid
                        break
            if target_obj_id is None and obj_state:
                target_obj_id = list(obj_state.keys())[0]

            # ── Set target on reward wrapper AND gripper control ────
            if target_obj_id and target_obj_id in obj_state:
                kuka_env.set_target_object(target_obj_id)  # Tell gripper what to grasp
                _bg_env._target_pos    = np.array(obj_state[target_obj_id]["pos"], dtype=np.float32)
                _bg_env._target_obj_id = target_obj_id
                _bg_env._prev_dist     = float(np.linalg.norm(ee_pos - _bg_env._target_pos))
                print(f"[Pipeline] Target: {obj_state[target_obj_id]['color']} "
                      f"{obj_state[target_obj_id]['shape']} "
                      f"@ {[round(v,3) for v in obj_state[target_obj_id]['pos']]}")

            # ── Set NLP embedding ──────────────────────────────────
            _bg_env.env.set_embedding(embedding)
            
            # ── Disable physics dropout during inference ───────────
            _bg_env.env._inference_mode = True

            # ── Build observation for policy ───────────────────────
            info = {
                "ee_position":  ee_pos.tolist(),
                "object_state": obj_state,
            }
            obs = _bg_env.env.observation(raw_obs, info)
            obs["nlp"] = embedding
            
            # Debug: Check observation shapes
            print(f"[Pipeline] Obs vision shape: {obs['vision'].shape} | nlp shape: {obs['nlp'].shape}")
            print(f"[Pipeline] Vision stats: min={obs['vision'].min():.3f} max={obs['vision'].max():.3f} "
                  f"mean={obs['vision'].mean():.3f} std={obs['vision'].std():.3f}")
            print(f"[Pipeline] NLP stats: min={obs['nlp'].min():.3f} max={obs['nlp'].max():.3f} "
                  f"mean={obs['nlp'].mean():.3f}")
            if np.any(np.isnan(obs['vision'])) or np.any(np.isnan(obs['nlp'])):
                print("[Pipeline] ✗ WARNING: NaN found in initial observation!")
            if np.allclose(obs['vision'], 0, atol=1e-3):
                print("[Pipeline] ✗ WARNING: Vision observation is all zeros!")
            if np.any(np.isinf(obs['vision'])):
                print("[Pipeline] ✗ WARNING: Infinity in vision observation!")

            # ── Snapshot scene for CV panel ────────────────────────
            detected_objects = [
                {
                    "id":    oid,
                    "color": st["color"],
                    "shape": st["shape"],
                    "pos":   [round(float(v), 3) for v in st["pos"]],
                }
                for oid, st in obj_state.items()
            ]
            print(f"[Pipeline] Scene ({len(detected_objects)} objects): "
                  f"{[(o['color'], o['shape']) for o in detected_objects]}")

        # ── Episode loop ───────────────────────────────────────────
        for step in range(max_steps):
            action, _ = _model.predict(obs, deterministic=True)

            # Debug: Check for NaN or stuck actions
            if np.any(np.isnan(action)):
                print(f"[Pipeline] ✗ WARNING: NaN in action at step {step}")
                break
            if np.allclose(action, 0, atol=1e-3):
                print(f"[Pipeline] ✗ WARNING: Zero action at step {step} (policy stuck?)")
                if step < 20:  # Only break if stuck early
                    break

            with _stream.get_stream_lock():
                obs, reward, terminated, truncated, info = _bg_env.step(action)
                frame = kuka_env._get_camera_image()

            distance = info.get("distance_to_target", float('inf'))

            if step % 50 == 0 or step < 5:
                action_str = f"[{','.join([f'{a:.2f}' for a in action])}]"
                obs_vision_mean = obs['vision'].mean() if isinstance(obs, dict) else 0
                print(f"[Pipeline] Step {step}/{max_steps} action={action_str} dist={distance:.3f} "
                      f"vision_mean={obs_vision_mean:.4f}")

            if frame is not None:
                frames.append(frame)
                _stream._latest_frame = frame

            ee_trajectory.append(info.get("ee_position", [0, 0, 0]))

            if info.get("is_success", False):
                success    = True
                terminated = True
                print(f"[Pipeline] ✓ SUCCESS at step {step}, dist={distance:.3f}")

            if terminated or truncated:
                break

        print(f"[Pipeline] Done: success={success} steps={step+1} "
              f"dist={distance:.3f} frames={len(frames)}")

    except Exception as e:
        import traceback
        print(f"[Pipeline] Episode error: {e}")
        traceback.print_exc()
        raise
    finally:
        _bg_env.env._inference_mode = False  # Re-enable dropout for training
        _stream.resume_stream()

    return {
        "frames":              frames,
        "success":             success,
        "steps":               step + 1,
        "matched_instruction": matched,
        "ee_trajectory":       ee_trajectory,
        "distance_final":      float(distance) if distance != float('inf') else -1.0,
        "detected_objects":    detected_objects,
        "command_type":        _parse_command(instruction_text),
        "target":              f"{target_colour or 'nearest'} object",
        "confidence":          0.95,
    }


def get_env():
    _load()
    return _bg_env


def get_model():
    _load()
    return _model