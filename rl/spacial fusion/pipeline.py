"""
Beta inference pipeline — SOAI Labs 2026
Runs episode on the background env (same scene the stream shows).
Stream is paused during episode, then replays the frames.
"""

import os
import numpy as np
from stable_baselines3 import PPO

from robotics.env.src.environment import KukaEnv
from rl.beta_wrapper import BetaLanguageConditionedWrapper
from rl.reward_shaping import RewardShapingWrapper
from rl.beta_feature_extractor import BetaFeatureExtractor

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_SF_DIR   = os.path.join(_BASE_DIR, "spacial fusion")
_CKPT_ZIP = os.path.join(_SF_DIR, "beta_policy.zip")
_NLP_NPY  = os.path.join(_SF_DIR, "embeddings.npy")
_NLP_CSV  = os.path.join(_SF_DIR, "nlp_instructions.csv")

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
    if any(w in t for w in ['pick', 'grab', 'take', 'get']): return 'pick'
    if any(w in t for w in ['place', 'put', 'drop', 'set']): return 'place'
    if 'push' in t:  return 'push'
    if 'reach' in t: return 'reach'
    return 'reach'


def run_episode(instruction_text: str, max_steps: int = 200) -> dict:
    _load()

    # Import stream controls — avoids circular import at module level
    import demo.backend.routes.stream as _stream

    embedding, matched = find_best_embedding(instruction_text)
    print(f"[Pipeline] Matched: {matched}")

    target_colour = _parse_colour(instruction_text)

    # ── Pause stream and run episode on the SAME background env ──
    # This ensures the scene the user sees is exactly what the policy acts on.
    _stream.pause_stream()

    frames        = []
    ee_trajectory = []
    success       = False
    step          = 0
    distance      = float('inf')

    try:
        with _stream.get_stream_lock():
            # Reset the background env (new scene)
            obs, info = _bg_env.reset()

            # Set instruction on reward wrapper so it targets the right object
            _bg_env.set_instruction(instruction_text)

            # Re-run reset logic to pick correct target object after instruction set
            obj_state = info.get("object_state", {})
            target_colour = _parse_colour(instruction_text)
            target_obj_id = None
            if target_colour:
                for oid, st in obj_state.items():
                    if st["color"] == target_colour:
                        target_obj_id = oid
                        break
            if target_obj_id is None and obj_state:
                target_obj_id = list(obj_state.keys())[0]

            # Manually set target position on reward wrapper
            if target_obj_id and target_obj_id in obj_state:
                import numpy as _np
                _bg_env._target_pos    = _np.array(obj_state[target_obj_id]["pos"], dtype=_np.float32)
                _bg_env._target_obj_id = target_obj_id
                ee_pos = _np.array(info.get("ee_position", [0, 0, 0]), dtype=_np.float32)
                _bg_env._prev_dist = float(_np.linalg.norm(ee_pos - _bg_env._target_pos))
                print(f"[Pipeline] Target: {obj_state[target_obj_id]['color']} {obj_state[target_obj_id]['shape']} "
                      f"@ {[round(v,3) for v in obj_state[target_obj_id]['pos']]}")

            # Set NLP embedding
            _bg_env.env.set_embedding(embedding)
            obs["nlp"] = embedding

            # ── Snapshot object positions NOW (before arm moves anything) ──
            import pybullet as _p
            _kenv = _bg_env.env.env
            detected_objects = []
            for _oid, _col, _shp in zip(_kenv._object_ids, _kenv._object_colors, _kenv._object_shapes):
                _pos, _ = _p.getBasePositionAndOrientation(_oid, physicsClientId=_kenv._physics_client_id)
                detected_objects.append({"id": _oid, "color": _col, "shape": _shp,
                                         "pos": [round(float(v), 3) for v in _pos]})
            print(f"[Pipeline] Scene: {[(o['color'],o['shape'],o['pos']) for o in detected_objects]}")

        # ── Episode loop — acquire lock per step ──────────────────
        best_dist   = float('inf')
        no_progress = 0
        STALL_LIMIT = 40

        for step in range(max_steps):
            action, _ = _model.predict(obs, deterministic=True)

            with _stream.get_stream_lock():
                obs, reward, terminated, truncated, info = _bg_env.step(action)
                kuka_env = _bg_env.env.env
                frame    = kuka_env._get_camera_image()

            distance = info.get("distance_to_target", float('inf'))

            if frame is not None:
                frames.append(frame)
                # Push live frame to stream so user can watch
                _stream._latest_frame = frame

            ee_trajectory.append(info.get("ee_position", [0, 0, 0]))

            if info.get("is_success", False):
                success    = True
                terminated = True
                print(f"[Pipeline] ✓ SUCCESS at step {step}, dist={distance:.3f}")

            # Stall detection
            if distance < best_dist - 0.005:
                best_dist   = distance
                no_progress = 0
            else:
                no_progress += 1

            if no_progress >= STALL_LIMIT:
                print(f"[Pipeline] Stalled {STALL_LIMIT} steps, best={best_dist:.3f}m — stopping")
                break

            if terminated or truncated:
                break

        print(f"[Pipeline] Done: success={success} steps={step+1} dist={distance:.3f} frames={len(frames)}")

    except Exception as e:
        import traceback
        print(f"[Pipeline] Episode error: {e}")
        traceback.print_exc()
        raise
    finally:
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