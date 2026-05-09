import time
import threading
import cv2
import numpy as np
from flask import Blueprint, Response, jsonify

from rl.pipeline import get_env, get_model

stream_bp = Blueprint('stream', __name__)

# ── State machine ──────────────────────────────────────────────
# IDLE:      arm holds position, stream shows last frame
# RUNNING:   pipeline is running an episode (stream updated by pipeline)
# REPLAYING: pipeline done, stream replays captured frames
_state         = 'IDLE'
_latest_frame  = None
_replay_buffer = []
_replay_idx    = 0
_lock          = threading.Lock()
_paused        = False


def pause_stream():
    global _paused
    _paused = True

def resume_stream():
    global _paused, _state, _replay_idx
    _paused     = False
    _replay_idx = 0
    _state      = 'REPLAYING'

def get_stream_lock():
    return _lock

def set_replay_frames(frames: list):
    global _replay_buffer, _replay_idx, _state
    _replay_buffer = frames
    _replay_idx    = 0
    _state         = 'REPLAYING'


def _render_home_frame(env):
    try:
        return env.env.env._get_camera_image()  # RewardShaping -> BetaWrapper -> KukaEnv
    except Exception as e:
        print(f"[Stream] Home render error: {e}")
        return None


def _simulation_loop():
    global _latest_frame, _state, _replay_idx

    env = get_env()

    with _lock:
        env.reset()
        frame = _render_home_frame(env)
        if frame is not None:
            _latest_frame = frame

    print("[Stream] Initial home frame rendered ✓")

    while True:
        if _paused:
            # Pipeline is running — it updates _latest_frame directly
            time.sleep(0.03)
            continue

        if _state == 'IDLE':
            time.sleep(0.1)

        elif _state == 'REPLAYING':
            if _replay_idx < len(_replay_buffer):
                _latest_frame = _replay_buffer[_replay_idx]
                _replay_idx  += 1
                time.sleep(1 / 30)
            else:
                # Replay done
                with _lock:
                    frame = _render_home_frame(env)
                    if frame is not None:
                        _latest_frame = frame
                _state = 'IDLE'
                print("[Stream] Replay complete → IDLE")


def start_background_stepper():
    t = threading.Thread(target=_simulation_loop, daemon=True)
    t.start()
    start = time.time()
    while _latest_frame is None and (time.time() - start) < 15:
        time.sleep(0.1)
    print("[Stream] Background stepper started ✓")


def generate_frames():
    while True:
        frame = _latest_frame
        if frame is not None:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n'
                   + buf.tobytes()
                   + b'\r\n')
        time.sleep(1 / 30)


@stream_bp.route('/api/stream')
def stream():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@stream_bp.route('/api/scene')
def get_scene():
    """Returns current object state without resetting. Always returns fresh positions."""
    try:
        env      = get_env()
        kuka_env = env.env.env
        import pybullet as p
        
        objects = []
        # Always try to read current positions, even if objects have moved
        for oid, col, shp in zip(kuka_env._object_ids, kuka_env._object_colors, kuka_env._object_shapes):
            try:
                pos, _ = p.getBasePositionAndOrientation(oid, physicsClientId=kuka_env._physics_client_id)
                objects.append({
                    "id": oid, 
                    "color": col, 
                    "shape": shp,
                    "pos": [round(float(v), 3) for v in pos]
                })
            except Exception as e:
                # Skip objects that can't be read (grasped or deleted)
                print(f"[Scene] Skipping object {oid}: {e}")
                continue
                
        return jsonify({
            "status": "ok", 
            "detected_objects": objects, 
            "n_visible": len(objects),
            "model": "ResNet18-4ch Beta"
        }), 200
    except Exception as e:
        print(f"[Scene] Error reading scene: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "ok", 
            "detected_objects": [], 
            "n_visible": 0,
            "error": str(e)
        }), 200  # Return 200 OK even on error so frontend keeps polling


@stream_bp.route('/api/reset', methods=['POST'])
def reset_scene():
    global _latest_frame, _state, _replay_buffer
    try:
        env = get_env()
        with _lock:
            obs, info = env.reset()
            frame = _render_home_frame(env)
            if frame is not None:
                _latest_frame = frame
        _replay_buffer = []
        _state = 'IDLE'

        kuka_env = env.env.env
        import pybullet as p
        objects = []
        for oid, col, shp in zip(kuka_env._object_ids, kuka_env._object_colors, kuka_env._object_shapes):
            pos, _ = p.getBasePositionAndOrientation(oid, physicsClientId=kuka_env._physics_client_id)
            objects.append({"id": oid, "color": col, "shape": shp,
                            "pos": [round(float(v), 3) for v in pos]})
        return jsonify({"status": "ok", "detected_objects": objects})
    except Exception as e:
        import traceback
        return jsonify({"status": "error", "error": str(e),
                        "trace": traceback.format_exc()}), 500


@stream_bp.route('/api/status')
def status():
    return jsonify({
        "status":   "ok",
        "strategy": "Beta — Early Spatial Fusion",
        "stream_state": _state,
        "modules":  {"pybullet": "live", "beta_cnn": "live",
                     "ppo_policy": "live", "nlp": "live"}
    })