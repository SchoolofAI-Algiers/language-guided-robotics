import io
import base64
import threading
import numpy as np
import cv2
from flask import Blueprint, request, jsonify

from rl.pipeline import run_episode, find_best_embedding, get_env

instruction_bp = Blueprint('instruction', __name__)
_lock = threading.Lock()


@instruction_bp.route('/api/instruction', methods=['POST'])
def run_instruction():
    data        = request.json or {}
    instruction = data.get('instruction', '').strip()

    if not instruction:
        return jsonify({"status": "error", "error": "No instruction provided"}), 400

    acquired = _lock.acquire(timeout=5)
    if not acquired:
        return jsonify({"status": "error", "error": "Simulation busy, try again"}), 503

    try:
        result = run_episode(instruction, max_steps=500)
    except Exception as e:
        import traceback
        return jsonify({"status": "error", "error": str(e),
                        "trace": traceback.format_exc()}), 500
    finally:
        _lock.release()

    # Encode frames as base64 JPEGs for the frontend
    encoded_frames = []
    for frame in result["frames"][::3]:   # send every 3rd frame to keep payload small
        _, buf = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                              [cv2.IMWRITE_JPEG_QUALITY, 70])
        encoded_frames.append(base64.b64encode(buf.tobytes()).decode('utf-8'))

    return jsonify({
        "status":               "ok",
        "received":             instruction,
        "matched_instruction":  result["matched_instruction"],
        "success":              result["success"],
        "steps":                result["steps"],
        "distance_final":       round(result["distance_final"], 4),
        "ee_trajectory":        result["ee_trajectory"][::5],  # downsample
        "frames":               encoded_frames,
        "frame_count":          len(encoded_frames),
        "strategy":             "Beta — Early Spatial Fusion",
    })


@instruction_bp.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        "status":   "ok",
        "strategy": "Beta — Early Spatial Fusion (4-channel ResNet18)",
        "modules":  {
            "pybullet":  "live",
            "beta_cnn":  "live",
            "ppo_policy": "live",
            "nlp_embeddings": "live",
        }
    })