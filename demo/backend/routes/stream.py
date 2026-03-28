from flask import Blueprint, Response
import cv2
import time
import numpy as np

stream_bp = Blueprint('stream', __name__)

_env = None
_step_count = 0

def init_stream(env):
    global _env
    _env = env

def wave_policy(obs, step):
    t = step * 0.05
    return np.array([
        0.6 * np.sin(t),
        0.4 * np.sin(t + 1.0),
        0.6 * np.sin(t + 2.0),
        0.5 * np.cos(t),
        0.6 * np.sin(t + 3.0),
        0.4 * np.cos(t + 1.5),
        0.3 * np.sin(t * 2.0),
    ], dtype=np.float32)

def generate_frames():
    global _step_count
    obs = _env.reset()[0]

    while True:
        action = wave_policy(obs, _step_count)
        obs, _, _, truncated, _ = _env.step(action)
        _step_count += 1

        if truncated:
            obs = _env.reset()[0]
            _step_count = 0

        frame = _env.render()
        _, buffer = cv2.imencode('.jpg', frame)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'
               + buffer.tobytes()
               + b'\r\n')

        time.sleep(1 / 30)

@stream_bp.route('/api/stream')
def stream():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )