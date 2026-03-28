import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../robotics')))
from flask import Flask
from flask_cors import CORS
from env.src.environment import KukaEnv
from logger import init_run

# ─────────────────────────────────────────────────────────────
# Shared env instance — one simulation, used by all routes
# stream.py reads frames from it continuously
# instruction.py reads obs from it on each instruction
# ─────────────────────────────────────────────────────────────
shared_env = KukaEnv(render_mode="rgb_array")
shared_env.reset()
init_run(run_name="phase2-stream", config={
    "phase": 2,
    "policy": "wave",
    "sim_timestep": 1/240,
    "control_hz": 30,
})

# import routes AFTER env is created so they can import it
from routes.stream import stream_bp, init_stream
from routes.instruction import instruction_bp, init_instruction

app = Flask(__name__)
CORS(app)

# pass shared env into each route module
init_stream(shared_env)
init_instruction(shared_env)

app.register_blueprint(stream_bp)
app.register_blueprint(instruction_bp)

if __name__ == '__main__':
    app.run(debug=False, port=5000, threaded=True)