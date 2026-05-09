import sys
import os

# Make all modules importable regardless of working directory
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from flask import Flask
from flask_cors import CORS

from demo.backend.routes.stream import stream_bp, start_background_stepper
from demo.backend.routes.instruction import instruction_bp

app = Flask(__name__)
CORS(app)

app.register_blueprint(stream_bp)
app.register_blueprint(instruction_bp)

# Start live inference stream
start_background_stepper()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port, threaded=True)