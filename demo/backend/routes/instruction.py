from flask import Blueprint, request, jsonify
import numpy as np
from logger import log_instruction

instruction_bp = Blueprint('instruction', __name__)

_env = None
_instruction_count = 0  # local counter for logging

def init_instruction(env):
    global _env
    _env = env

@instruction_bp.route('/api/instruction', methods=['POST'])
def run_instruction():
    global _instruction_count

    data = request.json
    instruction = data.get('instruction', '')

    obs = _env._get_observation()
    physics_state = obs["state"]

    joint_positions = physics_state[:7].tolist()
    joint_velocities = physics_state[7:14].tolist()
    ee_position = physics_state[14:17].tolist()
    ee_orientation = physics_state[17:21].tolist()

    joint_angles_deg = [
        {"joint": i, "angle": round(np.degrees(joint_positions[i]), 2)}
        for i in range(7)
    ]

    # log to W&B
    log_instruction(
        instruction=instruction,
        joint_angles=joint_angles_deg,
        ee_position=ee_position,
        step=_instruction_count
    )
    _instruction_count += 1

    return jsonify({
        "received": instruction,
        "status": "ok",
        "phase": 2,
        "joint_angles": joint_angles_deg,
        "ee_position": ee_position,
        "ee_orientation": ee_orientation,
        "nlp": None,
        "cv": None,
        "rl": None,
    })