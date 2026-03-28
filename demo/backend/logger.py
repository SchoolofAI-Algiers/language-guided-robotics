from torch.utils.tensorboard import SummaryWriter
import os

_writer = None

def init_run(run_name="demo-stream", config=None):
    global _writer
    log_dir = os.path.join(os.path.dirname(__file__), '../../experiments/runs', run_name)
    _writer = SummaryWriter(log_dir=log_dir)
    print(f"[Logger] TensorBoard logging to {log_dir}")

def log_instruction(instruction, joint_angles, ee_position, step):
    if not _writer: return
    _writer.add_scalar("ee/x", ee_position[0], step)
    _writer.add_scalar("ee/y", ee_position[1], step)
    _writer.add_scalar("ee/z", ee_position[2], step)
    for i, j in enumerate(joint_angles):
        _writer.add_scalar(f"joints/J{i+1}_deg", j["angle"], step)

def log_training(reward, success_rate, episode, policy_loss=None):
    if not _writer: return
    _writer.add_scalar("train/reward", reward, episode)
    _writer.add_scalar("train/success_rate", success_rate, episode)
    if policy_loss is not None:
        _writer.add_scalar("train/policy_loss", policy_loss, episode)

def finish():
    if _writer: _writer.close()