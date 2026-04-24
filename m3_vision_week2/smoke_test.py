"""
smoke_test.py — Quick sanity check for the M3 Vision Week 2 pipeline
======================================================================
Run with:
    python smoke_test.py

Expected output:
    [PASS] Observation shape : torch.Size([546])
    [PASS] Dummy policy      : (1, 7)
"""

import torch
import torch.nn as nn

from pipeline import build_pipeline, run_step
from src.observation import OBS_DIM

ACTION_DIM = 7


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}\n")

    # Build environment + backbone
    kuka_id, backbone = build_pipeline(device=device)

    # Run one observation step (training mode, random mask)
    obs = run_step(kuka_id, backbone, train=True, device=device)

    # Shape check
    assert obs.shape[0] == OBS_DIM, \
        f"[FAIL] Expected obs dim {OBS_DIM}, got {obs.shape[0]}"
    print(f"[PASS] Observation shape : {obs.shape}")

    # Dummy policy forward pass
    dummy_policy = nn.Sequential(
        nn.Linear(OBS_DIM, 256),
        nn.ReLU(),
        nn.Linear(256, ACTION_DIM),
        nn.Tanh(),
    ).to(device)

    with torch.no_grad():
        action = dummy_policy(obs.unsqueeze(0))

    assert action.shape == (1, ACTION_DIM), \
        f"[FAIL] Unexpected action shape {action.shape}"
    print(f"[PASS] Dummy policy      : {tuple(action.shape)}")

    # Eval mode (mask always = 1)
    obs_eval = run_step(kuka_id, backbone, train=False, mask=1.0, device=device)
    assert obs_eval.shape[0] == OBS_DIM
    print(f"[PASS] Eval obs shape    : {obs_eval.shape}")


if __name__ == "__main__":
    main()
