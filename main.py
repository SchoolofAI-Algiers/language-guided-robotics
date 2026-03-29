import torch
import torch.nn as nn

from config   import OBS_DIM, ACTION_DIM, DEVICE
from pipeline import run_pipeline


def week2_check(observation: torch.Tensor) -> torch.Tensor:
    """
    Dummy linear policy: verifies the observation tensor feeds into a
    7-DOF arm policy without shape errors.

    This is the Week-2 handoff contract — replace with the real PPO policy.

    Parameters
    ----------
    observation : (281,) tensor

    Returns
    -------
    action : (1, 7) tensor
    """
    policy = nn.Sequential(
        nn.Linear(OBS_DIM, 256),
        nn.ReLU(),
        nn.Linear(256, ACTION_DIM),
    ).to(DEVICE)

    with torch.no_grad():
        action = policy(observation.unsqueeze(0))   # (1, 7)

    return action


def main():
    print("=" * 50)
    print("M3 Vision — Week 1 Pipeline")
    print("=" * 50)

    observation = run_pipeline(verbose=True)

    print()
    action = week2_check(observation)

    print()
    print("WEEK 2 CHECK PASSED")
    print(f"  {observation.shape}  →  dummy policy  →  {action.shape}")
    print()
    print("Summary")
    print(f"  RGB frame          (224, 224, 3)")
    print(f"  Depth + seg mask   free from PyBullet")
    print(f"  Visual features    {256}")
    print(f"  Detection vector   {25}  (5 boxes × 5 features)")
    print(f"  Observation tensor {tuple(observation.shape)}  → RL team")


if __name__ == "__main__":
    main()