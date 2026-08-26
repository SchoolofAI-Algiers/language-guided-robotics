"""
scripts/check_checkpoint.py
 
Issue #4: Write a checkpoint-verification script.
 
Usage:
    python scripts/check_checkpoint.py path/to/checkpoint.zip
 
Prints num_timesteps, n_envs, action_space, features_extractor_class,
and observation_space for a Stable-Baselines3 PPO checkpoint zip,
without needing to load the model onto a GPU or re-create the
training environment.

"""
 
import sys
from pathlib import Path
 
from stable_baselines3.common.save_util import load_from_zip_file
 
 
def check_checkpoint(zip_path: str) -> None:
    path = Path(zip_path)
    print(f"=== {path.name} ===")
 
    if not path.exists():
        print(f"  ERROR: file does not exist at {path}")
        return
 
    if path.stat().st_size < 100:
        print(f"  ERROR: file is only {path.stat().st_size} bytes — "
              f"this is not a valid checkpoint (looks corrupted/empty).")
        return
 
    try:
        data, params, pytorch_variables = load_from_zip_file(str(path))
    except Exception as e:
        print(f"  ERROR: failed to load checkpoint — {type(e).__name__}: {e}")
        return
 
    features_extractor_class = None
    policy_kwargs = data.get("policy_kwargs")
    if isinstance(policy_kwargs, dict):
        fe_class = policy_kwargs.get("features_extractor_class")
        features_extractor_class = getattr(fe_class, "__name__", fe_class)
 
    print(f"  num_timesteps           : {data.get('num_timesteps')}")
    print(f"  n_envs                  : {data.get('n_envs')}")
    print(f"  action_space            : {data.get('action_space')}")
    print(f"  observation_space       : {data.get('observation_space')}")
    print(f"  features_extractor_class: {features_extractor_class}")
    print()
 
 
def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/check_checkpoint.py <checkpoint.zip> [more.zip ...]")
        sys.exit(1)
 
    for zip_path in sys.argv[1:]:
        check_checkpoint(zip_path)
 
 
if __name__ == "__main__":
    main()
