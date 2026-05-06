import argparse
from env.src.environment import KukaEnv

def check_environment():
    env = KukaEnv(render_mode="rgb_array")
    obs, info = env.reset()
    print("[Env Check] object_state:", info.get("object_state"))
    # Perform a single step with zero action
    action = [0.0] * 7
    obs, reward, terminated, truncated, step_info = env.step(action)
    print("[Env Check] step info:", step_info)
    env.close()

def main():
    parser = argparse.ArgumentParser(description="Language Guided Robotics entry point")
    parser.add_argument("--check-env", action="store_true", help="Run environment verification checks")
    args = parser.parse_args()
    if args.check_env:
        check_environment()
    else:
        print("Language-Guided Robotics project initialized!")

if __name__ == "__main__":
    main()