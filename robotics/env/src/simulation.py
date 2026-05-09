import argparse
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from env.src.environment import KukaEnv
from env.src.config import RENDER_WIDTH, RENDER_HEIGHT, MAX_EPISODE_STEPS, RENDER_FPS, RenderMode
def random_policy(obs, step):
    return np.random.uniform(-1.0, 1.0, size=7).astype(np.float32)

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

POLICIES = {
    "random": random_policy,
    "wave": wave_policy,
}

def run(policy_name="wave", max_steps=MAX_EPISODE_STEPS, render_mode=RenderMode.RGB_ARRAY.value):
    policy = POLICIES[policy_name]
    env = KukaEnv(render_mode=render_mode)
    obs, _ = env.reset()

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))
    img_array = np.zeros((RENDER_HEIGHT, RENDER_WIDTH, 3), dtype=np.uint8)
    img_plot = ax.imshow(img_array)
    ax.set_axis_off()
    ax.set_title(f"Kuka IIWA - {policy_name} policy")
    fig.tight_layout()

    try:
        for step in range(max_steps):
            action = policy(obs, step)
            obs, reward, terminated, truncated, info = env.step(action)

            # render every frame (env already runs at 30 Hz control)
            rgb = env.render()
            img_plot.set_data(rgb)
            ax.set_title(
                f"Kuka IIWA - {policy_name} policy | "
                f"step {step + 1}/{max_steps} | "
                f"ee: [{info['ee_position'][0]:+.2f}, "
                f"{info['ee_position'][1]:+.2f}, "
                f"{info['ee_position'][2]:+.2f}]"
            )
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            if terminated or truncated:
                break

        print(f"Simulation done - {step + 1} steps.")
        print(f"Final ee position: {info['ee_position']}")
        print("Close the window to exit.")
        plt.ioff()
        plt.show()
    finally:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kuka IIWA live simulation")
    parser.add_argument(
        "--policy",
        type=str,
        default="wave",
        choices=POLICIES.keys(),
        help="Which policy to run (default: wave)",
    )
    parser.add_argument(
        "--render_mode",
        type=str,
        default=RenderMode.RGB_ARRAY.value,
        choices=[mode.value for mode in RenderMode],
        help="Render mode for the environment (default: rgb_array)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=MAX_EPISODE_STEPS,
        help=f"Maximum steps to run the simulation (default: {MAX_EPISODE_STEPS})",
    )
    args = parser.parse_args()
    run(policy_name=args.policy, max_steps=args.max_steps, render_mode=args.render_mode)