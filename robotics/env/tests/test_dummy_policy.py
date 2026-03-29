import sys
import os
import numpy as np
import torch
import torch.nn as nn

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from robotics.env.src.environment import KukaEnv

class DummyPolicyNetwork(nn.Module):
    """
    A bare minimum, lightweight PyTorch model representing an RL actor network.
    Verifies that the concatenated observation tensor mathematically matches
    without throwing shape mismatch errors during a forward pass.
    """
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh() # bounds output roughly to [-1, 1], conforming to env action space
        )
        
    def forward(self, obs_array):
        # Convert raw numpy array from env into a float batched tensor
        obs_tensor = torch.tensor(obs_array, dtype=torch.float32).unsqueeze(0)
        action_tensor = self.net(obs_tensor)
        return action_tensor.squeeze(0).detach().numpy()

def test_ablation_dummy_policy(obs_mode):
    print(f"\n[Test] Initializing KukaEnv with obs_mode='{obs_mode}'")
    env = KukaEnv(render_mode="rgb_array", obs_mode=obs_mode)
    
    print("[Test] Calling env.reset()...")
    obs, info = env.reset()
    
    # Handle Dict vs Box spaces for dimensionality
    if isinstance(env.observation_space, gym.spaces.Dict):
        print(f"[Test] Observation is Dict Space: {env.observation_space}")
        obs_dim = env.observation_space["state"].shape[0] # Just base off state shape for printing
        action_dim = env.action_space.shape[0]
        # Quick validation
        assert "pixels" in obs and "state" in obs, "Missing dictionary keys for pixels mode"
    else:
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        print(f"[Test] Observation Shape: {obs.shape}")
        print(f"[Test] Expected Shape: {env.observation_space.shape}")
        assert obs.shape == env.observation_space.shape, f"Shape mismatch! Got {obs.shape}, expected {env.observation_space.shape}"
        
        # Initialize the Neural Network Policy
        print(f"[Test] Initializing DummyPolicyNetwork(input_dim={obs_dim}, output_dim={action_dim})...")
        policy = DummyPolicyNetwork(obs_dim, action_dim)

    print("[Test] Running 3 step loops...")
    for step_idx in range(3):
        # The visual + state tensor flows directly into the dummy policy
        if isinstance(env.observation_space, gym.spaces.Dict):
            # For Dict spaces (like pixels), we just dummy sample to prevent writing a CNN here
            predicted_action = env.action_space.sample()
        else:
            try:
                predicted_action = policy(obs)
            except Exception as e:
                print(f"[Test Failed] Shape error during network forward pass: {e}")
                raise e
        
        obs, reward, terminated, truncated, info = env.step(predicted_action)
        
        print(f"  Step {step_idx + 1} | Action [NN]: {np.round(predicted_action, 2)}")
        print(f"           | Obs Type: {type(obs)}")
        
        if terminated or truncated:
            print("[Test] Episode ended early.")
            break
            
    print(f"[Test] SUCCESS: obs_mode='{obs_mode}' flows without shape errors ✅")
    env.close()

if __name__ == "__main__":
    import gymnasium as gym
    test_ablation_dummy_policy("visual_joints")
    test_ablation_dummy_policy("visual_statepybullet")
    test_ablation_dummy_policy("visual_joints_statepybullet")
    test_ablation_dummy_policy("visual_only")
    test_ablation_dummy_policy("pixels")
    test_ablation_dummy_policy("state")
