_Your goal right now is to verify the environment and kick off the blind training run so it runs while you sleep._

**1. Shared Environment & Infrastructure Checks:**

- [ ] Run `env.reset()` and print `info["object_state"]`. Visually verify there are 4 real objects with non-zero XYZ positions.
    
- [ ] Run a `step()` and verify `info["position"]` is returning the real Kuka end-effector XYZ.
    
- [ ] Verify the reward target is locked to a _real_ spawned object position (not a random workspace point).
    
- [ ] Ensure `embeddings_350.npy` is loaded and the instruction randomizes on every reset.
    
- [ ] Set up your saving logic: Ensure model checkpoints are saving every 10,000 steps.
    
- [ ] Initialize your specific TensorBoard run (name it something clear like `Alpha_ResNet18_Run1`).
    
- [ ] Ensure PyBullet is set to CPU and CNN forward passes are on the Kaggle T4 GPU.
    

**2. RL Checkpoint 1 (Initial Training Verification):**

- [ ] Start the training loop.
    
- [ ] Monitor the first 10k steps on TensorBoard.
    
- [ ] **Crucial:** Look at `ep_rew_mean`. It MUST show an upward trend.
    
- [ ] _Troubleshooting:_ If the reward curve is completely flat, stop the run. You have a reward function bug. Fix it before proceeding.
    