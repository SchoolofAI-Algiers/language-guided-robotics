# Language-Conditioned RL Module 

This directory isolates the Reinforcement Learning responsibilities from the broader Robotics, NLP, and Vision modules.

## Architecture

The RL component interacts with the continuous-action Gymnasium environment via Stable-Baselines3. We built a custom multi-modal feature extractor for the observation dictionaries produced by merging the NLP team's instruction embeddings and the Vision team's visual features + state tracker.

### Feature Extractor

`feature_extractor.py` defines the `LanguageConditionedFeatureExtractor` based on SB3's `BaseFeaturesExtractor`.
- **Vision branch:** Projects `(521,)` vector (ResNet18 CNN features `(512,)` + PyBullet object state `(9,)`) through a dedicated Linear → LayerNorm → ReLU head.
- **NLP branch:** Projects `(384,)` instruction embedding from `all-MiniLM-L6-v2` through a dedicated Linear → LayerNorm → ReLU head.
- **Fusion:** Concatenates both branch outputs and passes through a shared Linear → LayerNorm → ReLU layer to produce the final `(features_dim,)` representation.

This two-branch architecture lets each modality learn its own representation before fusion, and LayerNorm stabilises training when input magnitudes differ.

### Integration Wrapper

`env_wrapper.py` implements a `LanguageConditionedWrapper(gym.ObservationWrapper)` designed to hook seamlessly onto the `KukaEnv` from the Robotics Track.
- Automatically connects PyBullet observations to identical representations provided by the Vision Track.
- Dynamically loads `Elbatoul-NLP-W1-instruction-embeddings/embeddings_125.npy` and `.csv` from the NLP Track.
- Randomizes instructions per episode and aligns real-time simulation output.

### Reward Shaping

`reward_shaping.py` implements a `RewardShapingWrapper(gym.Wrapper)` that adds dense reward on top of the base environment (which returns `reward=0.0`):
- **Potential-based shaping:** `reward += (prev_dist - curr_dist)` — positive when the end-effector gets closer to a random target.
- **Time penalty:** `reward -= 0.001` per step — encourages efficient trajectories.
- **Success bonus:** `reward += 1.0` when the end-effector reaches within 5 cm of the target (episode terminates).
- Target is re-sampled each episode within the robot's reachable workspace.



## How to execute

> **Must be run from the project root**, not from inside `rl/`:

```bash
# from language-guided-robotics/
python -m rl.train
```

### TensorBoard Logging

Training metrics (episode reward, episode length, policy loss, value loss) are logged into `./logs/`.
To start TensorBoard:

```bash
tensorboard --logdir=logs
```

## Dependencies (RL-specific)

These are now part of the root `requirements.txt`:
- `stable-baselines3` — PPO algorithm
- `tensorboard` — training dashboard
- `pandas` — loads `nlp_instructions_125.csv` in the wrapper

