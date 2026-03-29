import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import sys
import os

# Add project root to sys.path to allow importing from vision module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from vision.vision_pipeline import resnet_features

try:
    from robotics.env.src.config import GraphicalMode, NUM_JOINTS, END_EFFECTOR_LINK_INDEX, MAX_FORCE, SIM_TIMESTEP, SIM_STEPS_PER_ACTION, MAX_EPISODE_STEPS, JOINT_LOWER_LIMITS, JOINT_UPPER_LIMITS, HOME_POSITION, MAX_JOINT_VELOCITY, WORKSPACE_LOW, WORKSPACE_HIGH, CAM_DISTANCE, CAM_YAW, CAM_PITCH, CAM_TARGET, RENDER_WIDTH, RENDER_HEIGHT, RENDER_FPS, RenderMode
except ModuleNotFoundError:
    from env.src.config import GraphicalMode, NUM_JOINTS, END_EFFECTOR_LINK_INDEX, MAX_FORCE, SIM_TIMESTEP, SIM_STEPS_PER_ACTION, MAX_EPISODE_STEPS, JOINT_LOWER_LIMITS, JOINT_UPPER_LIMITS, HOME_POSITION, MAX_JOINT_VELOCITY, WORKSPACE_LOW, WORKSPACE_HIGH, CAM_DISTANCE, CAM_YAW, CAM_PITCH, CAM_TARGET, RENDER_WIDTH, RENDER_HEIGHT, RENDER_FPS, RenderMode

class KukaEnv(gym.Env):
    """Gymnasium-compatible environment for the Kuka IIWA 7-DOF robot arm.

    Action:      7-dim joint position targets (clipped to joint limits)
    Observation: Variable depending on `obs_mode`.
                 - 'state': 21-dim vector = [joint_pos(7), joint_vel(7), ee_position(3), ee_orientation(4)]
                 - 'visual_only': 512-dim CNN features
                 - 'visual_joints': 533-dim vector = 512-dim CNN features + 21-dim state
                 - 'visual_statepybullet': 521-dim vector = 512-dim CNN features + 9-dim pybullet object state
                 - 'visual_joints_statepybullet': 542-dim vector = 512-dim CNN features + 21-dim state + 9-dim pybullet object state
                 - 'pixels': Image observation (H, W, 3) + 21-dim state (requires Dict space)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": RENDER_FPS}

    def __init__(self, render_mode=RenderMode.RGB_ARRAY.value, obs_mode="visual_joints_statepybullet"):
        super().__init__()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        # Determine observation mode ("state", "visual_only", "visual_state", or "pixels")
        self.obs_mode = obs_mode


        # Normalized action space: policy outputs in [-1, 1], scaled to joint limits in step()
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(NUM_JOINTS,), dtype=np.float32
        )

        # Finite observation bounds: [joint_pos(7), joint_vel(7), ee_pos(3), ee_orn(4)]
        state_low = np.concatenate([
            JOINT_LOWER_LIMITS,
            np.full(NUM_JOINTS, -MAX_JOINT_VELOCITY, dtype=np.float32),
            WORKSPACE_LOW,
            np.full(4, -1.0, dtype=np.float32),
        ])
        state_high = np.concatenate([
            JOINT_UPPER_LIMITS,
            np.full(NUM_JOINTS, MAX_JOINT_VELOCITY, dtype=np.float32),
            WORKSPACE_HIGH,
            np.full(4, 1.0, dtype=np.float32),
        ])

        # 9-dim pybullet object state bounds: [x,y,z, vx,vy,vz, roll,pitch,yaw]
        obj_low = np.full(9, -10.0, dtype=np.float32)
        obj_high = np.full(9, 10.0, dtype=np.float32)

        feat_low = np.full(512, -1.0, dtype=np.float32)
        feat_high = np.full(512, 1.0, dtype=np.float32)

        # Adjust the observation space based on the selected mode
        if self.obs_mode == "state":
            # Pure physics state (21 dimensions). Suitable for state-based RL baselines.
            self.observation_space = spaces.Box(
                low=state_low, high=state_high, shape=(21,), dtype=np.float32
            )
        elif self.obs_mode == "visual_only":
            # Only Visual features (512 dims)
            self.observation_space = spaces.Box(
                low=feat_low, high=feat_high, shape=(512,), dtype=np.float32
            )
        elif self.obs_mode == "visual_joints" or self.obs_mode == "visual_state" or self.obs_mode == "visual":
            # CNN Visual features (512 dims) concatenated with physics state (21 dims) -> 533 dims
            obs_low = np.concatenate([feat_low, state_low])
            obs_high = np.concatenate([feat_high, state_high])

            self.observation_space = spaces.Box(
                low=obs_low, high=obs_high, shape=(533,), dtype=np.float32
            )
        elif self.obs_mode == "visual_statepybullet":
            # CNN Visual features (512 dims) + PyBullet Target Object State (9 dims) -> 521 dims
            obs_low = np.concatenate([feat_low, obj_low])
            obs_high = np.concatenate([feat_high, obj_high])

            self.observation_space = spaces.Box(
                low=obs_low, high=obs_high, shape=(521,), dtype=np.float32
            )
        elif self.obs_mode == "visual_joints_statepybullet":
            # CNN (512) + Arm State (21) + Object State (9) -> 542 dims
            obs_low = np.concatenate([feat_low, state_low, obj_low])
            obs_high = np.concatenate([feat_high, state_high, obj_high])

            self.observation_space = spaces.Box(
                low=obs_low, high=obs_high, shape=(542,), dtype=np.float32
            )
        elif self.obs_mode == "pixels":
            # Dictionary space containing raw normalized/unnormalized image pixels and physics state.
            self.observation_space = spaces.Dict({
                "pixels": spaces.Box(low=0, high=255, shape=(RENDER_HEIGHT, RENDER_WIDTH, 3), dtype=np.uint8),
                "state": spaces.Box(low=state_low, high=state_high, shape=(21,), dtype=np.float32)
            })
        else:
            raise ValueError(f"Unknown obs_mode: {self.obs_mode}. Choose from 'state', 'visual_only', 'visual_state', or 'pixels'.")

        self._physics_client_id = -1
        self._kuka_id = None
        self._plane_id = None
        self._step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self._physics_client_id < 0:
            mode = (
                GraphicalMode.GUI
                if self.render_mode == "human"
                else GraphicalMode.DIRECT
            )
            self._physics_client_id = p.connect(mode.value)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.resetSimulation(physicsClientId=self._physics_client_id)
        p.setGravity(0, 0, -10, physicsClientId=self._physics_client_id)
        p.setTimeStep(SIM_TIMESTEP, physicsClientId=self._physics_client_id)

        self._plane_id = p.loadURDF(
            "plane.urdf", physicsClientId=self._physics_client_id
        )
        self._kuka_id = p.loadURDF(
            "kuka_iiwa/model.urdf",
            basePosition=[0, 0, 0],
            useFixedBase=True,
            physicsClientId=self._physics_client_id,
        )

        for i in range(NUM_JOINTS):
            p.resetJointState(
                self._kuka_id,
                i,
                HOME_POSITION[i],
                physicsClientId=self._physics_client_id,
            )

        self._step_count = 0

        for _ in range(SIM_STEPS_PER_ACTION):
            p.stepSimulation(physicsClientId=self._physics_client_id)

        return self._get_observation(), {}

    def step(self, action):
        # Scale normalized [-1, 1] action to joint position limits
        action = np.clip(action, -1.0, 1.0)
        midpoint = (JOINT_UPPER_LIMITS + JOINT_LOWER_LIMITS) / 2.0
        half_range = (JOINT_UPPER_LIMITS - JOINT_LOWER_LIMITS) / 2.0
        action = midpoint + action * half_range

        for i in range(NUM_JOINTS):
            p.setJointMotorControl2(
                bodyUniqueId=self._kuka_id,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=float(action[i]),
                force=MAX_FORCE,
                physicsClientId=self._physics_client_id,
            )

        for _ in range(SIM_STEPS_PER_ACTION):
            p.stepSimulation(physicsClientId=self._physics_client_id)

        self._step_count += 1

        observation = self._get_observation()
        reward = 0.0
        terminated = False
        truncated = self._step_count >= MAX_EPISODE_STEPS
        
        # Get true physics state to extract ee_position for the info dict
        if self.obs_mode == "pixels":
            ee_pos = observation["state"][14:17].tolist()
        elif self.obs_mode == "visual_joints":
            ee_pos = observation[512+14:512+17].tolist()
        elif self.obs_mode == "visual_joints_statepybullet":
            ee_pos = observation[512+14:512+17].tolist()
        elif self.obs_mode == "visual_only" or self.obs_mode == "visual_statepybullet":
            # For visual only or modes without arm state, we compute it dynamically
            ee_state = p.getLinkState(self._kuka_id, END_EFFECTOR_LINK_INDEX, physicsClientId=self._physics_client_id)
            ee_pos = list(ee_state[0])
        elif self.obs_mode == "state":
            ee_pos = observation[14:17].tolist()
        else:
            # Fallback
            ee_state = p.getLinkState(self._kuka_id, END_EFFECTOR_LINK_INDEX, physicsClientId=self._physics_client_id)
            ee_pos = list(ee_state[0])
            
        info = {
            "step_count": self._step_count,
            "ee_position": ee_pos,
        }

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._get_camera_image()
        return None

    def close(self):
        if self._physics_client_id >= 0:
            p.disconnect(physicsClientId=self._physics_client_id)
            self._physics_client_id = -1

    def _get_observation(self):
        joint_positions = np.zeros(NUM_JOINTS, dtype=np.float32)
        joint_velocities = np.zeros(NUM_JOINTS, dtype=np.float32)

        for i in range(NUM_JOINTS):
            state = p.getJointState(
                self._kuka_id, i, physicsClientId=self._physics_client_id
            )
            joint_positions[i] = state[0]
            joint_velocities[i] = state[1]

        ee_state = p.getLinkState(
            self._kuka_id,
            END_EFFECTOR_LINK_INDEX,
            physicsClientId=self._physics_client_id,
        )
        ee_position = np.array(ee_state[0], dtype=np.float32)
        ee_orientation = np.array(ee_state[1], dtype=np.float32)

        # Assemble the raw physics state (21-dimensional vector)
        physics_state = np.concatenate([
            joint_positions,
            joint_velocities,
            ee_position,
            ee_orientation,
        ])
        
        # We need to compute appropriate bounds for the physics state ONLY to clip it
        # (self.observation_space.low/high might include visual features, so we can't use them directly here)
        state_low = np.concatenate([
            JOINT_LOWER_LIMITS,
            np.full(NUM_JOINTS, -MAX_JOINT_VELOCITY, dtype=np.float32),
            WORKSPACE_LOW,
            np.full(4, -1.0, dtype=np.float32),
        ])
        state_high = np.concatenate([
            JOINT_UPPER_LIMITS,
            np.full(NUM_JOINTS, MAX_JOINT_VELOCITY, dtype=np.float32),
            WORKSPACE_HIGH,
            np.full(4, 1.0, dtype=np.float32),
        ])
        
        physics_state = np.clip(physics_state, state_low, state_high)

        # Mock Object State since we don't have objects yet:
        # Ground truth [x, y, z, vx, vy, vz, roll, pitch, yaw]
        # TODO: Replace this with real physics queries when target blocks are spawned
        object_state = np.zeros(9, dtype=np.float32)

        # Determine the final observation based on the requested obs_mode
        if self.obs_mode == "state":
            # Return just the physics state
            return physics_state

        elif self.obs_mode == "visual_only":
            img = self._get_camera_image()
            import torch
            with torch.no_grad():
                vision_tensor = resnet_features([img]).cpu().numpy()[0]
            return vision_tensor

        elif self.obs_mode == "visual_joints" or self.obs_mode == "visual_state" or self.obs_mode == "visual":
            img = self._get_camera_image()
            import torch
            with torch.no_grad():
                vision_tensor = resnet_features([img]).cpu().numpy()[0]
            return np.concatenate([vision_tensor, physics_state])

        elif self.obs_mode == "visual_statepybullet":
            img = self._get_camera_image()
            import torch
            with torch.no_grad():
                vision_tensor = resnet_features([img]).cpu().numpy()[0]
            return np.concatenate([vision_tensor, object_state])

        elif self.obs_mode == "visual_joints_statepybullet":
            img = self._get_camera_image()
            import torch
            with torch.no_grad():
                vision_tensor = resnet_features([img]).cpu().numpy()[0]
            return np.concatenate([vision_tensor, physics_state, object_state])

        elif self.obs_mode == "pixels":
            # Return dictionary containing raw pixels and physics state
            img = self._get_camera_image()
            return {
                "pixels": img,
                "state": physics_state
            }

    def _get_camera_image(self):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=CAM_TARGET,
            distance=CAM_DISTANCE,
            yaw=CAM_YAW,
            pitch=CAM_PITCH,
            roll=0,
            upAxisIndex=2,
            physicsClientId=self._physics_client_id,
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=RENDER_WIDTH / RENDER_HEIGHT,
            nearVal=0.1,
            farVal=100,
            physicsClientId=self._physics_client_id,
        )
        _, _, rgb, _, _ = p.getCameraImage(
            RENDER_WIDTH,
            RENDER_HEIGHT,
            view_matrix,
            proj_matrix,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self._physics_client_id,
        )
        return np.array(rgb, dtype=np.uint8).reshape(
            RENDER_HEIGHT, RENDER_WIDTH, 4
        )[:, :, :3]
