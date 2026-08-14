import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data

from env.src.config import GraphicalMode, NUM_JOINTS, END_EFFECTOR_LINK_INDEX, MAX_FORCE, SIM_TIMESTEP, SIM_STEPS_PER_ACTION, MAX_EPISODE_STEPS, JOINT_LOWER_LIMITS, JOINT_UPPER_LIMITS, HOME_POSITION, MAX_JOINT_VELOCITY, WORKSPACE_LOW, WORKSPACE_HIGH, CAM_DISTANCE, CAM_YAW, CAM_PITCH, CAM_TARGET, RENDER_WIDTH, RENDER_HEIGHT, RENDER_FPS, RenderMode
from env.src.config import NUM_OBJECTS, OBJECT_COLORS, OBJECT_SIZE_MIN, OBJECT_SIZE_MAX,TABLE_POSITION, TABLE_HALF_EXTENTS, TABLE_SURFACE_Z, SPAWN_RANGE , OBJECT_SHAPES


class KukaEnv(gym.Env):
    """Gymnasium-compatible environment for the Kuka IIWA 7-DOF robot arm.

     Phase 2 additions:
    - Multi-object scene (3 colored blocks, randomized on every reset)
    - Object state exposed via info["object_state"]:
        {obj_id: {"pos": [x, y, z], "color": str}}

    Action:      7-dim joint position targets (clipped to joint limits)
    Observation:  21-dim vector = [joint_pos(7), joint_vel(7),
                                   ee_position(3), ee_orientation(4)]
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": RENDER_FPS}

    def __init__(self, render_mode=RenderMode.RGB_ARRAY.value):
        super().__init__()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Normalized action space: policy outputs in [-1, 1], scaled to joint limits in step()
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(NUM_JOINTS,), dtype=np.float32
        )

        # Finite observation bounds: [joint_pos(7), joint_vel(7), ee_pos(3), ee_orn(4)]
        obs_low = np.concatenate([
            JOINT_LOWER_LIMITS,
            np.full(NUM_JOINTS, -MAX_JOINT_VELOCITY, dtype=np.float32),
            WORKSPACE_LOW,
            np.full(4, -1.0, dtype=np.float32),
        ])
        obs_high = np.concatenate([
            JOINT_UPPER_LIMITS,
            np.full(NUM_JOINTS, MAX_JOINT_VELOCITY, dtype=np.float32),
            WORKSPACE_HIGH,
            np.full(4, 1.0, dtype=np.float32),
        ])
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, shape=(21,), dtype=np.float32
        )

        self._physics_client_id = -1
        self._kuka_id = None
        self._plane_id = None
        self._step_count = 0

        # object tracking 
        self._object_ids = []      # list of PyBullet object IDs
        self._object_colors = []   # list of color name strings

    # private helpers

    def _load_table(self):
        """Load a static table in front of the robot."""
        col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=TABLE_HALF_EXTENTS,
            physicsClientId=self._physics_client_id,
        )
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=TABLE_HALF_EXTENTS,
            rgbaColor=[0.6, 0.4, 0.2, 1.0],
            physicsClientId=self._physics_client_id,
        )
        p.createMultiBody(
            baseMass=0,  # static ΓÇö won't move
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=TABLE_POSITION,
            physicsClientId=self._physics_client_id,
        )

    def _load_objects(self):
        self._object_ids = []
        self._object_colors = []
        self._object_shapes = []

        color_names = list(OBJECT_COLORS.keys())
        table_cx, table_cy = TABLE_POSITION[0], TABLE_POSITION[1]
        MIN_DISTANCE = 0.10
        placed_positions = []

        for i in range(NUM_OBJECTS):
            size  = float(self.np_random.uniform(OBJECT_SIZE_MIN, OBJECT_SIZE_MAX))
            color_name = color_names[i % len(color_names)]
            rgba  = OBJECT_COLORS[color_name]

            # pick a random shape for this object
            shape_name = OBJECT_SHAPES[ int(self.np_random.integers(0, len(OBJECT_SHAPES))) ]

            if shape_name == "box":
                col = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=[size, size, size],
                    physicsClientId=self._physics_client_id,
                )
                vis = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[size, size, size],
                    rgbaColor=rgba,
                    physicsClientId=self._physics_client_id,
                )

            elif shape_name == "sphere":
                col = p.createCollisionShape(
                    p.GEOM_SPHERE,
                    radius=size,
                    physicsClientId=self._physics_client_id,
                )
                vis = p.createVisualShape(
                    p.GEOM_SPHERE,
                    radius=size,
                    rgbaColor=rgba,
                    physicsClientId=self._physics_client_id,
                )

            elif shape_name == "cylinder":
                col = p.createCollisionShape(
                    p.GEOM_CYLINDER,
                    radius=size,
                    height=size * 2,
                    physicsClientId=self._physics_client_id,
                )
                vis = p.createVisualShape(
                    p.GEOM_CYLINDER,
                    radius=size,
                    length=size * 2,
                    rgbaColor=rgba,
                    physicsClientId=self._physics_client_id,
                )

        
            # Find a valid position 
            for _ in range(50):
                ox = float(self.np_random.uniform(-SPAWN_RANGE, SPAWN_RANGE))
                oy = float(self.np_random.uniform(-SPAWN_RANGE, SPAWN_RANGE))
                candidate = [table_cx + ox, table_cy + oy]
                too_close = any(
                    np.sqrt((candidate[0] - p_[0])**2 + (candidate[1] - p_[1])**2) < MIN_DISTANCE
                    for p_ in placed_positions
                )
                if not too_close:
                    break

            position = [candidate[0], candidate[1], TABLE_SURFACE_Z]
            placed_positions.append(candidate)

            obj_id = p.createMultiBody(
                baseMass=0.1,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=position,
                physicsClientId=self._physics_client_id,
            )

            self._object_ids.append(obj_id)
            self._object_colors.append(color_name)
            self._object_shapes.append(shape_name)

    def _get_object_state(self):
        """
        Returns the object state dict the Vision team needs:
            {obj_id: {"pos": [x, y, z], "color": str}}
        """
        state = {}
        for obj_id, color, shape in zip(
            self._object_ids, self._object_colors, self._object_shapes
        ):
            pos, _ = p.getBasePositionAndOrientation(
                obj_id, physicsClientId=self._physics_client_id
            )
            state[obj_id] = {
                "pos":   list(pos),
                "color": color,
                "shape": shape,      
            }
        return state

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
        
        #load table and objects 
        self._load_table()
        self._load_objects()

        self._step_count = 0

        for _ in range(SIM_STEPS_PER_ACTION):
            p.stepSimulation(physicsClientId=self._physics_client_id)

        obs = self._get_observation()
        info = {
            "step_count": self._step_count,
            "ee_position": obs[14:17].tolist(),
            "object_state": self._get_object_state(),  
        }
        return obs, info

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
        info = {
            "step_count": self._step_count,
            "ee_position": observation[14:17].tolist(),
            "object_state": self._get_object_state(), 
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

        obs = np.concatenate([
            joint_positions,
            joint_velocities,
            ee_position,
            ee_orientation,
        ])
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

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
