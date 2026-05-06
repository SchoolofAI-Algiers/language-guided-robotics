import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import os

try:
    from robotics.env.src.config import (
        GraphicalMode, NUM_JOINTS, END_EFFECTOR_LINK_INDEX,
        MAX_FORCE, SIM_TIMESTEP, SIM_STEPS_PER_ACTION,
        MAX_EPISODE_STEPS, JOINT_LOWER_LIMITS, JOINT_UPPER_LIMITS,
        HOME_POSITION, MAX_JOINT_VELOCITY, WORKSPACE_LOW, WORKSPACE_HIGH,
        CAM_DISTANCE, CAM_YAW, CAM_PITCH, CAM_TARGET,
        RENDER_WIDTH, RENDER_HEIGHT, RENDER_FPS, RenderMode,
        NUM_OBJECTS, OBJECT_COLORS, OBJECT_SHAPES,
        OBJECT_SIZE_MIN, OBJECT_SIZE_MAX,
        TABLE_POSITION, TABLE_HALF_EXTENTS, TABLE_SURFACE_Z, SPAWN_RANGE,
        GRASP_DISTANCE,
    )
except ModuleNotFoundError:
    from env.src.config import (
        GraphicalMode, NUM_JOINTS, END_EFFECTOR_LINK_INDEX,
        MAX_FORCE, SIM_TIMESTEP, SIM_STEPS_PER_ACTION,
        MAX_EPISODE_STEPS, JOINT_LOWER_LIMITS, JOINT_UPPER_LIMITS,
        HOME_POSITION, MAX_JOINT_VELOCITY, WORKSPACE_LOW, WORKSPACE_HIGH,
        CAM_DISTANCE, CAM_YAW, CAM_PITCH, CAM_TARGET,
        RENDER_WIDTH, RENDER_HEIGHT, RENDER_FPS, RenderMode,
        NUM_OBJECTS, OBJECT_COLORS, OBJECT_SHAPES,
        OBJECT_SIZE_MIN, OBJECT_SIZE_MAX,
        TABLE_POSITION, TABLE_HALF_EXTENTS, TABLE_SURFACE_Z, SPAWN_RANGE,
        GRASP_DISTANCE,
    )


class KukaEnv(gym.Env):
    """Merged Kuka IIWA environment with:
    - Dict observation: {pixels, state, object_state}
    - Multi-object scene with randomized positions / colors / shapes
    - Magnetic gripper (constraint-based snap)
    - 9-dim target object state: [x,y,z, vx,vy,vz, roll,pitch,yaw]
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": RENDER_FPS}

    def __init__(self, render_mode=RenderMode.RGB_ARRAY.value):
        super().__init__()
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(NUM_JOINTS,), dtype=np.float32
        )

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
        obj_low = np.full(9, -10.0, dtype=np.float32)
        obj_high = np.full(9, 10.0, dtype=np.float32)

        self.observation_space = spaces.Dict({
            "pixels": spaces.Box(
                low=0, high=255,
                shape=(RENDER_HEIGHT, RENDER_WIDTH, 3), dtype=np.uint8
            ),
            "state": spaces.Box(
                low=state_low, high=state_high, shape=(21,), dtype=np.float32
            ),
            "object_state": spaces.Box(
                low=obj_low, high=obj_high, shape=(9,), dtype=np.float32
            ),
        })

        self._physics_client_id = -1
        self._kuka_id = None
        self._plane_id = None
        self._step_count = 0
        self._object_ids = []
        self._object_colors = []
        self._object_shapes = []
        self._target_id = None
        self._grasp_constraint = None
        self._grasped_object_id = None

    def _load_table(self):
        col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=TABLE_HALF_EXTENTS,
            physicsClientId=self._physics_client_id,
        )
        vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=TABLE_HALF_EXTENTS,
            rgbaColor=[0.6, 0.4, 0.2, 1.0],
            physicsClientId=self._physics_client_id,
        )
        p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis, basePosition=TABLE_POSITION,
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
            size = float(self.np_random.uniform(OBJECT_SIZE_MIN, OBJECT_SIZE_MAX))
            color_name = color_names[i % len(color_names)]
            rgba = OBJECT_COLORS[color_name]
            shape_name = OBJECT_SHAPES[
                int(self.np_random.integers(0, len(OBJECT_SHAPES)))
            ]

            if shape_name == "box":
                col = p.createCollisionShape(
                    p.GEOM_BOX, halfExtents=[size, size, size],
                    physicsClientId=self._physics_client_id,
                )
                vis = p.createVisualShape(
                    p.GEOM_BOX, halfExtents=[size, size, size],
                    rgbaColor=rgba, physicsClientId=self._physics_client_id,
                )
            elif shape_name == "sphere":
                col = p.createCollisionShape(
                    p.GEOM_SPHERE, radius=size,
                    physicsClientId=self._physics_client_id,
                )
                vis = p.createVisualShape(
                    p.GEOM_SPHERE, radius=size,
                    rgbaColor=rgba, physicsClientId=self._physics_client_id,
                )
            elif shape_name == "cylinder":
                col = p.createCollisionShape(
                    p.GEOM_CYLINDER, radius=size, height=size * 2,
                    physicsClientId=self._physics_client_id,
                )
                vis = p.createVisualShape(
                    p.GEOM_CYLINDER, radius=size, length=size * 2,
                    rgbaColor=rgba, physicsClientId=self._physics_client_id,
                )

            for _ in range(50):
                ox = float(self.np_random.uniform(-SPAWN_RANGE, SPAWN_RANGE))
                oy = float(self.np_random.uniform(-SPAWN_RANGE, SPAWN_RANGE))
                candidate = [table_cx + ox, table_cy + oy]
                too_close = any(
                    np.sqrt((candidate[0] - px)**2 + (candidate[1] - py)**2)
                    < MIN_DISTANCE
                    for px, py in placed_positions
                )
                if not too_close:
                    break

            position = [candidate[0], candidate[1], TABLE_SURFACE_Z]
            placed_positions.append(candidate)

            obj_id = p.createMultiBody(
                baseMass=0.1, baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis, basePosition=position,
                physicsClientId=self._physics_client_id,
            )
            self._object_ids.append(obj_id)
            self._object_colors.append(color_name)
            self._object_shapes.append(shape_name)

        self._target_id = int(
            self.np_random.integers(0, len(self._object_ids))
        )

    def _get_target_state(self):
        if self._target_id is not None and self._target_id < len(self._object_ids):
            pos, orn = p.getBasePositionAndOrientation(
                self._object_ids[self._target_id],
                physicsClientId=self._physics_client_id,
            )
            vel, ang_vel = p.getBaseVelocity(
                self._object_ids[self._target_id],
                physicsClientId=self._physics_client_id,
            )
            roll, pitch, yaw = p.getEulerFromQuaternion(orn)
            return np.array(
                list(pos) + list(vel) + [roll, pitch, yaw],
                dtype=np.float32,
            )
        return np.zeros(9, dtype=np.float32)

    def _get_all_objects_info(self):
        state = {}
        for obj_id, color, shape in zip(
            self._object_ids, self._object_colors, self._object_shapes
        ):
            pos, _ = p.getBasePositionAndOrientation(
                obj_id, physicsClientId=self._physics_client_id,
            )
            state[obj_id] = {"pos": list(pos), "color": color, "shape": shape}
        return state

    def _try_grasp(self):
        if self._grasp_constraint is not None:
            return
        ee_state = p.getLinkState(
            self._kuka_id, END_EFFECTOR_LINK_INDEX,
            physicsClientId=self._physics_client_id,
        )
        ee_pos = np.array(ee_state[0])
        for obj_id in self._object_ids:
            obj_pos, _ = p.getBasePositionAndOrientation(
                obj_id, physicsClientId=self._physics_client_id,
            )
            if np.linalg.norm(ee_pos - np.array(obj_pos)) < GRASP_DISTANCE:
                self._grasp_constraint = p.createConstraint(
                    self._kuka_id, END_EFFECTOR_LINK_INDEX,
                    obj_id, -1,
                    p.JOINT_FIXED, [0,0,0], [0,0,0], [0,0,0],
                    physicsClientId=self._physics_client_id,
                )
                p.changeConstraint(
                    self._grasp_constraint, maxForce=500,
                    physicsClientId=self._physics_client_id,
                )
                self._grasped_object_id = obj_id
                break

    def _release(self):
        if self._grasp_constraint is not None:
            p.removeConstraint(
                self._grasp_constraint,
                physicsClientId=self._physics_client_id,
            )
            self._grasp_constraint = None
            self._grasped_object_id = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self._physics_client_id < 0:
            mode = GraphicalMode.GUI if self.render_mode == "human" else GraphicalMode.DIRECT
            self._physics_client_id = p.connect(mode.value)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.resetSimulation(physicsClientId=self._physics_client_id)
        p.setGravity(0, 0, -10, physicsClientId=self._physics_client_id)
        p.setTimeStep(SIM_TIMESTEP, physicsClientId=self._physics_client_id)

        self._plane_id = p.loadURDF("plane.urdf", physicsClientId=self._physics_client_id)
        self._kuka_id = p.loadURDF(
            "kuka_iiwa/model.urdf", basePosition=[0, 0, 0],
            useFixedBase=True, physicsClientId=self._physics_client_id,
        )

        for i in range(NUM_JOINTS):
            p.resetJointState(
                self._kuka_id, i, HOME_POSITION[i],
                physicsClientId=self._physics_client_id,
            )

        self._load_table()
        self._load_objects()

        self._step_count = 0
        self._grasp_constraint = None
        self._grasped_object_id = None

        for _ in range(SIM_STEPS_PER_ACTION):
            p.stepSimulation(physicsClientId=self._physics_client_id)

        obs = self._get_observation()
        info = {
            "step_count": self._step_count,
            "ee_position": obs["state"][14:17].tolist(),
            "target_object_id": self._target_id,
            "all_objects": self._get_all_objects_info(),
        }
        return obs, info

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        midpoint = (JOINT_UPPER_LIMITS + JOINT_LOWER_LIMITS) / 2.0
        half_range = (JOINT_UPPER_LIMITS - JOINT_LOWER_LIMITS) / 2.0
        action = midpoint + action * half_range

        for i in range(NUM_JOINTS):
            p.setJointMotorControl2(
                bodyUniqueId=self._kuka_id, jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=float(action[i]),
                force=MAX_FORCE,
                physicsClientId=self._physics_client_id,
            )

        for _ in range(SIM_STEPS_PER_ACTION):
            p.stepSimulation(physicsClientId=self._physics_client_id)

        self._step_count += 1
        self._try_grasp()

        observation = self._get_observation()
        reward = 0.0
        terminated = False
        truncated = self._step_count >= MAX_EPISODE_STEPS

        info = {
            "step_count": self._step_count,
            "ee_position": observation["state"][14:17].tolist(),
            "target_object_id": self._target_id,
            "all_objects": self._get_all_objects_info(),
        }
        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        joint_positions = np.zeros(NUM_JOINTS, dtype=np.float32)
        joint_velocities = np.zeros(NUM_JOINTS, dtype=np.float32)
        for i in range(NUM_JOINTS):
            state = p.getJointState(
                self._kuka_id, i, physicsClientId=self._physics_client_id,
            )
            joint_positions[i] = state[0]
            joint_velocities[i] = state[1]

        ee_state = p.getLinkState(
            self._kuka_id, END_EFFECTOR_LINK_INDEX,
            physicsClientId=self._physics_client_id,
        )
        ee_position = np.array(ee_state[0], dtype=np.float32)
        ee_orientation = np.array(ee_state[1], dtype=np.float32)

        physics_state = np.clip(
            np.concatenate([joint_positions, joint_velocities, ee_position, ee_orientation]),
            self.observation_space["state"].low,
            self.observation_space["state"].high,
        )

        return {
            "pixels": self._get_camera_image(),
            "state": physics_state,
            "object_state": self._get_target_state(),
        }

    def render(self):
        if self.render_mode == "rgb_array":
            return self._get_camera_image()
        return None

    def close(self):
        if self._physics_client_id >= 0:
            p.disconnect(physicsClientId=self._physics_client_id)
            self._physics_client_id = -1

    def _get_camera_image(self):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=CAM_TARGET, distance=CAM_DISTANCE,
            yaw=CAM_YAW, pitch=CAM_PITCH, roll=0, upAxisIndex=2,
            physicsClientId=self._physics_client_id,
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=RENDER_WIDTH / RENDER_HEIGHT,
            nearVal=0.1, farVal=100,
            physicsClientId=self._physics_client_id,
        )
        _, _, rgb, _, _ = p.getCameraImage(
            RENDER_WIDTH, RENDER_HEIGHT, view_matrix, proj_matrix,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self._physics_client_id,
        )
        return np.array(rgb, dtype=np.uint8).reshape(
            RENDER_HEIGHT, RENDER_WIDTH, 4
        )[:, :, :3]