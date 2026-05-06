import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data

from env.src.config import (
    GraphicalMode,
    NUM_JOINTS,
    END_EFFECTOR_LINK_INDEX,
    MAX_FORCE,
    SIM_TIMESTEP,
    SIM_STEPS_PER_ACTION,
    MAX_EPISODE_STEPS,
    JOINT_LOWER_LIMITS,
    JOINT_UPPER_LIMITS,
    HOME_POSITION,
    MAX_JOINT_VELOCITY,
    WORKSPACE_LOW,
    WORKSPACE_HIGH,
    CAM_DISTANCE,
    CAM_YAW,
    CAM_PITCH,
    CAM_TARGET,
    RENDER_WIDTH,
    RENDER_HEIGHT,
    RENDER_FPS,
    RenderMode,
    NUM_OBJECTS,
    OBJECT_COLORS,
    OBJECT_SHAPES,
    OBJECT_SIZE_MIN,
    OBJECT_SIZE_MAX,
    TABLE_POSITION,
    TABLE_HALF_EXTENTS,
    TABLE_SURFACE_Z,
    SPAWN_RANGE,
)

GRASP_DISTANCE = 0.15  # meters — how close EE must be to snap to object

class KukaEnv(gym.Env):
    """
    Gymnasium-compatible environment for the Kuka IIWA 7-DOF robot arm.

    Phase 1: Basic arm + Gymnasium wrapper
    Phase 2: Multi-object scene with randomized positions/colors/shapes
    Phase 3: Magnetic gripper — constraint-based grasping (Option C)

    Action:      7-dim joint position targets (normalized [-1, 1])
    Observation: 22-dim vector =
                 [joint_pos(7), joint_vel(7),
                  ee_position(3), ee_orientation(4), gripper_state(1)]

    gripper_state: 1.0 = holding object, 0.0 = empty
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": RENDER_FPS}

    def __init__(self, render_mode=RenderMode.RGB_ARRAY.value):
        super().__init__()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Action space — 7 joints normalized
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(NUM_JOINTS,), dtype=np.float32
        )

        # Observation space — 22 dims (21 original + 1 gripper state)
        obs_low = np.concatenate([
            JOINT_LOWER_LIMITS,
            np.full(NUM_JOINTS, -MAX_JOINT_VELOCITY, dtype=np.float32),
            WORKSPACE_LOW,
            np.full(4, -1.0, dtype=np.float32),
            np.array([0.0], dtype=np.float32),  # gripper state min
        ])
        obs_high = np.concatenate([
            JOINT_UPPER_LIMITS,
            np.full(NUM_JOINTS, MAX_JOINT_VELOCITY, dtype=np.float32),
            WORKSPACE_HIGH,
            np.full(4, 1.0, dtype=np.float32),
            np.array([1.0], dtype=np.float32),  # gripper state max
        ])
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, shape=(22,), dtype=np.float32
        )

        # Physics state
        self._physics_client_id = -1
        self._kuka_id = None
        self._plane_id = None
        self._step_count = 0

        # Object tracking
        self._object_ids = []
        self._object_colors = []
        self._object_shapes = []

        # Gripper state
        self._grasp_constraint = None
        self._grasped_object_id = None

    # ─── TABLE ────────────────────────────────────────────────────────────────

    def _load_table(self):
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
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=TABLE_POSITION,
            physicsClientId=self._physics_client_id,
        )

    # ─── OBJECTS ──────────────────────────────────────────────────────────────

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
            shape_name = OBJECT_SHAPES[int(self.np_random.integers(0, len(OBJECT_SHAPES)))]

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

            # Anti-overlap placement
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

    # ─── OBJECT STATE ─────────────────────────────────────────────────────────

    def _get_object_state(self):
        state = {}
        for obj_id, color, shape in zip(
            self._object_ids, self._object_colors, self._object_shapes
        ):
            pos, _ = p.getBasePositionAndOrientation(
                obj_id, physicsClientId=self._physics_client_id
            )
            state[obj_id] = {
                "pos": list(pos),
                "color": color,
                "shape": shape,
            }
        return state

    # ─── GRIPPER ──────────────────────────────────────────────────────────────

    def _try_grasp(self):
        """Snap the nearest object to the EE if within GRASP_DISTANCE."""
        if self._grasp_constraint is not None:
            return  # already holding something

        ee_state = p.getLinkState(
            self._kuka_id,
            END_EFFECTOR_LINK_INDEX,
            physicsClientId=self._physics_client_id,
        )
        ee_pos = np.array(ee_state[0])

        for obj_id in self._object_ids:
            obj_pos, _ = p.getBasePositionAndOrientation(
                obj_id,
                physicsClientId=self._physics_client_id,
            )
            dist = np.linalg.norm(ee_pos - np.array(obj_pos))
            #print(f"  dist to obj {obj_id}: {dist:.4f}m")  # debug

            if dist < GRASP_DISTANCE:
                #print(f"✅ SNAPPING obj {obj_id}!")
                self._grasp_constraint = p.createConstraint(
                    self._kuka_id,
                    END_EFFECTOR_LINK_INDEX,
                    obj_id,
                    -1,
                    p.JOINT_FIXED,
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    physicsClientId=self._physics_client_id,
                )
                p.changeConstraint(
                    self._grasp_constraint,
                    maxForce=500,
                    physicsClientId=self._physics_client_id,
                )
                self._grasped_object_id = obj_id
                break

    def _release(self):
        """Release whatever the gripper is holding."""
        if self._grasp_constraint is not None:
            p.removeConstraint(
                self._grasp_constraint,
                physicsClientId=self._physics_client_id,
            )
            self._grasp_constraint = None
            self._grasped_object_id = None

    # ─── RESET ────────────────────────────────────────────────────────────────

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
            "ee_position": obs[14:17].tolist(),
            "object_state": self._get_object_state(),
            "gripper_state": 0.0,
        }
        return obs, info

    # ─── STEP ─────────────────────────────────────────────────────────────────

    def step(self, action):
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
        self._try_grasp()

        observation = self._get_observation()
        gripper_state = 1.0 if self._grasp_constraint is not None else 0.0

        reward = 0.0
        terminated = False
        truncated = self._step_count >= MAX_EPISODE_STEPS
        info = {
            "step_count": self._step_count,
            "ee_position": observation[14:17].tolist(),
            "object_state": self._get_object_state(),
            "gripper_state": gripper_state,
            "grasped_object": self._grasped_object_id,
        }

        return observation, reward, terminated, truncated, info

    # ─── OBSERVATION ──────────────────────────────────────────────────────────

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

        gripper_state = np.array(
            [1.0 if self._grasp_constraint is not None else 0.0],
            dtype=np.float32
        )

        obs = np.concatenate([
            joint_positions,
            joint_velocities,
            ee_position,
            ee_orientation,
            gripper_state,
        ])
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    # ─── RENDER ───────────────────────────────────────────────────────────────

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