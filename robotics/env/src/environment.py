import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data

from robotics.env.src.config import (
    GraphicalMode, NUM_JOINTS, END_EFFECTOR_LINK_INDEX, MAX_FORCE,
    SIM_TIMESTEP, SIM_STEPS_PER_ACTION, MAX_EPISODE_STEPS,
    JOINT_LOWER_LIMITS, JOINT_UPPER_LIMITS, HOME_POSITION,
    MAX_JOINT_VELOCITY, WORKSPACE_LOW, WORKSPACE_HIGH,
    CAM_DISTANCE, CAM_YAW, CAM_PITCH, CAM_TARGET,
    RENDER_WIDTH, RENDER_HEIGHT, RENDER_FPS, RenderMode,
    NUM_OBJECTS, OBJECT_COLORS, OBJECT_SIZE_MIN, OBJECT_SIZE_MAX,
    TABLE_POSITION, TABLE_HALF_EXTENTS, TABLE_SURFACE_Z, SPAWN_RANGE,
    OBJECT_SHAPES, GRIPPER_ATTACH_DISTANCE, GRIPPER_RELEASE_DISTANCE
)


class KukaEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": RENDER_FPS}

    def __init__(self, render_mode=RenderMode.RGB_ARRAY.value):
        super().__init__()
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # 8-dim action space: 7 joints + 1 gripper command
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(NUM_JOINTS + 1,), dtype=np.float32)

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
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, shape=(21,), dtype=np.float32)

        self._physics_client_id = -1
        self._kuka_id = None
        self._plane_id = None
        self._step_count = 0
        self._object_ids = []
        self._object_colors = []
        self._object_shapes = []
        self._grasped_object_id = None
        self._grasp_constraint = None
        self._target_object_id = None  # For instruction-aware grasping

    def set_target_object(self, obj_id):
        """Set which object is the target to grasp (prevents grasping wrong objects)."""
        self._target_object_id = obj_id

    def _load_table(self):
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=TABLE_HALF_EXTENTS,
                                     physicsClientId=self._physics_client_id)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=TABLE_HALF_EXTENTS,
                                  rgbaColor=[0.6, 0.4, 0.2, 1.0],
                                  physicsClientId=self._physics_client_id)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                          baseVisualShapeIndex=vis, basePosition=TABLE_POSITION,
                          physicsClientId=self._physics_client_id)

    def _load_objects(self):
        self._object_ids = []
        self._object_colors = []
        self._object_shapes = []
        self._grasped_object_id = None
        self._grasp_constraint = None

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
                col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size, size, size],
                                             physicsClientId=self._physics_client_id)
                vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[size, size, size],
                                          rgbaColor=rgba, physicsClientId=self._physics_client_id)
            elif shape_name == "sphere":
                col = p.createCollisionShape(p.GEOM_SPHERE, radius=size,
                                             physicsClientId=self._physics_client_id)
                vis = p.createVisualShape(p.GEOM_SPHERE, radius=size,
                                          rgbaColor=rgba, physicsClientId=self._physics_client_id)
            else:  # cylinder
                col = p.createCollisionShape(p.GEOM_CYLINDER, radius=size, height=size * 2,
                                             physicsClientId=self._physics_client_id)
                vis = p.createVisualShape(p.GEOM_CYLINDER, radius=size, length=size * 2,
                                          rgbaColor=rgba, physicsClientId=self._physics_client_id)

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

            obj_id = p.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=col,
                                       baseVisualShapeIndex=vis, basePosition=position,
                                       physicsClientId=self._physics_client_id)
            self._object_ids.append(obj_id)
            self._object_colors.append(color_name)
            self._object_shapes.append(shape_name)

    def _get_object_state(self):
        state = {}
        for obj_id, color, shape in zip(self._object_ids, self._object_colors, self._object_shapes):
            pos, _ = p.getBasePositionAndOrientation(obj_id, physicsClientId=self._physics_client_id)
            state[obj_id] = {"pos": list(pos), "color": color, "shape": shape}
        return state

    def capture_scene(self):
        """Capture current scene: object colors, shapes, positions. Returns dict for later replay."""
        scene_data = []
        for obj_id, color, shape in zip(self._object_ids, self._object_colors, self._object_shapes):
            pos, _ = p.getBasePositionAndOrientation(obj_id, physicsClientId=self._physics_client_id)
            scene_data.append({
                "color": color,
                "shape": shape,
                "pos": list(pos)
            })
        return scene_data

    def restore_scene(self, scene_data):
        """Load objects from captured scene data at their original positions."""
        self._object_ids = []
        self._object_colors = []
        self._object_shapes = []
        self._grasped_object_id = None
        self._grasp_constraint = None

        for item in scene_data:
            color_name = item["color"]
            shape_name = item["shape"]
            position = item["pos"]
            rgba = OBJECT_COLORS[color_name]
            
            # Infer size from shape (rough approximation - 0.03m box size)
            size = 0.03
            
            if shape_name == "box":
                col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size, size, size],
                                             physicsClientId=self._physics_client_id)
                vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[size, size, size],
                                          rgbaColor=rgba, physicsClientId=self._physics_client_id)
            elif shape_name == "sphere":
                col = p.createCollisionShape(p.GEOM_SPHERE, radius=size,
                                             physicsClientId=self._physics_client_id)
                vis = p.createVisualShape(p.GEOM_SPHERE, radius=size,
                                          rgbaColor=rgba, physicsClientId=self._physics_client_id)
            else:  # cylinder
                col = p.createCollisionShape(p.GEOM_CYLINDER, radius=size, height=size * 2,
                                             physicsClientId=self._physics_client_id)
                vis = p.createVisualShape(p.GEOM_CYLINDER, radius=size, length=size * 2,
                                          rgbaColor=rgba, physicsClientId=self._physics_client_id)

            obj_id = p.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=col,
                                       baseVisualShapeIndex=vis, basePosition=position,
                                       physicsClientId=self._physics_client_id)
            self._object_ids.append(obj_id)
            self._object_colors.append(color_name)
            self._object_shapes.append(shape_name)

    def _try_grasp(self):
        if self._grasped_object_id is not None:
            return
        ee_state = p.getLinkState(self._kuka_id, END_EFFECTOR_LINK_INDEX,
                                  physicsClientId=self._physics_client_id)
        ee_pos = np.array(ee_state[0])
        
        # CRITICAL: Only attempt to grasp TARGET object (set by instruction)
        # This prevents grasping wrong objects when multiple are nearby
        if self._target_object_id is None:
            return  # No target set, don't grasp anything
        
        if self._target_object_id not in self._object_ids:
            return  # Target doesn't exist, don't grasp
            
        obj_pos, _ = p.getBasePositionAndOrientation(self._target_object_id, 
                                                     physicsClientId=self._physics_client_id)
        if np.linalg.norm(ee_pos - np.array(obj_pos)) < GRIPPER_ATTACH_DISTANCE:
            self._grasp_constraint = p.createConstraint(
                parentBodyUniqueId=self._kuka_id,
                parentLinkIndex=END_EFFECTOR_LINK_INDEX,
                childBodyUniqueId=self._target_object_id,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0],
                physicsClientId=self._physics_client_id)
            self._grasped_object_id = self._target_object_id

    def _release_grasp(self):
        """Release the grasped object by removing the constraint."""
        if self._grasp_constraint is not None:
            p.removeConstraint(self._grasp_constraint, physicsClientId=self._physics_client_id)
            self._grasp_constraint = None
        self._grasped_object_id = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self._physics_client_id < 0:
            # Always DIRECT — no display on Railway
            self._physics_client_id = p.connect(p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.resetSimulation(physicsClientId=self._physics_client_id)
        p.setGravity(0, 0, -10, physicsClientId=self._physics_client_id)
        p.setTimeStep(SIM_TIMESTEP, physicsClientId=self._physics_client_id)

        self._plane_id = p.loadURDF("plane.urdf", physicsClientId=self._physics_client_id)
        self._kuka_id = p.loadURDF("kuka_iiwa/model.urdf", basePosition=[0, 0, 0],
                                   useFixedBase=True, physicsClientId=self._physics_client_id)

        for i in range(NUM_JOINTS):
            p.resetJointState(self._kuka_id, i, HOME_POSITION[i],
                              physicsClientId=self._physics_client_id)

        self._grasped_object_id = None
        self._grasp_constraint = None
        self._target_object_id = None  # Reset target for new episode
        self._load_table()
        
        # Check if scene_data is provided in options to restore a previous scene
        scene_data = options.get("scene_data") if options else None
        if scene_data:
            self.restore_scene(scene_data)
        else:
            self._load_objects()
        self._step_count = 0

        for _ in range(SIM_STEPS_PER_ACTION):
            p.stepSimulation(physicsClientId=self._physics_client_id)

        self._try_grasp()
        self._step_count += 1

        obs = self._get_observation()
        info = {
            "step_count": self._step_count,
            "ee_position": obs[14:17].tolist(),
            "object_state": self._get_object_state(),
            "grasped_object": self._grasped_object_id,
        }
        return obs, info

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        
        # Extract gripper command (dimension 7: >0.5 = close/grasp, <-0.5 = open/release)
        gripper_command = action[7] if len(action) > 7 else 0.0
        
        # Joint control (dimensions 0-6)
        midpoint = (JOINT_UPPER_LIMITS + JOINT_LOWER_LIMITS) / 2.0
        half_range = (JOINT_UPPER_LIMITS - JOINT_LOWER_LIMITS) / 2.0
        joint_actions = midpoint + action[:NUM_JOINTS] * half_range

        for i in range(NUM_JOINTS):
            p.setJointMotorControl2(bodyUniqueId=self._kuka_id, jointIndex=i,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=float(joint_actions[i]),
                                    force=MAX_FORCE,
                                    physicsClientId=self._physics_client_id)

        for _ in range(SIM_STEPS_PER_ACTION):
            p.stepSimulation(physicsClientId=self._physics_client_id)

        # Handle gripper based on command (close=grasp, open=release)
        if gripper_command > 0.5:
            self._try_grasp()
        elif gripper_command < -0.5:
            self._release_grasp()

        self._step_count += 1
        observation = self._get_observation()
        truncated = self._step_count >= MAX_EPISODE_STEPS
        
        # Check if object was successfully grasped
        is_success = self._grasped_object_id is not None
        
        info = {
            "step_count": self._step_count,
            "ee_position": observation[14:17].tolist(),
            "object_state": self._get_object_state(),
            "grasped_object_id": self._grasped_object_id,
            "is_success": is_success,
        }
        return observation, 0.0, False, truncated, info

    def render(self):
        return self._get_camera_image()

    def get_segmentation(self):
        """Returns (rgb_frame, seg_map) for Beta vision pipeline."""
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=CAM_TARGET, distance=CAM_DISTANCE,
            yaw=CAM_YAW, pitch=CAM_PITCH, roll=0, upAxisIndex=2,
            physicsClientId=self._physics_client_id)
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=RENDER_WIDTH / RENDER_HEIGHT,
            nearVal=0.1, farVal=100, physicsClientId=self._physics_client_id)
        _, _, rgb, _, seg = p.getCameraImage(
            RENDER_WIDTH, RENDER_HEIGHT, view_matrix, proj_matrix,
            renderer=p.ER_TINY_RENDERER, physicsClientId=self._physics_client_id)
        frame = np.array(rgb, dtype=np.uint8).reshape(RENDER_HEIGHT, RENDER_WIDTH, 4)[:, :, :3]
        seg_map = np.array(seg, dtype=np.int32).reshape(RENDER_HEIGHT, RENDER_WIDTH)
        return frame, seg_map

    def close(self):
        if self._physics_client_id >= 0:
            p.disconnect(physicsClientId=self._physics_client_id)
            self._physics_client_id = -1

    def _get_observation(self):
        joint_positions = np.zeros(NUM_JOINTS, dtype=np.float32)
        joint_velocities = np.zeros(NUM_JOINTS, dtype=np.float32)
        for i in range(NUM_JOINTS):
            state = p.getJointState(self._kuka_id, i, physicsClientId=self._physics_client_id)
            joint_positions[i] = state[0]
            joint_velocities[i] = state[1]
        ee_state = p.getLinkState(self._kuka_id, END_EFFECTOR_LINK_INDEX,
                                  physicsClientId=self._physics_client_id)
        ee_position = np.array(ee_state[0], dtype=np.float32)
        ee_orientation = np.array(ee_state[1], dtype=np.float32)
        obs = np.concatenate([joint_positions, joint_velocities, ee_position, ee_orientation])
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _get_camera_image(self):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=CAM_TARGET, distance=CAM_DISTANCE,
            yaw=CAM_YAW, pitch=CAM_PITCH, roll=0, upAxisIndex=2,
            physicsClientId=self._physics_client_id)
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=RENDER_WIDTH / RENDER_HEIGHT,
            nearVal=0.1, farVal=100, physicsClientId=self._physics_client_id)
        _, _, rgb, _, _ = p.getCameraImage(
            RENDER_WIDTH, RENDER_HEIGHT, view_matrix, proj_matrix,
            renderer=p.ER_TINY_RENDERER, physicsClientId=self._physics_client_id)
        return np.array(rgb, dtype=np.uint8).reshape(RENDER_HEIGHT, RENDER_WIDTH, 4)[:, :, :3]