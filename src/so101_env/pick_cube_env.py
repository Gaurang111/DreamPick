"""Gymnasium environment: SO101 arm picking up a cube, in MuJoCo.

Observation (Dict):
    wrist_image:    uint8 (64, 64, 3)  - camera mounted on the gripper, looking at the jaws
    overhead_image: uint8 (64, 64, 3)  - fixed camera that keeps the whole arm and the cube in view.
                    (Named "overhead" from when it looked down from behind the arm; it is now a low
                    side view - see overhead_cam in assets/scene_pick_cube.xml and
                    docs/training_run_1/README.md, section 10, for why and where it sits.)
    joint_pos:      float32 (6,)       - current angle of each arm joint + the gripper

Action: Box(-1, 1, shape=(6,)), float32
    Each entry is a *delta* applied to that joint's position target:
        new_target = clip(old_target + action * max_delta, joint_low, joint_high)
    The position target is separate from the joint's actual (physical) position -
    it's the setpoint we hand to the position actuator/PD controller, same as how
    the real servos are commanded.

Reward stages (reward_stage constructor arg): "reach", "grasp", "lift".
Each stage's reward includes everything from the stage before it, and every stage
subtracts a penalty for dragging the cube toward the arm's base.

Cube start: a random spot in cube_x_range x cube_y_range (metres out from / across
the arm's base) with a random rotation. The default zone starts 21cm out because
the real arm can only lift a cube that is at least about 20cm out (its wrist range
is limited by an encoder rollover) - see docs/training_run_1/README.md, section 11.
"""

from pathlib import Path

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

_SCENE_PATH = Path(__file__).resolve().parents[2] / "assets" / "scene_pick_cube.xml"

_IMAGE_SIZE = 64
_REACH_DISTANCE_SCALE = 10.0  # larger = reward falls off faster with distance
_GRASP_BONUS = 1.0
# Lift stage (raised for the lift retry: run 2's first lift stage plateaued at 2.4cm, carrying the
# cube ~20cm sideways instead of raising it - docs/training_run_1/README.md, section 17):
_LIFT_BONUS_WEIGHT = 10.0  # paid per step in proportion to height, up to success_height (2.0 per cm; was 1.2)
_HOLD_BONUS = 4.0          # extra per step while the cube is at/above success_height (was 2.0)
# Carrying the cube sideways earns nothing in a real lift, so in the lift stage the cube moving away from
# where it started in ANY direction is penalised (the drag penalty below only covered "toward the base").
_SHIFT_PENALTY = 5.0       # per step at full effect: more than the ~4.8 a 2.4cm lift pays, so sideways carrying loses
_SHIFT_FREE = 0.03         # metres of free movement (a straight lift shifts the cube a centimetre or two)
_SHIFT_FULL = 0.08
# Lift stage, third round (smoothing the jerky lift and stopping slips, docs section 17d):
_SMOOTH_PENALTY = 0.0        # OFF (tried at 2.0 and abandoned, docs 17d). x the mean |change in action| over the 6 joints (0 to 2) from one step to the next
_SLIP_PENALTY = 0.0          # OFF (tried at 5.0 and abandoned, docs 17d). per step at full effect: opening the gripper while the cube is off the table
_SLIP_AIR_HEIGHT = 0.01      # metres: the cube counts as "in the air" above this (at the previous step)
_SLIP_FULL_OPEN_SPEED = 0.02  # gripper joint opening (rad per step) that earns the full slip penalty
# Dragging: the first policies all pulled the cube in toward the arm instead of lifting it
# (docs/training_run_1/README.md, section 7). Moving the cube this far toward the base from
# where it started is free (carrying it up needs a little), then the penalty ramps up.
_DRAG_PENALTY = 3.0
_DRAG_FREE = 0.02
_DRAG_FULL = 0.08

# Joint ranges (degrees) the REAL arm can use, so a policy trained in the sim never relies on
# motions the real arm can't follow: the calibrated ranges cut at the encoder rollover points
# (shoulder_lift and wrist_flex lose their top end), minus a margin. Intersected with the model's own limits.
_REAL_ARM_LIMITS_DEG = {
    "shoulder_pan": (-109.9, 109.9),
    "shoulder_lift": (-101.8, 79.1),
    "elbow_flex": (-93.7, 93.7),
    "wrist_flex": (-98.0, 65.9),
}

# A relaxed "ready" pose: elbow/wrist bent so the gripper hovers above the table
# in front of the base, rather than starting folded up or pointing at the sky.
_HOME_QPOS = np.array([0.0, -0.6, 1.0, 0.4, 0.0, 0.4], dtype=np.float64)


class SO101PickCubeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        reward_stage: str = "reach",
        frame_skip: int = 25,
        max_episode_steps: int = 200,
        max_delta: float = 0.05,
        success_height: float = 0.05,
        terminate_on_success: bool | None = None,
        reset_noise: float = 0.02,
        render_mode: str | None = None,
        cube_x_range: tuple[float, float] = (0.21, 0.29),
        cube_y_range: tuple[float, float] = (-0.04, 0.04),
        randomize_cube_yaw: bool = True,
        use_real_arm_limits: bool = True,
    ):
        if reward_stage not in ("reach", "grasp", "lift"):
            raise ValueError(f"reward_stage must be 'reach', 'grasp' or 'lift', got {reward_stage!r}")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode must be one of {self.metadata['render_modes']}, got {render_mode!r}")

        self.reward_stage = reward_stage
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.max_delta = max_delta
        self.success_height = success_height
        self.terminate_on_success = (reward_stage == "lift") if terminate_on_success is None else terminate_on_success
        self.reset_noise = reset_noise
        self.render_mode = render_mode
        self.cube_x_range = cube_x_range
        self.cube_y_range = cube_y_range
        self.randomize_cube_yaw = randomize_cube_yaw

        self.model = mujoco.MjModel.from_xml_path(str(_SCENE_PATH))
        self.data = mujoco.MjData(self.model)

        self.n_joints = self.model.nu
        # qpos address of each actuator's driven joint, in actuator order.
        self._joint_qpos_adr = np.array(
            [self.model.jnt_qposadr[self.model.actuator_trnid[i, 0]] for i in range(self.n_joints)]
        )
        self._joint_low = self.model.actuator_ctrlrange[:, 0].copy()
        self._joint_high = self.model.actuator_ctrlrange[:, 1].copy()
        if use_real_arm_limits:
            for i in range(self.n_joints):
                limits = _REAL_ARM_LIMITS_DEG.get(self.model.actuator(i).name)
                if limits:
                    self._joint_low[i] = max(self._joint_low[i], np.radians(limits[0]))
                    self._joint_high[i] = min(self._joint_high[i], np.radians(limits[1]))
        # Observations use the joints' physical range, not the (tighter) command range: a joint
        # can sit a hair past its commanded limit under load.
        joint_ids = self.model.actuator_trnid[:, 0]
        self._obs_low = self.model.jnt_range[joint_ids, 0].astype(np.float32)
        self._obs_high = self.model.jnt_range[joint_ids, 1].astype(np.float32)

        self._cube_body_id = self.model.body("cube").id
        self._cube_qpos_adr = self.model.joint("cube_freejoint").qposadr[0]
        self._cube_resting_z = 0.015  # table top (z=0) + half the cube's 3cm size
        self._gripperframe_site_id = self.model.site("gripperframe").id
        # The cube only ever touches the gripper through these two pads (the jaw
        # meshes' convex hulls are excluded from cube contact - see
        # scene_pick_cube.xml), so grasping and the grasp center are defined by them.
        self._fixed_pad_body_id = self.model.body("fixed_jaw_pad").id
        self._moving_pad_body_id = self.model.body("moving_jaw_pad").id
        self._cube_geom_id = self.model.geom("cube_geom").id

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "wrist_image": spaces.Box(0, 255, (_IMAGE_SIZE, _IMAGE_SIZE, 3), dtype=np.uint8),
                "overhead_image": spaces.Box(0, 255, (_IMAGE_SIZE, _IMAGE_SIZE, 3), dtype=np.uint8),
                "joint_pos": spaces.Box(
                    low=self._obs_low,
                    high=self._obs_high,
                    shape=(self.n_joints,),
                    dtype=np.float32,
                ),
            }
        )

        # One small renderer reused for both 64x64 observation cameras, and a
        # bigger one created lazily only if someone asks for render_mode="rgb_array".
        self._obs_renderer = mujoco.Renderer(self.model, height=_IMAGE_SIZE, width=_IMAGE_SIZE)
        self._human_viewer = None
        self._rgb_renderer = None

        self._ctrl_target = _HOME_QPOS.copy()
        self._step_count = 0
        self._cube_start_radius = 0.22
        self._cube_start_xy = np.array([0.22, 0.0])
        self._last_action = np.zeros(self.n_joints)
        self._last_cube_height = 0.0
        self._held_steps = 0

    # --- Gymnasium API -----------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        noise = self.np_random.uniform(-self.reset_noise, self.reset_noise, size=self.n_joints)
        start_qpos = np.clip(_HOME_QPOS + noise, self._joint_low, self._joint_high)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._joint_qpos_adr] = start_qpos
        # Cube: a random spot in the start zone, random rotation about the vertical (free joint: 3 pos + 4 quat).
        cube_x = self.np_random.uniform(*self.cube_x_range)
        cube_y = self.np_random.uniform(*self.cube_y_range)
        yaw = self.np_random.uniform(0.0, 2.0 * np.pi) if self.randomize_cube_yaw else 0.0
        self.data.qpos[self._cube_qpos_adr : self._cube_qpos_adr + 3] = [cube_x, cube_y, self._cube_resting_z]
        self.data.qpos[self._cube_qpos_adr + 3 : self._cube_qpos_adr + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        self._cube_start_radius = float(np.hypot(cube_x, cube_y))  # distance from the arm's base (the origin)
        self._cube_start_xy = np.array([cube_x, cube_y])
        self._last_action = np.zeros(self.n_joints)
        self._last_cube_height = 0.0
        self._held_steps = 0
        self.data.qvel[:] = 0.0
        self._ctrl_target = start_qpos.copy()
        self.data.ctrl[:] = self._ctrl_target
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0

        obs = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self.render()
        return obs, info

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self._ctrl_target = np.clip(self._ctrl_target + action * self.max_delta, self._joint_low, self._joint_high)
        self.data.ctrl[:] = self._ctrl_target
        gripper_before = float(self.data.qpos[self._joint_qpos_adr[-1]])

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        obs = self._get_obs()
        info = self._get_info()
        # Used by the lift reward only. The gripper is the last joint; a larger value is more open.
        info["action_change"] = float(np.mean(np.abs(action - self._last_action)))
        info["gripper_open_speed"] = max(0.0, float(self.data.qpos[self._joint_qpos_adr[-1]]) - gripper_before)
        info["was_in_air"] = self._last_cube_height > _SLIP_AIR_HEIGHT
        self._last_action = action
        self._last_cube_height = info["cube_height"]
        reward = self._compute_reward(info)
        terminated = bool(info["is_success"] and self.terminate_on_success)
        truncated = self._step_count >= self.max_episode_steps

        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            if self._rgb_renderer is None:
                self._rgb_renderer = mujoco.Renderer(self.model, height=720, width=960)
            # viewer_cam is a diagonal 3/4 view meant purely for humans watching the
            # arm work - it is not part of the observation space fed to the agent.
            self._rgb_renderer.update_scene(self.data, camera="viewer_cam")
            return self._rgb_renderer.render()
        if self.render_mode == "human":
            if self._human_viewer is None:
                self._human_viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._human_viewer.sync()
            return None
        return None

    def close(self):
        if self._obs_renderer is not None:
            self._obs_renderer.close()
            self._obs_renderer = None
        if self._rgb_renderer is not None:
            self._rgb_renderer.close()
            self._rgb_renderer = None
        if self._human_viewer is not None:
            self._human_viewer.close()
            self._human_viewer = None

    # --- helpers -------------------------------------------------------------

    def _get_obs(self):
        self._obs_renderer.update_scene(self.data, camera="wrist_cam")
        wrist_image = self._obs_renderer.render()
        self._obs_renderer.update_scene(self.data, camera="overhead_cam")
        overhead_image = self._obs_renderer.render()
        joint_pos = self.data.qpos[self._joint_qpos_adr].astype(np.float32)
        return {
            "wrist_image": wrist_image,
            "overhead_image": overhead_image,
            "joint_pos": joint_pos,
        }

    def _get_info(self):
        # Midpoint between the pads: 0 distance = cube centered between the jaws.
        # (gripperframe sits on the fixed jaw's face, so measuring from it would
        # reward pushing that jaw into the cube.)
        gripper_pos = 0.5 * (self.data.xpos[self._fixed_pad_body_id] + self.data.xpos[self._moving_pad_body_id])
        cube_pos = self.data.xpos[self._cube_body_id]
        distance = float(np.linalg.norm(gripper_pos - cube_pos))
        cube_height = float(max(0.0, cube_pos[2] - self._cube_resting_z))
        is_grasped = self._check_grasped()
        is_success = cube_height > self.success_height
        self._held_steps = self._held_steps + 1 if cube_height >= self.success_height else 0
        return {
            "gripper_cube_distance": distance,
            "is_grasped": is_grasped,
            "cube_height": cube_height,
            "is_success": is_success,
            # How far the cube has been pulled toward the arm's base from where it started (>= 0).
            "cube_drag": float(max(0.0, self._cube_start_radius - np.hypot(cube_pos[0], cube_pos[1]))),
            # How far the cube has moved horizontally from where it started, in any direction (m).
            "cube_shift": float(np.linalg.norm(cube_pos[:2] - self._cube_start_xy)),
            # Consecutive steps at/above the target height (20 steps = 1 second).
            "held_steps": self._held_steps,
        }

    def _check_grasped(self) -> bool:
        """True when the cube is touched by both jaw pads at once - i.e. it's
        pinched between them, not just resting against one side."""
        touches_fixed_jaw = False
        touches_moving_jaw = False
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            bodies = {
                self.model.geom_bodyid[contact.geom1],
                self.model.geom_bodyid[contact.geom2],
            }
            geoms = {contact.geom1, contact.geom2}
            if self._cube_geom_id not in geoms:
                continue
            if self._fixed_pad_body_id in bodies:
                touches_fixed_jaw = True
            if self._moving_pad_body_id in bodies:
                touches_moving_jaw = True
        return touches_fixed_jaw and touches_moving_jaw

    def _compute_reward(self, info: dict) -> float:
        reach_reward = 1.0 - np.tanh(_REACH_DISTANCE_SCALE * info["gripper_cube_distance"])
        drag_fraction = np.clip((info.get("cube_drag", 0.0) - _DRAG_FREE) / (_DRAG_FULL - _DRAG_FREE), 0.0, 1.0)
        drag_penalty = _DRAG_PENALTY * drag_fraction
        if self.reward_stage == "reach":
            return float(reach_reward - drag_penalty)

        grasp_bonus = _GRASP_BONUS if info["is_grasped"] else 0.0
        if self.reward_stage == "grasp":
            return float(reach_reward + grasp_bonus - drag_penalty)

        # "lift": proportional to height up to success_height, plus a bonus for being up there,
        # minus whichever is larger: the toward-the-base drag penalty or the any-direction shift penalty
        lift_bonus = _LIFT_BONUS_WEIGHT * np.clip(info["cube_height"] / self.success_height, 0.0, 1.0)
        hold_bonus = _HOLD_BONUS if info["cube_height"] >= self.success_height else 0.0
        shift_fraction = np.clip((info.get("cube_shift", 0.0) - _SHIFT_FREE) / (_SHIFT_FULL - _SHIFT_FREE), 0.0, 1.0)
        penalty = max(drag_penalty, _SHIFT_PENALTY * shift_fraction)
        # Smooth actions, and no letting go of the cube once it is off the table.
        smooth_penalty = _SMOOTH_PENALTY * info.get("action_change", 0.0)
        slip_fraction = np.clip(info.get("gripper_open_speed", 0.0) / _SLIP_FULL_OPEN_SPEED, 0.0, 1.0)
        slip_penalty = _SLIP_PENALTY * slip_fraction if info.get("was_in_air", False) else 0.0
        return float(reach_reward + grasp_bonus + lift_bonus + hold_bonus - penalty - smooth_penalty - slip_penalty)
