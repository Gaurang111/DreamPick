"""Run 3 environment: SO101 picks up a cube, from state (no images). See docs/training_run_3_plan.md.

Observation (Dict)
    proprio  float32 (6,)  joint angles (rad) as the real arm would report them: + constant per-episode calibration
                           offset + noise (both only at the "full" difficulty).
    cube     float32 (5,)  what the overhead camera gives: [x, y] in the arm's frame (normalised), [cos 4*yaw, sin 4*yaw]
                           (the cube is square), and a `visible` flag. At "wide"/"full" difficulty the pose has noise
                           and latency, a height parallax, and it freezes at its last value (visible = 0) while the
                           gripper is over the cube. No cube height, no images, nothing else.

Action: Box(-1, 1, (6,)) = desired joint speed as a fraction of v_max. The environment limits the commanded speed
    (v_max) and how fast it may change (a_max), and at "full" difficulty delays it by 0-2 steps, so the policy
    learns motion the real arm can execute. Real joint limits are always on.

Difficulty presets (what is randomised):
    easy   narrow cube zone, start near the sim's home pose, exact observations, nominal physics.
    wide   full cube zone, start anywhere between the real rest pose and home, observation noise, latency,
           parallax, occlusion.
    full   + servo gain/damping, friction, cube mass/size, gripper force, action delay, constant joint-angle and
           camera calibration errors.

Reward stages "reach" / "grasp" / "lift" reuse the terms and constants of the run 1-2 environment (pick_cube_env.py)
with an added penalty for pressing hard on the table.
"""
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from . import pick_cube_env as old

_SCENE_PATH = Path(__file__).resolve().parents[2] / "assets" / "scene_pick_cube.xml"

# The real arm's folded rest pose (rad), from the user's photo and the earlier demo: lift/elbow/wrist near their stops.
REST_QPOS = np.array([0.0, -1.74, 1.60, 1.0, 0.0, 0.1])
HOME_QPOS = old._HOME_QPOS.copy()

CAMERA_FOOT_XY = np.array([0.23, 0.0])   # where the overhead camera looks straight down (about the mat's middle)
CUBE_X0, CUBE_SCALE = 0.24, 0.10         # observation normalisation of the cube position (m)
OCCLUSION_RADIUS = 0.05                  # gripper within this horizontal distance of the cube hides it from above
TABLE_FORCE_FREE, TABLE_FORCE_FULL, TABLE_PENALTY = 10.0, 40.0, 2.0   # N, N, reward per step
SPEED_DEADBAND = 0.07      # fraction of full speed (0.063 rad/s); a held policy's tiny residual commands made the arm creep ~0.3 cm/s
SHIFT_PENALTY, SHIFT_FREE, SHIFT_FULL = 12.0, 0.01, 0.03   # per step; the scripted solver keeps the held cube within 0.5 cm, success allows 3 cm
# The model's gripper can squeeze with 2.94 N*m (~49 N at the pads), which crushes the cube out of the jaws. The real
# arm's bridge limits the gripper torque to about half (Max_Torque_Limit 500 of 1000), so the sim's is set to this.
GRIPPER_FORCE_LIMIT = 1.0     # N*m


@dataclass(frozen=True)
class Difficulty:
    cube_x: tuple = (0.21, 0.27)
    cube_y: tuple = (-0.03, 0.03)
    start: str = "home"            # "home" | "family"
    obs_noise: bool = False        # camera noise, latency, parallax
    occlusion: bool = False        # the cube is hidden (last pose held) while the gripper is over it
    show_flag: bool = True         # with occlusion: also tell the policy it is hidden (visible flag 0); False = flag stays 1
    physics: bool = False          # servo / friction / mass / size / force randomisation
    calibration: bool = False      # constant joint-angle and camera offsets, action delay
    calib_scale: float = 1.0       # scales the offsets (1.0: joint up to 3 deg, camera up to 10 mm)


# The zone starts 21 cm out: the real arm's wrist (encoder rollover) cannot lift a cube closer than about 20 cm.
PRESETS = {
    "easy": Difficulty(),
    "wide": Difficulty(cube_x=(0.21, 0.31), cube_y=(-0.10, 0.10), start="family", obs_noise=True, occlusion=True, show_flag=False),
    "full": Difficulty(cube_x=(0.21, 0.31), cube_y=(-0.10, 0.10), start="family", obs_noise=True, occlusion=True, show_flag=False, physics=True, calibration=True),
    # the final stage: everything on, but with calibration errors a checked real arm and camera can stay within (1.5 deg, 5 mm)
    "polish": Difficulty(cube_x=(0.21, 0.31), cube_y=(-0.10, 0.10), start="family", obs_noise=True, occlusion=True, show_flag=False, physics=True, calibration=True, calib_scale=0.5),
    # curriculum steps between easy and wide (run 3, stage 4): one new difficulty at a time
    "wide_exact": Difficulty(cube_x=(0.21, 0.31), cube_y=(-0.10, 0.10), start="family"),
    "wide_noisy": Difficulty(cube_x=(0.21, 0.31), cube_y=(-0.10, 0.10), start="family", obs_noise=True),
}


class SO101StatePickEnv(gym.Env):
    ctrl_hook = None        # see step()

    metadata = {"render_modes": [], "render_fps": 20}

    def __init__(self, reward_stage="lift", difficulty="easy", max_episode_steps=300, v_max=0.9, a_max=6.0,
                 frame_skip=25, success_height=0.05, use_real_arm_limits=True):
        if reward_stage not in ("reach", "grasp", "lift"):
            raise ValueError(f"reward_stage must be reach, grasp or lift, got {reward_stage!r}")
        self.reward_stage = reward_stage
        self.diff = PRESETS[difficulty] if isinstance(difficulty, str) else difficulty
        self.max_episode_steps, self.v_max, self.a_max = max_episode_steps, v_max, a_max
        self.frame_skip, self.success_height = frame_skip, success_height

        self.model = mujoco.MjModel.from_xml_path(str(_SCENE_PATH))
        self.data = mujoco.MjData(self.model)
        self.dt = frame_skip * self.model.opt.timestep
        self.n = self.model.nu
        self._adr = np.array([self.model.jnt_qposadr[self.model.actuator_trnid[i, 0]] for i in range(self.n)])
        self._low = self.model.actuator_ctrlrange[:, 0].copy()
        self._high = self.model.actuator_ctrlrange[:, 1].copy()
        if use_real_arm_limits:
            for i in range(self.n):
                lim = old._REAL_ARM_LIMITS_DEG.get(self.model.actuator(i).name)
                if lim:
                    self._low[i] = max(self._low[i], np.radians(lim[0]))
                    self._high[i] = min(self._high[i], np.radians(lim[1]))

        m = self.model
        self._cube_body = m.body("cube").id
        self._cube_adr = m.joint("cube_freejoint").qposadr[0]
        self._cube_geom = m.geom("cube_geom").id
        self._table_geom = m.geom("table_top").id
        self._fixed_pad = m.body("fixed_jaw_pad").id
        self._moving_pad = m.body("moving_jaw_pad").id
        m.actuator_forcerange[self.n - 1] = [-GRIPPER_FORCE_LIMIT, GRIPPER_FORCE_LIMIT]
        # nominal values the randomisation is applied to (and restored from)
        self._nom = dict(gain=m.actuator_gainprm.copy(), bias=m.actuator_biasprm.copy(), force=m.actuator_forcerange.copy(),
                         friction=m.geom_friction[self._cube_geom].copy(), mass=float(m.body_mass[self._cube_body]),
                         inertia=m.body_inertia[self._cube_body].copy(), size=m.geom_size[self._cube_geom].copy())

        self.action_space = spaces.Box(-1.0, 1.0, (self.n,), np.float32)
        self.observation_space = spaces.Dict({
            "proprio": spaces.Box(-4.0, 4.0, (self.n,), np.float32),
            "cube": spaces.Box(-6.0, 6.0, (5,), np.float32),
        })
        self._reset_state()

    # --- helpers -----------------------------------------------------------

    def _reset_state(self):
        self._step_count = 0
        self._target = HOME_QPOS.copy()
        self._v = np.zeros(self.n)
        self._queue = deque()
        self._delay = 0
        self._joint_offset = np.zeros(self.n)
        self._cam_offset = np.zeros(2)
        self._cam_height = 0.55
        self._latency = 0
        self._cube_hist = deque(maxlen=4)
        self._frozen = np.array([CUBE_X0, 0.0, 1.0, 0.0])
        self._cube_start_xy = np.array([0.22, 0.0])
        self._cube_start_radius = 0.22
        self._held_steps = 0
        self._cube_rest_z = 0.015
        self.params = {}

    def _apply_physics(self, rng):
        m, nom = self.model, self._nom
        m.actuator_gainprm[:] = nom["gain"]
        m.actuator_biasprm[:] = nom["bias"]
        m.actuator_forcerange[:] = nom["force"]
        m.geom_friction[self._cube_geom] = nom["friction"]
        m.body_mass[self._cube_body] = nom["mass"]
        m.body_inertia[self._cube_body] = nom["inertia"]
        m.geom_size[self._cube_geom] = nom["size"]
        p = dict(kp=1.0, kv=1.0, force=1.0, friction=1.0, mass=1.0, size_delta=0.0)
        if self.diff.physics:
            p = dict(kp=rng.uniform(0.3, 1.0), kv=rng.uniform(0.5, 1.5), force=rng.uniform(0.6, 1.2),
                     friction=rng.uniform(0.6, 1.5), mass=rng.uniform(0.7, 1.5), size_delta=rng.uniform(-0.0015, 0.0015))
            m.actuator_gainprm[:, 0] = nom["gain"][:, 0] * p["kp"]
            m.actuator_biasprm[:, 1] = nom["bias"][:, 1] * p["kp"]
            m.actuator_biasprm[:, 2] = nom["bias"][:, 2] * p["kv"]
            m.actuator_forcerange[self.n - 1] = nom["force"][self.n - 1] * p["force"]   # gripper only
            m.geom_friction[self._cube_geom, 0] = nom["friction"][0] * p["friction"]
            m.body_mass[self._cube_body] = nom["mass"] * p["mass"]
            m.body_inertia[self._cube_body] = nom["inertia"] * p["mass"]
            m.geom_size[self._cube_geom] = nom["size"] + p["size_delta"]
        self._cube_rest_z = float(m.geom_size[self._cube_geom][2])
        return p

    def _start_pose(self, rng):
        self._start_u = 1.0
        if self.diff.start == "home":
            q = HOME_QPOS + rng.uniform(-0.03, 0.03, self.n)
        else:
            u = rng.uniform(0.0, 1.0)
            self._start_u = float(u)
            q = (1 - u) * REST_QPOS + u * HOME_QPOS + rng.normal(0.0, 0.06, self.n)
            q[0] = rng.uniform(-0.3, 0.3)
            q[5] = rng.uniform(0.0, 0.7)
        return np.clip(q, self._low, self._high)

    def pad_mid(self):
        return 0.5 * (self.data.xpos[self._fixed_pad] + self.data.xpos[self._moving_pad])

    def cube_pos(self):
        return self.data.xpos[self._cube_body]

    def cube_yaw(self):
        r = self.data.xmat[self._cube_body].reshape(3, 3)
        return float(np.arctan2(r[1, 0], r[0, 0]))

    def _grasped(self):
        a = b = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if self._cube_geom not in (c.geom1, c.geom2):
                continue
            bodies = {self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]}
            a |= self._fixed_pad in bodies
            b |= self._moving_pad in bodies
        return a and b

    def _table_force(self):
        worst, f6 = 0.0, np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if self._table_geom in (c.geom1, c.geom2) and self._cube_geom not in (c.geom1, c.geom2):
                mujoco.mj_contactForce(self.model, self.data, i, f6)
                worst = max(worst, float(f6[0]))
        return worst

    # --- observation -------------------------------------------------------

    def _observe(self):
        rng, d = self.np_random, self.diff
        q = self.data.qpos[self._adr].copy()
        pos, yaw = self.cube_pos(), self.cube_yaw()
        if d.obs_noise:
            q = q + rng.normal(0.0, np.radians(0.3), self.n)
        proprio = (q + self._joint_offset).astype(np.float32)

        xy, h = pos[:2].copy(), float(pos[2])
        if d.obs_noise:
            xy = xy + (xy - CAMERA_FOOT_XY) * (h / (self._cam_height - h))          # parallax
            xy = xy + self._cam_offset + rng.normal(0.0, 0.0025, 2)
            yaw = yaw + rng.normal(0.0, np.radians(3.0))
        self._cube_hist.append(np.array([xy[0], xy[1], np.cos(4 * yaw), np.sin(4 * yaw)]))
        seen = self._cube_hist[max(-len(self._cube_hist), -1 - self._latency)]
        visible = True
        if d.occlusion:
            visible = float(np.linalg.norm(self.pad_mid()[:2] - pos[:2])) > OCCLUSION_RADIUS
        if visible:
            self._frozen = seen
        cube = np.array([(self._frozen[0] - CUBE_X0) / CUBE_SCALE, self._frozen[1] / CUBE_SCALE,
                         self._frozen[2], self._frozen[3], 1.0 if (visible or not d.show_flag) else 0.0], dtype=np.float32)
        return {"proprio": proprio, "cube": cube}

    # --- gym API -----------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng, d = self.np_random, self.diff
        self._reset_state()
        self.params = self._apply_physics(rng)
        mujoco.mj_resetData(self.model, self.data)

        q0 = self._start_pose(rng)
        self.data.qpos[self._adr] = q0
        x, y = rng.uniform(*d.cube_x), rng.uniform(*d.cube_y)
        yaw = rng.uniform(0.0, 2 * np.pi)
        self.data.qpos[self._cube_adr:self._cube_adr + 7] = [x, y, self._cube_rest_z, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self._cube_start_xy, self._cube_start_radius = np.array([x, y]), float(np.hypot(x, y))
        self.data.qvel[:] = 0.0
        self._target = q0.copy()
        self.data.ctrl[:] = self._target

        if d.calibration:
            self._joint_offset = np.clip(rng.normal(0.0, np.radians(1.5), self.n), -np.radians(3), np.radians(3)) * d.calib_scale
            self._cam_offset = np.clip(rng.normal(0.0, 0.004, 2), -0.010, 0.010) * d.calib_scale
            self._delay = int(rng.integers(0, 3))
        if d.obs_noise:
            self._cam_height = rng.uniform(0.45, 0.65)
            self._latency = int(rng.integers(1, 3))
        self._queue = deque([np.zeros(self.n)] * self._delay)
        self.params.update(delay=self._delay, latency=self._latency, joint_offset_deg=np.degrees(np.abs(self._joint_offset)).max(),
                           cam_offset_mm=float(np.abs(self._cam_offset).max() * 1000), x=x, y=y, yaw=yaw, start_u=self._start_u)
        mujoco.mj_forward(self.model, self.data)
        return self._observe(), self._info()

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        a = np.sign(a) * np.maximum(np.abs(a) - SPEED_DEADBAND, 0.0) / (1.0 - SPEED_DEADBAND)   # servo deadband: tiny commands do nothing
        if self._delay:
            self._queue.append(a)
            a = self._queue.popleft()
        want = a * self.v_max
        limit = self.a_max * self.dt
        self._v = self._v + np.clip(want - self._v, -limit, limit)
        new = self._target + self._v * self.dt
        clipped = np.clip(new, self._low, self._high)
        self._v[clipped != new] = 0.0
        self._target = clipped
        # Optional post-processing between the joint target and the servo (smoothing study, scripts/run3_smooth_eval.py):
        # ctrl_hook(target) -> array (frame_skip, n) of servo goals for the sub-steps of this control step.
        seq = self.ctrl_hook(self._target) if self.ctrl_hook is not None else None
        self.data.ctrl[:] = self._target
        for k in range(self.frame_skip):
            if seq is not None:
                self.data.ctrl[:] = seq[k]
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        info = self._info()
        reward = self._reward(info)
        truncated = self._step_count >= self.max_episode_steps
        return self._observe(), reward, False, truncated, info

    def _info(self):
        pos = self.cube_pos()
        height = float(max(0.0, pos[2] - self._cube_rest_z))
        self._held_steps = self._held_steps + 1 if height >= self.success_height else 0
        return {
            "gripper_cube_distance": float(np.linalg.norm(self.pad_mid() - pos)),
            "is_grasped": self._grasped(),
            "cube_height": height,
            "cube_drag": float(max(0.0, self._cube_start_radius - np.hypot(pos[0], pos[1]))),
            "cube_shift": float(np.linalg.norm(pos[:2] - self._cube_start_xy)),
            "held_steps": self._held_steps,
            "table_force": self._table_force(),
            "joint_pos": self.data.qpos[self._adr].copy(),
        }

    def _reward(self, info):
        reach = 1.0 - np.tanh(old._REACH_DISTANCE_SCALE * info["gripper_cube_distance"])
        drag = old._DRAG_PENALTY * np.clip((info["cube_drag"] - old._DRAG_FREE) / (old._DRAG_FULL - old._DRAG_FREE), 0, 1)
        table = TABLE_PENALTY * np.clip((info["table_force"] - TABLE_FORCE_FREE) / (TABLE_FORCE_FULL - TABLE_FORCE_FREE), 0, 1)
        if self.reward_stage == "reach":
            return float(reach - drag - table)
        grasp = old._GRASP_BONUS if info["is_grasped"] else 0.0
        if self.reward_stage == "grasp":
            return float(reach + grasp - drag - table)
        lift = old._LIFT_BONUS_WEIGHT * np.clip(info["cube_height"] / self.success_height, 0, 1)
        hold = old._HOLD_BONUS if info["cube_height"] >= self.success_height else 0.0
        shift = SHIFT_PENALTY * np.clip((info["cube_shift"] - SHIFT_FREE) / (SHIFT_FULL - SHIFT_FREE), 0, 1)
        return float(reach + grasp + lift + hold - max(drag, shift) - table)
