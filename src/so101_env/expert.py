"""Scripted solver for SO101StatePickEnv (privileged: it reads the true cube pose and arm state).

Used to (1) prove the task is solvable inside the environment's speed/acceleration limits and fix the cube zone, and
(2) put successful episodes into the replay buffer at the start of training.

Plan: open the jaws, go above the cube with the jaws' closing axis square to a cube face, descend, close, lift, hold.
Control is resolved-rate: every step the actual position of the jaws' centre is compared with the goal and turned into
joint speeds through the arm's Jacobian, so gravity sag, servo lag and randomised gains are corrected automatically.
Position has priority; the gripper's tilt is followed with the arm's spare freedom. Joints at their limits are not
commanded outward. Roll is chosen once, at the grasp pose, and kept; the tilt of each waypoint comes from inverse kinematics
(smallest reachable tilt not below TILT_MIN).
"""
import math

import mujoco
import numpy as np

ARM = 5                   # joints 0-4: pan, lift, elbow, wrist_flex, roll
GRIP_OPEN, GRIP_CLOSED = 0.45, -0.17     # approach opening: pads ~5 cm apart for a 3 cm cube
KX = 5.0                  # 1/s: jaw-centre speed = KX * position error
KP = 3.5                  # 1/s: gripper / roll chase
KO = 3.0                  # tilt following gain
LIMIT_MARGIN = 0.03       # rad


def _wrap(a, period):
    return (a + period / 2) % period - period / 2


class ScriptedExpert:
    def __init__(self, env, grasp_z=0.0165, tilt_min=0.0, approach_z=0.075, lift_z=0.095, lift_speed=0.10):
        self.env, self.grasp_z, self.tilt_min = env, grasp_z, tilt_min
        self.approach_z, self.lift_z, self.lift_speed = approach_z, lift_z, lift_speed
        self.m = env.model
        self.d = mujoco.MjData(env.model)
        self.site = self.m.site("gripperframe").id
        self.dof = [self.m.joint(self.m.actuator_trnid[i, 0]).dofadr[0] for i in range(ARM)]
        self.plan, self.phase, self.phase_steps, self.grasp_steps = [], 0, 0, 0
        self.roll_goal, self.xy, self.radial = 0.0, np.zeros(2), np.array([1.0, 0.0])
        self.ok = True

    # --- kinematics on a scratch copy ---------------------------------------

    def _fk(self, q, grip=GRIP_OPEN, full_state=None):
        d = self.d
        if full_state is not None:
            d.qpos[:] = full_state
        d.qpos[self.env._adr[:ARM]] = q[:ARM]
        d.qpos[self.env._adr[5]] = grip
        mujoco.mj_forward(self.m, d)
        fixed, moving = d.xpos[self.env._fixed_pad], d.xpos[self.env._moving_pad]
        # control point: horizontally between the pads, vertically at the FIXED pad (the moving jaw swings on a hinge,
        # so its pad is much higher than the fixed one when the jaws are open and comes down to meet it as they close)
        pad = np.array([0.5 * (fixed[0] + moving[0]), 0.5 * (fixed[1] + moving[1]), fixed[2]])
        axis = d.site_xmat[self.site].reshape(3, 3)[:, 0]
        closing = moving - fixed
        return pad, axis, closing

    def _jacobians(self):
        """3x4 position Jacobian of the jaws' centre and 3x4 rotation Jacobian of the pointing axis (joints 0-3)."""
        m, d = self.m, self.d
        ja, jb, jr, junk = (np.zeros((3, m.nv)) for _ in range(4))
        mujoco.mj_jacBody(m, d, ja, junk, self.env._fixed_pad)
        mujoco.mj_jacBody(m, d, jb, junk, self.env._moving_pad)
        mujoco.mj_jacSite(m, d, junk, jr, self.site)
        jp = 0.5 * (ja + jb)
        jp[2] = ja[2]
        return jp[:, self.dof[:4]], jr[:, self.dof[:4]]

    def _want_axis(self, tilt_deg):
        t = math.radians(tilt_deg)
        return np.array([math.sin(t) * self.radial[0], math.sin(t) * self.radial[1], -math.cos(t)])

    def _ik(self, target, q0, tilt_deg, roll):
        want = self._want_axis(tilt_deg)
        q = np.array(q0, dtype=float)
        q[4] = roll
        lo, hi = self.env._low[:4], self.env._high[:4]
        for _ in range(80):
            pad, axis, _ = self._fk(q)
            jp, jr = self._jacobians()
            err = np.concatenate([target - pad, 0.25 * np.cross(axis, want)])
            jac = np.vstack([jp, 0.25 * jr])
            q[:4] = np.clip(q[:4] + jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), err), lo, hi)
        pad, axis, _ = self._fk(q)
        return q, bool(np.linalg.norm(target - pad) < 0.002 and np.linalg.norm(np.cross(axis, want)) < 0.05)

    def _best_roll(self, q, yaw):
        best, best_err = 0.0, 9.0
        for r in np.radians(np.arange(-90, 91, 3)):
            qq = np.array(q, dtype=float)
            qq[4] = r
            _, _, closing = self._fk(qq)
            err = abs(_wrap(math.atan2(closing[1], closing[0]) - yaw, math.pi / 2))
            if err < best_err:
                best, best_err = r, err
        return best

    def _tilt_for(self, z, q_seed, roll, tilt_from):
        """Smallest tilt >= tilt_from at which the jaws' centre can be at height z over the cube. Returns (tilt, q, ok)."""
        qq = q_seed
        for tilt in range(int(tilt_from), 71, 3):
            qq, ok = self._ik(np.array([self.xy[0], self.xy[1], z]), q_seed, tilt, roll)
            if ok:
                return float(tilt), qq, True
        return float(tilt_from), qq, False

    # --- plan ---------------------------------------------------------------

    def reset(self):
        env = self.env
        cube = env.cube_pos().copy()
        yaw = env.cube_yaw()
        self.xy = cube[:2].copy()
        self.radial = self.xy / np.linalg.norm(self.xy)
        q = env.data.qpos[env._adr].copy()
        self._fk(q, full_state=env.data.qpos.copy())
        roll = self._best_roll(q, yaw)
        tilt_g, q_g, ok = self._tilt_for(self.grasp_z, q, roll, self.tilt_min)
        roll = self._best_roll(q_g, yaw)
        tilt_g, q_g, ok = self._tilt_for(self.grasp_z, q_g, roll, self.tilt_min)
        self.roll_goal, self.ok = float(roll), bool(ok)
        tilts = {self.grasp_z: tilt_g}
        q_up = q_g
        for z in (0.030, 0.050, self.approach_z, self.lift_z):          # build upward from the grasp pose: same branch
            t, q_up, ok = self._tilt_for(z, q_up, roll, tilt_g)
            self.ok &= ok
            tilts[z] = t
        # (name, jaw-centre height, gripper goal, tilt, position tolerance m, max steps, max jaw speed m/s)
        self.plan = [("approach", self.approach_z, GRIP_OPEN, tilts[self.approach_z], 0.012, 160, 0.25),
                     ("descend1", 0.050, GRIP_OPEN, tilts[0.050], 0.008, 60, 0.12),
                     ("descend2", 0.030, GRIP_OPEN, tilts[0.030], 0.006, 60, 0.08),
                     ("descend3", self.grasp_z, GRIP_OPEN, tilt_g, 0.004, 80, 0.06),
                     ("close", self.grasp_z, GRIP_CLOSED, tilt_g, None, 50, 0.06),
                     ("lift", self.lift_z, GRIP_CLOSED, tilts[self.lift_z], 0.01, 120, self.lift_speed),
                     ("hold", self.lift_z, GRIP_CLOSED, tilts[self.lift_z], None, 10 ** 6, self.lift_speed)]
        self.phase, self.phase_steps, self.grasp_steps = 0, 0, 0
        return self.ok

    def act(self):
        env = self.env
        name, z, g_goal, tilt, tol, max_steps, vmax_pad = self.plan[self.phase]
        q_now = env.data.qpos[env._adr]
        pad, axis, _ = self._fk(q_now, grip=q_now[5], full_state=env.data.qpos)
        err = np.array([self.xy[0], self.xy[1], z]) - pad
        jp, jr = self._jacobians()

        # joints at a limit are not commanded further out
        v_task = np.clip(KX * err, -vmax_pad, vmax_pad)
        err_ori = np.cross(axis, self._want_axis(tilt))
        qdot = np.zeros(4)
        free = np.ones(4, bool)
        for _ in range(3):
            jpf = jp[:, free]
            pinv = jpf.T @ np.linalg.inv(jpf @ jpf.T + 1e-5 * np.eye(3))
            dq = np.zeros(4)
            dq[free] = pinv @ v_task
            null = np.eye(free.sum()) - pinv @ jpf                                  # spare freedom of the free joints
            jrf = jr[:, free]
            dq[free] += null @ (jrf.T @ np.linalg.solve(jrf @ jrf.T + 1e-3 * np.eye(3), KO * err_ori))
            qdot = dq
            lo, hi = env._low[:4] + LIMIT_MARGIN, env._high[:4] - LIMIT_MARGIN
            blocked = ((q_now[:4] <= lo) & (qdot < 0)) | ((q_now[:4] >= hi) & (qdot > 0))
            if not blocked.any() or not (free & ~blocked).any():
                break
            free = free & ~blocked
            qdot = np.where(free, qdot, 0.0)
        peak = np.abs(qdot).max() / env.v_max
        action = np.zeros(env.n)
        action[:4] = qdot / env.v_max / max(1.0, peak)
        action[4] = KP * (self.roll_goal - env._target[4]) / env.v_max
        action[5] = KP * (g_goal - env._target[5]) / env.v_max
        action = np.clip(action, -1.0, 1.0)

        self.phase_steps += 1
        if name == "close":
            self.grasp_steps = self.grasp_steps + 1 if env._grasped() else 0
            done = self.grasp_steps >= 8 or self.phase_steps >= max_steps
        else:
            roll_ok = abs(q_now[4] - self.roll_goal) < 0.08
            done = (tol is not None and np.linalg.norm(err) < tol and roll_ok) or self.phase_steps >= max_steps
        if done and self.phase < len(self.plan) - 1:
            self.phase += 1
            self.phase_steps = 0
        return action.astype(np.float32)
