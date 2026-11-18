"""Running a run-3 state policy outside the training environment (on the real arm, or any arm that reports joint angles).

The policy outputs a joint speed fraction per joint. The training environment turns that into a joint target (dead
band, speed and acceleration limits, joint limits). `LiveController` repeats exactly that, step for step, so a policy
can be driven by measured joint angles and a camera cube pose and give back the joint target to send to the arm.
tests/test_deploy.py checks it against the environment.

Cube pose from the camera: centimetres out from the arm's base (forward) and to the left, plus the cube's yaw in
degrees (locate_cube.py's numbers; the sim frame). The policy was trained with the cube held at its last seen pose
once the gripper covers it, with no "hidden" flag, so a fixed pose taken before the arm moves is a valid input.
"""
import numpy as np

from so101_env.state_pick_env import CUBE_SCALE, CUBE_X0, SPEED_DEADBAND


class LiveController:
    def __init__(self, low, high, dt, v_max=0.9, a_max=6.0, deadband=SPEED_DEADBAND):
        self.low, self.high = np.asarray(low, float), np.asarray(high, float)
        self.dt, self.v_max, self.a_max, self.deadband = dt, v_max, a_max, deadband
        self.target = None
        self.v = np.zeros(len(self.low))

    @classmethod
    def from_env(cls, env):
        return cls(env._low, env._high, env.dt, env.v_max, env.a_max)

    def reset(self, q):
        """Start from the arm's measured joint angles (radians), standing still."""
        self.target = np.clip(np.asarray(q, float), self.low, self.high)
        self.v = np.zeros(len(self.low))
        return self.target.copy()

    def step(self, action):
        """Policy action in [-1, 1] per joint -> the next joint target (radians); same arithmetic as the environment."""
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        a = np.sign(a) * np.maximum(np.abs(a) - self.deadband, 0.0) / (1.0 - self.deadband)
        want = a * self.v_max
        limit = self.a_max * self.dt
        self.v = self.v + np.clip(want - self.v, -limit, limit)
        new = self.target + self.v * self.dt
        clipped = np.clip(new, self.low, self.high)
        self.v[clipped != new] = 0.0
        self.target = clipped
        return self.target.copy()


def observation(q, cube_cm_cm_deg):
    """The policy's two inputs. q: measured joint angles in radians (6). cube_cm_cm_deg: (x_cm, y_cm, yaw_deg)."""
    x, y, yaw = cube_cm_cm_deg
    x, y, yaw = x / 100.0, y / 100.0, np.radians(yaw)
    cube = np.array([(x - CUBE_X0) / CUBE_SCALE, y / CUBE_SCALE, np.cos(4 * yaw), np.sin(4 * yaw), 1.0], dtype=np.float32)
    return {"proprio": np.asarray(q, dtype=np.float32), "cube": cube}
