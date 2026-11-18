"""Smoothing between the policy's joint targets and the servos (no retraining: the policy is untouched).

The policy asks for a joint speed every 1/20 s; deploy.LiveController integrates that into a joint target (speed and
acceleration limited). Between steps that target can still change acceleration abruptly, and a servo that is sent a new
goal every 50 ms moves in small hard steps. `TargetSmoother` sits after the controller: it takes each new joint target
and returns the servo goals for the `sub` sub-steps of that control step (sim: frame_skip sub-steps; real arm: send
goals at 20*sub Hz), so the servo sees a continuous, jerk-limited path instead of 20 Hz steps.

Modes (per joint group, see make_smoother):
  hold    zero-order hold, one goal per control step (what training used; the baseline)
  interp  straight line from the previous target to the new one over the step (1 step = 50 ms behind)
  ema     first-order lag, time constant tau
  crit    critically damped second-order lag, time constant tau (continuous speed, so no speed steps; ~2 tau behind)
"""
import numpy as np


class TargetSmoother:
    def __init__(self, n, dt, sub, mode="crit", tau=0.1):
        self.n, self.dt, self.sub, self.mode = n, dt, sub, mode
        self.tau = np.broadcast_to(np.asarray(tau, float), (n,)).copy()
        self.reset(np.zeros(n))

    def reset(self, q):
        self.x = np.asarray(q, float).copy()           # smoothed goal
        self.v = np.zeros(self.n)                      # its speed (crit)
        self.u_prev = self.x.copy()                    # previous target (interp)

    def step(self, target):
        """New joint target (rad, n) -> servo goals (sub, n), the last row being where the goal ends this step."""
        u = np.asarray(target, float)
        out = np.empty((self.sub, self.n))
        h = self.dt / self.sub
        if self.mode == "hold":
            out[:] = u
        elif self.mode == "interp":
            for k in range(self.sub):
                out[k] = self.u_prev + (u - self.u_prev) * (k + 1) / self.sub
        else:
            tau = np.maximum(self.tau, 1e-6)
            for k in range(self.sub):
                if self.mode == "ema":
                    self.x = u + (self.x - u) * np.exp(-h / tau)
                elif self.mode == "crit":          # exact step of x'' = w^2 (u - x) - 2 w x'
                    w = 1.0 / tau
                    e0, v0 = self.x - u, self.v
                    c = v0 + w * e0
                    decay = np.exp(-w * h)
                    self.x = u + (e0 + c * h) * decay
                    self.v = (v0 - w * c * h) * decay
                else:
                    raise ValueError(self.mode)
                out[k] = self.x
        self.u_prev = u.copy()
        if self.mode in ("hold", "interp"):
            self.x = out[-1].copy()
        return out


def make_smoother(n, dt, sub, mode="crit", tau_arm=0.1, tau_gripper=None):
    """Same mode for every joint; the gripper (last joint) can have its own time constant (None = same as the arm)."""
    tau = np.full(n, tau_arm, float)
    if tau_gripper is not None:
        tau[-1] = tau_gripper
    return TargetSmoother(n, dt, sub, mode, tau)


def smoothness(goal, q, dt=0.05):
    """Per-joint roughness of a (T, n) goal path and of the measured joints, sampled each control step.
    Returns dict with p99 |accel| (rad/s^2), p99 |jerk| (rad/s^3), max goal jump per step (deg), and velocity reversals per second."""
    out = {}
    for name, x in (("goal", np.asarray(goal, float)), ("meas", np.asarray(q, float))):
        v = np.diff(x, axis=0) / dt
        a = np.diff(v, axis=0) / dt
        j = np.diff(a, axis=0) / dt
        moving = (np.abs(v[1:]) > 0.03) & (np.abs(v[:-1]) > 0.03)
        flips = (np.sign(v[1:]) != np.sign(v[:-1])) & moving
        out[name] = dict(
            accel_p99=float(np.percentile(np.abs(a[:, :5]), 99)),
            jerk_p99=float(np.percentile(np.abs(j[:, :5]), 99)),
            jerk_max=float(np.abs(j[:, :5]).max()),
            jump_max_deg=float(np.degrees(np.abs(np.diff(x[:, :5], axis=0))).max()),
            flips_per_s=float(flips[:, :5].sum() / (len(x) * dt)),
        )
    return out
