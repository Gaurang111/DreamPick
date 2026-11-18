"""Episode metrics shared by the scripted solver, the training gates and the final acceptance test.

Success = the cube is held at >= 5 cm for at least HOLD_STEPS consecutive steps (3 s at 20 Hz) and does not move more than
3 cm sideways while it is held. "Still held at the end" is reported separately: in the simulator a pinched cube creeps
slowly between the pads (about 1 mm/s) and can drop after several seconds, which says little about the policy.
"""
import numpy as np

HOLD_STEPS = 60
SHIFT_LIMIT = 0.05
LIFT_HEIGHT = 0.05


def episode_metrics(heights, shifts, table_forces=None, joint_pos=None, dt=0.05):
    h, sh = np.asarray(heights), np.asarray(shifts)
    up = h >= LIFT_HEIGHT
    best, run, best_end = 0, 0, -1
    for i, x in enumerate(up):
        run = run + 1 if x else 0
        if run > best:
            best, best_end = run, i
    held_shift = float(sh[best_end - best + 1:best_end + 1].max()) if best > 0 else float(sh.max())
    out = {
        "success": bool(best >= HOLD_STEPS and held_shift <= SHIFT_LIMIT),
        "reached": bool(up.any()),
        "held_steps": int(best),
        "held_at_end": bool(up[-1]),
        "peak_cm": float(h.max() * 100),
        "shift_while_held_cm": held_shift * 100,
        "final_shift_cm": float(sh[-1] * 100),
        "time_up_share": float(up.mean()),
    }
    if table_forces is not None:
        out["table_force"] = float(np.max(table_forces))
    if joint_pos is not None:
        q = np.asarray(joint_pos)[:, :5]
        speed = np.diff(q, axis=0) / dt
        accel = np.diff(speed, axis=0) / dt
        out["peak_accel"] = float(np.abs(accel).max()) if len(accel) else 0.0
        if best >= 2:                                    # tremor while holding: direction flips of the arm joints
            seg = np.diff(q[best_end - best + 1:best_end + 1], axis=0)
            flips = np.sign(seg[1:]) != np.sign(seg[:-1])
            out["hold_flips"] = float(flips.mean()) if flips.size else 0.0
        else:
            out["hold_flips"] = float("nan")
    return out
