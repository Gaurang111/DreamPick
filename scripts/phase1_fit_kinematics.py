"""Phase 1b: calibrate the real arm's joint zeros AND the mat-to-arm frame together, from the hand grasps in outputs/phase1/samples_v2.jsonl.

A grasped cube sits at a fixed spot in the FIXED jaw pad's own frame (grasp_in_fixed_pad.json, from scripts/grasp_frame_stats.py).
So for each grasp:   forward_kinematics(real joints - offsets).fixed_pad_frame * that spot   must equal   the cube the camera saw, moved
into the arm's frame (a rigid 2D move: rotation psi + shift t) at the cube's height (15 mm).  Unknowns: offsets of lift, elbow, wrist_flex,
wrist_roll (deg), psi, tx, ty. (pan's offset is the same thing as psi, so pan keeps 0.)  Levenberg-Marquardt with a weak prior on the offsets
(sigma 4 deg).  Reports fit and leave-one-out errors in mm, in 3D, and writes outputs/phase1/kinematics.json.

    .venv\\Scripts\\python.exe scripts\\phase1_fit_kinematics.py
"""
import json
import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import mujoco  # noqa: E402
from so101_env.state_pick_env import SO101StatePickEnv  # noqa: E402

OUT = ROOT / "outputs" / "phase1"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
FIT_JOINTS = [1, 2, 3, 4]                       # lift, elbow, wrist_flex, wrist_roll (pan's offset == frame rotation psi)
PRIOR_DEG = 12.0                                  # weak: only picks the smallest offsets among combinations the data cannot tell apart
CUBE_Z = 0.015
CUBE_YAW_NOTE = "psi also corrects the camera's cube yaw"

env = SO101StatePickEnv("lift", "easy")
env.reset(seed=0)
V = np.array(json.loads((OUT / "grasp_in_fixed_pad.json").read_text())["v_local_m"])


def cube_from_q(q_real_deg, offsets_deg):
    """FK: where the cube is when held, given the real joint readings (deg, first 5) and offsets (deg, length 5)."""
    q = np.radians(np.asarray(q_real_deg[:5], float) - np.asarray(offsets_deg, float))
    env.data.qpos[env._adr[:5]] = np.clip(q, env._low[:5], env._high[:5])
    env.data.qpos[env._adr[5]] = 0.3
    mujoco.mj_forward(env.model, env.data)
    R = env.data.xmat[env._fixed_pad].reshape(3, 3)
    return env.data.xpos[env._fixed_pad] + R @ V


def unpack(p):
    off = np.zeros(5)
    off[FIT_JOINTS] = p[:4]
    return off, math.radians(p[4]), p[5:7] / 100.0        # offsets deg, psi rad, t in m (p holds cm)


def residuals(p, rows):
    off, psi, t = unpack(p)
    c, s = math.cos(psi), math.sin(psi)
    out = []
    for r in rows:
        cam = np.array([r["cam_x_cm"], r["cam_y_cm"]]) / 100.0
        obs = np.array([c * cam[0] - s * cam[1] + t[0], s * cam[0] + c * cam[1] + t[1], CUBE_Z])
        out.extend((cube_from_q(r["q_real"], off) - obs) * 1000.0)            # mm
    out.extend(p[:4] / PRIOR_DEG * 5.0)                                         # weak prior on the offsets (5 mm per sigma)
    return np.array(out)


def lm(rows, p0=None, iters=60):
    p = np.zeros(7) if p0 is None else p0.copy()
    lam = 1e-2
    f = residuals(p, rows)
    for _ in range(iters):
        J = np.empty((len(f), len(p)))
        for k in range(len(p)):
            d = np.zeros(len(p)); d[k] = 1e-3
            J[:, k] = (residuals(p + d, rows) - f) / 1e-3
        step = np.linalg.solve(J.T @ J + lam * np.diag(np.diag(J.T @ J) + 1e-9), -J.T @ f)
        fn = residuals(p + step, rows)
        if fn @ fn < f @ f:
            p, f, lam = p + step, fn, lam * 0.5
            if np.abs(step).max() < 1e-5:
                break
        else:
            lam *= 4.0
    return p


def predict_error(p, r):
    off, psi, t = unpack(p)
    c, s = math.cos(psi), math.sin(psi)
    cam = np.array([r["cam_x_cm"], r["cam_y_cm"]]) / 100.0
    obs = np.array([c * cam[0] - s * cam[1] + t[0], s * cam[0] + c * cam[1] + t[1], CUBE_Z])
    return (cube_from_q(r["q_real"], off) - obs) * 1000.0


def main():
    f = OUT / "samples_v2.jsonl"
    skip = set(sys.argv[1:])                                    # tags to leave out, e.g.  s1 s7
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.exists() else []
    rows = [r for r in rows if "q_real" in r and r["tag"] not in skip]
    if len(rows) < 6:
        raise SystemExit(f"need at least 6 grasps with joint readings, have {len(rows)}")
    p = lm(rows)
    errs = np.array([predict_error(p, r) for r in rows])
    print(f"{len(rows)} grasps")
    off, psi, t = unpack(p)
    print("joint offsets (deg; sim angle = real reading - offset):", {j: round(float(o), 2) for j, o in zip(JOINTS, off)})
    print(f"mat -> arm frame: rotate {math.degrees(psi):+.2f} deg, shift ({t[0] * 100:+.2f}, {t[1] * 100:+.2f}) cm")
    print(f"fit error  : horizontal mean {np.hypot(errs[:, 0], errs[:, 1]).mean():.1f} mm, worst {np.hypot(errs[:, 0], errs[:, 1]).max():.1f} mm | height mean |dz| {np.abs(errs[:, 2]).mean():.1f} mm, worst {np.abs(errs[:, 2]).max():.1f} mm")
    # leave-one-out
    loo = []
    for i in range(len(rows)):
        keep = [r for j, r in enumerate(rows) if j != i]
        pi = lm(keep, p)
        loo.append(predict_error(pi, rows[i]))
    loo = np.array(loo)
    print(f"unseen spots: horizontal mean {np.hypot(loo[:, 0], loo[:, 1]).mean():.1f} mm, worst {np.hypot(loo[:, 0], loo[:, 1]).max():.1f} mm | height mean |dz| {np.abs(loo[:, 2]).mean():.1f} mm, worst {np.abs(loo[:, 2]).max():.1f} mm")
    for r, e in zip(rows, errs):
        print(f"   {r['tag']:<4} camera ({r['cam_x_cm']:5.1f},{r['cam_y_cm']:+5.1f})  off by dx {e[0]:+5.1f} dy {e[1]:+5.1f} dz {e[2]:+5.1f} mm")
    out = dict(joint_offset_deg={j: float(o) for j, o in zip(JOINTS, off)}, psi_deg=math.degrees(psi), t_cm=[float(t[0] * 100), float(t[1] * 100)],
               fit_horizontal_mean_mm=float(np.hypot(errs[:, 0], errs[:, 1]).mean()), unseen_horizontal_mean_mm=float(np.hypot(loo[:, 0], loo[:, 1]).mean()),
               unseen_height_mean_mm=float(np.abs(loo[:, 2]).mean()), n=len(rows), note=CUBE_YAW_NOTE)
    (OUT / "kinematics.json").write_text(json.dumps(out, indent=1))
    print("saved", OUT / "kinematics.json")
    # the camera -> arm map is now a plain rotation + shift (the joint offsets absorb the rest); keep the earlier affine map as a backup
    old = OUT / "camera_to_arm.json"
    if old.exists() and not (OUT / "camera_to_arm.affine_v1.json").exists():
        (OUT / "camera_to_arm.affine_v1.json").write_text(old.read_text())
    c, s = math.cos(psi), math.sin(psi)
    old.write_text(json.dumps(dict(model="rigid+joint_offsets", A=[[c, -s], [s, c]], b=[float(t[0] * 100), float(t[1] * 100)], yaw_offset_deg=math.degrees(psi),
                                   samples=len(rows), fit_mean_mm=out["fit_horizontal_mean_mm"], predict_unseen_mean_mm=out["unseen_horizontal_mean_mm"]), indent=1))
    print("saved", old, "(old affine map kept as camera_to_arm.affine_v1.json)")


if __name__ == "__main__":
    main()
