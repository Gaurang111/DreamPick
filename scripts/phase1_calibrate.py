"""Phase 1: make the camera's cube position match the arm's (= the simulator's) frame, using the arm itself as the ruler.

For each cube spot:  (1) the camera measures the cube; (2) you close the gripper by hand around the cube (torque OFF the whole
time, nothing moves by itself) and the real joint angles go through the simulator's forward kinematics to give where the
gripper's centre is in the sim frame, which is where the policy believes a grasped cube is.

Run in the LeRobot venv from the project root:
    ...\\lerobot\\venv\\Scripts\\python.exe scripts\\phase1_calibrate.py            collect samples (appends to outputs/phase1/samples.jsonl)
    ...\\lerobot\\venv\\Scripts\\python.exe scripts\\phase1_calibrate.py --fit      fit the camera-to-arm map from the samples
Suggested spots (cm out, cm left of the arm's axis, roughly; the exact spot does not matter, only that they spread over the zone):
    (22,-8) (22,0) (22,8) (26,-9) (26,9) (30,-8) (30,0) (30,8)   -- and turn the cube a different way each time.
"""
import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "outputs" / "phase1"
SAMPLES = OUT / "samples_v2.jsonl"          # v2 also stores the joint readings and the pad height (v1: samples.jsonl, positions only)
MAP_FILE = OUT / "camera_to_arm.json"


def load_samples():
    return [json.loads(l) for l in SAMPLES.read_text().splitlines() if l.strip()] if SAMPLES.exists() else []


# ------------------------------------------------------------------ fitting

def fit_models(cam, arm):
    """cam, arm: (n, 2) cm. Returns {name: (apply_fn_params, residuals)} for translation / rigid / similarity / affine."""
    n = len(cam)
    out = {}
    t = (arm - cam).mean(axis=0)
    out["translation"] = dict(A=np.eye(2).tolist(), b=t.tolist())
    cc, ac = cam.mean(0), arm.mean(0)
    H = (cam - cc).T @ (arm - ac)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, d]) @ U.T
    out["rigid"] = dict(A=R.tolist(), b=(ac - R @ cc).tolist())
    s = S.sum() / ((cam - cc) ** 2).sum()
    out["similarity"] = dict(A=(s * R).tolist(), b=(ac - s * R @ cc).tolist())
    X = np.hstack([cam, np.ones((n, 1))])
    sol, *_ = np.linalg.lstsq(X, arm, rcond=None)
    out["affine"] = dict(A=sol[:2].T.tolist(), b=sol[2].tolist())
    return out


def apply_map(m, pts):
    return np.asarray(pts) @ np.array(m["A"]).T + np.array(m["b"])


def fit(args):
    rows = load_samples()
    if len(rows) < 4:
        raise SystemExit(f"need at least 4 samples, have {len(rows)}")
    cam = np.array([[r["cam_x_cm"], r["cam_y_cm"]] for r in rows])
    arm = np.array([[r["arm_x_cm"], r["arm_y_cm"]] for r in rows])
    raw = np.linalg.norm(arm - cam, axis=1)
    print(f"{len(rows)} samples. Camera vs arm with no correction: mean {raw.mean() * 10:.1f} mm, worst {raw.max() * 10:.1f} mm")
    models = fit_models(cam, arm)
    best, best_loo = None, 1e9
    for name, m in models.items():
        res = np.linalg.norm(apply_map(m, cam) - arm, axis=1)
        loo = []
        for i in range(len(rows)):                           # leave one spot out: how well does the map predict a spot it never saw?
            keep = np.arange(len(rows)) != i
            if keep.sum() < (2 if name == "translation" else 3 if name in ("rigid", "similarity") else 4):
                continue
            mi = fit_models(cam[keep], arm[keep])[name]
            loo.append(np.linalg.norm(apply_map(mi, cam[i:i + 1]) - arm[i:i + 1], axis=1)[0])
        loo = np.array(loo) if loo else res
        print(f"  {name:<12} fit error: mean {res.mean() * 10:5.1f} mm  worst {res.max() * 10:5.1f} mm | predicting unseen spots: mean {loo.mean() * 10:5.1f} mm  worst {loo.max() * 10:5.1f} mm")
        if loo.mean() < best_loo - 0.02:                      # prefer the simpler model unless the richer one clearly predicts better
            best, best_loo = name, loo.mean()
    m = models[best]
    res = np.linalg.norm(apply_map(m, cam) - arm, axis=1)
    print(f"using '{best}': A = {np.round(m['A'], 4).tolist()}, b = {np.round(m['b'], 3).tolist()} cm")
    for r, e, c in zip(rows, res, apply_map(m, cam)):
        print(f"    spot {r['tag']:<8} camera ({r['cam_x_cm']:5.1f},{r['cam_y_cm']:+5.1f}) -> mapped ({c[0]:5.1f},{c[1]:+5.1f})  arm says ({r['arm_x_cm']:5.1f},{r['arm_y_cm']:+5.1f})   off by {e * 10:4.1f} mm")
    # yaw: the closing axis of the gripper against the camera's cube yaw (the cube repeats every 90 deg)
    d = [((r["grip_yaw_deg"] - r["cam_yaw_deg"] + 45) % 90) - 45 for r in rows if "grip_yaw_deg" in r]
    if d:
        circ = np.degrees(np.angle(np.mean(np.exp(1j * 4 * np.radians(d))))) / 4.0
        print(f"yaw: gripper minus camera = {circ:+.1f} deg on average (spread {np.std(d):.1f} deg)")
    MAP_FILE.write_text(json.dumps(dict(model=best, A=m["A"], b=m["b"], samples=len(rows), fit_mean_mm=float(res.mean() * 10),
                                       predict_unseen_mean_mm=float(best_loo * 10), yaw_offset_deg=float(circ) if d else 0.0), indent=1))
    print("saved", MAP_FILE)


# ------------------------------------------------------------------ collecting

def collect(args):
    import msvcrt
    import mujoco
    import run3_real_client as rc
    from sim_real_bridge import JOINTS, Bridge
    from so101_env.state_pick_env import SO101StatePickEnv

    env = SO101StatePickEnv("lift", "easy")                # only for the arm's forward kinematics
    bridge = Bridge("COM3", "my_follower", 3.0)
    OUT.mkdir(parents=True, exist_ok=True)

    def fk():
        real = bridge.read_real()
        for j, a in zip(JOINTS, env._adr):
            env.data.qpos[a] = bridge.real_to_sim(j, real[j])
        mujoco.mj_forward(env.model, env.data)
        mid = env.pad_mid()
        d = env.data.xpos[env._moving_pad] - env.data.xpos[env._fixed_pad]
        fk.last_real = dict(real)
        fk.last_z = float(mid[2] * 100)
        return mid[0] * 100, mid[1] * 100, math.degrees(math.atan2(d[1], d[0])), real["gripper"]

    bridge.connect()                                       # torque stays OFF; mirror-style read access
    n0 = len(load_samples())
    try:
        k = n0
        while True:
            k += 1
            tag = f"s{k}"
            ans = input(f"\n[{tag}] Put the cube down at a new spot in the zone (any spot, turn it a different way each time), arm out of the way,\n"
                        f"      then press ENTER to measure it with the camera (type q + ENTER to finish): ")
            if ans.strip().lower().startswith("q"):
                break
            try:
                cx, cy, cyaw = rc.measure_cube(base_x=0.0, base_y=0.0)
            except Exception as e:
                print("   camera:", e, "- try again")
                k -= 1
                continue
            print(f"   camera says: {cx:.2f} cm out, {cy:+.2f} cm left, yaw {cyaw:+.1f} deg")
            print("   Now close the gripper BY HAND around the cube (jaws squeezing it, cube centred between the pads), hold the arm still.")
            print("   Live gripper centre below. Press ENTER when it is steady and gripping; ESC skips this spot.")
            skip = False
            while True:
                x, y, yaw, g = fk()
                print(f"     gripper centre {x:6.2f} cm out, {y:+6.2f} cm left | jaw {g:5.1f}%      ", end="\r", flush=True)
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if ch in ("\r", "\n"):
                        break
                    if ch == "\x1b":
                        skip = True
                        break
                time.sleep(0.08)
            print()
            if skip:
                k -= 1
                continue
            reads, qs, zs = [], [], []
            for _ in range(12):                            # average a second of readings
                reads.append(fk())
                qs.append([fk.last_real[j] for j in JOINTS])
                zs.append(fk.last_z)
                time.sleep(0.08)
            a = np.array(reads)
            sd = a[:, :2].std(axis=0) * 10
            row = dict(tag=tag, cam_x_cm=cx, cam_y_cm=cy, cam_yaw_deg=cyaw, arm_x_cm=float(a[:, 0].mean()), arm_y_cm=float(a[:, 1].mean()),
                       arm_z_cm=float(np.mean(zs)), q_real=[float(v) for v in np.mean(qs, axis=0)],       # joint readings in real units (deg; gripper %)
                       grip_yaw_deg=float(a[:, 2].mean()), jaw_pct=float(a[:, 3].mean()), arm_std_mm=[float(sd[0]), float(sd[1])], t=time.time())
            with open(SAMPLES, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"   saved {tag}: camera ({cx:.1f},{cy:+.1f}) vs arm ({row['arm_x_cm']:.1f},{row['arm_y_cm']:+.1f}) -> differ by "
                  f"{math.hypot(row['arm_x_cm'] - cx, row['arm_y_cm'] - cy) * 10:.1f} mm   (arm reading steadiness {sd[0]:.1f}/{sd[1]:.1f} mm)")
            print("   Let go of the cube and the arm; the arm stays limp (torque off).")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        bridge.release()
    print(f"\n{len(load_samples())} samples saved in {SAMPLES}. Run again with --fit when you have 6 or more.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", action="store_true")
    args = ap.parse_args()
    fit(args) if args.fit else collect(args)
