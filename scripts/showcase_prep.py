"""Showcase video prep, step 1 (LeRobot venv): from a recorded run folder, find
  - the last N seconds of the cube-placement clip as indices into the full-resolution top_camera.mp4, with the camera's cube detections for each frame,
  - for every step of the replay, which full-resolution top-camera frame matches the frame stored in sim_to_real.mp4 (the recorder's timestamps are not kept,
    so the match is made by comparing small grey pictures, moving forward only).
Writes outputs/showcase/<run>/prep.json.

    ...\\lerobot\\venv\\Scripts\\python.exe scripts\\showcase_prep.py run_20261115_212001 [--last 6]
"""
import argparse
import json
import pathlib
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import locate_cube as lc  # noqa: E402
from so101_env import camera_map  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--last", type=float, default=6.0, help="seconds of the placement clip to keep (from its end)")
ap.add_argument("--window", type=int, nargs=2, metavar=("FIRST", "END"), help="use these camera-frame indices for the placement clip instead")
args = ap.parse_args()
RUN = ROOT / "outputs" / "real_live" / args.run
OUT = ROOT / "outputs" / "showcase" / args.run
OUT.mkdir(parents=True, exist_ok=True)

# ---- placement clip: the last `last` seconds of real_to_sim.mp4 (its frames are the first frames of top_camera.mp4, in order)
r2s = cv2.VideoCapture(str(RUN / "real_to_sim.mp4"))
n_r2s, fps_r2s = int(r2s.get(cv2.CAP_PROP_FRAME_COUNT)), r2s.get(cv2.CAP_PROP_FPS)
r2s.release()
i1 = n_r2s
i0 = max(0, n_r2s - int(round(args.last * fps_r2s)))
if args.window:
    i0, i1 = args.window
top = cv2.VideoCapture(str(RUN / "top_camera.mp4"))
fl, _ = lc.get_flattener(cv2.imread(str(ROOT / "outputs" / "phase1" / "top_cube_raw.jpg")))
dets = []
top.set(cv2.CAP_PROP_POS_FRAMES, i0)
for i in range(i0, i1):
    ok, f = top.read()
    if not ok:
        break
    res, _ = lc.analyse(f.copy(), 0.0, 0.0, fl)
    pose = camera_map.to_arm_frame(res["x_cm"], res["y_cm"], res["yaw_deg"]) if res else None
    dets.append(dict(frame=i, pose=pose))
    print(f"placement frame {i}: {'%.1f, %+.1f, yaw %+.0f' % pose if pose else 'no cube'}", flush=True)
found = [d["pose"] for d in dets[-10:] if d["pose"]]
lock = [float(np.median([p[0] for p in found])), float(np.median([p[1] for p in found])),
        float(np.degrees(np.angle(np.mean(np.exp(1j * 4 * np.radians([p[2] for p in found]))))) / 4.0)] if found else None
print("RESULT locked cube (policy frame):", lock)

# ---- replay alignment: sim_to_real.mp4's left half (768x432 real camera) vs the full-resolution frames of top_camera.mp4
def small(img):
    g = cv2.cvtColor(cv2.resize(img, (192, 108)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    g[:12] = 0                                                # the text label at the top of the stored composite
    return g


s2r = cv2.VideoCapture(str(RUN / "sim_to_real.mp4"))
refs = []
while True:
    ok, f = s2r.read()
    if not ok:
        break
    refs.append(small(f[:, :768]))
s2r.release()
n_top = int(top.get(cv2.CAP_PROP_FRAME_COUNT))
top.set(cv2.CAP_PROP_POS_FRAMES, 0)
tops = []
while True:
    ok, f = top.read()
    if not ok:
        break
    tops.append(small(f))
top.release()
tops = np.array(tops)
print(f"{len(refs)} replay frames, {len(tops)} camera frames", flush=True)
# first frame: global search over the camera frames after the placement clip; then only forward, at most 8 frames per step
start = max(i1, n_r2s)                                        # the replay comes after the whole placement/planning part of the recording
cost0 = np.abs(tops[start:] - refs[0]).mean(axis=(1, 2))
idx = [start + int(np.argmin(cost0))]
for k in range(1, len(refs)):
    lo, hi = idx[-1], min(len(tops), idx[-1] + 9)
    c = np.abs(tops[lo:hi] - refs[k]).mean(axis=(1, 2))
    idx.append(lo + int(np.argmin(c)))
idx = np.array(idx)
print("RESULT alignment: first camera frame", idx[0], "last", idx[-1], "| steps per camera frame (median)", float(np.median(np.diff(idx))))
(OUT / "prep.json").write_text(json.dumps(dict(run=args.run, placement=dict(first=i0, last=i1 - 1, fps=fps_r2s, detections=dets, lock=lock),
                                              align=[int(v) for v in idx], n_top=len(tops))))
print("saved", OUT / "prep.json")
