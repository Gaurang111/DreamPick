"""One look through the overhead camera: saves the raw frame and the flattened, annotated top-down picture, and reports how
steady the cube's measured position is over N frames (repeatability, before any accuracy question).

    C:\\Users\\goran\\PycharmProjects\\so101\\lerobot\\venv\\Scripts\\python.exe scripts\\camera_snapshot.py [--frames 30] [--tag spot1]
Output: outputs/phase1/<tag>_raw.jpg, <tag>_flat.jpg and one line in outputs/phase1/readings.jsonl
"""
import argparse
import json
import pathlib
import sys
import time

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import locate_cube as lc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--frames", type=int, default=30)
ap.add_argument("--tag", default="snap")
args = ap.parse_args()
out = ROOT / "outputs" / "phase1"
out.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(lc.find_camera(), cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
for _ in range(25):
    cap.read()
    time.sleep(0.03)
ok, img = cap.read()
if not ok:
    raise SystemExit("could not read the camera")
flattener, model = lc.get_flattener(img)
poses, shown_last, n = [], None, 0
while n < args.frames:
    ok, frame = cap.read()
    if not ok:
        continue
    n += 1
    res, shown = lc.analyse(frame, lc.BASE_X_CM, lc.BASE_Y_CM, flattener)
    shown_last = shown
    if res:
        poses.append((res["x_cm"], res["y_cm"], res["yaw_deg"], *res["pixel"]))
cap.release()
cv2.imwrite(str(out / f"{args.tag}_raw.jpg"), img)
cv2.imwrite(str(out / f"{args.tag}_flat.jpg"), cv2.resize(shown_last, None, fx=0.6, fy=0.6))
if len(poses) < 0.5 * args.frames:
    print(f"RESULT {args.tag}: cube seen in only {len(poses)} of {args.frames} frames")
else:
    a = np.array(poses)
    yaw = np.degrees(np.angle(np.mean(np.exp(1j * 4 * np.radians(a[:, 2]))))) / 4.0
    row = dict(tag=args.tag, frames=len(poses), x_cm=float(np.median(a[:, 0])), y_cm=float(np.median(a[:, 1])), yaw_deg=float(yaw),
               std_x_mm=float(a[:, 0].std() * 10), std_y_mm=float(a[:, 1].std() * 10), px=[float(np.median(a[:, 3])), float(np.median(a[:, 4]))],
               k1=model["k1"], k2=model["k2"])
    with open(out / "readings.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"RESULT {args.tag}: x {row['x_cm']:.2f} cm out, y {row['y_cm']:+.2f} cm left, yaw {yaw:+.1f} deg | "
          f"steadiness over {len(poses)} frames: x {row['std_x_mm']:.2f} mm, y {row['std_y_mm']:.2f} mm (1 sigma)")
