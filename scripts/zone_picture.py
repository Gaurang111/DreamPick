"""One picture from the overhead camera with the cube zone drawn on the real mat (and the cube's position if it is there).
Zone = what the policy was trained on: 21-31 cm straight out from the arm's rotation axis, 10 cm to each side.

    ...\\lerobot\\venv\\Scripts\\python.exe scripts\\zone_picture.py [--out outputs/phase1/zone.jpg]
"""
import argparse
import pathlib
import sys
import time

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import locate_cube as lc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(ROOT / "outputs" / "phase1" / "zone.jpg"))
args = ap.parse_args()
cap = cv2.VideoCapture(lc.find_camera(), cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
for _ in range(25):
    cap.read()
    time.sleep(0.03)
ok, img = cap.read()
cap.release()
if not ok:
    raise SystemExit("could not read the camera")
fl, _ = lc.get_flattener(img)


sys.path.insert(0, str(ROOT / "src"))
from so101_env import camera_map  # noqa: E402

USE_MAP = camera_map.available()
if USE_MAP:                                               # draw the policy's zone where it really is on the mat
    _m = camera_map.load()
    _A, _b = np.array(_m["A"]), np.array(_m["b"])


def to_raw(x_cm, y_cm):
    """Policy-frame position (cm out, cm left) -> pixel in the camera's own picture (lens distortion included)."""
    if USE_MAP:
        x_cm, y_cm = np.linalg.solve(_A, np.array([x_cm, y_cm]) - _b)      # back to the camera's mat frame (arm origin 0/0)
        base_x, base_y = 0.0, 0.0
    else:
        base_x, base_y = lc.BASE_X_CM, lc.BASE_Y_CM
    px = lc.RECT_W / 2 - (y_cm + base_y) * lc.PX_PER_CM
    py = lc.RECT_H - (x_cm + base_x) * lc.PX_PER_CM
    px, py = int(np.clip(px, 0, lc.RECT_W - 1)), int(np.clip(py, 0, lc.RECT_H - 1))
    return int(fl.map_x[py, px]), int(fl.map_y[py, px])


def poly(pts, color, thick=3):
    cv2.polylines(img, [np.array(pts, np.int32)], True, color, thick)


xs, ys = lc.ZONE["x"], lc.ZONE["y"]
edge = []
for t in np.linspace(0, 1, 30):
    edge.append(to_raw(xs[0], ys[0] + t * (ys[1] - ys[0])))
for t in np.linspace(0, 1, 30):
    edge.append(to_raw(xs[0] + t * (xs[1] - xs[0]), ys[1]))
for t in np.linspace(1, 0, 30):
    edge.append(to_raw(xs[1], ys[0] + t * (ys[1] - ys[0])))
for t in np.linspace(1, 0, 30):
    edge.append(to_raw(xs[0] + t * (xs[1] - xs[0]), ys[0]))
poly(edge, (255, 200, 0), 3)
for x in (21, 26, 31):                                    # cm marks along the middle line
    cv2.putText(img, f"{x} cm", (to_raw(x, 0)[0] + 6, to_raw(x, 0)[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
for y, name in ((-10, "10 cm right"), (10, "10 cm left")):
    p = to_raw(xs[0] - 2.2, y)
    cv2.putText(img, name, (p[0] - 40, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
cv2.line(img, to_raw(0.5, 0), to_raw(34, 0), (0, 255, 255), 1)
res, _ = lc.analyse(img.copy(), 0.0 if USE_MAP else lc.BASE_X_CM, 0.0 if USE_MAP else lc.BASE_Y_CM, fl)
if res:
    cx, cy, _ = camera_map.to_arm_frame(res["x_cm"], res["y_cm"], 0.0) if USE_MAP else (res["x_cm"], res["y_cm"], 0.0)
    inside = xs[0] <= cx <= xs[1] and ys[0] <= cy <= ys[1]
    cv2.putText(img, f"cube (policy frame): {cx:.1f} out, {cy:+.1f} left - {'INSIDE zone' if inside else 'OUTSIDE zone'}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if inside else (0, 0, 255), 3)
    res = dict(x_cm=cx, y_cm=cy)
cv2.imwrite(args.out, img)
print("RESULT saved", args.out, "| cube:", f"{res['x_cm']:.1f}, {res['y_cm']:+.1f}" if res else "none seen")
