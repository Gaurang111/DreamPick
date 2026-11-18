"""Camera -> arm frame correction (phase 1). `scripts/phase1_calibrate.py --fit` writes outputs/phase1/camera_to_arm.json from cubes
the real arm gripped by hand; this applies it to what the overhead camera reports.

The camera's picture (locate_cube.py with the arm origin at the mat's edge, BASE 0/0) is in the mat's frame; the policy works in the
simulator's frame, in which the real arm's joint angles (set by hand at calibration) put the gripper a few cm and a few degrees
away from where the real gripper really is. The map is fitted on grasps, so a cube seen at camera position p is handed to the policy
at the sim-frame position where the real arm's joints close on it.
"""
import json
import pathlib

import numpy as np

MAP_FILE = pathlib.Path(__file__).resolve().parents[2] / "outputs" / "phase1" / "camera_to_arm.json"
_cache = {}


def load(path=MAP_FILE):
    key = str(path)
    if key not in _cache:
        p = pathlib.Path(path)
        _cache[key] = json.loads(p.read_text()) if p.exists() else None
    return _cache[key]


def available(path=MAP_FILE):
    return load(path) is not None


def to_arm_frame(x_cm, y_cm, yaw_deg, path=MAP_FILE):
    """Camera (x out, y left, yaw) with the camera's arm origin at 0/0 -> the policy's frame. Unchanged if no map is saved."""
    m = load(path)
    if m is None:
        return float(x_cm), float(y_cm), float(yaw_deg)
    p = np.array(m["A"]) @ np.array([x_cm, y_cm]) + np.array(m["b"])
    yaw = (yaw_deg + m.get("yaw_offset_deg", 0.0) + 45.0) % 90.0 - 45.0         # the square repeats every 90 deg
    return float(p[0]), float(p[1]), float(yaw)
