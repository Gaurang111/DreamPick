"""Usable joint ranges implied by a LeRobot calibration file (what sim_real_bridge.Bridge will allow), next to the ranges the
policy was trained with (so101_env.pick_cube_env._REAL_ARM_LIMITS_DEG). Read-only, no hardware.

    C:\\Users\\goran\\PycharmProjects\\so101\\lerobot\\venv\\Scripts\\python.exe scripts\\calib_report.py [--calib_dir DIR --id my_follower]
"""
import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from sim_real_bridge import JOINTS, Bridge  # noqa: E402

TRAINED = {"shoulder_pan": (-109.9, 109.9), "shoulder_lift": (-101.8, 79.1), "elbow_flex": (-93.7, 93.7), "wrist_flex": (-98.0, 65.9)}

ap = argparse.ArgumentParser()
ap.add_argument("--id", default="my_follower")
args = ap.parse_args()
b = Bridge("COM3", args.id, 3.0)                      # builds the bus object only; nothing is opened
print(f"calibration id '{args.id}'  (degrees, 0 = middle of the calibrated range)")
print(f"{'joint':<14}{'calibrated':>20}{'encoder window':>20}{'usable now':>20}{'policy trained on':>22}")
for j in JOINTS:
    c = b.cal[j]
    cal_lo, cal_hi = b.ticks_to_real(j, c.range_min), b.ticks_to_real(j, c.range_max)
    w_lo, w_hi = b.encoder_window(j)
    win = (b.ticks_to_real(j, w_lo), b.ticks_to_real(j, w_hi))
    tr = TRAINED.get(j)
    note = ""
    if tr and (b.real_lo[j] > tr[0] + 0.5 or b.real_hi[j] < tr[1] - 0.5):
        note = "  <- narrower than trained"
    print(f"{j:<14}{cal_lo:>9.1f}..{cal_hi:<8.1f}{win[0]:>9.1f}..{win[1]:<8.1f}{b.real_lo[j]:>9.1f}..{b.real_hi[j]:<8.1f}"
          f"{(f'{tr[0]:.1f}..{tr[1]:.1f}' if tr else '-'):>22}{note}")
