"""Fit the static sag of the gravity-loaded joints from follow runs: (real - goal), lag-aligned, regressed on the gravity load at the goal pose.
The error is measured with the previous sag compensation already switched on, so the NEW compensation is  k_new = k_prev - slope.

    ...\\lerobot\\venv\\Scripts\\python.exe scripts\\fit_sag.py --lag 0 follow_A.json follow_B.json     (runs recorded with --lead 4: lag 0)
    ...                                                  --lag 4  (older runs without the lead)
Writes outputs/real_live/sag_k.json (the old file is kept as sag_k.prev.json) unless --dry.
"""
import argparse
import glob
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from sim_real_bridge import JOINTS, SAG_JOINTS, SAG_K_MAX, Bridge  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="*")
ap.add_argument("--lag", type=int, default=0, help="steps the servos trail their goal in these runs (0 with the 4-step lead, 4 without)")
ap.add_argument("--from_step", type=int, default=15)
ap.add_argument("--to_step", type=int, default=140, help="stop before the carry phase (the wrist hits its limit there)")
ap.add_argument("--dry", action="store_true")
args = ap.parse_args()
OUT = ROOT / "outputs" / "real_live"
files = args.files or sorted(glob.glob(str(OUT / "follow_*.json")))[-2:]
prev = json.loads((OUT / "sag_k.json").read_text()) if (OUT / "sag_k.json").exists() else {j: 0.0 for j in SAG_JOINTS}
b = Bridge("COM3", "my_follower", 3.0)     # model only, nothing is opened
num = {j: 0.0 for j in SAG_JOINTS}
den = {j: 0.0 for j in SAG_JOINTS}
mean_err = {j: [] for j in SAG_JOINTS}
for f in files:
    f = pathlib.Path(f) if pathlib.Path(f).is_absolute() or pathlib.Path(f).exists() else OUT / f
    fo = json.loads(f.read_text())
    plan = json.loads((OUT / fo["plan"]).read_text())
    goals, real = np.array(plan["goals"]), np.array(fo["real_q"])
    n = min(len(real), len(goals) - 1, args.to_step)
    for k in range(max(args.from_step, args.lag), n):
        g_real = {j: b.sim_to_real(j, goals[k - args.lag + 1][i]) for i, j in enumerate(JOINTS)}
        load = b.gravity_load(g_real)
        r_real = {j: b.sim_to_real(j, real[k][i]) for i, j in enumerate(JOINTS)}
        for j in SAG_JOINTS:
            e = r_real[j] - g_real[j]
            mean_err[j].append(e)
            if abs(load[j]) > 0.05:
                num[j] += e * load[j]
                den[j] += load[j] ** 2
slope = {j: (num[j] / den[j] if den[j] else 0.0) for j in SAG_JOINTS}
new = {j: float(np.clip(prev.get(j, 0.0) - slope[j], -SAG_K_MAX, SAG_K_MAX)) for j in SAG_JOINTS}
print(f"RESULT {len(files)} run(s), steps {args.from_step}-{args.to_step}, lag {args.lag}")
for j in SAG_JOINTS:
    print(f"RESULT  {j:<14} mean (real - goal) {np.mean(mean_err[j]):+5.2f} deg | slope {slope[j]:+5.2f} deg/Nm | k {prev.get(j, 0.0):+5.2f} -> {new[j]:+5.2f}")
if not args.dry:
    if (OUT / "sag_k.json").exists():
        (OUT / "sag_k.prev.json").write_text((OUT / "sag_k.json").read_text())
    (OUT / "sag_k.json").write_text(json.dumps(new, indent=1))
    print("RESULT written", OUT / "sag_k.json")
